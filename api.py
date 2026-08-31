"""
MLOps/rag/api.py

RAG 五個主要功能，共用同一套編碼邏輯（同一個 SentenceTransformer 模型
實例、同一個 "passage: "/"query: " 前綴慣例）與同一個 sqlite-vec 資料庫：

  1. create_document() 新增全新文件——連檔案本身都還不存在，由這個函式
                        負責寫 frontmatter+內文到磁碟，寫完委派給
                        store_document() 編碼寫入索引。要求 source_path
                        指到的檔案還不存在（避免誤蓋既有內容），跟
                        edit_document() 的「要求資料庫裡已有紀錄」互補
  2. store_document()  索引一份已經存在磁碟上的文件——只傳 source_path，
                        函式自己去讀那個檔案、解析 frontmatter、編碼內文。
                        同一個 source_path 已存在就先刪除再重新寫入
                        （upsert 語意）。create_document()/edit_document()
                        都是委派到這裡做實際的編碼寫入
  3. search()           向量搜尋——給 query 做語意相似度 KNN 搜尋；
                        給 source_path 則是精確比對、不做向量運算
  4. edit_document()    編輯既有文件——內部依序呼叫 search(source_path=...)
                        確認文件存在，再呼叫 store_document() 重新讀檔、
                        重新編碼覆蓋。本質是「確認存在 + 重新讀檔存」，
                        不是獨立的第三套邏輯
  5. delete_document()  刪除既有文件的索引——只動資料庫（documents +
                        vec_chunks），不會刪磁碟上的檔案本身。跟
                        store_document() 內 upsert 用的「先刪舊資料」共用
                        同一段 _delete_by_id() 邏輯，不是各自複製一份

架構決策：SQL 只存 source_path 跟 metadata（title/tags/type/status），
**不存文本本身**。文本永遠只有檔案這一份真相（single source of truth），
「編輯」就是改檔案、重新讀取編碼——不會有資料庫裡的文字跟檔案內容兜不起來
的問題。search() 回傳結果時才即時讀檔案內容附上，方便呼叫端直接使用，
但那份文字不會被寫回資料庫。create_document() 雖然會寫檔案，但寫完之後
一樣是委派給 store_document() 做編碼寫入，SQL 裡存的仍然只有路徑+metadata，
不是文字本身，跟這個架構決策沒有衝突。

模型只在第一次呼叫時載入一次（模組層級快取），同一個 process 裡多次呼叫
這幾個函數不會重複載入模型。

CLI 用法：
    python rag/api.py create <source_path> --title <T> --tags a,b,c --type <T> --status <S>   # 內文從 stdin 讀入
    python rag/api.py store <source_path>
    python rag/api.py edit <source_path>
    python rag/api.py delete <source_path>
    python rag/api.py search "<查詢字串>" [--top-k N]
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import sqlite_vec
from sentence_transformers import SentenceTransformer

from frontmatter import parse_frontmatter, render_frontmatter

RAG_DIR    = Path(__file__).parent
MODEL_DIR  = RAG_DIR / "models" / "multilingual-e5-small"
MODEL_NAME = "intfloat/multilingual-e5-small"
DB_PATH    = RAG_DIR / "index" / "vectors.sqlite"

_model_cache: SentenceTransformer | None = None


def _load_model(model_dir: Path = MODEL_DIR) -> SentenceTransformer:
    global _model_cache
    if _model_cache is None:
        if model_dir.exists():
            _model_cache = SentenceTransformer(str(model_dir))
        else:
            print(f"[提示] 找不到本機模型 {model_dir}，改從 {MODEL_NAME} 下載並存檔...",
                  file=sys.stderr)
            model = SentenceTransformer(MODEL_NAME)
            model.save(str(model_dir))
            _model_cache = model
    return _model_cache


def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            embedding float[384]
        )
    """)
    # 注意：沒有 chunk_text 欄位——文本只存在檔案裡，資料庫只存路徑+metadata。
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY,
            source_path TEXT NOT NULL UNIQUE,
            title TEXT,
            tags TEXT,
            type TEXT,
            status TEXT
        )
    """)
    conn.commit()
    return conn


def _resolve_path(source_path: str, rag_dir: Path = RAG_DIR) -> Path:
    """把 source_path 轉成實際檔案路徑。絕對路徑（例如跨目錄索引外部檔案時
    存的絕對路徑）直接照用；相對路徑則沿用既有慣例，視為相對於 rag_dir。
    兩種形式共存於同一個資料庫裡，讓「專案內文件」跟「外部目錄索引進來的
    文件」可以並存，不用互相遷就成同一種路徑格式。"""
    p = Path(source_path)
    return p if p.is_absolute() else rag_dir / p


def _read_source(source_path: str, rag_dir: Path = RAG_DIR) -> tuple[dict, str]:
    """讀 source_path 指到的檔案，回傳 (frontmatter 欄位, 內文)。source_path
    可以是相對於 rag_dir 的路徑，也可以是絕對路徑（見 _resolve_path）。"""
    full_path = _resolve_path(source_path, rag_dir)
    if not full_path.exists():
        raise FileNotFoundError(f"找不到檔案：{full_path}")
    return parse_frontmatter(full_path.read_text(encoding="utf-8"))


def _delete_by_id(conn: sqlite3.Connection, doc_id: int) -> None:
    """刪除 documents/vec_chunks 裡 id=doc_id 的紀錄。不 commit、不 close——
    呼叫端決定何時 commit，讓 store_document() 的 upsert 跟 delete_document()
    可以共用這段邏輯，各自套自己的交易邊界。"""
    conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (doc_id,))
    conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


MEMORY_TYPE = "memory"  # frontmatter 的 type: memory 表示這是模型自己寫入的記憶文件


def _warn_if_too_long(source_path: str, passage: str, model: SentenceTransformer,
                       doc_type: str = "") -> None:
    """超過模型 max_seq_length 的內容會被 model.encode() 默默截斷、不會進
    向量（不會報錯，chunk_text/檔案內容本身不受影響，只有拿去算相似度的
    那個向量只看得到前面 limit 個 token）。只印警告、不擋寫入。

    警告文字依 doc_type 是否為 MEMORY_TYPE（frontmatter `type: memory`）分兩種：
    - 記憶文件（模型自己寫入用來記事的文件）：可以考慮拆成續篇文件，但要不
      要拆仍由模型臨場判斷後段內容是否豐富到值得獨立成篇。
    - 非記憶文件（原始資料、非模型為了記憶而產生的文件）：不可拆分/修改原
      始檔案本身——原檔案的完整性優先於索引方便；如果之後發現多次搜尋不到
      只存在後段的內容，可以另外新增一份記憶文件來輔助搜尋，但不動原檔案。
    """
    limit = getattr(model, "max_seq_length", 512)
    n_tokens = len(model.tokenizer.encode(passage))
    if n_tokens <= limit:
        return
    is_memory = doc_type.strip().lower() == MEMORY_TYPE
    base = (
        f"⚠️ 警告：{source_path} 編碼後有 {n_tokens} tokens，超過模型上限"
        f"（{limit}）。超過的部分會被模型默默截斷、不會進入向量，可能讓這份"
        f"文件在語意搜尋時漏掉只出現在後段的內容（讀取內容本身仍是完整全文，"
        f"只有搜尋用的向量受影響，較難被搜尋到）。\n"
    )
    if is_memory:
        base += (
            f"    這是記憶文件（type: {MEMORY_TYPE}），可以考慮把後段拆成一份"
            f"獨立的續篇文件——如果真的拆開：(1) 在這篇文件最後補上一段指向"
            f"續篇的連結（續篇的標題/tags/關鍵字）；(2) 續篇那份文件也要標明"
            f"自己是接續於這一篇（哪個 source_path/標題），不能只有單向連結；"
            f"(3) 幫續篇加上「續篇」相關的 tag（例如 continuation）。但要不要"
            f"拆完全取決於臨場判斷——關鍵不是超過多少，而是「後段/新增的內容"
            f"是否豐富到足以獨立成一個篇章」：內容單薄的話硬拆出一份檔案反而"
            f"奇怪，不拆、保留現狀也是合理選擇；只有後段本身夠豐富、站得住腳"
            f"當一個獨立主題時，才值得拆成續篇。"
        )
    else:
        base += (
            f"    這不是記憶文件（type 不是 {MEMORY_TYPE}），不可以拆分或修改"
            f"這份原始檔案本身——原檔案內容的完整性優先於索引方便，不要為了"
            f"塞進 token 上限而動使用者的原始資料。如果之後發現多次搜尋都找"
            f"不到只出現在後段的內容，可以另外新增一份 type: {MEMORY_TYPE} 的"
            f"記憶文件，摘要或指向那段內容來輔助搜尋，但原檔案維持不動。"
        )
    print(base, file=sys.stderr)


def _row_to_dict(row: tuple, distance: float | None, chunk_text: str | None) -> dict:
    doc_id, source_path, title, tags, doc_type, status = row
    return {
        "id": doc_id,
        "source_path": source_path,
        "title": title,
        "tags": json.loads(tags) if tags else [],
        "type": doc_type,
        "status": status,
        "chunk_text": chunk_text,  # 即時讀檔附上，不是資料庫裡存的
        "distance": distance,      # None：精確查找（source_path），不是向量搜尋結果
    }


# ── 0. 新增全新文件 ──────────────────────────────────────────────────────────

def create_document(source_path: str, body: str, *, title: str | None = None,
                     tags: list[str] | None = None, type: str = "", status: str = "",
                     rag_dir: Path = RAG_DIR, db_path: Path = DB_PATH,
                     model_dir: Path = MODEL_DIR) -> int:
    """新增一份全新文件：先在磁碟寫出 source_path 指到的 .md 檔案
    （frontmatter + body），再委派給 store_document() 做編碼寫入索引——
    不是獨立的第三套編碼邏輯，跟 edit_document() 一樣是「確認條件 +
    委派 store_document()」的模式，只是確認方向相反：edit 要求資料庫裡
    已有紀錄才能編輯，create 要求磁碟上還沒有這個檔案才能新增，兩者剛好
    互補，不會踩到彼此負責的範圍。

    source_path 指到的檔案必須還不存在，避免不小心覆蓋既有內容——要修改
    既有文件的內容，請先手動改檔案，再呼叫 edit_document() 同步索引。
    """
    full_path = _resolve_path(source_path, rag_dir)
    if full_path.exists():
        raise FileExistsError(
            f"檔案已存在：{full_path}（新增不會覆蓋既有檔案，修改既有內容請改檔案後用 edit_document()）"
        )

    if not type.strip():
        print(
            f"⚠️ 提醒：{source_path} 沒有指定 type（目前是空字串）。如果這份"
            f"文件是模型自己寫來記事的記憶文件，請加上 type=\"{MEMORY_TYPE}\"——"
            f"沒標的話，之後內容超過模型 token 上限時會被當成非記憶文件處理"
            f"（只會提醒新增輔助記憶文件，不會建議拆成續篇）。",
            file=sys.stderr,
        )

    meta = {
        "title":  title or Path(source_path).stem,
        "tags":   tags or [],
        "type":   type,
        "status": status,
    }
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(render_frontmatter(meta) + "\n" + body.strip() + "\n", encoding="utf-8")

    return store_document(source_path, rag_dir=rag_dir, db_path=db_path, model_dir=model_dir)


# ── 1. 索引既有文件 ──────────────────────────────────────────────────────────

def store_document(source_path: str, rag_dir: Path = RAG_DIR,
                    db_path: Path = DB_PATH, model_dir: Path = MODEL_DIR) -> int:
    """讀 source_path 指到的檔案（frontmatter + 內文），編碼並寫入，回傳
    documents.id。只存 source_path/title/tags/type/status，不存內文本身。

    upsert 語意：同一個 source_path 已存在就先刪除舊的 documents/vec_chunks
    資料再重新寫入，不會累積重複。
    """
    meta, body = _read_source(source_path, rag_dir)
    model = _load_model(model_dir)
    conn = _connect(db_path)

    # E5 系列模型的既定慣例：被檢索的內容要用 "passage: " 前綴。
    passage = f"passage: {body}"
    _warn_if_too_long(source_path, passage, model, meta.get("type", ""))
    embedding = model.encode(passage, normalize_embeddings=True)

    row = conn.execute(
        "SELECT id FROM documents WHERE source_path = ?", (source_path,)
    ).fetchone()
    if row is not None:
        _delete_by_id(conn, row[0])

    cur = conn.execute(
        "INSERT INTO documents (source_path, title, tags, type, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            source_path,
            meta.get("title", Path(source_path).stem),
            json.dumps(meta.get("tags", []), ensure_ascii=False),
            meta.get("type", ""),
            meta.get("status", ""),
        ),
    )
    doc_id = cur.lastrowid
    conn.execute(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        (doc_id, sqlite_vec.serialize_float32(embedding.tolist())),
    )
    conn.commit()
    conn.close()
    return doc_id


# ── 1b. 跨目錄索引外部資料夾（不複製檔案，直接以絕對路徑索引原地檔案） ──────

def index_external_dir(root_dir: Path, extensions: tuple[str, ...] = (".md", ".txt"),
                        rag_dir: Path = RAG_DIR, db_path: Path = DB_PATH,
                        model_dir: Path = MODEL_DIR) -> list[tuple[str, int | str]]:
    """走訪 root_dir 底下所有檔案，副檔名符合 extensions 的，直接以絕對路徑
    當 source_path 呼叫 store_document() 建索引——檔案留在原地，不複製、
    不搬進 corpus/。root_dir 是呼叫端傳入的執行期參數，不是寫死在程式碼裡
    的路徑，所以可以指向 rag_dir 以外的任何位置（這正是跨目錄索引的用途）。

    外部檔案通常沒有 frontmatter，parse_frontmatter() 對沒有 frontmatter 的
    內容回傳空 meta，store_document() 會自動 fallback：title 用檔名、
    tags/type/status 用空值——不需要事先幫外部檔案加 frontmatter。

    回傳 [(絕對路徑字串, doc_id 或錯誤訊息字串), ...]，方便呼叫端統計/印出
    成功與失敗的檔案，不會因為單一檔案讀取失敗就中斷整批索引。
    """
    root_dir = root_dir.resolve()
    results: list[tuple[str, int | str]] = []
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        source_path = str(path.resolve())
        try:
            doc_id = store_document(source_path, rag_dir=rag_dir, db_path=db_path, model_dir=model_dir)
            results.append((source_path, doc_id))
        except Exception as e:
            results.append((source_path, f"錯誤：{e}"))
    return results


# ── 2. 向量搜尋 ──────────────────────────────────────────────────────────────

def search(query: str | None = None, top_k: int = 5, source_path: str | None = None,
           rag_dir: Path = RAG_DIR, db_path: Path = DB_PATH,
           model_dir: Path = MODEL_DIR) -> list[dict]:
    """
    給 query：把 query 加上 "query: " 前綴編碼成向量，在 vec_chunks 做語意
    相似度 KNN 搜尋，join 回 documents，回傳最相近的 top_k 筆（含 distance，
    數字越小越相近）。

    給 source_path（不給 query）：精確比對 documents.source_path，不做任何
    向量運算，回傳該檔案目前在資料庫裡的 metadata（distance 固定是 None）。
    edit_document() 用這個模式確認文件是否存在。

    兩種模式都會即時讀 source_path 指到的檔案，把內文放進結果的
    chunk_text——這份文字不是資料庫存的，是查詢當下讀檔案讀出來的。
    """
    conn = _connect(db_path)

    if source_path is not None:
        rows = conn.execute(
            "SELECT id, source_path, title, tags, type, status "
            "FROM documents WHERE source_path = ?",
            (source_path,),
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            try:
                _, body = _read_source(r[1], rag_dir)
            except FileNotFoundError:
                body = None
            results.append(_row_to_dict(r, distance=None, chunk_text=body))
        return results

    if query is None:
        conn.close()
        raise ValueError("search() 需要 query 或 source_path 其中一個")

    model = _load_model(model_dir)
    q_embedding = model.encode(f"query: {query}", normalize_embeddings=True)

    rows = conn.execute(
        """
        SELECT documents.id, documents.source_path, documents.title, documents.tags,
               documents.type, documents.status, vec_chunks.distance
        FROM vec_chunks
        JOIN documents ON documents.id = vec_chunks.rowid
        WHERE vec_chunks.embedding MATCH ? AND k = ?
        ORDER BY vec_chunks.distance
        """,
        (sqlite_vec.serialize_float32(q_embedding.tolist()), top_k),
    ).fetchall()
    conn.close()

    results = []
    for r in rows:
        try:
            _, body = _read_source(r[1], rag_dir)
        except FileNotFoundError:
            body = None
        results.append(_row_to_dict(r[:6], distance=r[6], chunk_text=body))
    return results


# ── 3. 編輯既有文件 ──────────────────────────────────────────────────────────

def edit_document(source_path: str, rag_dir: Path = RAG_DIR,
                   db_path: Path = DB_PATH, model_dir: Path = MODEL_DIR) -> int:
    """編輯既有文件：確認 source_path 已經在資料庫裡（否則報錯，要新增請用
    store_document()），然後重新讀檔、重新編碼覆蓋——檔案本身要先改好，
    這個函式只負責把資料庫同步到檔案目前的內容。
    """
    existing = search(source_path=source_path, rag_dir=rag_dir,
                       db_path=db_path, model_dir=model_dir)
    if not existing:
        raise ValueError(
            f"找不到既有文件：{source_path}，無法編輯"
            "（全新文件請用 create_document() 新增；若檔案已存在磁碟上但還沒索引過，用 store_document()）"
        )

    return store_document(source_path, rag_dir=rag_dir, db_path=db_path, model_dir=model_dir)


# ── 4. 刪除既有文件 ──────────────────────────────────────────────────────────

def delete_document(source_path: str, db_path: Path = DB_PATH) -> bool:
    """刪除 source_path 在資料庫裡的索引（documents + vec_chunks）。只動
    資料庫，不會刪磁碟上的檔案本身。回傳是否真的刪到東西（source_path
    不存在就回傳 False，不報錯）。
    """
    conn = _connect(db_path)
    row = conn.execute(
        "SELECT id FROM documents WHERE source_path = ?", (source_path,)
    ).fetchone()
    if row is None:
        conn.close()
        return False
    _delete_by_id(conn, row[0])
    conn.commit()
    conn.close()
    return True


# ── 互動 search 用的檢視視窗（子進程） ──────────────────────────────────────

def _spawn_viewer() -> tuple[subprocess.Popen, Path]:
    """開一個獨立子進程（rag/viewer.py）跳出視窗，回傳 (子進程, 狀態檔路徑)。
    之後把想顯示的文字寫進狀態檔，視窗會自己輪詢更新，不用重開視窗。"""
    fd, state_path_str = tempfile.mkstemp(suffix=".txt", prefix="rag_viewer_")
    os.close(fd)
    state_path = Path(state_path_str)
    viewer_script = RAG_DIR / "viewer.py"
    proc = subprocess.Popen([sys.executable, str(viewer_script), str(state_path)])
    return proc, state_path


def _append_viewer_state(state_path: Path, text: str) -> None:
    """累積寫入，不是覆蓋——視窗要跟終端機一樣，看得到從頭到尾所有階段的
    內容，不是只看到「當下這一段」。"""
    with state_path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli():
    # Windows 主控台預設用 cp950，corpus 文件內文可能含 cp950 編碼範圍外的
    # 字元（例如 emoji）——印完整內容時若不重設編碼會直接 UnicodeEncodeError
    # 崩潰。errors="replace" 讓印不出來的字元退化成替代符號，不會整個中斷。
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    # stdin 同樣需要重設——create 子指令的內文從 stdin 讀入，Windows 預設用
    # cp950 解碼會讓非 cp950 範圍的字元（例如透過 UTF-8 heredoc 餵入的中文）
    # 變成代理字元（surrogate），寫檔時 UnicodeEncodeError 崩潰。明確指定
    # UTF-8 才能跟 heredoc/管線實際送進來的位元組對上。
    sys.stdin.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="rag/api.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="新增全新文件（連檔案本身都還不存在，內文從 stdin 讀入）")
    p_create.add_argument("source_path")
    p_create.add_argument("--title")
    p_create.add_argument("--tags", help="逗號分隔，例如 bug,cli,gotcha")
    p_create.add_argument("--type", default="")
    p_create.add_argument("--status", default="")

    p_store = sub.add_parser("store", help="索引一份已經存在磁碟上的文件（source_path 可相對於 rag/，也可以是絕對路徑）")
    p_store.add_argument("source_path")

    p_index_dir = sub.add_parser(
        "index-dir",
        help="跨目錄索引：走訪指定資料夾，直接以絕對路徑索引符合副檔名的檔案（不複製進 corpus/）",
    )
    p_index_dir.add_argument("directory", help="要索引的外部資料夾路徑")
    p_index_dir.add_argument("--ext", default="md,txt", help="要索引的副檔名，逗號分隔（預設 md,txt）")

    p_edit = sub.add_parser("edit", help="編輯既有文件（重新讀檔並覆蓋資料庫）")
    p_edit.add_argument("source_path")

    p_search = sub.add_parser("search", help="向量搜尋")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)

    p_delete = sub.add_parser("delete", help="刪除既有文件的索引（不刪磁碟檔案）")
    p_delete.add_argument("source_path")

    args = parser.parse_args()

    if args.command == "create":
        body = sys.stdin.read()
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        doc_id = create_document(args.source_path, body, title=args.title, tags=tags,
                                  type=args.type, status=args.status)
        print(f"已新增：{args.source_path}（id={doc_id}）")
    elif args.command == "store":
        doc_id = store_document(args.source_path)
        print(f"已儲存：{args.source_path}（id={doc_id}）")
    elif args.command == "index-dir":
        exts = tuple(f".{e.strip().lstrip('.').lower()}" for e in args.ext.split(",") if e.strip())
        results = index_external_dir(Path(args.directory), extensions=exts)
        ok = [r for r in results if isinstance(r[1], int)]
        failed = [r for r in results if not isinstance(r[1], int)]
        print(f"索引完成：成功 {len(ok)}/{len(results)} 份（來源：{args.directory}）")
        for sp, err in failed:
            print(f"  失敗：{sp} → {err}")
    elif args.command == "edit":
        doc_id = edit_document(args.source_path)
        print(f"已更新：{args.source_path}（id={doc_id}）")
    elif args.command == "delete":
        ok = delete_document(args.source_path)
        if ok:
            print(f"已刪除索引：{args.source_path}")
        else:
            print(f"找不到既有索引：{args.source_path}（無需刪除）")
    elif args.command == "search":
        results = search(query=args.query, top_k=args.top_k)
        if not results:
            print("沒有找到任何結果（資料庫可能是空的）")
            return
        # 互動兩段式：先只列路徑/metadata（TOP-K 清單），輸入編號才顯示該筆
        # 的完整內容（即時讀檔）；看完內容按 Enter 回到清單重選，不必重新
        # 查詢一次。清單跟內容不是同時列出——內容永遠在「選定之後」才出現。
        #
        # 同一份文字同時印到終端機、也寫進 viewer 子進程的狀態檔，視窗顯示
        # 的內容跟終端機看到的保證一致，不是另外組一份摘要。

        def _list_text() -> str:
            lines = [f"查詢：{args.query}"]
            for i, r in enumerate(results, 1):
                lines.append(f"[{i}] {r['title']}  (distance={r['distance']:.4f})")
                lines.append(f"    source_path: {r['source_path']}")
                lines.append(f"    tags: {r['tags']}")
            return "\n".join(lines)

        def _content_text(r: dict) -> str:
            body = r["chunk_text"] if r["chunk_text"] is not None else "（找不到對應檔案，內容無法讀取）"
            return f"───── {r['title']}（{r['source_path']}）─────\n{body}"

        # 視窗生命週期不歸這個迴圈管——CLI 這次呼叫結束後視窗依然留著，只能
        # 被使用者手動關閉（見 viewer.py 的 WM_DELETE_WINDOW 處理，狀態暫存
        # 檔也是視窗自己關閉時才清）。
        _, state_path = _spawn_viewer()

        # 沒有真人在敲鍵盤時（例如被其他程式/agent 非互動呼叫，包括 Claude
        # 自己透過工具呼叫這支 CLI）input() 讀不到東西會丟 EOFError——這裡
        # 直接接住，當作「看完這次的清單/內容就好」正常結束，不要讓例外炸
        # 出去變成看起來像卡死或崩潰。sys.stdin.isatty() 在有些呼叫環境
        # （例如某些工具用 pty 包裝過的 shell）並不可靠，所以不靠它判斷，
        # 直接處理 input() 實際失敗的那一刻。視窗仍然會開著讓人類看到這次
        # 查到了什麼。
        # 這兩句提示文字一定要跟終端機、視窗兩邊都同步——不然只看視窗的人
        # （例如旁觀 Claude 查 RAG 的使用者）會只看到清單/內容本身，完全
        # 不知道下一步該打什麼、也不知道怎麼結束，看起來就像卡死。
        PROMPT_SELECT = "\n輸入編號查看完整內容，或按 Enter 結束："
        PROMPT_BACK = "\n按 Enter 回到列表..."

        while True:
            list_block = f"\n{_list_text()}"
            print(list_block)
            _append_viewer_state(state_path, list_block)
            _append_viewer_state(state_path, PROMPT_SELECT)
            try:
                choice = input(PROMPT_SELECT).strip()
            except EOFError:
                break
            if not choice:
                break
            if not choice.isdigit() or not (1 <= int(choice) <= len(results)):
                print("輸入無效，請重新選擇。")
                continue
            r = results[int(choice) - 1]
            content_block = f"\n{_content_text(r)}"
            print(content_block)
            _append_viewer_state(state_path, content_block)
            _append_viewer_state(state_path, PROMPT_BACK)
            try:
                input(PROMPT_BACK)
            except EOFError:
                break


if __name__ == "__main__":
    _cli()
