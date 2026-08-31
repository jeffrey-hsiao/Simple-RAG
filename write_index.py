"""
MLOps/rag/write_index.py

只負責寫入資料庫：讀 encode_corpus.py 產生的中繼檔
rag/index/encoded_corpus.jsonl，寫進 rag/index/vectors.sqlite
（sqlite-vec 擴充）。不碰模型、不做任何編碼——編碼是 encode_corpus.py 的
責任，兩者分開後才能各自獨立重跑。

用法：
    python rag/write_index.py
"""
import json
import sqlite3
from pathlib import Path

import sqlite_vec

RAG_DIR = Path(__file__).parent
IN_PATH = RAG_DIR / "index" / "encoded_corpus.jsonl"
DB_PATH = RAG_DIR / "index" / "vectors.sqlite"


def _connect(db_path: Path) -> sqlite3.Connection:
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
    # 注意：沒有 chunk_text 欄位——文本只存在檔案裡（rag/corpus/），資料庫
    # 只存路徑+metadata，跟 api.py 的 schema 保持一致（同一個資料庫檔案）。
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


def write_index(in_path: Path = IN_PATH, db_path: Path = DB_PATH) -> int:
    """讀 in_path 的 JSONL 中繼檔，寫進 db_path，回傳寫入的文件數。

    可重複執行：同一個 source_path 會先刪除舊資料再重新寫入，不會累積重複。
    """
    conn = _connect(db_path)
    count = 0
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            row = conn.execute(
                "SELECT id FROM documents WHERE source_path = ?", (rec["source_path"],)
            ).fetchone()
            if row is not None:
                doc_id = row[0]
                conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (doc_id,))
                conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

            cur = conn.execute(
                "INSERT INTO documents (source_path, title, tags, type, status) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    rec["source_path"],
                    rec["title"],
                    json.dumps(rec["tags"], ensure_ascii=False),
                    rec["type"],
                    rec["status"],
                ),
            )
            doc_id = cur.lastrowid
            conn.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
                (doc_id, sqlite_vec.serialize_float32(rec["embedding"])),
            )
            count += 1

    conn.commit()
    conn.close()
    return count


if __name__ == "__main__":
    n = write_index()
    print(f"已寫入 {n} 份文件到 {DB_PATH}")
