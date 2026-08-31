"""
MLOps/rag/encode_corpus.py

只負責編碼：讀 rag/corpus/ 底下的 Markdown 文件，用
rag/models/multilingual-e5-small 算出向量，寫成中繼檔案
rag/index/encoded_corpus.jsonl。不碰資料庫——寫進 vectors.sqlite 是
write_index.py 的責任，兩者分開後才能各自獨立重跑（例如只想重新編碼但不想
動資料庫，或反過來拿舊的編碼結果重寫資料庫）。

用法：
    python rag/encode_corpus.py
"""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

from api import _warn_if_too_long
from frontmatter import parse_frontmatter

RAG_DIR    = Path(__file__).parent
CORPUS_DIR = RAG_DIR / "corpus"
MODEL_DIR  = RAG_DIR / "models" / "multilingual-e5-small"
OUT_PATH   = RAG_DIR / "index" / "encoded_corpus.jsonl"


def encode_corpus(corpus_dir: Path = CORPUS_DIR, out_path: Path = OUT_PATH,
                   model_dir: Path = MODEL_DIR) -> int:
    """編碼 corpus_dir 底下所有 .md 檔，寫成 JSONL 中繼檔，回傳處理的文件數。

    每個檔案整份當一個 chunk，不再往下切段落——corpus 裡的文件本身已經是
    刻意拆過的「一份文件一個主題」細粒度，不需要也不應該再切更細。
    """
    model = SentenceTransformer(str(model_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = sorted(corpus_dir.glob("*.md"))
    with out_path.open("w", encoding="utf-8") as f:
        for path in md_files:
            text = path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(text)
            source_path = str(path.relative_to(corpus_dir.parent)).replace("\\", "/")

            # E5 系列模型的既定慣例：被檢索的內容要加 "passage: " 前綴，查詢時
            # 改用 "query: " 前綴——這兩種前綴在訓練時是分開處理的，前綴不
            # 一致會讓相似度分數失真，不能省略或混用。
            passage = f"passage: {body}"
            _warn_if_too_long(source_path, passage, model)
            embedding = model.encode(passage, normalize_embeddings=True)

            record = {
                "source_path": source_path,
                "title": meta.get("title", path.stem),
                "tags": meta.get("tags", []),
                "type": meta.get("type", ""),
                "status": meta.get("status", ""),
                "chunk_text": body,
                "embedding": embedding.tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(md_files)


if __name__ == "__main__":
    n = encode_corpus()
    print(f"已編碼 {n} 份文件，中繼檔存於 {OUT_PATH}")
