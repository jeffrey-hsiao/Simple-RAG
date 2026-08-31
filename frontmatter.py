"""
MLOps/rag/frontmatter.py

共用的簡易 YAML frontmatter 解析器，只處理 rag/corpus/ 這批文件實際用到的
"key: value" / "key: [a, b, c]" 兩種格式，不依賴 PyYAML。被 encode_corpus.py
跟 api.py 共用，避免同一段解析邏輯寫兩份。
"""
import re

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.S)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """回傳 (frontmatter 欄位字典, 內文)。沒有 frontmatter 就回傳 ({}, 原文)。"""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    raw_fm, body = m.group(1), m.group(2)
    meta: dict = {}
    for line in raw_fm.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if val.startswith("[") and val.endswith("]"):
            meta[key] = [v.strip() for v in val[1:-1].split(",") if v.strip()]
        else:
            meta[key] = val
    return meta, body.strip()


def render_frontmatter(meta: dict) -> str:
    """`parse_frontmatter()` 的反向操作：把 meta 字典序列化成
    `"---\\n...\\n---\\n"` 格式，值是 list 就渲染成 `"[a, b, c]"`，其餘當純
    量字串輸出。只支援 `parse_frontmatter` 看得懂的這兩種型態，刻意跟 parse
    共用同一份格式規則，讓 `parse_frontmatter(render_frontmatter(meta))[0] == meta`
    可以 round-trip。
    """
    lines = ["---"]
    for key, val in meta.items():
        if isinstance(val, list):
            lines.append(f"{key}: [{', '.join(val)}]")
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines) + "\n"
