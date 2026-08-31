# Simple RAG

一個輕量、本機執行的 RAG（檢索增強生成）系統：Markdown 語料 + 多語言
sentence-embedding 模型（`multilingual-e5-small`）+ `sqlite-vec` 向量索引。
不需要外部向量資料庫服務，一個 sqlite 檔案就是整個索引。

## 架構

- **文字只有一份真相**：向量資料庫（`index/vectors.sqlite`）只存
  `source_path` 與 metadata（title / tags / type / status），**不存內文本
  身**。內文永遠只存在 `corpus/` 底下的 `.md` 檔案裡。查詢/搜尋回傳結果時
  才即時讀檔附上內文；「編輯」文件的本質是「改檔案 → 重新讀檔編碼覆蓋索
  引」，不會有資料庫文字跟檔案內容對不起來的問題。
- **每份文件是一個 chunk**：不做段落切分。語料設計上假設每個 `.md` 檔案
  本身就是一個夠細的主題單位。
- **E5 前綴慣例**：`multilingual-e5-small` 系列模型訓練時區分「被檢索的
  內容」與「查詢」，編碼前分別加上 `"passage: "` / `"query: "` 前綴，兩者
  不可混用，否則相似度分數會失真。
- **upsert 語意**：同一個 `source_path` 重複寫入時，會先刪除舊的索引紀錄
  再寫入新的，不會累積重複。

## 檔案結構

```
api.py             單筆文件的 CRUD + 向量搜尋，也是主要的 CLI 入口
encode_corpus.py   批次編碼 corpus/ 下所有 .md，輸出中繼檔 index/encoded_corpus.jsonl
write_index.py     讀 encoded_corpus.jsonl，寫入 index/vectors.sqlite
frontmatter.py     極簡 YAML frontmatter 解析／序列化（自製，不依賴 PyYAML）
viewer.py          search 時彈出的 tkinter 檢視視窗（被 api.py 用子行程啟動）
corpus/            你的 Markdown 語料（未納入版控，見下方「語料格式」）
index/             編碼中繼檔與 sqlite 索引（未納入版控，執行後自動產生）
models/            嵌入模型（未納入版控，見下方「安裝」）
```

## 安裝

```bash
pip install -r requirements.txt
```

嵌入模型會在第一次呼叫時自動處理：`_load_model()` 檢查
`models/multilingual-e5-small` 是否存在，不存在的話會自動從
`intfloat/multilingual-e5-small` 下載並存檔到該路徑，之後就跟本機模型
一樣直接讀，不會每次都重新下載。也可以手動先下載好：

```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("intfloat/multilingual-e5-small").save("models/multilingual-e5-small")
```

## 語料格式

`corpus/` 底下每個 `.md` 檔案開頭是簡易 frontmatter：

```markdown
---
title: 文件標題
tags: [tag1, tag2]
type: 任意分類字串
status: 任意狀態字串
---

內文（Markdown）……
```

`title` / `tags` / `type` / `status` 都是選填的 metadata，用來輔助
`search()` 結果的顯示與篩選，不影響向量本身（向量只編碼內文）。

**要求**：如果這份文件是模型自己為了記事而寫入的「記憶文件」，建立時
（`create_document()` / `python api.py create ...`）**必須**在 frontmatter
加上 `type: memory`。這是系統判斷「內容超長時能不能考慮拆成續篇」的唯一
依據（見下方「已知限制」）——沒標記的話會被當成一般文件，超長時會被要求
不可拆分原始檔案。系統不會自動幫你判斷、也不會擋下漏標的寫入，只有沒填
`type` 時才會印提醒（見「已知限制」），填了別的值不會有任何提示。非記憶
用途的文件，`type` 可以填任意分類字串，不影響行為。

## 使用方式

### 批次建索引（第一次使用、或想整批重建時）

```bash
cd rag  # 或你放這些檔案的目錄
python encode_corpus.py   # 編碼 corpus/ 下所有 .md → index/encoded_corpus.jsonl
python write_index.py     # 把中繼檔寫進 index/vectors.sqlite
```

兩支腳本分開執行：`encode_corpus.py` 只做編碼、不碰資料庫；
`write_index.py` 只做資料庫寫入、不做任何編碼。可以只重編碼不重寫資料庫
（反之亦然），例如想拿舊的編碼結果重寫一份新資料庫時很有用。

### 單筆文件操作（CLI）

```bash
# 新增全新文件——連檔案本身都還不存在，內文從 stdin 讀入
python api.py create <source_path> --title <T> --tags a,b,c --type <T> --status <S>

# 索引一份已經存在磁碟上、但還沒進索引的文件
python api.py store <source_path>

# 編輯既有文件：檔案內容先改好，再執行這行同步索引
python api.py edit <source_path>

# 刪除文件的索引（只動資料庫，不刪磁碟上的檔案）
python api.py delete <source_path>

# 向量搜尋
python api.py search "<查詢字串>" --top-k 5
```

`source_path` 可以是相對於這批程式檔案所在目錄的路徑（例如
`corpus/foo.md`），也可以是絕對路徑——絕對路徑用來索引專案目錄以外的
檔案，檔案留在原地、不會被複製進 `corpus/`，資料庫只存那個絕對路徑當
指標，`search()`/`edit_document()` 讀內容時會直接讀原始位置。

```bash
# 跨目錄索引：走訪外部資料夾，直接以絕對路徑索引符合副檔名的檔案
python api.py index-dir "<外部資料夾路徑>" --ext md,txt
```

外部檔案通常沒有 frontmatter，這沒關係——`create_document()`/
`store_document()` 對缺少 frontmatter 的內容會自動 fallback：
title 用檔名、tags/type/status 用空值。

`search` 是互動式的：先列出 top-k 筆的路徑/metadata，輸入編號可以看該筆
完整內容（即時讀檔），同時會另外彈出一個 tkinter 視窗顯示同樣的內容，方
便旁觀者一起看搜尋過程。非互動呼叫（例如被其他程式呼叫、沒有真人在敲鍵
盤）時，看完這次的清單/內容就會自動結束，不會卡住；視窗本身要靠使用者
手動關閉。

**給 LLM/agent 呼叫者的提醒**：拿到清單後，請直接把要看的編號透過 stdin
餵給同一個程序把互動流程走完（例如 `echo "2" | python api.py search "查詢
字串" --top-k 5`），而不是找到 `source_path` 之後就改用其他通用的讀檔工
具去讀那個路徑。這支 CLI 本身就是設計成一路用到底的，繞過它、只把它當
成「查路徑用的工具」是沒有必要的額外步驟。

### 當函式庫用

上述五個操作也都是 `api.py` 裡可以直接 import 的函式：
`create_document()` / `store_document()` / `edit_document()` /
`delete_document()` / `search()`，簽章與行為細節見各函式的 docstring。

## 已知限制

- 內文長度超過模型的 `max_seq_length`（多語言 e5-small 預設 512 tokens）
  時，超過的部分會被模型默默截斷、不會進向量（讀取到的內容本身不受影
  響，只有語意搜尋可能漏掉後段內容）。`store_document()` /
  `encode_corpus.py` 都會在超過時印警告到 stderr，但不會擋下寫入。警告文
  字依 `type` 是否為 `memory` 分兩種：記憶文件（`type: memory`，模型自己
  寫來記事的文件）可以考慮拆成續篇文件，拆不拆由模型臨場判斷後段內容是
  否豐富到值得獨立成篇；非記憶文件（其他 `type`）則**不可拆分或修改原始
  檔案**，只能另外新增一份記憶文件來輔助之後搜尋不到的內容，原檔案維持
  不動。
- `type` 是否為 `memory` 完全靠呼叫端手動標記，沒有機制驗證標記正確。
  `create_document()` 唯一的防呆是：呼叫時沒帶 `type` 參數（空字串）會印
  一行提醒到 stderr，告訴你這份文件之後會被當成非記憶文件處理——但不會
  擋下寫入，也無法辨識「type 填了別的值、但其實是記憶文件」這種情況。
- `frontmatter.py` 只支援 `key: value` 與 `key: [a, b, c]` 兩種格式，不
  是完整的 YAML 解析器。
