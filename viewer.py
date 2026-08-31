"""
MLOps/rag/viewer.py

被 api.py 的互動式 search 流程呼叫的獨立子進程：開一個 tkinter 視窗，定期
輪詢一個狀態檔案（純文字），內容變了就重新整理畫面。視窗顯示的文字就是
api.py 印到終端機的同一份字串（同一個變數寫進狀態檔），不是另外組一份
摘要——確保「你在終端機看到的」跟「這個視窗顯示的」永遠一致：TOP-K 清單
狀態時顯示清單，選定某一筆時顯示該筆完整內容。

用法（通常不會手動執行，是被 api.py 用 subprocess.Popen 啟動）：
    python rag/viewer.py <state_file_path>
"""
import sys
import tkinter as tk
from pathlib import Path

POLL_MS = 300


def main():
    state_path = Path(sys.argv[1])

    root = tk.Tk()
    root.title("RAG 搜尋檢視")
    root.geometry("800x600")

    text = tk.Text(root, wrap="word", font=("Consolas", 11))
    text.pack(fill="both", expand=True)

    last_content = None

    def refresh():
        nonlocal last_content
        try:
            content = state_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            content = ""
        if content != last_content:
            last_content = content
            text.delete("1.0", "end")
            text.insert("1.0", content)
            text.see("end")  # 狀態檔是累積式的，永遠捲到最新一段
        root.after(POLL_MS, refresh)

    def on_close():
        # 視窗只由使用者手動關閉（不會被 CLI 端自動 terminate）。關閉時才
        # 清掉狀態暫存檔——CLI 那邊已經不管這個檔案的生命週期了。
        state_path.unlink(missing_ok=True)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
