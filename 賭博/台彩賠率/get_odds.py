"""
台灣運彩賠率即時抓取程式 v6
使用 Playwright 攔截真實 Chrome 的 WebSocket 訊息
完全繞過 Cloudflare 和壓縮問題
用法：
  pip install playwright
  playwright install chromium
  python get_odds.py
"""

import json
import time
from datetime import datetime
from playwright.sync_api import sync_playwright


# ── 設定 ──────────────────────────────────────────────
TARGET_URL = "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1"

# 要攔截的頻道關鍵字
WATCH_TYPES = {"eventGroup", "boNavigationList", "inplayLiveDataEventSummaryListByTournament"}


# ── 工具函式 ───────────────────────────────────────────
def fraction_to_decimal(up, down) -> float:
    try:
        return round(1 + int(up) / int(down), 2)
    except (ValueError, ZeroDivisionError):
        return 0.0

def format_odds(events: list, group_name: str = "") -> str:
    now = datetime.now().strftime("%m/%d %H:%M:%S")
    lines = [f"{'='*40}", f"📡 {group_name}  {now}", f"{'='*40}"]
    for event in events:
        away = event.get("participantname_away", "").strip()
        home = event.get("participantname_home", "").strip()
        ts   = event.get("tsstart", "")
        t    = ts[11:16] if len(ts) >= 16 else ""
        lines.append(f"\n⏰ {t}  {away} @ {home}")
        for mkt in event.get("markets", []):
            sels  = mkt.get("selections", [])
            parts = [
                f"{s.get('name','').strip()} "
                f"{fraction_to_decimal(s.get('currentpriceup','0'), s.get('currentpricedown','1'))}"
                for s in sels
            ]
            lines.append(f"  {mkt['name']}：{'  /  '.join(parts)}")
    return "\n".join(lines)


# ── 處理 WebSocket 訊息 ────────────────────────────────
def parse_sockjs(raw: str) -> list:
    """解析 SockJS frame，回傳 payload 列表"""
    if not raw or raw[0] not in ("a", "m"):
        return []
    try:
        frames = json.loads(raw[1:])
        results = []
        for f in (frames if isinstance(frames, list) else [frames]):
            try:
                results.append(json.loads(f))
            except Exception:
                pass
        return results
    except Exception:
        return []

def handle_payload(payload: dict):
    ntype = payload.get("notificationType", "")

    if ntype == "LISTENING_STARTED":
        print("[✓] 訂閱成功，等待資料推送...")
        return

    if ntype != "CONTENT_CHANGES":
        return

    for item in payload.get("data", []):
        ctype  = item.get("contentId", {}).get("type", "")
        change = item.get("change", {})

        if ctype == "eventGroup":
            events = change.get("events", [])
            name   = change.get("name", "")
            if events:
                print(format_odds(events, name))

        elif ctype == "inplayLiveDataEventSummaryListByTournament":
            summaries = change.get("eventSummaries", [])
            if summaries:
                print(f"\n📺 即時賽事：{len(summaries)} 場")

        elif ctype == "boNavigationList":
            name = change.get("name", "")
            num  = change.get("numevents", "0")
            if int(num) > 0:
                print(f"[nav] {name}：{num} 場賽事")

def on_frame_received(payload: str):
    """WebSocket 收到訊息時的 callback"""
    if not payload or payload in ("o", "h"):
        return
    if payload.startswith("c"):
        print(f"[ws closed] {payload[:50]}")
        return

    for p in parse_sockjs(payload):
        handle_payload(p)


# ── 主程式 ────────────────────────────────────────────
def run():
    with sync_playwright() as p:
        print("[browser] 啟動 Chrome...")
        browser = p.chromium.launch(
            headless=False,   # 顯示瀏覽器視窗（可改 True 隱藏）
            args=["--no-sandbox"]
        )
        context = browser.new_context()
        page    = context.new_page()

        # 攔截 WebSocket
        ws_connected = False

        def on_websocket(ws):
            nonlocal ws_connected
            if "notification" not in ws.url:
                return
            ws_connected = True
            print(f"[ws] 連線：{ws.url}")

            ws.on("framereceived", lambda f: on_frame_received(
                f.payload if isinstance(f.payload, str) else f.payload.decode("utf-8", errors="ignore")
            ))
            ws.on("close", lambda: print("[ws] 連線關閉"))

        page.on("websocket", on_websocket)

        print(f"[browser] 開啟：{TARGET_URL}")
        response = page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)
        print(f"[browser] 頁面 status: {response.status}")
        print(f"[browser] 頁面 URL: {page.url}")
        print(f"[browser] 頁面標題: {page.title()}")

        # 印出頁面內容前 500 字
        content = page.content()
        print(f"[browser] 頁面完整內容:\n{content[:2000]}")

        print("[browser] 等待賠率推送...\n")
        print("[browser] 頁面載入完成，等待賠率推送...\n")

        # 持續等待（按 Ctrl+C 停止）
        try:
            while True:
                time.sleep(1)
                # 如果頁面被關掉就重開
                if page.is_closed():
                    print("[browser] 頁面關閉，重新開啟...")
                    page = context.new_page()
                    page.on("websocket", on_websocket)
                    page.goto(TARGET_URL, wait_until="networkidle", timeout=30000)
        except KeyboardInterrupt:
            pass

        browser.close()


def main():
    while True:
        try:
            run()
        except KeyboardInterrupt:
            print("\n[stop] 程式已停止")
            break
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")
            print("30 秒後重試...")
            time.sleep(30)


if __name__ == "__main__":
    print("=" * 40)
    print("  台灣運彩即時賠率抓取程式 v6")
    print("  按 Ctrl+C 停止")
    print("=" * 40 + "\n")
    main()