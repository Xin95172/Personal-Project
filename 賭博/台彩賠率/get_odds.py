"""
台灣運彩賠率即時抓取程式 v2
根據真實瀏覽器行為還原所有參數
用法：
  pip install websockets httpx
  python taiwan_lottery_odds.py
"""

import asyncio
import json
import random
import string
import websockets
from datetime import datetime


# ── 設定 ──────────────────────────────────────────────
BASE_WS = "wss://velnt-talo-ssb-pr.sportslottery.com.tw/notification"

# 從 DevTools 確認的真實 Headers
REAL_HEADERS = {
    "Origin":                   "https://www.sportslottery.com.tw",
    "Referer":                  "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1",
    "Accept-Language":          "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding":          "gzip, deflate, br, zstd",
    "Cache-Control":            "no-cache",
    "Pragma":                   "no-cache",
    "Sec-WebSocket-Extensions": "permessage-deflate; client_max_window_bits",
    "Sec-WebSocket-Version":    "13",
    "User-Agent":               "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}

# 要訂閱的頻道（從 DevTools WebSocket Messages 確認）
SUBSCRIPTIONS = [
    {"type": "eventGroup",      "id": "60067.1"},   # NBA
    {"type": "eventGroup",      "id": "60063.1"},   # MLB（若有開賽）
    {"type": "boNavigationList","id": "1355/top"},  # 籃球導覽總覽
    {"type": "inplayLiveDataEventSummaryListByTournament", "id": "23527"},  # 即時賽事
]


# ── 工具函式 ───────────────────────────────────────────
def generate_subscriber_id() -> str:
    """模擬瀏覽器產生的 subscriberId：0f50{8碼隨機hex}0001"""
    middle = "".join(random.choices("0123456789abcdef", k=8))
    return f"0f50{middle}0001"

def random_server_id() -> str:
    return str(random.randint(100, 999))

def random_session_id() -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def fraction_to_decimal(up: str, down: str) -> float:
    """台彩賠率分數格式轉小數（含本金）"""
    try:
        return round(1 + int(up) / int(down), 2)
    except (ValueError, ZeroDivisionError):
        return 0.0


# ── 解析函式 ───────────────────────────────────────────
def parse_sockjs_frame(raw: str) -> list:
    """
    SockJS frame 格式：
      'o'      → 連線開啟
      'h'      → heartbeat
      'a[...]' → 資料陣列
      'c[...]' → 連線關閉
    """
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

def format_odds(events: list, group_name: str = "") -> str:
    """把賠率清單格式化成可讀文字"""
    now = datetime.now().strftime("%m/%d %H:%M:%S")
    lines = [
        f"{'='*40}",
        f"📡 {group_name}  {now}",
        f"{'='*40}",
    ]
    for event in events:
        away = event.get("participantname_away", "").strip()
        home = event.get("participantname_home", "").strip()
        ts   = event.get("tsstart", "")
        t    = ts[11:16] if len(ts) >= 16 else ""

        lines.append(f"\n⏰ {t}  {away} @ {home}")

        for mkt in event.get("markets", []):
            mname = mkt.get("name", "")
            sels  = mkt.get("selections", [])
            parts = []
            for s in sels:
                dec = fraction_to_decimal(
                    s.get("currentpriceup",   "0"),
                    s.get("currentpricedown", "1"),
                )
                parts.append(f"{s.get('name','').strip()} {dec}")
            lines.append(f"  {mname}：{'  /  '.join(parts)}")

    return "\n".join(lines)


# ── 處理 Payload ───────────────────────────────────────
async def handle_payload(payload: dict):
    ntype = payload.get("notificationType", "")

    if ntype == "LISTENING_STARTED":
        print("[✓] 訂閱成功")
        return

    if ntype != "CONTENT_CHANGES":
        return

    for item in payload.get("data", []):
        content_id = item.get("contentId", {})
        ctype  = content_id.get("type", "")
        change = item.get("change", {})

        # eventGroup → 有完整賠率
        if ctype == "eventGroup":
            events     = change.get("events", [])
            group_name = change.get("name", content_id.get("id", ""))
            if events:
                print(format_odds(events, group_name))

        # inplayLiveData → 即時賽事清單
        elif ctype == "inplayLiveDataEventSummaryListByTournament":
            items = change.get("eventSummaries", [])
            if items:
                print(f"\n📺 即時賽事更新：{len(items)} 場")
                for e in items:
                    print(f"  {e.get('home','')} vs {e.get('away','')}  "
                          f"比分 {e.get('score','')}")

        # boNavigationList → 導覽結構（不含賠率數字）
        elif ctype == "boNavigationList":
            name = change.get("name", "")
            num  = change.get("numevents", "0")
            print(f"[nav] {name}：{num} 場賽事")


# ── 主要 WebSocket 邏輯 ────────────────────────────────
async def run():
    server_id     = random_server_id()
    session_id    = random_session_id()
    subscriber_id = generate_subscriber_id()
    url = f"{BASE_WS}/listen/{server_id}/{session_id}/websocket"

    print(f"[connect] {url}")
    print(f"[subscriberId] {subscriber_id}")

    async with websockets.connect(
        url,
        additional_headers=REAL_HEADERS,
        ping_interval=20,
        ping_timeout=10,
    ) as ws:
        # 等待 SockJS 開啟幀
        opening = await ws.recv()
        if opening != "o":
            print(f"[warn] 預期 'o'，收到 '{opening}'")
        else:
            print("[connected] WebSocket 已連線 ✓")

        # 訂閱所有頻道
        for sub in SUBSCRIPTIONS:
            msg = json.dumps([json.dumps({
                "subscriberId": subscriber_id,
                "contentId":    sub,
                "clientContext": {
                    "language":  "ZH",
                    "ipAddress": "134.208.96.189",
                },
            })])
            await ws.send(msg)
            print(f"[subscribe] {sub['type']} / {sub['id']}")
            await asyncio.sleep(0.3)   # 每次訂閱間隔，模擬真實瀏覽器

        print("\n[waiting] 等待賠率推送...\n")

        # 持續接收
        async for raw in ws:
            if raw == "h":
                continue
            if raw == "o":
                continue
            if raw.startswith("c"):
                print(f"[closed] 伺服器關閉：{raw}")
                break

            payloads = parse_sockjs_frame(raw)
            for payload in payloads:
                await handle_payload(payload)


# ── 自動重連 ──────────────────────────────────────────
async def main():
    while True:
        try:
            await run()
        except websockets.ConnectionClosed as e:
            print(f"\n[disconnect] 連線中斷：{e}，5 秒後重連...")
            await asyncio.sleep(5)
        except ConnectionResetError as e:
            print(f"\n[reset] 連線被重設：{e}，5 秒後重連...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}，10 秒後重試...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    print("=" * 40)
    print("  台灣運彩即時賠率抓取程式 v2")
    print("  按 Ctrl+C 停止")
    print("=" * 40 + "\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[stop] 程式已停止")