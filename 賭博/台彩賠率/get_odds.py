"""
台灣運彩賠率即時抓取程式 v3
加入 SockJS /info 握手，模擬真實瀏覽器行為
用法：
  pip install websockets httpx
  python taiwan_lottery_odds.py
"""

import asyncio
import json
import random
import string
import time
import httpx
import websockets
from datetime import datetime


# ── 設定 ──────────────────────────────────────────────
BASE    = "https://velnt-talo-ssb-pr.sportslottery.com.tw/notification"
BASE_WS = "wss://velnt-talo-ssb-pr.sportslottery.com.tw/notification"

REAL_HEADERS = {
    "Origin":          "https://www.sportslottery.com.tw",
    "Referer":         "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36",
}

SUBSCRIPTIONS = [
    {"type": "eventGroup", "id": "60067.1"},  # NBA
    {"type": "eventGroup", "id": "60563.1"},  # United League
]


# ── 工具函式 ───────────────────────────────────────────
def generate_subscriber_id() -> str:
    middle = "".join(random.choices("0123456789abcdef", k=8))
    return f"0f50{middle}0001"

def fraction_to_decimal(up: str, down: str) -> float:
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

def parse_sockjs_frame(raw: str) -> list:
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


# ── 處理 Payload ───────────────────────────────────────
async def handle_payload(payload: dict):
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


# ── SockJS /info 握手 ──────────────────────────────────
async def get_sockjs_info() -> bool:
    """模擬瀏覽器行為：先打兩次 /info 再建立 WebSocket"""
    t = int(time.time() * 1000)
    try:
        async with httpx.AsyncClient(headers=REAL_HEADERS, timeout=10) as client:
            r1 = await client.get(f"{BASE}/info")
            print(f"[info] status={r1.status_code}")
            await asyncio.sleep(0.5)
            r2 = await client.get(f"{BASE}/info?t={t}")
            print(f"[info?t] status={r2.status_code}")
        return r1.status_code == 200
    except Exception as e:
        print(f"[info error] {e}")
        return False


# ── 主要 WebSocket 邏輯 ────────────────────────────────
async def run():
    # Step 1：SockJS 握手
    ok = await get_sockjs_info()
    if not ok:
        print("[error] /info 失敗（403），等待 5 分鐘後重試...")
        await asyncio.sleep(300)  # 被封鎖就等 5 分鐘
        return

    await asyncio.sleep(1.0)  # 模擬瀏覽器載入頁面的延遲

    # Step 2：建立 WebSocket
    server_id     = str(random.randint(100, 999))
    session_id    = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    subscriber_id = generate_subscriber_id()
    url = f"{BASE_WS}/listen/{server_id}/{session_id}/websocket"

    print(f"[connect] {url}")
    print(f"[subscriberId] {subscriber_id}")

    ws_headers = {
        **REAL_HEADERS,
        "Cache-Control":            "no-cache",
        "Pragma":                   "no-cache",
        "Sec-WebSocket-Extensions": "permessage-deflate; client_max_window_bits",
        "Sec-WebSocket-Version":    "13",
    }

    async with websockets.connect(
        url,
        additional_headers=ws_headers,
        ping_interval=20,
        ping_timeout=10,
    ) as ws:
        opening = await ws.recv()
        print(f"[frame] '{opening}'")

        if opening != "o":
            print("[warn] 開啟幀不是 'o'，連線可能有問題")
            return

        print("[connected] WebSocket 已連線 ✓")

        # Step 3：訂閱頻道
        for sub in SUBSCRIPTIONS:
            msg = json.dumps([json.dumps({
                "subscriberId": subscriber_id,
                "contentId":    sub,
                "clientContext": {"language": "ZH"},
            })])
            await ws.send(msg)
            print(f"[subscribe] {sub['type']} / {sub['id']}")
            await asyncio.sleep(1.0)

        print("\n[waiting] 等待賠率推送...\n")

        # Step 4：持續接收
        async for raw in ws:
            if raw in ("h", "o"):
                continue
            if raw.startswith("c"):
                print(f"[closed] 伺服器關閉：{raw}")
                break
            for payload in parse_sockjs_frame(raw):
                await handle_payload(payload)


# ── 自動重連（保守間隔）────────────────────────────────
async def main():
    while True:
        try:
            await run()
        except websockets.ConnectionClosed as e:
            print(f"\n[disconnect] 連線中斷：{e}")
            print("30 秒後重連...")
            await asyncio.sleep(30)   # 正常斷線等 30 秒
        except ConnectionResetError as e:
            print(f"\n[reset] 連線被重設：{e}")
            print("30 秒後重連...")
            await asyncio.sleep(30)
        except Exception as e:
            print(f"\n[error] {type(e).__name__}: {e}")
            print("60 秒後重試...")
            await asyncio.sleep(60)   # 其他錯誤等 60 秒


if __name__ == "__main__":
    print("=" * 40)
    print("  台灣運彩即時賠率抓取程式 v3")
    print("  按 Ctrl+C 停止")
    print("=" * 40 + "\n")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[stop] 程式已停止")
