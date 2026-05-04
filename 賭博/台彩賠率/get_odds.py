"""
台灣運彩賠率即時抓取程式
透過 WebSocket (SockJS) 連線取得即時賠率
用法：python taiwan_lottery_odds.py
"""

import asyncio
import json
import random
import string
import httpx
import websockets
from datetime import datetime

# ── 設定 ──────────────────────────────────────────────
BASE_HTTP  = "https://velnt-talo-ssb-pr.sportslottery.com.tw/notification"
BASE_WS    = "wss://velnt-talo-ssb-pr.sportslottery.com.tw/notification"

HEADERS = {
    "Origin":          "https://www.sportslottery.com.tw",
    "Referer":         "https://www.sportslottery.com.tw/",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8",
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/147 Safari/537.36",
}

# 要訂閱的賠率頻道（可同時訂多個）
SUBSCRIPTIONS = [
    {"type": "eventGroup", "id": "60067.1"},   # NBA
    {"type": "eventGroup", "id": "60063.1"},   # MLB（依實際 id 調整）
    # {"type": "boNavigationList", "id": "1355/34765.1"},  # 籃球導覽
]

LINE_BOT_TOKEN = ""   # 填入你的 LINE Bot token（留空則只印到 console）
LINE_GROUP_ID  = ""   # 填入你的 LINE 群組 ID


# ── 工具函式 ───────────────────────────────────────────
def random_session_id():
    """SockJS 需要隨機 session id（8 個英數字）"""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))

def random_server_id():
    """SockJS server id（3 位數字）"""
    return str(random.randint(100, 999))

def fraction_to_decimal(up: str, down: str) -> float:
    """台彩賠率格式：分子/分母 → 小數（含本金）"""
    try:
        return round(1 + int(up) / int(down), 2)
    except (ValueError, ZeroDivisionError):
        return 0.0

def format_odds_message(events: list) -> str:
    """把賠率清單格式化成 LINE 訊息"""
    now = datetime.now().strftime("%m/%d %H:%M")
    lines = [f"🏀 台灣運彩即時賠率  {now}", "=" * 22]

    for event in events:
        away = event.get("participantname_away", "")
        home = event.get("participantname_home", "").strip()
        ts   = event.get("tsstart", "")
        time_str = ts[11:16] if len(ts) >= 16 else ""

        lines.append(f"\n⏰ {time_str}  {away} @ {home}")

        for market in event.get("markets", []):
            mname = market.get("name", "")
            sels  = market.get("selections", [])
            parts = []
            for sel in sels:
                dec = fraction_to_decimal(
                    sel.get("currentpriceup", "0"),
                    sel.get("currentpricedown", "1"),
                )
                parts.append(f"{sel['name'].strip()} {dec}")
            lines.append(f"  {mname}：{'  /  '.join(parts)}")

    return "\n".join(lines)

async def send_to_line(message: str):
    """推送訊息到 LINE 群組（選用）"""
    if not LINE_BOT_TOKEN or not LINE_GROUP_ID:
        return
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={"Authorization": f"Bearer {LINE_BOT_TOKEN}"},
            json={"to": LINE_GROUP_ID, "messages": [{"type": "text", "text": message}]},
        )


# ── SockJS 握手 ────────────────────────────────────────
async def get_sockjs_session() -> tuple[str, str]:
    """
    SockJS 協議需先打 GET /info 確認可用，
    再用 /{server_id}/{session_id}/websocket 建立 WS。
    """
    server_id  = random_server_id()
    session_id = random_session_id()

    # 確認伺服器可連（可選，失敗也繼續）
    try:
        async with httpx.AsyncClient(headers=HEADERS, timeout=5) as client:
            r = await client.get(f"{BASE_HTTP}/info")
            print(f"[info] SockJS info: {r.status_code}")
    except Exception as e:
        print(f"[info] 略過 SockJS info 確認：{e}")

    return server_id, session_id


# ── 解析 WebSocket 訊息 ────────────────────────────────
def parse_message(raw: str) -> list[dict]:
    """
    SockJS frame 格式：
      'o'        → 開啟
      'h'        → heartbeat
      'a[...]'   → 資料陣列（JSON 字串）
      'c[...]'   → 關閉
    """
    if not raw or raw[0] not in ("a", "m"):
        return []

    inner = raw[1:]   # 去掉首字母
    try:
        frames = json.loads(inner)   # 解出字串陣列
    except json.JSONDecodeError:
        return []

    results = []
    for frame in (frames if isinstance(frames, list) else [frames]):
        try:
            obj = json.loads(frame)
            results.append(obj)
        except Exception:
            pass
    return results


# ── 主要 WebSocket 邏輯 ────────────────────────────────
async def run():
    server_id, session_id = await get_sockjs_session()
    ws_url = f"{BASE_WS}/listen/{server_id}/{session_id}/websocket"
    print(f"[connect] {ws_url}")

    subscriber_id = "".join(random.choices(string.hexdigits[:16], k=16))

    async with websockets.connect(
        ws_url,
        additional_headers=HEADERS,
        ping_interval=20,
        ping_timeout=10,
    ) as ws:
        print("[connected] WebSocket 已連線")

        # 等待 SockJS 開啟幀 'o'
        opening = await ws.recv()
        print(f"[frame] {opening}")

        # 訂閱所有頻道
        for sub in SUBSCRIPTIONS:
            msg = json.dumps([json.dumps({
                "subscriberId": subscriber_id,
                "contentId": sub,
                "clientContext": {"language": "ZH"},
            })])
            await ws.send(msg)
            print(f"[subscribe] {sub}")

        # 持續接收
        async for raw in ws:
            if raw == "h":          # heartbeat，忽略
                continue
            if raw.startswith("c"): # 伺服器關閉
                print("[closed] 伺服器關閉連線")
                break

            payloads = parse_message(raw)
            for payload in payloads:
                await handle_payload(payload)


async def handle_payload(payload: dict):
    """處理一則 WebSocket payload"""
    ntype = payload.get("notificationType")

    # ── LISTENING_STARTED：訂閱成功確認 ──
    if ntype == "LISTENING_STARTED":
        print("[ok] 訂閱成功")
        return

    # ── CONTENT_CHANGES：賠率更新 ──
    if ntype != "CONTENT_CHANGES":
        return

    for item in payload.get("data", []):
        content_id = item.get("contentId", {})
        ctype = content_id.get("type", "")
        cid   = content_id.get("id", "")
        change = item.get("change", {})

        if ctype == "eventGroup":
            events = change.get("events", [])
            if not events:
                continue

            group_name = change.get("name", cid)
            print(f"\n{'='*40}")
            print(f"📡 {group_name}  更新  {datetime.now().strftime('%H:%M:%S')}")

            msg = format_odds_message(events)
            print(msg)

            # 推送到 LINE（若有設定）
            await send_to_line(msg)


# ── 自動重連 ──────────────────────────────────────────
async def main():
    while True:
        try:
            await run()
        except (websockets.ConnectionClosed, ConnectionResetError) as e:
            print(f"[disconnect] 連線中斷：{e}，5 秒後重連...")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[error] {e}，10 秒後重試...")
            await asyncio.sleep(10)


if __name__ == "__main__":
    print("台灣運彩即時賠率抓取程式啟動")
    print("按 Ctrl+C 停止\n")
    asyncio.run(main())