import json
import time
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError


TARGET_URL = "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1"

DEBUG_HTML_PREVIEW_LEN = 1500
DEBUG_BODY_PREVIEW_LEN = 500
DEBUG_FRAME_BODY_PREVIEW_LEN = 300
WAIT_AFTER_GOTO_MS = 10000


def ts():
    return datetime.now().strftime("%H:%M:%S")


def safe_print(*args):
    try:
        print(*args)
    except Exception:
        pass


def fraction_to_decimal(up, down) -> float:
    try:
        return round(1 + int(up) / int(down), 2)
    except (ValueError, ZeroDivisionError, TypeError):
        return 0.0


def format_odds(events: list, group_name: str = "") -> str:
    now = datetime.now().strftime("%m/%d %H:%M:%S")
    lines = [f"{'='*50}", f"📡 {group_name}  {now}", f"{'='*50}"]

    for event in events:
        away = event.get("participantname_away", "").strip()
        home = event.get("participantname_home", "").strip()
        tsstart = event.get("tsstart", "")
        t = tsstart[11:16] if isinstance(tsstart, str) and len(tsstart) >= 16 else ""

        lines.append(f"\n⏰ {t}  {away} @ {home}")

        for mkt in event.get("markets", []):
            market_name = mkt.get("name", "<no market name>")
            sels = mkt.get("selections", [])
            parts = []

            for s in sels:
                sel_name = s.get("name", "").strip()
                up = s.get("currentpriceup", "0")
                down = s.get("currentpricedown", "1")
                dec = fraction_to_decimal(up, down)
                parts.append(f"{sel_name} {dec}")

            lines.append(f"  {market_name}：{' / '.join(parts)}")

    return "\n".join(lines)


def parse_sockjs(raw: str) -> list:
    """
    SockJS frame:
    - o : open
    - h : heartbeat
    - a[...] : array of JSON strings
    """
    if not raw:
        return []

    if raw in ("o", "h"):
        return []

    if raw.startswith("c"):
        return [{"__control__": raw}]

    if raw[0] not in ("a", "m"):
        return []

    try:
        frames = json.loads(raw[1:])
    except Exception as e:
        safe_print(f"[{ts()}] [sockjs parse error] raw_head={raw[:120]!r} err={e}")
        return []

    if not isinstance(frames, list):
        frames = [frames]

    results = []
    for f in frames:
        try:
            if isinstance(f, str):
                results.append(json.loads(f))
            else:
                results.append(f)
        except Exception as e:
            safe_print(f"[{ts()}] [inner json parse error] frame_head={str(f)[:120]!r} err={e}")

    return results


def handle_payload(payload: dict):
    if "__control__" in payload:
        safe_print(f"[{ts()}] [ws control] {payload['__control__']}")
        return

    ntype = payload.get("notificationType", "")

    if ntype == "LISTENING_STARTED":
        safe_print(f"[{ts()}] [✓] 訂閱成功，等待資料推送...")
        return

    if ntype != "CONTENT_CHANGES":
        # 先保留 debug
        safe_print(f"[{ts()}] [payload] notificationType={ntype}")
        return

    for item in payload.get("data", []):
        ctype = item.get("contentId", {}).get("type", "")
        change = item.get("change", {})

        safe_print(f"[{ts()}] [content] type={ctype}")

        if ctype == "eventGroup":
            events = change.get("events", [])
            name = change.get("name", "")
            if events:
                safe_print(format_odds(events, name))
            else:
                safe_print(f"[{ts()}] [eventGroup] no events")

        elif ctype == "inplayLiveDataEventSummaryListByTournament":
            summaries = change.get("eventSummaries", [])
            safe_print(f"[{ts()}] [live summaries] count={len(summaries)}")

        elif ctype == "boNavigationList":
            name = change.get("name", "")
            num = change.get("numevents", "0")
            safe_print(f"[{ts()}] [nav] {name}: {num}")

        else:
            # 未知類型先印前面一小段
            preview = json.dumps(item, ensure_ascii=False)[:300]
            safe_print(f"[{ts()}] [unknown content] {preview}")


def run():
    with sync_playwright() as p:
        safe_print("=" * 60)
        safe_print("台灣運彩 debug 版")
        safe_print("=" * 60)

        browser = p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = browser.new_context()
        page = context.new_page()

        ws_connected = False
        last_ws_message_time = None

        # ----------------------------
        # Request / Response / Frame Debug
        # ----------------------------
        def on_request(req):
            url = req.url
            if any(k in url for k in ["sportslottery", "talo-ssb", "notification", "sockjs"]):
                safe_print(f"[{ts()}] [req] {req.resource_type:10} {url}")

        def on_response(res):
            url = res.url
            if any(k in url for k in ["sportslottery", "talo-ssb", "notification", "sockjs"]):
                safe_print(f"[{ts()}] [res] {res.status:3} {url}")

                if "talo-ssb" in url:
                    try:
                        headers = res.headers
                        picked = {
                            "x-frame-options": headers.get("x-frame-options"),
                            "content-security-policy": headers.get("content-security-policy"),
                            "location": headers.get("location"),
                        }
                        safe_print(f"[{ts()}] [res headers] {picked}")
                    except Exception as e:
                        safe_print(f"[{ts()}] [res headers error] {e}")

        def on_request_failed(req):
            safe_print(f"[{ts()}] [req failed] {req.resource_type:10} {req.url} :: {req.failure}")

        def on_frame_navigated(frame):
            safe_print(f"[{ts()}] [frame nav] {frame.url}")

        page.on("request", on_request)
        page.on("response", on_response)
        page.on("requestfailed", on_request_failed)
        page.on("framenavigated", on_frame_navigated)

        # ----------------------------
        # WebSocket Debug
        # ----------------------------
        def on_frame_received(frame):
            nonlocal last_ws_message_time
            last_ws_message_time = time.time()

            payload = frame.payload
            if not isinstance(payload, str):
                payload = payload.decode("utf-8", errors="ignore")

            if not payload:
                return

            if payload in ("o", "h"):
                safe_print(f"[{ts()}] [ws heartbeat/open] {payload}")
                return

            if payload.startswith("c"):
                safe_print(f"[{ts()}] [ws closed] {payload[:100]}")
                return

            safe_print(f"[{ts()}] [ws recv raw] {payload[:200]}")

            for pld in parse_sockjs(payload):
                handle_payload(pld)

        def on_websocket(ws):
            nonlocal ws_connected
            ws_connected = True

            safe_print(f"[{ts()}] [ws found] {ws.url}")

            ws.on("framereceived", on_frame_received)
            ws.on("framesent", lambda f: safe_print(
                f"[{ts()}] [ws sent] {(f.payload if isinstance(f.payload, str) else f.payload.decode('utf-8', errors='ignore'))[:200]}"
            ))
            ws.on("close", lambda: safe_print(f"[{ts()}] [ws close] {ws.url}"))

        page.on("websocket", on_websocket)

        # ----------------------------
        # Goto
        # ----------------------------
        safe_print(f"[{ts()}] [browser] 開啟：{TARGET_URL}")
        response = page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)

        if response is not None:
            safe_print(f"[{ts()}] [browser] 頁面 status: {response.status}")
        else:
            safe_print(f"[{ts()}] [browser] response is None")

        safe_print(f"[{ts()}] [browser] 頁面 URL: {page.url}")
        safe_print(f"[{ts()}] [browser] 頁面標題: {page.title()}")

        try:
            html = page.content()
            safe_print(f"[{ts()}] [browser] HTML 前段:\n{html[:DEBUG_HTML_PREVIEW_LEN]}")
        except Exception as e:
            safe_print(f"[{ts()}] [browser] content() error: {e}")

        safe_print(f"[{ts()}] [browser] 等待 {WAIT_AFTER_GOTO_MS/1000:.0f} 秒讓 app 初始化...\n")
        page.wait_for_timeout(WAIT_AFTER_GOTO_MS)

        # ----------------------------
        # Body text debug
        # ----------------------------
        try:
            body_text = page.locator("body").inner_text(timeout=3000)
            safe_print(f"[{ts()}] [body text]\n{body_text[:DEBUG_BODY_PREVIEW_LEN]}")
        except Exception as e:
            safe_print(f"[{ts()}] [body text error] {e}")

        # ----------------------------
        # Frame debug
        # ----------------------------
        frames = page.frames
        safe_print(f"[{ts()}] [frames] 共 {len(frames)} 個 frame")

        for i, f in enumerate(frames):
            safe_print(f"[{ts()}]   frame[{i}] url={f.url!r}")
            try:
                txt = f.locator("body").inner_text(timeout=2000)
                safe_print(f"[{ts()}]   frame[{i}] text[:{DEBUG_FRAME_BODY_PREVIEW_LEN}] = {txt[:DEBUG_FRAME_BODY_PREVIEW_LEN]!r}")
            except Exception as e:
                safe_print(f"[{ts()}]   frame[{i}] body read error = {e}")

        safe_print(f"[{ts()}] [ws_connected] {ws_connected}")
        safe_print(f"[{ts()}] [debug] 進入持續監控，按 Ctrl+C 停止\n")

        # ----------------------------
        # Main loop
        # ----------------------------
        try:
            while True:
                time.sleep(1)

                if page.is_closed():
                    safe_print(f"[{ts()}] [browser] 頁面已關閉")
                    break

                if last_ws_message_time is not None:
                    idle = time.time() - last_ws_message_time
                    if idle > 20:
                        safe_print(f"[{ts()}] [warn] websocket 超過 {int(idle)} 秒沒收到資料")
                else:
                    # 完全沒出現 ws
                    pass

        except KeyboardInterrupt:
            safe_print(f"\n[{ts()}] [stop] 使用者中止")

        finally:
            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass


if __name__ == "__main__":
    run()