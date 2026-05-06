import time
from datetime import datetime
from playwright.sync_api import sync_playwright

TARGET_URL = "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1"
POLL_SECONDS = 3


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def frac_to_decimal(up: str, down: str) -> str:
    try:
        return f"{1 + int(up) / int(down):.2f}"
    except Exception:
        return ""


def clean_text(s: str) -> str:
    return (s or "").replace("\r", "").strip()


def normalize_event_group_json(data: dict):
    """
    把 content/get 回來的 eventGroup JSON 轉成乾淨的賠率資料
    只抓 market.name == '不讓分'
    """
    out = []

    group = data.get("data", {})
    events = group.get("events", [])

    for event in events:
        away = clean_text(event.get("participantname_away"))
        home = clean_text(event.get("participantname_home"))
        tsstart = event.get("tsstart", "")
        time_str = tsstart[11:16] if len(tsstart) >= 16 else ""
        event_id = event.get("idfoevent", "")

        markets = event.get("markets", [])
        moneyline = None

        for m in markets:
            if clean_text(m.get("name")) == "不讓分":
                moneyline = m
                break

        if not moneyline:
            continue

        listcode = moneyline.get("listcode", "")
        selections = moneyline.get("selections", [])

        away_odds = None
        home_odds = None
        away_status = "LOCK"
        home_status = "LOCK"

        for sel in selections:
            had = sel.get("hadvalue")
            up = sel.get("currentpriceup")
            down = sel.get("currentpricedown")
            price = frac_to_decimal(up, down) if up and down else None

            # 開盤/鎖盤狀態
            bolife = sel.get("idfobolifestate", "")
            tradable = sel.get("idfoselectionsuspensiontype", "") == "" and bolife in ("N", "O")

            if had == "A":
                away_odds = price
                away_status = "OPEN" if price and tradable else "LOCK"
            elif had == "H":
                home_odds = price
                home_status = "OPEN" if price and tradable else "LOCK"

        out.append({
            "event_group_name": clean_text(group.get("name")),
            "event_id": event_id,
            "listcode": listcode,
            "time": time_str,
            "away": away,
            "home": home,
            "away_odds": away_odds,
            "home_odds": home_odds,
            "away_status": away_status,
            "home_status": home_status,
        })

    return out


def row_key(row):
    return (row["event_id"], row["listcode"], row["time"], row["away"], row["home"])


def row_value(row):
    return (
        row["away_odds"],
        row["home_odds"],
        row["away_status"],
        row["home_status"],
    )


def format_row(row):
    away_text = row["away_odds"] if row["away_status"] == "OPEN" and row["away_odds"] else "LOCK"
    home_text = row["home_odds"] if row["home_status"] == "OPEN" and row["home_odds"] else "LOCK"
    return f'{row["listcode"]}  {row["time"]}  {row["away"]} @ {row["home"]}  客:{away_text} / 主:{home_text}'


def print_snapshot(rows):
    print(f"\n[{now_str()}] 初始賠率")
    for r in sorted(rows, key=lambda x: (x["time"], x["listcode"])):
        print(format_row(r))


def print_changes(old_rows, new_rows):
    old_map = {row_key(r): row_value(r) for r in old_rows}
    new_map = {row_key(r): row_value(r) for r in new_rows}
    new_lookup = {row_key(r): r for r in new_rows}

    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in (old_keys & new_keys) if old_map[k] != new_map[k])

    if not added and not removed and not changed:
        return False

    print(f"\n[{now_str()}] 賠率更新")

    for k in added:
        print("[新增]", format_row(new_lookup[k]))

    for k in changed:
        old_away, old_home, old_away_status, old_home_status = old_map[k]
        r = new_lookup[k]
        new_away = r["away_odds"] if r["away_status"] == "OPEN" and r["away_odds"] else "LOCK"
        new_home = r["home_odds"] if r["home_status"] == "OPEN" and r["home_odds"] else "LOCK"
        old_away_text = old_away if old_away_status == "OPEN" and old_away else "LOCK"
        old_home_text = old_home if old_home_status == "OPEN" and old_home else "LOCK"

        print(
            f'[變動] {r["listcode"]}  {r["time"]}  {r["away"]} @ {r["home"]}  '
            f'客:{old_away_text}->{new_away} / 主:{old_home_text}->{new_home}'
        )

    for k in removed:
        event_id, listcode, time_str, away, home = k
        old_away, old_home, old_away_status, old_home_status = old_map[k]
        old_away_text = old_away if old_away_status == "OPEN" and old_away else "LOCK"
        old_home_text = old_home if old_home_status == "OPEN" and old_home else "LOCK"
        print(f"[移除] {listcode}  {time_str}  {away} @ {home}  客:{old_away_text} / 主:{old_home_text}")

    return True


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = browser.new_context()
        page = context.new_page()

        subscribed_event_groups = set()

        def on_request(req):
            if "/services/content/subscribe" not in req.url:
                return
            try:
                payload = req.post_data_json
                content_id = payload.get("contentId", {})
                if content_id.get("type") == "eventGroup":
                    egid = content_id.get("id")
                    if egid:
                        subscribed_event_groups.add(egid)
            except Exception:
                pass

        page.on("request", on_request)

        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(8000)

        print("[eventGroup ids]", sorted(subscribed_event_groups))

        if not subscribed_event_groups:
            print("沒有抓到 eventGroup id")
            browser.close()
            return

        def fetch_event_group(group_id: str):
            # 在頁面上下文 fetch，直接沿用瀏覽器 cookie / session / cf 狀態
            return page.evaluate(
                """
                async ({ groupId }) => {
                  const res = await fetch("https://www-talo-ssb-pr.sportslottery.com.tw/services/content/get", {
                    method: "POST",
                    headers: { "content-type": "application/json" },
                    body: JSON.stringify({
                      contentId: { type: "eventGroup", id: groupId },
                      clientContext: { language: "ZH", ipAddress: "0.0.0.0" }
                    })
                  });
                  return await res.json();
                }
                """,
                {"groupId": group_id}
            )

        prev_rows = None

        try:
            while True:
                all_rows = []

                for egid in sorted(subscribed_event_groups):
                    try:
                        data = fetch_event_group(egid)
                        rows = normalize_event_group_json(data)
                        all_rows.extend(rows)
                    except Exception as e:
                        print(f"[{now_str()}] [error] eventGroup {egid}: {e}")

                # 去重
                dedup = {}
                for r in all_rows:
                    dedup[row_key(r)] = r
                rows = list(dedup.values())

                if prev_rows is None:
                    print_snapshot(rows)
                    prev_rows = rows
                else:
                    if print_changes(prev_rows, rows):
                        prev_rows = rows
                time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("\n停止監控")

        finally:
            browser.close()


if __name__ == "__main__":
    main()