import re
import time
from datetime import datetime
from playwright.sync_api import sync_playwright

TARGET_URL = "https://www.sportslottery.com.tw/sportsbook/sport/%E7%B1%83%E7%90%83/34765.1"
POLL_SECONDS = 3


def now_str():
    return datetime.now().strftime("%H:%M:%S")


def get_sportsbook_frame(page, timeout_ms=15000):
    deadline = time.time() + timeout_ms / 1000
    while time.time() < deadline:
        for frame in page.frames:
            if frame.url and "www-talo-ssb-pr.sportslottery.com.tw/sport/" in frame.url:
                return frame
        page.wait_for_timeout(500)
    raise TimeoutError("找不到 sportsbook iframe")


def extract_text(frame):
    return frame.locator("body").inner_text(timeout=10000)


def clean_lines(text: str):
    return [x.strip() for x in text.splitlines() if x.strip()]


def is_event_start(lines, i):
    """
    判斷某一行是否為一場賽事的起點
    目前依照你抓到的格式：
    307
    2
    06 5月
    07:00
    """
    if i + 3 >= len(lines):
        return False

    return (
        re.fullmatch(r"\d{3,4}", lines[i]) is not None and
        re.fullmatch(r"\d+", lines[i + 1]) is not None and
        re.fullmatch(r"\d{2}\s+\d+月", lines[i + 2]) is not None and
        re.fullmatch(r"\d{2}:\d{2}", lines[i + 3]) is not None
    )


def split_event_blocks(lines):
    starts = []
    for i in range(len(lines)):
        if is_event_start(lines, i):
            starts.append(i)

    blocks = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        blocks.append(lines[s:e])

    return blocks


def parse_event_block(block):
    """
    block 範例：
    307
    2
    06 5月
    07:00
    +
    26
    底特律活塞
    克里夫蘭騎士
    1.94
    1.53
    """
    if len(block) < 6:
        return None

    event_id = block[0]
    time_str = None
    date_str = None

    for x in block:
        if time_str is None and re.fullmatch(r"\d{2}:\d{2}", x):
            time_str = x
        if date_str is None and re.fullmatch(r"\d{2}\s+\d+月", x):
            date_str = x

    # 找賠率
    odds = [x for x in block if re.fullmatch(r"\d+\.\d{2}", x)]

    # 過濾非隊名
    team_candidates = []
    skip_exact = {
        "+", "主場", "客場", "賽事", "冠軍及特別項目", "媒體",
        "目前無直播服務", "登入後觀看", "僅開放會員登入後觀看",
        "投注單", "我的投注", "投注單空", "新增選項來下注",
        "載入投注代碼", "輸入您的投注代碼", "載入"
    }

    for x in block:
        if x in skip_exact:
            continue
        if re.fullmatch(r"\d{3,4}", x):
            continue
        if re.fullmatch(r"\d+", x):
            continue
        if re.fullmatch(r"\d{2}\s+\d+月", x):
            continue
        if re.fullmatch(r"\d{2}:\d{2}", x):
            continue
        if re.fullmatch(r"\d+\.\d{2}", x):
            continue
        if re.fullmatch(r"\(\d+\)", x):
            continue
        if len(x) < 2:
            continue

        team_candidates.append(x)

    # 只取最前面兩個隊名
    # 這裡比原本穩，因為只在單場 block 內抓，不會吃到下一場
    if len(team_candidates) >= 2:
        team1 = team_candidates[0]
        team2 = team_candidates[1]
    else:
        team1 = None
        team2 = None

    # 有些場次是鎖住，沒有 decimal odds
    if len(odds) >= 2:
        left_odds = odds[0]
        right_odds = odds[1]
        status = "OPEN"
    else:
        left_odds = None
        right_odds = None
        status = "LOCK"

    return {
        "event_id": event_id,
        "date": date_str,
        "time": time_str,
        "team1": team1,
        "team2": team2,
        "left_odds": left_odds,
        "right_odds": right_odds,
        "status": status,
    }


def parse_odds_from_text(text):
    lines = clean_lines(text)
    blocks = split_event_blocks(lines)

    rows = []
    for block in blocks:
        row = parse_event_block(block)
        if row and row["time"] and row["team1"] and row["team2"]:
            rows.append(row)

    return rows


def to_map(rows):
    return {
        (r["event_id"], r["time"], r["team1"], r["team2"]): (
            r["left_odds"],
            r["right_odds"],
            r["status"],
        )
        for r in rows
    }


def format_row(row):
    if row["status"] == "LOCK":
        return f'{row["event_id"]}  {row["time"]}  {row["team1"]} vs {row["team2"]}  LOCK'
    return f'{row["event_id"]}  {row["time"]}  {row["team1"]} vs {row["team2"]}  {row["left_odds"]} / {row["right_odds"]}'


def print_full_snapshot(rows):
    print(f"\n[{now_str()}] 初始賠率")
    for row in rows:
        print(format_row(row))


def print_changes(old_map, new_rows):
    new_map = to_map(new_rows)
    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())

    added = sorted(new_keys - old_keys)
    removed = sorted(old_keys - new_keys)
    changed = sorted(k for k in (old_keys & new_keys) if old_map[k] != new_map[k])

    if not added and not removed and not changed:
        return old_map, False

    print(f"\n[{now_str()}] 賠率更新")

    lookup = {(r["event_id"], r["time"], r["team1"], r["team2"]): r for r in new_rows}

    for k in added:
        print("[新增]", format_row(lookup[k]))

    for k in changed:
        old_left, old_right, old_status = old_map[k]
        new_left, new_right, new_status = new_map[k]
        event_id, time_str, team1, team2 = k

        old_text = "LOCK" if old_status == "LOCK" else f"{old_left} / {old_right}"
        new_text = "LOCK" if new_status == "LOCK" else f"{new_left} / {new_right}"

        print(f"[變動] {event_id}  {time_str}  {team1} vs {team2}  {old_text} -> {new_text}")

    for k in removed:
        event_id, time_str, team1, team2 = k
        old_left, old_right, old_status = old_map[k]
        old_text = "LOCK" if old_status == "LOCK" else f"{old_left} / {old_right}"
        print(f"[移除] {event_id}  {time_str}  {team1} vs {team2}  {old_text}")

    return new_map, True


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

        page.goto(TARGET_URL, wait_until="domcontentloaded", timeout=30000)
        frame = get_sportsbook_frame(page)

        print("[frame url]", frame.url)

        prev_map = None

        try:
            while True:
                text = extract_text(frame)
                rows = parse_odds_from_text(text)
                curr_map = to_map(rows)

                if prev_map is None:
                    print_full_snapshot(rows)
                    prev_map = curr_map
                else:
                    prev_map, _ = print_changes(prev_map, rows)

                time.sleep(POLL_SECONDS)

        except KeyboardInterrupt:
            print("\n停止監控")

        finally:
            browser.close()


if __name__ == "__main__":
    main()