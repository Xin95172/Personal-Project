# 這個要直接在 terminal 跑

import re
import json
import pandas as pd
from playwright.sync_api import sync_playwright


def first_present(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def scalar_value(value):
    """Convert nested API values into a stable scalar for output/deduping."""
    if value is None:
        return None

    if isinstance(value, list):
        if not value:
            return None
        return scalar_value(value[0])

    if isinstance(value, dict):
        for key in (
            "value",
            "text",
            "name",
            "label",
            "address",
            "area",
            "price",
            "total",
            "date",
            "month",
            "id",
        ):
            if key in value:
                nested = scalar_value(value[key])
                if nested is not None:
                    return nested

        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def dedupe_value(value):
    value = scalar_value(value)

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return value


def make_dedupe_key(item):
    return (
        dedupe_value(first_present(item.get("id"), item.get("case_id"), item.get("house_id"))),
        dedupe_value(item.get("address")),
        dedupe_value(first_present(item.get("trans_date"), item.get("month"))),
        dedupe_value(first_present(item.get("total_price_v"), item.get("total_price"))),
        dedupe_value(first_present(item.get("build_area_v"), item.get("build_area"))),
        dedupe_value(
            first_present(
                item.get("original_shift_floor"),
                item.get("shift_floor"),
                item.get("shift_floor_val"),
            )
        ),
    )


def to_number(x):
    x = scalar_value(x)

    if x is None:
        return None

    x = str(x).strip()
    x = x.replace("萬", "")
    x = x.replace("坪", "")
    x = x.replace("樓", "")
    x = x.replace(",", "")

    nums = re.findall(r"-?\d+\.?\d*", x)

    if not nums:
        return None

    return pd.to_numeric(nums[0], errors="coerce")


def parse_floor(x):
    """
    5樓 -> 5
    1,2樓 -> 1
    地下1樓 -> -1
    """
    x = scalar_value(x)

    if x is None:
        return None

    s = str(x).strip()

    if "地下" in s or "B" in s.upper():
        nums = re.findall(r"\d+", s)
        if nums:
            return -int(nums[0])
        return None

    nums = re.findall(r"\d+", s)

    if not nums:
        return None

    nums = [int(n) for n in nums]

    return min(nums)


def to_date_tw(x):
    x = scalar_value(x)

    if x is None:
        return pd.NaT

    s = str(x).strip().replace("/", "-")
    nums = re.findall(r"\d+", s)

    if len(nums) < 2:
        return pd.NaT

    year = int(nums[0])
    month = int(nums[1])
    day = int(nums[2]) if len(nums) >= 3 else 1

    if year < 1911:
        year += 1911

    return pd.to_datetime(f"{year}-{month}-{day}", errors="coerce")


def calc_age(trans_date, build_date):
    trans_date = to_date_tw(trans_date)
    build_date = to_date_tw(build_date)

    if pd.isna(trans_date) or pd.isna(build_date):
        return None

    age = (trans_date - build_date).days / 365.25

    if age < 0:
        return None

    return round(age, 2)


def is_transaction_item(obj):
    """
    判斷某個 dict 是否像是一筆實價登錄資料。
    """
    if not isinstance(obj, dict):
        return False

    keys = set(obj.keys())

    signals = {
        "id",
        "total_price_v",
        "trans_date",
        "build_area_v",
        "shift_floor_val",
        "original_total_floor",
        "has_park",
        "park_count",
        "address",
    }

    return len(keys & signals) >= 3


def find_transaction_items(obj, results=None, depth=0):
    """
    遞迴搜尋 JSON 裡所有像實價登錄交易的 dict。
    """
    if results is None:
        results = []

    if depth > 12:
        return results

    if isinstance(obj, dict):
        if is_transaction_item(obj):
            results.append(obj)

        for value in obj.values():
            find_transaction_items(value, results, depth + 1)

    elif isinstance(obj, list):
        for item in obj:
            find_transaction_items(item, results, depth + 1)

    return results


def extract_nuxt_items(page):
    js = """
    () => {
        const nuxt = window.__NUXT__;
        return nuxt || {};
    }
    """

    try:
        nuxt_data = page.evaluate(js)
        return find_transaction_items(nuxt_data)
    except Exception:
        return []


def get_mid_high_floor_dummy(item, floor, total_floor):
    tags = item.get("tag") or item.get("tags") or []

    if isinstance(tags, str):
        tags = [tags]

    tag_text = " ".join([str(t) for t in tags])

    # 優先相信 591 自己的樓層標籤
    if "高樓層" in tag_text or "中樓層" in tag_text:
        return 1

    if "低樓層" in tag_text:
        return 0

    # 沒有標籤時，才用樓層比例判斷
    if floor is not None and total_floor is not None:
        if pd.notna(floor) and pd.notna(total_floor) and total_floor > 0:
            return 1 if floor >= total_floor / 2 else 0

    return 0


def normalize_items(raw_items):
    item_map = {}

    for item in raw_items:
        if not isinstance(item, dict):
            continue

        key = make_dedupe_key(item)
        item_map[key] = item

    return list(item_map.values())


def build_dataframe(raw_items):
    rows = []

    for item in raw_items:
        case_id = scalar_value(
            item.get("id")
            or item.get("case_id")
            or item.get("house_id")
        )

        month = scalar_value(
            item.get("month")
            or item.get("trans_month")
            or item.get("year_month")
        )

        trans_date = scalar_value(
            item.get("trans_date")
            or item.get("deal_date")
            or month
        )

        build_date = scalar_value(
            item.get("build_date")
            or item.get("building_date")
            or item.get("complete_date")
        )

        total_price = to_number(
            item.get("total_price_v")
            or item.get("total_price")
            or item.get("price")
        )

        build_area = to_number(
            item.get("build_area_v")
            or item.get("build_area")
            or item.get("area")
        )

        floor = parse_floor(
            item.get("original_shift_floor")
            or item.get("shift_floor")
            or item.get("shift_floor_val")
            or item.get("floor")
            or item.get("floor_v")
        )

        total_floor = to_number(
            item.get("original_total_floor")
            or item.get("total_floor")
            or item.get("total_floor_v")
        )

        age = calc_age(trans_date, build_date)
        age_sq = age ** 2 if age is not None else None

        park_count = to_number(item.get("park_count"))

        has_park_raw = (
            item.get("has_park")
            or item.get("is_has_park")
            or item.get("parking")
        )
        has_park_raw = scalar_value(has_park_raw)

        has_parking = 1 if str(has_park_raw) == "1" or (park_count is not None and park_count > 0) else 0

        mid_high_floor = get_mid_high_floor_dummy(item, floor, total_floor)

        rows.append({
            "Case_ID": case_id,
            "成交年月": month,
            "成交日期": trans_date,
            "成交總價_含車位_萬元": total_price,
            "建坪_含車位": build_area,
            "樓層": floor,
            "總樓層": total_floor,
            "屋齡": age,
            "屋齡平方": age_sq,
            "有車位虛擬變數": has_parking,
            "中高樓層虛擬變數": mid_high_floor,
            "地址": scalar_value(item.get("address")),
        })

    df = pd.DataFrame(rows)

    if "Case_ID" in df.columns:
        df = df[df["Case_ID"].notna()].copy()

    df = df.drop_duplicates(
        subset=["Case_ID", "成交日期", "成交總價_含車位_萬元", "建坪_含車位", "地址"],
        keep="first"
    )    
    return df


def scrape_data(
    url="https://market.591.com.tw/5899993/price?trans_type=1",
    output_file="591_實價登錄整理.xlsx",
    headless=False
):
    collected_items = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)

        page = browser.new_page(
            viewport={"width": 1366, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )

        def handle_response(response):
            try:
                content_type = response.headers.get("content-type", "")

                if "application/json" not in content_type:
                    return

                data = response.json()
                found = find_transaction_items(data)

                if found:
                    collected_items.extend(found)

            except Exception:
                pass

        page.on("response", handle_response)

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        # 先抓初始資料
        nuxt_items = extract_nuxt_items(page)
        collected_items.extend(nuxt_items)

        # 往下滑，觸發後面分頁 / lazy load API
        max_scroll = 120
        no_new_limit = 5
        target_count = 46

        last_unique_count = 0
        no_new_count = 0

        # 先把滑鼠移到畫面中央偏下，模擬人真的在列表區滑動
        page.mouse.move(700, 700)
        page.wait_for_timeout(500)

        for i in range(max_scroll):
            # 用小段慢慢滑，不要一次滑太大
            for _ in range(5):
                page.mouse.wheel(0, 700)
                page.wait_for_timeout(500)

            raw_items_now = normalize_items(collected_items)
            current_unique_count = len(raw_items_now)

            print(f"第 {i + 1} 次滑動，目前去重後 {current_unique_count} 筆")

            if current_unique_count >= target_count:
                print(f"已抓到目標筆數 {target_count}，停止滑動")
                break

            if current_unique_count > last_unique_count:
                last_unique_count = current_unique_count
                no_new_count = 0
            else:
                no_new_count += 1

            if no_new_count >= no_new_limit:
                print(f"連續 {no_new_limit} 次沒有新增資料，停止滑動")
                break

        page.wait_for_timeout(3000)
        print("\n===== DOM 畫面列檢查 =====")

        dom_rows = page.locator(".realprice-list-row")
        dom_count = dom_rows.count()

        print(f"DOM 畫面列數：{dom_count}")

        for i in range(dom_count):
            text = dom_rows.nth(i).inner_text()
            print("=" * 60)
            print(f"DOM row {i + 1}")
            print(text)
            
        browser.close()
    
    def debug_collected_items(raw_items, output_file="raw_debug_591.xlsx"):
        rows = []

        for idx, item in enumerate(raw_items):
            case_id = scalar_value(
                first_present(item.get("id"), item.get("case_id"), item.get("house_id"))
            )

            dedupe_key = make_dedupe_key(item)

            rows.append({
                "raw_index": idx,
                "dedupe_key": str(dedupe_key),
                "Case_ID": case_id,
                "month": scalar_value(item.get("month")),
                "trans_date": scalar_value(item.get("trans_date")),
                "total_price_v": scalar_value(item.get("total_price_v")),
                "build_area_v": scalar_value(item.get("build_area_v")),
                "floor": scalar_value(
                    first_present(
                        item.get("original_shift_floor"),
                        item.get("shift_floor"),
                        item.get("shift_floor_val"),
                    )
                ),
                "total_floor": scalar_value(
                    first_present(item.get("original_total_floor"), item.get("total_floor"))
                ),
                "address": scalar_value(item.get("address")),
            })

        debug_df = pd.DataFrame(rows)

        print("\n===== 原始資料檢查 =====")
        print(f"原始筆數：{len(debug_df)}")
        print(f"Case_ID 不重複筆數：{debug_df['Case_ID'].nunique()}")
        print(f"去重鍵不重複筆數：{debug_df['dedupe_key'].nunique()}")

        dup_keys = debug_df["dedupe_key"][debug_df["dedupe_key"].duplicated(keep=False)]

        if len(dup_keys) > 0:
            print("\n重複的去重鍵：")
            print(dup_keys.value_counts())

            print("\n重複資料明細：")
            dup_detail = debug_df[debug_df["dedupe_key"].isin(dup_keys)].sort_values("dedupe_key")
            print(dup_detail)
        else:
            print("\n沒有重複去重鍵。")

        debug_df.to_excel(output_file, index=False)
        print(f"\n已輸出原始檢查檔：{output_file}")

        return debug_df

    debug_collected_items(collected_items)

    raw_items = normalize_items(collected_items)

    print(f"抓到原始交易資料：{len(collected_items)} 筆")
    print(f"去重後交易資料：{len(raw_items)} 筆")

    df = build_dataframe(raw_items)

    # 排序：成交日期新到舊
    if "成交日期" in df.columns:
        df["_成交日期排序"] = pd.to_datetime(df["成交日期"], errors="coerce")
        df = df.sort_values("_成交日期排序", ascending=False)
        df = df.drop(columns=["_成交日期排序"])

    if output_file:
        df.to_excel(output_file, index=False)

    return df


if __name__ == "__main__":
    url = "https://market.591.com.tw/5899993/price?trans_type=1"

    df = scrape_data(
        url=url,
        output_file="591_實價登錄整理.xlsx",
        headless=True
    )

    print(df.head())
    print(f"完成，共輸出 {len(df)} 筆資料")
    print("已輸出：591_實價登錄整理.xlsx")
