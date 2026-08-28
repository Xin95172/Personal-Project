import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from dataclasses import dataclass, field


# ─── 設定 ───────────────────────────────────────────────────
DEFAULT_URL = "https://magicbox.mg/ajax/post"  # ← 請確認實際 endpoint

COOKIES = {
    "user_session": "c15a9a9ddc82191eece3da260af43823",
    "PHPSESSID": "5ut3hq1b48odele2bqqc2lhdlu",
    "c_user": "13918",
    "xs": "cf2d691f2b02069a814bd43b71b954ff",
    "user_jwt": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
}

HEADERS = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    "X-Requested-With": "XMLHttpRequest",
    "Origin": "https://magicbox.mg",
    "Referer": "https://magicbox.mg/posts/183385",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
}


# ─── 結果資料結構 ───────────────────────────────────────────
@dataclass
class StressTestResult:
    """壓力測試結果"""
    mode: str = ""                              # 測試模式名稱
    total_requests: int = 0                     # 總請求數
    success_count: int = 0                      # 成功數 (HTTP 200)
    fail_count: int = 0                         # 失敗數 (非 200 或例外)
    timeout_count: int = 0                      # 逾時次數
    total_elapsed: float = 0.0                  # 總耗時 (秒)
    request_times: list = field(default_factory=list)       # 每次請求耗時
    status_codes: list = field(default_factory=list)        # HTTP 狀態碼列表
    response_sizes: list = field(default_factory=list)      # 回應大小 (bytes)
    response_valid: list = field(default_factory=list)      # 回應內容是否合法
    errors: list = field(default_factory=list)              # 錯誤訊息列表

    # ── 計算屬性 ──
    @property
    def rps(self):
        """每秒請求數 (吞吐量)"""
        return self.success_count / self.total_elapsed if self.total_elapsed > 0 else 0

    @property
    def avg_time(self):
        return statistics.mean(self.request_times) if self.request_times else 0

    @property
    def min_time(self):
        return min(self.request_times) if self.request_times else 0

    @property
    def max_time(self):
        return max(self.request_times) if self.request_times else 0

    @property
    def p50(self):
        return self._percentile(50)

    @property
    def p95(self):
        return self._percentile(95)

    @property
    def p99(self):
        return self._percentile(99)

    @property
    def timeout_rate(self):
        """逾時率 (%)"""
        return (self.timeout_count / self.total_requests * 100) if self.total_requests > 0 else 0

    @property
    def status_code_dist(self):
        """HTTP 狀態碼分布"""
        return dict(Counter(self.status_codes))

    @property
    def avg_response_size(self):
        """平均回應大小 (bytes)"""
        return statistics.mean(self.response_sizes) if self.response_sizes else 0

    @property
    def valid_rate(self):
        """回應內容合法率 (%)"""
        if not self.response_valid:
            return 0
        return sum(self.response_valid) / len(self.response_valid) * 100

    def _percentile(self, p):
        if not self.request_times:
            return 0
        sorted_times = sorted(self.request_times)
        idx = int(len(sorted_times) * p / 100)
        idx = min(idx, len(sorted_times) - 1)
        return sorted_times[idx]

    def print_report(self):
        """印出完整報告"""
        print("\n" + "=" * 55)
        print(f"  壓力測試結果 — {self.mode}")
        print("=" * 55)

        # 基本統計
        print(f"  總請求數:         {self.total_requests}")
        print(f"  成功發文數:       {self.success_count}")
        print(f"  失敗次數:         {self.fail_count}")
        print(f"  逾時次數:         {self.timeout_count}  ({self.timeout_rate:.1f}%)")
        print(f"  總耗時:           {self.total_elapsed:.3f} 秒")
        print(f"  吞吐量 (RPS):     {self.rps:.2f} 請求/秒")

        # 延遲分布
        print("-" * 55)
        print(f"  平均上傳時間:     {self.avg_time:.3f} 秒")
        print(f"  最快一次:         {self.min_time:.3f} 秒")
        print(f"  最慢一次:         {self.max_time:.3f} 秒")
        print(f"  P50 延遲:         {self.p50:.3f} 秒")
        print(f"  P95 延遲:         {self.p95:.3f} 秒")
        print(f"  P99 延遲:         {self.p99:.3f} 秒")

        # 狀態碼分布
        print("-" * 55)
        print("  HTTP 狀態碼分布:")
        for code, count in sorted(self.status_code_dist.items()):
            pct = count / self.total_requests * 100 if self.total_requests else 0
            label = {200: "成功", 429: "被限速", 500: "伺服器錯誤", 403: "被封鎖"}.get(code, "")
            print(f"    {code} {label}: {count} ({pct:.1f}%)")

        # 回應內容
        print("-" * 55)
        print(f"  平均回應大小:     {self.avg_response_size:.0f} bytes")
        print(f"  回應合法率:       {self.valid_rate:.1f}%")

        # 錯誤摘要
        if self.errors:
            print("-" * 55)
            error_counts = Counter(self.errors)
            print(f"  錯誤摘要 (共 {len(self.errors)} 個):")
            for err, cnt in error_counts.most_common(5):
                print(f"    [{cnt}x] {err}")

        print("=" * 55)

    def to_dict(self):
        """轉成 dict 方便後續分析"""
        return {
            "mode": self.mode,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "timeout_count": self.timeout_count,
            "timeout_rate": self.timeout_rate,
            "total_elapsed": self.total_elapsed,
            "rps": self.rps,
            "avg_time": self.avg_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "status_code_dist": self.status_code_dist,
            "avg_response_size": self.avg_response_size,
            "valid_rate": self.valid_rate,
            "request_times": self.request_times,
        }


# ─── 驗證回應內容 ──────────────────────────────────────────
def _validate_response(response):
    """
    檢查回應內容是否代表真正發文成功。
    依你的 API 實際回傳格式調整判斷邏輯。
    """
    try:
        data = response.json()
        # 常見判斷: status/success 欄位、或沒有 error key
        if isinstance(data, dict):
            if data.get("status") == "error" or "error" in data:
                return False
        return True
    except Exception:
        # 回傳不是 JSON → 可能是錯誤頁面
        return False


# ─── 單次請求 (給所有模式共用) ──────────────────────────────
def _send_one(session, url, request_id):
    """
    發送一次 POST，回傳 (耗時, 狀態碼, 回應大小, 是否合法, 錯誤訊息|None)
    """
    payload = {
        "handle": "post",
        "id": "183385",
        "message": f"python-test-{int(time.time())}-{request_id}",
    }

    req_start = time.perf_counter()
    try:
        resp = session.post(url, data=payload, timeout=10)
        elapsed = time.perf_counter() - req_start
        is_valid = _validate_response(resp)
        return elapsed, resp.status_code, len(resp.content), is_valid, None
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - req_start
        return elapsed, None, 0, False, "Timeout"
    except requests.exceptions.ConnectionError as e:
        elapsed = time.perf_counter() - req_start
        return elapsed, None, 0, False, f"ConnectionError: {e}"
    except Exception as e:
        elapsed = time.perf_counter() - req_start
        return elapsed, None, 0, False, f"{type(e).__name__}: {e}"


def _make_session():
    """建立帶 cookies/headers 的 Session (連線復用)"""
    s = requests.Session()
    s.headers.update(HEADERS)
    s.cookies.update(COOKIES)
    return s


def _collect(result, elapsed, status_code, size, is_valid, error):
    """把單次結果收進 StressTestResult"""
    result.request_times.append(elapsed)
    result.response_sizes.append(size)
    result.response_valid.append(is_valid)

    if error:
        result.errors.append(error)
        result.fail_count += 1
        if "Timeout" in str(error):
            result.timeout_count += 1
    elif status_code == 200:
        result.success_count += 1
        result.status_codes.append(status_code)
    else:
        result.fail_count += 1
        result.status_codes.append(status_code)
        # 偵測 Rate Limiting
        if status_code == 429:
            result.timeout_count += 0  # 不算 timeout，但狀態碼會被記錄


# ═══════════════════════════════════════════════════════════
#  模式 1: 單線程連續測試 (基本款)
# ═══════════════════════════════════════════════════════════
def run_basic(url=DEFAULT_URL, total_requests=10000, report_every=100):
    """
    單線程連續發送，最基本的壓力測試。
    """
    result = StressTestResult(mode="單線程連續測試", total_requests=total_requests)
    session = _make_session()

    total_start = time.perf_counter()
    for i in range(1, total_requests + 1):
        elapsed, status, size, valid, error = _send_one(session, url, i)
        _collect(result, elapsed, status, size, valid, error)

        if i % report_every == 0:
            print(
                f"[進度] {i}/{total_requests} | "
                f"成功: {result.success_count} | 失敗: {result.fail_count} | "
                f"平均: {result.avg_time:.3f}s | RPS: {result.success_count / (time.perf_counter() - total_start):.1f}"
            )

    result.total_elapsed = time.perf_counter() - total_start
    session.close()
    result.print_report()
    return result


# ═══════════════════════════════════════════════════════════
#  模式 2: 併發測試
# ═══════════════════════════════════════════════════════════
def run_concurrent(url=DEFAULT_URL, total_requests=10000, workers=10, report_every=100):
    """
    用 ThreadPoolExecutor 開 N 個線程同時發送，模擬多人同時操作。
    """
    result = StressTestResult(
        mode=f"併發測試 ({workers} 線程)",
        total_requests=total_requests,
    )
    session = _make_session()
    completed = 0

    total_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_send_one, session, url, i): i
            for i in range(1, total_requests + 1)
        }

        for future in as_completed(futures):
            elapsed, status, size, valid, error = future.result()
            _collect(result, elapsed, status, size, valid, error)
            completed += 1

            if completed % report_every == 0:
                wall = time.perf_counter() - total_start
                print(
                    f"[進度] {completed}/{total_requests} | "
                    f"成功: {result.success_count} | 失敗: {result.fail_count} | "
                    f"平均: {result.avg_time:.3f}s | RPS: {result.success_count / wall:.1f}"
                )

    result.total_elapsed = time.perf_counter() - total_start
    session.close()
    result.print_report()
    return result


# ═══════════════════════════════════════════════════════════
#  模式 3: 漸進式壓力測試
# ═══════════════════════════════════════════════════════════
def run_gradual(url=DEFAULT_URL, stages=None, requests_per_stage=200):
    """
    逐步增加併發數，找出伺服器承受上限。

    Args:
        stages: 每個階段的線程數列表，預設 [1, 5, 10, 25, 50]
        requests_per_stage: 每個階段發幾篇
    """
    if stages is None:
        stages = [1, 5, 10, 25, 50]

    all_results = []
    print("=" * 55)
    print("  漸進式壓力測試")
    print("=" * 55)

    for workers in stages:
        print(f"\n>>> 階段: {workers} 線程 × {requests_per_stage} 請求")
        result = StressTestResult(
            mode=f"漸進式 - {workers} 線程",
            total_requests=requests_per_stage,
        )
        session = _make_session()
        total_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_send_one, session, url, i): i
                for i in range(1, requests_per_stage + 1)
            }
            for future in as_completed(futures):
                elapsed, status, size, valid, error = future.result()
                _collect(result, elapsed, status, size, valid, error)

        result.total_elapsed = time.perf_counter() - total_start
        session.close()

        print(
            f"    RPS: {result.rps:.1f} | "
            f"平均: {result.avg_time:.3f}s | "
            f"P95: {result.p95:.3f}s | "
            f"成功率: {result.success_count / result.total_requests * 100:.1f}%"
        )
        all_results.append(result)

    # 印出對比表
    print("\n" + "=" * 55)
    print("  漸進式壓力對比")
    print("=" * 55)
    print(f"  {'線程':>4} | {'RPS':>8} | {'平均':>8} | {'P95':>8} | {'P99':>8} | {'成功率':>6}")
    print("  " + "-" * 52)
    for r in all_results:
        sr = r.success_count / r.total_requests * 100 if r.total_requests else 0
        workers = r.mode.split(" - ")[1].replace(" 線程", "")
        print(f"  {workers:>4} | {r.rps:>7.1f} | {r.avg_time:>7.3f}s | {r.p95:>7.3f}s | {r.p99:>7.3f}s | {sr:>5.1f}%")
    print("=" * 55)

    return all_results


# ═══════════════════════════════════════════════════════════
#  模式 4: 持續時間測試
# ═══════════════════════════════════════════════════════════
def run_duration(url=DEFAULT_URL, duration_seconds=300, workers=5, report_every_sec=30):
    """
    不設定次數，持續跑指定秒數，觀察效能是否衰退。

    Args:
        duration_seconds: 持續秒數 (預設 300 = 5 分鐘)
        workers: 併發線程數
        report_every_sec: 每幾秒印一次中間報告
    """
    result = StressTestResult(mode=f"持續測試 ({duration_seconds}s, {workers} 線程)")
    session = _make_session()
    request_id = 0
    last_report = time.perf_counter()

    print(f"開始持續測試: {duration_seconds} 秒, {workers} 線程")
    total_start = time.perf_counter()
    deadline = total_start + duration_seconds

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 持續提交任務直到時間到
        futures = []
        while time.perf_counter() < deadline:
            request_id += 1
            f = executor.submit(_send_one, session, url, request_id)
            futures.append(f)

            # 避免 futures 堆積太多，定期清理已完成的
            if len(futures) >= workers * 10:
                done_futures = [ft for ft in futures if ft.done()]
                for ft in done_futures:
                    elapsed, status, size, valid, error = ft.result()
                    _collect(result, elapsed, status, size, valid, error)
                    result.total_requests += 1
                futures = [ft for ft in futures if not ft.done()]

            # 定期報告
            now = time.perf_counter()
            if now - last_report >= report_every_sec:
                wall = now - total_start
                print(
                    f"[{wall:.0f}s] "
                    f"已發: {result.total_requests} | "
                    f"成功: {result.success_count} | "
                    f"RPS: {result.success_count / wall:.1f} | "
                    f"平均: {result.avg_time:.3f}s"
                )
                last_report = now

        # 等待剩餘的 futures 完成
        for ft in futures:
            elapsed, status, size, valid, error = ft.result()
            _collect(result, elapsed, status, size, valid, error)
            result.total_requests += 1

    result.total_elapsed = time.perf_counter() - total_start
    session.close()
    result.print_report()
    return result


# ═══════════════════════════════════════════════════════════
#  模式 5: 突發流量測試
# ═══════════════════════════════════════════════════════════
def run_burst(url=DEFAULT_URL, idle_requests=50, burst_requests=500, burst_workers=50, cycles=3):
    """
    模擬突發流量: 平時慢慢發 → 突然一瞬間灌大量請求 → 再恢復平靜。

    Args:
        idle_requests: 平靜期每次發幾篇 (單線程)
        burst_requests: 突發期一次灌幾篇
        burst_workers: 突發期併發線程數
        cycles: 重複幾輪
    """
    all_results = []
    print("=" * 55)
    print("  突發流量測試")
    print(f"  {cycles} 輪 × (平靜 {idle_requests} 篇 → 突發 {burst_requests} 篇)")
    print("=" * 55)

    for cycle in range(1, cycles + 1):
        print(f"\n─── 第 {cycle} 輪 ───")

        # 平靜期 (單線程慢慢發)
        print(f"  平靜期: 單線程 × {idle_requests} 篇...")
        idle_result = StressTestResult(
            mode=f"突發-平靜期 (第{cycle}輪)",
            total_requests=idle_requests,
        )
        session = _make_session()
        t = time.perf_counter()
        for i in range(1, idle_requests + 1):
            elapsed, status, size, valid, error = _send_one(session, url, i)
            _collect(idle_result, elapsed, status, size, valid, error)
        idle_result.total_elapsed = time.perf_counter() - t
        print(f"    RPS: {idle_result.rps:.1f} | 平均: {idle_result.avg_time:.3f}s")

        # 突發期 (大量併發)
        print(f"  突發期: {burst_workers} 線程 × {burst_requests} 篇...")
        burst_result = StressTestResult(
            mode=f"突發-爆發期 (第{cycle}輪)",
            total_requests=burst_requests,
        )
        t = time.perf_counter()
        with ThreadPoolExecutor(max_workers=burst_workers) as executor:
            futures = {
                executor.submit(_send_one, session, url, i): i
                for i in range(1, burst_requests + 1)
            }
            for future in as_completed(futures):
                elapsed, status, size, valid, error = future.result()
                _collect(burst_result, elapsed, status, size, valid, error)
        burst_result.total_elapsed = time.perf_counter() - t
        session.close()

        sr = burst_result.success_count / burst_result.total_requests * 100
        print(
            f"    RPS: {burst_result.rps:.1f} | 平均: {burst_result.avg_time:.3f}s | "
            f"P99: {burst_result.p99:.3f}s | 成功率: {sr:.1f}%"
        )

        all_results.append({"idle": idle_result, "burst": burst_result})

    # 對比表
    print("\n" + "=" * 55)
    print("  突發流量對比")
    print("=" * 55)
    print(f"  {'輪次':>4} | {'階段':>6} | {'RPS':>7} | {'平均':>8} | {'P95':>8} | {'成功率':>6}")
    print("  " + "-" * 52)
    for i, r in enumerate(all_results, 1):
        for phase, res in r.items():
            label = "平靜" if phase == "idle" else "突發"
            sr = res.success_count / res.total_requests * 100 if res.total_requests else 0
            print(f"  {i:>4} | {label:>6} | {res.rps:>6.1f} | {res.avg_time:>7.3f}s | {res.p95:>7.3f}s | {sr:>5.1f}%")
    print("=" * 55)

    return all_results


# ═══════════════════════════════════════════════════════════
#  模式 6: Session 復用 vs 不復用 對比
# ═══════════════════════════════════════════════════════════
def run_session_comparison(url=DEFAULT_URL, total_requests=200):
    """
    比較 requests.Session (連線復用) vs 每次新建連線的效能差異。
    """
    print("=" * 55)
    print("  Session 復用 vs 不復用 對比")
    print("=" * 55)

    # ── 有 Session (連線復用) ──
    print(f"\n  使用 Session (連線復用) × {total_requests} 篇...")
    result_session = StressTestResult(mode="有 Session", total_requests=total_requests)
    session = _make_session()
    t = time.perf_counter()
    for i in range(1, total_requests + 1):
        elapsed, status, size, valid, error = _send_one(session, url, i)
        _collect(result_session, elapsed, status, size, valid, error)
    result_session.total_elapsed = time.perf_counter() - t
    session.close()

    # ── 無 Session (每次新建連線) ──
    print(f"  不使用 Session (每次新建) × {total_requests} 篇...")
    result_no_session = StressTestResult(mode="無 Session", total_requests=total_requests)
    t = time.perf_counter()
    for i in range(1, total_requests + 1):
        payload = {
            "handle": "post",
            "id": "183385",
            "message": f"python-test-{int(time.time())}-{i}",
        }
        req_start = time.perf_counter()
        try:
            resp = requests.post(
                url, headers=HEADERS, cookies=COOKIES, data=payload, timeout=10
            )
            el = time.perf_counter() - req_start
            valid = _validate_response(resp)
            _collect(result_no_session, el, resp.status_code, len(resp.content), valid, None)
        except Exception as e:
            el = time.perf_counter() - req_start
            err_msg = f"{type(e).__name__}: {e}"
            _collect(result_no_session, el, None, 0, False, err_msg)
    result_no_session.total_elapsed = time.perf_counter() - t

    # 對比
    print("\n" + "-" * 55)
    print(f"  {'':>14} | {'有 Session':>12} | {'無 Session':>12} | {'差異':>8}")
    print("  " + "-" * 52)
    diff_avg = ((result_no_session.avg_time - result_session.avg_time) / result_session.avg_time * 100) if result_session.avg_time else 0
    diff_rps = ((result_session.rps - result_no_session.rps) / result_no_session.rps * 100) if result_no_session.rps else 0
    print(f"  {'平均耗時':>10} | {result_session.avg_time:>10.3f}s | {result_no_session.avg_time:>10.3f}s | {diff_avg:>+6.1f}%")
    print(f"  {'RPS':>10} | {result_session.rps:>10.1f} | {result_no_session.rps:>10.1f} | {diff_rps:>+6.1f}%")
    print(f"  {'P95':>10} | {result_session.p95:>10.3f}s | {result_no_session.p95:>10.3f}s |")
    print("=" * 55)

    return {"with_session": result_session, "without_session": result_no_session}
