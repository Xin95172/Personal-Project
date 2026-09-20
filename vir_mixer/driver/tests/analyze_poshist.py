import argparse
import json
import re
from pathlib import Path


TRACE_RE = re.compile(
    r"VirMixer: TRACE "
    r"seq=(?P<seq>\d+) "
    r"epoch=(?P<epoch>\d+) "
    r"kind=(?P<kind>\d+) "
    r"side=(?P<side>\d+) "
    r"from=(?P<from>\d+) "
    r"to=(?P<to>\d+) "
    r"qpc=(?P<qpc>\d+).*?"
    r"req=(?P<req>\d+) "
    r"avail=(?P<avail>\d+) "
    r"take=(?P<take>\d+)"
)

POSHIST_RE = re.compile(
    r"VirMixer: POSHIST "
    r"seq=(?P<seq>\d+) "
    r"side=(?P<side>\d+) "
    r"index=(?P<index>\d+) "
    r"count=(?P<count>\d+) "
    r"qpc=(?P<qpc>\d+) "
    r"prev=(?P<prev>\d+) "
    r"now=(?P<now>\d+) "
    r"linear=(?P<linear>\d+) "
    r"disp=(?P<disp>\d+) "
    r"origin=(?P<origin>\d+)"
)


def load_trials(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    qpc_frequency = data["qpc_frequency"]

    trials = []

    for test in data["tests"]:
        if "repeat" not in test:
            continue

        trials.append(
            {
                "repeat": test["repeat"],
                "seed": test["seed"],
                "passed": test["passed"],
                "start": test["trial_start_qpc"],
                "end": test["trial_end_qpc"],
            }
        )

    return qpc_frequency, trials


def find_trial(qpc, trials):
    for trial in trials:
        if trial["start"] <= qpc <= trial["end"]:
            return trial

    return None


def parse_log(path):
    traces = {}
    histories = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = TRACE_RE.search(line)

            if m:
                item = {
                    key: int(value)
                    for key, value in m.groupdict().items()
                }

                if item["kind"] == 5:
                    traces[item["seq"]] = item

                continue

            m = POSHIST_RE.search(line)

            if m:
                item = {
                    key: int(value)
                    for key, value in m.groupdict().items()
                }

                seq = item["seq"]
                side = item["side"]

                histories.setdefault(seq, {0: [], 1: []})
                histories[seq][side].append(item)

    return traces, histories


def latest_before(history, qpc):
    candidates = [
        item
        for item in history
        if item["qpc"] <= qpc
    ]

    if not candidates:
        return None

    return max(candidates, key=lambda x: x["qpc"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("log")
    parser.add_argument("report")
    parser.add_argument(
        "--output",
        default="driver/out/poshist-analysis.json",
    )

    args = parser.parse_args()

    qpc_frequency, trials = load_trials(args.report)
    traces, histories = parse_log(args.log)

    rows = []

    for seq, trace in sorted(traces.items()):
        qpc = trace["qpc"]

        trial = find_trial(qpc, trials)

        if trial is None:
            continue

        history = histories.get(seq)

        if not history:
            continue

        render = latest_before(history[0], qpc)
        capture = latest_before(history[1], qpc)

        if render is None or capture is None:
            continue

        linear_delta = capture["linear"] - render["linear"]

        delta_ms = (
            linear_delta / 192.0
        )

        event_ms = (
            (qpc - trial["start"])
            / qpc_frequency
            * 1000.0
        )

        position_qpc_delta_ms = (
            (capture["qpc"] - render["qpc"])
            / qpc_frequency
            * 1000.0
        )

        shortage = trace["req"] - trace["take"]

        row = {
            "trial": trial["repeat"],
            "seed": trial["seed"],
            "result": "PASS" if trial["passed"] else "FAIL",

            "seq": seq,
            "event_ms": event_ms,

            "request_bytes": trace["req"],
            "available_bytes": trace["avail"],
            "taken_bytes": trace["take"],
            "shortage_bytes": shortage,
            "shortage_ms": shortage / 192.0,

            "render_qpc": render["qpc"],
            "capture_qpc": capture["qpc"],

            "render_linear": render["linear"],
            "capture_linear": capture["linear"],

            "capture_minus_render_bytes": linear_delta,
            "capture_minus_render_ms": delta_ms,

            "position_qpc_delta_ms": position_qpc_delta_ms,

            "render_displacement": render["disp"],
            "capture_displacement": capture["disp"],
        }

        rows.append(row)

    Path(args.output).write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )

    print()
    print(
        f"{'Trial':>5} "
        f"{'Result':>6} "
        f"{'Time(ms)':>9} "
        f"{'Short':>7} "
        f"{'Lead(B)':>8} "
        f"{'Lead(ms)':>9} "
        f"{'QPCΔ(ms)':>9} "
        f"{'Rdisp':>6} "
        f"{'Cdisp':>6}"
    )

    print("-" * 85)

    for row in rows:
        print(
            f"{row['trial']:5d} "
            f"{row['result']:>6} "
            f"{row['event_ms']:9.1f} "
            f"{row['shortage_bytes']:7d} "
            f"{row['capture_minus_render_bytes']:8d} "
            f"{row['capture_minus_render_ms']:9.2f} "
            f"{row['position_qpc_delta_ms']:9.2f} "
            f"{row['render_displacement']:6d} "
            f"{row['capture_displacement']:6d}"
        )

    print()
    print(f"Underruns analyzed: {len(rows)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()