"""Summarize bounded VirMixer DebugView/DbgView cable diagnostics.

Usage: python driver/tests/parse_cable_diagnostics.py path/to/debug.log
This reads a copied log only. It does not open audio devices or change Windows.
"""
import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


UNDERRUN = re.compile(
    r"VirMixer: UNDERRUN reset=(?P<reset>\d+) count=(?P<count>\d+) "
    r"req=(?P<requested>\d+) avail=(?P<available>\d+) take=(?P<taken>\d+) "
    r"primed=(?P<primed>[01]) totalW=(?P<total_written>\d+) "
    r"totalReq=(?P<total_requested>\d+) totalRead=(?P<total_read>\d+) "
    r"epochW=(?P<epoch_written>\d+) epochReq=(?P<epoch_requested>\d+) "
    r"epochRead=(?P<epoch_read>\d+)"
)


def parse(text):
    events = []
    for match in UNDERRUN.finditer(text):
        event = {name: int(value) for name, value in match.groupdict().items()}
        events.append(event)
    return events


def summarize(events):
    if not events:
        return {"underruns": 0, "classification": "No bounded VirMixer underrun records found."}
    resets = Counter(event["reset"] for event in events)
    reset_race = any(event["epoch_written"] == 0 for event in events)
    impossible_accounting = [event for event in events if event["epoch_read"] > event["epoch_written"]]
    consumer_ahead = [event for event in events if event["epoch_requested"] > event["epoch_written"]]
    partial = [event for event in events if 0 < event["taken"] < event["requested"]]
    empty = [event for event in events if event["available"] == 0]
    return {
        "underruns": len(events),
        "reset_epochs": dict(resets),
        "empty_reads": len(empty),
        "partial_reads": len(partial),
        "consumer_requested_ahead_of_producer": len(consumer_ahead),
        "impossible_actual_read_ahead_of_written": len(impossible_accounting),
        "possible_reset_race": reset_race,
        "classification": (
            "Actual reads exceeded epoch writes: investigate accounting before classifying timing."
            if impossible_accounting else
            "No actual-read overrun observed. Requested totals include startup prefill silence; "
            "their lead does not measure instantaneous clock lead. Empty epochs alone do not prove "
            "a reset race. Stream timestamps and displacement evidence are required."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    print(summarize(parse(args.log.read_text(encoding="utf-8", errors="replace"))))


if __name__ == "__main__":
    main()
