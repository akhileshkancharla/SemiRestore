"""Command-line entry point for the reproducible SemiRestore HTTP load harness."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from semirestore.platform.load_testing import (
    LoadEndpoint,
    LoadTestConfig,
    report_payload,
    run_load_test,
    write_report,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-url", default="http://127.0.0.1:8000")
    value.add_argument(
        "--endpoint",
        choices=[endpoint.value for endpoint in LoadEndpoint],
        required=True,
    )
    value.add_argument("--concurrency", type=int, default=1)
    value.add_argument("--duration", type=float, default=10.0, dest="duration_seconds")
    value.add_argument("--rate", type=float, default=1.0, dest="request_rate")
    value.add_argument("--timeout", type=float, default=30.0, dest="timeout_seconds")
    value.add_argument("--width", type=int, default=64)
    value.add_argument("--height", type=int, default=64)
    value.add_argument(
        "--report",
        type=Path,
        default=Path("load-results/latest.json"),
        help="Ignored JSON report destination.",
    )
    return value


def main() -> int:
    arguments = parser().parse_args()
    config = LoadTestConfig(
        base_url=arguments.base_url,
        endpoint=LoadEndpoint(arguments.endpoint),
        concurrency=arguments.concurrency,
        duration_seconds=arguments.duration_seconds,
        request_rate=arguments.request_rate,
        timeout_seconds=arguments.timeout_seconds,
        width=arguments.width,
        height=arguments.height,
    )
    run = asyncio.run(run_load_test(config))
    write_report(run, arguments.report)
    print(json.dumps(report_payload(run)["summary"], indent=2, sort_keys=True))
    return 0 if run.summary.failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
