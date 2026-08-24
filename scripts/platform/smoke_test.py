"""Run the SemiRestore health-to-restoration production smoke sequence."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence

from semirestore.platform.smoke_testing import (
    SmokeOperation,
    SmokeTestConfig,
    SmokeTestError,
    report_payload,
    run_smoke_test,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--base-url", default="http://127.0.0.1:8000")
    value.add_argument(
        "--operation",
        choices=[operation.value for operation in SmokeOperation],
        default=SmokeOperation.RESTORE_AND_ANALYZE.value,
    )
    value.add_argument("--timeout", type=float, default=120.0, dest="timeout_seconds")
    value.add_argument("--width", type=int, default=16)
    value.add_argument("--height", type=int, default=16)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        config = SmokeTestConfig(
            base_url=arguments.base_url,
            operation=SmokeOperation(arguments.operation),
            timeout_seconds=arguments.timeout_seconds,
            width=arguments.width,
            height=arguments.height,
        )
        report = asyncio.run(run_smoke_test(config))
    except (ValueError, SmokeTestError) as error:
        print(f"smoke test failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report_payload(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
