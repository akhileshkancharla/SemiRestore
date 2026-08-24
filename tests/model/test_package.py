from __future__ import annotations

import semirestore


def test_package_exposes_version() -> None:
    assert semirestore.__version__ == "0.1.0"
