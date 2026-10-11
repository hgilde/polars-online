"""The command line's progress line, drawn only on a terminal.

The Rust tests beside the binary (``crates/online-cli/tests/surface.rs``)
hold the piped side: a log or a pipe carries no progress line and none of
its carriage returns. This holds the other side, which needs a terminal, so
the binary runs here with a pseudo-terminal as its standard error.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

TIER = "essential"

# A pseudo-terminal is POSIX's: Windows has no `pty` module. The side every
# platform tests, the piped one, is the one a scheduled job meets.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="a pseudo-terminal needs POSIX's pty module"
)


def _drawn(online_cli: Path, config: Path, *args: str) -> str:
    """What the binary draws on a terminal standard error, run to its end."""
    import pty

    leader, follower = pty.openpty()
    # Read while it runs: a terminal's unread text can go with its last
    # writer, and its buffer is small.
    with subprocess.Popen(
        [str(online_cli), "--config", str(config), *args],
        stdout=subprocess.DEVNULL,
        stderr=follower,
    ) as proc:
        os.close(follower)  # the child holds the only writer now
        drawn = b""
        try:
            while chunk := os.read(leader, 4096):
                drawn += chunk
        except OSError:  # Linux ends a closed terminal's reads with EIO, macOS with b""
            pass
        finally:
            os.close(leader)
    assert proc.returncode == 0, drawn
    return drawn.decode("utf-8")


def test_the_progress_line_is_drawn_on_a_terminal_unless_quiet(online_cli, tmp_path):
    n = 50
    pl.DataFrame(
        {
            "t": [float(i) for i in range(n)],
            "x0": [float(i % 5) for i in range(n)],
            "y": [float(i % 3) for i in range(n)],
        }
    ).write_parquet(tmp_path / "in.parquet")
    config = tmp_path / "job.toml"
    config.write_text(
        f"input = '{tmp_path / 'in.parquet'}'\noutput = '{tmp_path / 'out.parquet'}'\n"
        "chunk_size = 7\n\n"
        '[[specs]]\nname = "ridge"\ntargets = ["y"]\nfeatures = ["x0"]\nclock = "t"\n'
        'half_life = 20.0\ngap_cap = 5.0\n[specs.model]\ntype = "ewridge"\n',
        encoding="utf-8",
    )
    drawn = _drawn(online_cli, config)
    # Rewritten in place, chunk by chunk, then ended with a newline, which
    # the terminal turns into a carriage return and a line feed.
    assert drawn.startswith("\r7 rows in 1 chunk\r14 rows in 2 chunks\r"), repr(drawn)
    assert drawn.endswith("\r50 rows in 8 chunks\r\n"), repr(drawn)
    assert _drawn(online_cli, config, "--quiet") == ""
