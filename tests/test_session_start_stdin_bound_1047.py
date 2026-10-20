"""#1047: SessionStart's stdin is not bounded, and everything downstream of it is
quadratic in the payload's length.

The hook reads stdin into `HOOK_STDIN` (a line loop, deliberately kept after
#1040 -- the payload is a few hundred bytes: ids, paths, the source) and then
hands that whole string to `_stdin_json_string_into`, whose first real step is

    _sjsi_rest=${_sjsi_raw#*\\"$_sjsi_key\\"}

`${var#*pattern}` with a leading `*` is quadratic in bash -- the pilot re-runs
the pattern matcher at every offset. Measured for #1047 on bash 5.2.15, one
synthetic string of n bytes with `"cwd"` at the end: 0.28 s at 12.5 KB, 0.95 s
at 25 KB, 3.55 s at 50 KB, 15.2 s at 100 KB. `case x in *"cwd"*)` on the same
strings is 2-6 ms, so it is the `#` operator, not pattern matching as such --
the asymmetry `lib-stdin.sh` already records for the newline strip (165 s for
1 MB).

Nothing in the hook picks its stdin. The reporter's box had 27 of these hooks
alive for 21 hours each, at 11% CPU, after child `claude` sessions (the harness
runs `CLAUDE_CODE_ENTRYPOINT=sdk-cli`) were handed their own transcript as fd 0:
6.8 MB, `"cwd"` at byte 3,353,224. Four extractions on that is hours, and a
`${var#...}` builtin runs without a signal check, so `SIGTERM` did nothing --
`SIGKILL` was the only way out.

What the tests are evidence of:

* DETERMINISTIC, every platform. The payload the hook publishes to a `hooks.d/`
  listener (`REMEMBER_HOOK_STDIN_FILE`) is the payload it kept, and that is
  what a 64 KB stdin must not be able to make large. Counted by execution, not
  by a clock, so a loaded box and a quiet one agree (#669's own reason for
  preferring a count).
* THE POSITIVE CONTROL, same fixture, same listener: a real SessionStart
  payload must arrive byte-for-byte as it did before, or a bound that crops
  everything would pass the first assertion too.
* A HANG CEILING. A transcript-sized (1 MB) stdin must finish. On macOS and
  Linux the surviving line loop costs about a second per MB (#1040), so the
  ceiling is huge on purpose: it is the bug's shape -- never finishing -- that
  it catches, not a duration. OBSERVED on macOS only; reasoned elsewhere, and
  stated as such rather than claimed.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

from .test_path_resolution import _create_full_plugin_copy, _create_full_project

SESSION = "1047abcd-0000-4000-8000-000000001047"

# What the hook keeps: one crop per side of the join, so the accumulator is
# bounded by twice the crop whatever the host sends.
CROP = 4096
BOUND = 2 * CROP

# A bound on a hang, not a measurement of anything.
CEILING = 120

# How long to wait for the deferred listener to write its two files. A bound on
# a hang, not a measurement of anything.
MARKER_TIMEOUT = 20

LISTENER = """#!/bin/bash
# Test stub for hooks.d/before_session_start/ (tests/test_session_start_stdin_bound_1047.py).
# The hook publishes the payload it consumed as a file; the size of that file is
# what these tests assert on.
cp "$REMEMBER_HOOK_STDIN_FILE" "$REMEMBER_TEST_STDIN_COPY"
printf '%s' "$REMEMBER_HOOK_STDIN" > "$REMEMBER_TEST_STDIN_ENV"
"""


def _setup(tmp_path: Path):
    plugin = tmp_path / "plugin"
    project = tmp_path / "project"
    home = tmp_path / "home"
    plugin.mkdir()
    project.mkdir()
    home.mkdir()
    _create_full_plugin_copy(str(plugin))
    _create_full_project(str(project))

    listener = plugin / "hooks.d" / "before_session_start" / "99-record-stdin.sh"
    listener.write_text(LISTENER, encoding="utf-8")
    listener.chmod(listener.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return plugin, project, home


def _run(tmp_path: Path, payload: str):
    plugin, project, home = _setup(tmp_path)
    copy = tmp_path / "stdin-copy"
    env_file = tmp_path / "stdin-env"
    env = {
        **os.environ,
        "HOME": str(home),
        "CLAUDE_PROJECT_DIR": str(project),
        "CLAUDE_PLUGIN_ROOT": str(plugin),
        "REMEMBER_TEST_STDIN_COPY": str(copy),
        "REMEMBER_TEST_STDIN_ENV": str(env_file),
    }
    env.pop("REMEMBER_NESTED_SUMMARIZER", None)
    t0 = time.perf_counter()
    result = subprocess.run(
        ["bash", str(plugin / "scripts" / "session-start-hook.sh")],
        input=payload, env=env, capture_output=True, text=True,
        timeout=CEILING, check=False,
    )
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, (
        f"session-start-hook.sh exited {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    # Named `99-record-stdin.sh`, not `50-git-restore.sh`: the hook defers a
    # LONE bundled listener (`& disown`, #660) and deletes the published file
    # on its way out, so a deferred copy races the cleanup. An unknown script
    # is dispatched in the foreground -- one of #660's own conditions -- and
    # the poll below is only a bound.
    deadline = time.time() + MARKER_TIMEOUT
    while time.time() < deadline and not copy.is_file():
        time.sleep(0.05)
    assert copy.is_file(), (
        f"the hooks.d listener never recorded the published payload\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    return copy.read_bytes(), env_file.read_bytes(), elapsed


def _payload(**over) -> str:
    body = {
        "session_id": SESSION,
        "transcript_path": f"/does/not/matter/{SESSION}.jsonl",
        "hook_event_name": "SessionStart",
        "source": "startup",
        "cwd": "/does/not/matter",
    }
    body.update(over)
    return json.dumps(body)


def test_a_64_kb_stdin_is_cropped_before_the_hook_publishes_it(tmp_path):
    """Over the crop, the payload the extractors see must stay bounded.

    Would this test still pass if the code did nothing? No: the hook used to
    keep every byte it read, so the published copy was the payload -- 65,607
    bytes here, against the 8,192 bound. Nothing else in the hook looks at its
    size.
    """
    payload = '{"session_id":"' + SESSION + '","pad":"' + "x" * 65536 + '","cwd":"/does/not/matter"}'
    copy, env_stdin, _ = _run(tmp_path, payload)

    assert len(payload) > BOUND, "fixture is not over the bound"
    assert len(copy) <= BOUND, (
        f"the hook published a {len(copy)}-byte payload; the bound is {BOUND}"
    )


def test_a_real_payload_arrives_byte_for_byte(tmp_path):
    """The positive control for the crop: a few-hundred-byte payload is untouched.

    Same fixture, same listener, same assertion surface -- only the size
    changes. Byte equality, not a prefix: the crop is only allowed to matter
    past its own bound.
    """
    payload = _payload()
    copy, env_stdin, _ = _run(tmp_path, payload)

    assert len(payload) <= CROP, "fixture is not under the crop"
    assert copy == payload.encode(), (
        f"a {len(payload)}-byte payload arrived as {len(copy)} bytes: {copy[:200]!r}"
    )
    assert env_stdin == payload.encode(), (
        "the payload in REMEMBER_HOOK_STDIN is not the one in the file"
    )


def test_a_transcript_sized_stdin_does_not_hang_the_hook(tmp_path):
    """The reported shape: one 1 MB line, the key at the end of it.

    Before the fix this does not finish in any wall clock a session survives:
    the strip to `"cwd"` alone is (1 MB / 100 KB)^2 * 15.2 s. The ceiling is
    the assertion -- this test is about hanging, not about speed.
    """
    payload = '{"type":"queue-operation","pad":"' + "y" * (1 << 20) + '","cwd":"/does/not/matter"}'
    copy, _, elapsed = _run(tmp_path, payload)

    assert elapsed < CEILING, f"took {elapsed:.1f}s"
    assert len(copy) <= BOUND, (
        f"the hook published a {len(copy)}-byte payload; the bound is {BOUND}"
    )
