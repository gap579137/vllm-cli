#!/usr/bin/env python3
"""
Platform-specific process management abstraction.

Wraps OS-specific process signals (SIGTERM / SIGKILL / os.killpg on POSIX,
taskkill on Windows) behind a small set of helpers so that the rest of
the codebase can call them without ``if sys.platform`` guards everywhere.

Note: vLLM only runs on Linux today, so the Windows paths are best-effort
stubs that follow the checklist design. They are gated behind ``is_posix``
/ ``is_windows`` checks and raise ``PlatformNotSupportedError`` only if
actually invoked on an unsupported platform.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Platform detection
# ---------------------------------------------------------------------------

_PLATFORM = sys.platform


def is_posix() -> bool:
    """True on Linux / macOS / other POSIX systems."""
    return _PLATFORM != "win32"


def is_windows() -> bool:
    """True on Windows."""
    return _PLATFORM == "win32"


class PlatformNotSupportedError(RuntimeError):
    """Raised when a platform-specific operation cannot be performed."""


# ---------------------------------------------------------------------------
#  Signal helpers
# ---------------------------------------------------------------------------

# Symbolic constants that work on both platforms
GRACEFUL_STOP = "graceful"
FORCEFUL_STOP = "forceful"


def send_signal_to_process(pid: int, forceful: bool = False) -> bool:
    """
    Send a stop signal to a single process.

    On POSIX: sends SIGTERM (graceful) or SIGKILL (forceful).
    On Windows: uses ``taskkill`` (graceful) or ``taskkill /F`` (forceful).

    Args:
        pid: Process ID to signal.
        forceful: If True, use an un-ignorable kill signal.

    Returns:
        True if the signal was sent successfully, False otherwise.
    """
    try:
        if is_posix():
            import signal as _signal

            sig = _signal.SIGKILL if forceful else _signal.SIGTERM
            os.kill(pid, sig)
            logger.debug(f"Sent {'SIGKILL' if forceful else 'SIGTERM'} to process {pid}")
        else:
            # Windows
            _windows_taskkill(pid, forceful=forceful)
        return True
    except ProcessLookupError:
        logger.debug(f"Process {pid} not found (already exited)")
        return False
    except PermissionError as exc:
        logger.warning(f"Permission denied sending signal to {pid}: {exc}")
        return False
    except OSError as exc:
        logger.warning(f"OS error sending signal to {pid}: {exc}")
        return False


def send_signal_to_process_group(pgid: int, forceful: bool = False) -> bool:
    """
    Send a stop signal to an entire process group.

    On POSIX: uses ``os.killpg`` with SIGTERM / SIGKILL.
    On Windows: falls back to ``send_signal_to_process`` because Windows
    does not have a native process-group kill equivalent. Callers that need
    to kill a tree should use ``terminate_process_tree`` instead.

    Args:
        pgid: Process group ID to signal.
        forceful: If True, use an un-ignorable kill signal.

    Returns:
        True if the signal was sent, False otherwise.
    """
    try:
        if is_posix():
            import signal as _signal

            sig = _signal.SIGKILL if forceful else _signal.SIGTERM
            os.killpg(pgid, sig)
            logger.debug(
                f"Sent {'SIGKILL' if forceful else 'SIGTERM'} to process group {pgid}"
            )
        else:
            # Windows: no real process groups — fall back to single-process kill
            logger.debug(
                "Windows: process group kill not supported; "
                f"falling back to single process {pgid}"
            )
            return send_signal_to_process(pgid, forceful=forceful)
        return True
    except ProcessLookupError:
        logger.debug(f"Process group {pgid} not found (already exited)")
        return False
    except PermissionError as exc:
        logger.warning(f"Permission denied for process group {pgid}: {exc}")
        return False
    except OSError as exc:
        logger.warning(f"OS error for process group {pgid}: {exc}")
        return False


def check_process_alive(pid: int) -> bool:
    """
    Check whether a process is still running.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process exists and is running.
    """
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:
        pass

    # Fallback without psutil
    if is_posix():
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False
    else:
        # Windows fallback: use tasklist
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return str(pid) in result.stdout
        except Exception:
            return False


# ---------------------------------------------------------------------------
#  High-level process lifecycle helpers
# ---------------------------------------------------------------------------


def graceful_stop_process_group(
    pgid: int,
    timeout: float = 10.0,
    wait_fn: Optional[callable] = None,
) -> bool:
    """
    Attempt a graceful stop of a process group, escalating to SIGKILL
    if the group does not exit within *timeout* seconds.

    Args:
        pgid: Process group ID (on POSIX, same as the leader's PID when
              ``start_new_session=True``).
        timeout: Seconds to wait after SIGTERM before force-killing.
        wait_fn: Optional callable that waits for the process to exit.
                 Signature: ``wait_fn(timeout) -> None``. Should raise
                 ``subprocess.TimeoutExpired`` if the process is still
                 running after *timeout*.

    Returns:
        True if the process (group) was stopped.
    """
    # Step 1: graceful
    if not send_signal_to_process_group(pgid, forceful=False):
        # Process already dead
        return True

    # Step 2: wait
    if wait_fn is not None:
        try:
            wait_fn(timeout)
            logger.info(f"Process group {pgid} terminated gracefully")
            return True
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Process group {pgid} did not terminate within {timeout}s, "
                "escalating to SIGKILL"
            )
    else:
        # No explicit wait function — sleep and check
        import time

        time.sleep(min(timeout, 2.0))
        if not check_process_alive(pgid):
            logger.info(f"Process group {pgid} terminated gracefully")
            return True
        logger.warning(
            f"Process group {pgid} still alive after {timeout}s, "
            "escalating to SIGKILL"
        )

    # Step 3: forceful
    send_signal_to_process_group(pgid, forceful=True)
    logger.info(f"Process group {pgid} force-killed")
    return True


def graceful_stop_process(
    pid: int,
    timeout: float = 10.0,
) -> bool:
    """
    Attempt a graceful stop of a single process, escalating to SIGKILL.

    Args:
        pid: Process ID.
        timeout: Seconds to wait after SIGTERM before force-killing.

    Returns:
        True if the process was stopped.
    """
    if not send_signal_to_process(pid, forceful=False):
        return True

    import time

    time.sleep(min(timeout, 2.0))
    if not check_process_alive(pid):
        return True

    send_signal_to_process(pid, forceful=True)
    return True


def start_subprocess_platform(cmd, **kwargs):
    """
    Start a subprocess with platform-appropriate defaults.

    On POSIX: uses ``start_new_session=True`` (creates new process group).
    On Windows: uses ``CREATE_NEW_PROCESS_GROUP`` creation flag.

    All extra *kwargs* are forwarded to ``subprocess.Popen``.

    Returns:
        subprocess.Popen instance.
    """
    if is_posix():
        kwargs.setdefault("start_new_session", True)
    else:
        # Windows: CREATE_NEW_PROCESS_GROUP
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        existing_flags = kwargs.get("creationflags", 0)
        kwargs["creationflags"] = existing_flags | CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(cmd, **kwargs)


# ---------------------------------------------------------------------------
#  Windows-specific internals
# ---------------------------------------------------------------------------


def _windows_taskkill(pid: int, forceful: bool = False) -> None:
    """
    Kill a process on Windows using ``taskkill``.

    Args:
        pid: Process ID.
        forceful: Whether to use ``/F`` (force) flag.

    Raises:
        OSError: If taskkill fails.
    """
    cmd = ["taskkill"]
    if forceful:
        cmd.append("/F")
    cmd.extend(["/PID", str(pid)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise OSError(
            f"taskkill failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    logger.debug(f"taskkill {'(forced) ' if forceful else ''}PID {pid}: {result.stdout.strip()}")
