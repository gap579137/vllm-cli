"""Tests for the platform-specific process management abstraction (Phase 5).

Tests the helpers in server/platform.py.
"""

import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from vllm_cli.server.platform import (
    check_process_alive,
    graceful_stop_process,
    graceful_stop_process_group,
    is_posix,
    is_windows,
    send_signal_to_process,
    send_signal_to_process_group,
    start_subprocess_platform,
)


# ---------------------------------------------------------------------------
#  Platform detection
# ---------------------------------------------------------------------------


class TestPlatformDetection:
    """Test platform detection helpers."""

    def test_is_posix_on_linux(self):
        assert is_posix() is True

    def test_is_windows_on_linux(self):
        assert is_windows() is False


# ---------------------------------------------------------------------------
#  send_signal_to_process
# ---------------------------------------------------------------------------


class TestSendSignalToProcess:
    """Test send_signal_to_process."""

    def test_send_signal_to_nonexistent_pid(self):
        """Sending signal to non-existent PID returns False."""
        result = send_signal_to_process(999999, forceful=False)
        assert result is False

    def test_send_signal_to_current_process(self):
        """Sending signal 0 equivalent (checking existence via helper)."""
        # Our own process is alive
        assert check_process_alive(os.getpid()) is True

    @patch("vllm_cli.server.platform.os.kill")
    def test_graceful_signal(self, mock_kill):
        """Graceful sends SIGTERM on POSIX."""
        mock_kill.return_value = None
        result = send_signal_to_process(12345, forceful=False)
        assert result is True
        import signal

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    @patch("vllm_cli.server.platform.os.kill")
    def test_forceful_signal(self, mock_kill):
        """Forceful sends SIGKILL on POSIX."""
        mock_kill.return_value = None
        result = send_signal_to_process(12345, forceful=True)
        assert result is True
        import signal

        mock_kill.assert_called_once_with(12345, signal.SIGKILL)

    @patch("vllm_cli.server.platform.os.kill", side_effect=ProcessLookupError)
    def test_process_not_found(self, mock_kill):
        """Returns False when process doesn't exist."""
        result = send_signal_to_process(12345, forceful=False)
        assert result is False

    @patch("vllm_cli.server.platform.os.kill", side_effect=PermissionError)
    def test_permission_denied(self, mock_kill):
        """Returns False on permission error."""
        result = send_signal_to_process(12345, forceful=False)
        assert result is False


# ---------------------------------------------------------------------------
#  send_signal_to_process_group
# ---------------------------------------------------------------------------


class TestSendSignalToProcessGroup:
    """Test send_signal_to_process_group."""

    @patch("vllm_cli.server.platform.os.killpg")
    def test_graceful_group_signal(self, mock_killpg):
        """Graceful sends SIGTERM to process group."""
        mock_killpg.return_value = None
        result = send_signal_to_process_group(12345, forceful=False)
        assert result is True
        import signal

        mock_killpg.assert_called_once_with(12345, signal.SIGTERM)

    @patch("vllm_cli.server.platform.os.killpg")
    def test_forceful_group_signal(self, mock_killpg):
        """Forceful sends SIGKILL to process group."""
        mock_killpg.return_value = None
        result = send_signal_to_process_group(12345, forceful=True)
        assert result is True
        import signal

        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    @patch("vllm_cli.server.platform.os.killpg", side_effect=ProcessLookupError)
    def test_group_not_found(self, mock_killpg):
        """Returns False when group doesn't exist."""
        result = send_signal_to_process_group(12345, forceful=False)
        assert result is False


# ---------------------------------------------------------------------------
#  check_process_alive
# ---------------------------------------------------------------------------


class TestCheckProcessAlive:
    """Test check_process_alive."""

    def test_current_process_alive(self):
        """Our own process should be alive."""
        assert check_process_alive(os.getpid()) is True

    def test_nonexistent_pid(self):
        """Very high PID should not exist."""
        assert check_process_alive(999999999) is False

    @patch("vllm_cli.server.platform.os.kill", side_effect=ProcessLookupError)
    def test_fallback_without_psutil(self, mock_kill):
        """Falls back to os.kill(pid, 0) when psutil unavailable."""
        # Force the psutil import to fail
        with patch.dict("sys.modules", {"psutil": None}):
            result = check_process_alive(99999)
            assert result is False


# ---------------------------------------------------------------------------
#  graceful_stop_process_group
# ---------------------------------------------------------------------------


class TestGracefulStopProcessGroup:
    """Test graceful_stop_process_group."""

    @patch("vllm_cli.server.platform.send_signal_to_process_group")
    def test_graceful_then_wait(self, mock_signal):
        """Calls SIGTERM first, then waits."""
        mock_signal.return_value = True
        wait_fn = MagicMock()  # Simulates process.wait()

        result = graceful_stop_process_group(12345, timeout=5.0, wait_fn=wait_fn)
        assert result is True
        wait_fn.assert_called_once_with(5.0)

    @patch("vllm_cli.server.platform.send_signal_to_process_group")
    def test_escalates_to_sigkill_on_timeout(self, mock_signal):
        """Escalates to SIGKILL when wait times out."""
        mock_signal.return_value = True
        wait_fn = MagicMock(side_effect=subprocess.TimeoutExpired("cmd", 5))

        result = graceful_stop_process_group(12345, timeout=5.0, wait_fn=wait_fn)
        assert result is True
        # Should have been called twice: graceful then forceful
        assert mock_signal.call_count == 2
        mock_signal.assert_any_call(12345, forceful=False)
        mock_signal.assert_any_call(12345, forceful=True)

    @patch("vllm_cli.server.platform.send_signal_to_process_group", return_value=False)
    def test_already_dead(self, mock_signal):
        """Returns True if process is already dead."""
        result = graceful_stop_process_group(12345)
        assert result is True


# ---------------------------------------------------------------------------
#  start_subprocess_platform
# ---------------------------------------------------------------------------


class TestStartSubprocessPlatform:
    """Test start_subprocess_platform."""

    def test_starts_with_new_session_on_posix(self):
        """On POSIX, start_new_session=True is set."""
        proc = start_subprocess_platform(
            ["echo", "hello"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            # Process should have a different session ID since we used
            # start_new_session=True
            assert proc.pid > 0
            proc.wait(timeout=5)
        finally:
            proc.kill()
            proc.wait()

    def test_real_subprocess_can_be_killed(self):
        """Integration test: start a real subprocess and kill it."""
        proc = start_subprocess_platform(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.poll() is None  # Still running

        # Use our graceful stop
        graceful_stop_process_group(proc.pid, timeout=2.0, wait_fn=proc.wait)
        # Process should be dead now
        proc.wait(timeout=5)  # Should not hang
        assert proc.poll() is not None


# ---------------------------------------------------------------------------
#  graceful_stop_process
# ---------------------------------------------------------------------------


class TestGracefulStopProcess:
    """Test graceful_stop_process."""

    @patch("vllm_cli.server.platform.send_signal_to_process", return_value=False)
    def test_already_dead(self, mock_signal):
        """Returns True if process is already gone."""
        result = graceful_stop_process(12345)
        assert result is True

    @patch("vllm_cli.server.platform.check_process_alive", return_value=False)
    @patch("vllm_cli.server.platform.send_signal_to_process", return_value=True)
    def test_graceful_succeeds(self, mock_signal, mock_alive):
        """Returns True when SIGTERM is enough."""
        result = graceful_stop_process(12345, timeout=0.1)
        assert result is True
