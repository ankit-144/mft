"""Keep asynchronous tests usable when the sandbox forbids socket writes."""

import asyncio
import errno
import socket
import pytest


@pytest.fixture(autouse=True)
def block_real_checkpoints(monkeypatch):
    """Fail before any backend can read or download a real checkpoint."""
    from model import tabfm_model
    def refuse(*args, **kwargs):
        raise AssertionError("Real checkpoint access is forbidden in automated tests")
    monkeypatch.setattr(tabfm_model, "_make_backend", refuse)
    monkeypatch.setattr(tabfm_model, "weights_available", refuse)


def pytest_configure(config):
    left, right = socket.socketpair()
    try:
        left.send(b"x")
    except OSError as error:
        if error.errno not in (errno.EPERM, errno.EACCES):
            raise
        original = asyncio.SelectorEventLoop.run_forever

        def run_forever(loop):
            def heartbeat():
                if loop.is_running():
                    loop.call_later(0.005, heartbeat)

            pulse = loop.call_later(0.005, heartbeat)
            try:
                return original(loop)
            finally:
                pulse.cancel()

        asyncio.SelectorEventLoop.run_forever = run_forever
        config._mft_restricted_socket_writes = True
    finally:
        left.close()
        right.close()


def pytest_report_header(config):
    if getattr(config, "_mft_restricted_socket_writes", False):
        return "MFT: test-only selector heartbeat enabled (sandbox denies socketpair writes)"
