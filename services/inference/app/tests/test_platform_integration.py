"""Exercise Go Parquet and execution through the Python production runtime."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import threading
import select

import httpx
import numpy as np
import pytest

from app.candles import DuckDBCandleStore
from app.config import InferenceConfig
from app.execution import ExecutionClient
from app.features import FeatureBuilder
from app.loop import MinuteScheduler, Outcome
from app.predictor import Predictor
from support import ConstantModel

ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def platform_binary(tmp_path_factory):
    binary = tmp_path_factory.mktemp("platform-bin") / "fixture"
    environment = dict(os.environ, GOMAXPROCS="2", GOCACHE="/tmp/mft-go-cache")
    subprocess.run(["go", "build", "-p", "2", "-o", str(binary), "./services/execution/cmd/platformfixture"],
                   cwd=ROOT, env=environment, check=True, capture_output=True, timeout=90)
    return binary


class Platform:
    """Adapt a socket-free Go HTTP fixture process to an HTTP client transport."""

    def __init__(self, binary, directory):
        self._process = subprocess.Popen([str(binary), "--data-dir", str(directory)],
                                         text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._lock = threading.Lock()
        line = self._readline()
        if not line:
            raise RuntimeError(self._process.stderr.read())
        self.metadata = json.loads(line)

    def _readline(self):
        if not select.select([self._process.stdout], [], [], 15)[0]:
            self._process.kill()
            self._process.wait(timeout=5)
            raise TimeoutError("Go platform fixture did not respond within 15 seconds")
        return self._process.stdout.readline()

    def request(self, method, path, body=None, token="fixture-token"):
        with self._lock:
            self._process.stdin.write(json.dumps({"method": method, "path": path, "body": body, "token": token}) + "\n")
            self._process.stdin.flush()
            line = self._readline()
            if not line:
                raise RuntimeError(self._process.stderr.read())
            return json.loads(line)

    def transport(self):
        async def handle(request):
            body = json.loads(request.content) if request.content else None
            response = await asyncio.to_thread(self.request, request.method, request.url.path, body)
            return httpx.Response(response["status"], json=response["body"])
        return httpx.MockTransport(handle)

    def close(self):
        self._process.stdin.close()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
            raise
        assert self._process.returncode == 0, self._process.stderr.read()


def test_go_parquet_feature_model_http_execution_and_restart(platform_binary, tmp_path):
    platform = Platform(platform_binary, tmp_path)
    async def run():
        store = DuckDBCandleStore(tmp_path / "candles")
        predictor = Predictor(ConstantModel(0.9), model_name="constant")
        client = httpx.AsyncClient(transport=platform.transport())
        sender = ExecutionClient("http://fixture.invalid", client=client, api_token="fixture-token")
        try:
            rows = await store.tail("RELIANCE", 200)
            assert len(rows) == 200
            assert rows[-1].timestamp.isoformat().replace("+00:00", "Z") == platform.metadata["as_of"]
            features = FeatureBuilder().build(rows)
            assert list(features.columns) == platform.metadata["columns"]
            np.testing.assert_allclose(features.frame.to_numpy(), platform.metadata["features"], rtol=1e-10, atol=1e-10)
            await predictor.load()
            config = InferenceConfig(model="heuristic", instruments=["RELIANCE"], dry_run=False, order_quantity=10,
                                     score_threshold=0.5, cursor_path=str(tmp_path / "inference" / "cursors.json"))
            scheduler = MinuteScheduler(config, store, predictor, sender, max_context_age=None)
            decision = (await scheduler.tick())[0]
            assert decision.outcome is Outcome.SENT
            assert decision.order_id
            replay = await sender.submit(decision.signal)
            assert replay.duplicate and replay.order_id == decision.order_id
            assert (await scheduler.tick())[0].outcome is Outcome.UNCHANGED
            portfolio = platform.request("GET", "/v1/portfolio")["body"]
            assert portfolio["open_positions"]["RELIANCE"] == 10
            assert portfolio["cash"] < 1_000_000
            return decision.signal, decision.order_id
        finally:
            await client.aclose()
            await predictor.aclose()
            await store.aclose()
    try:
        signal, order_id = asyncio.run(run())
    finally:
        platform.close()
    restored = Platform(platform_binary, tmp_path)
    try:
        portfolio = restored.request("GET", "/v1/portfolio")["body"]
        assert portfolio["open_positions"]["RELIANCE"] == 10
        replay = restored.request("POST", "/v1/signals", signal.to_wire())
        assert replay["status"] == 409
        assert replay["body"]["order_id"] == order_id
    finally:
        restored.close()


def test_http_invalid_signal_is_rejected_without_changing_book(platform_binary, tmp_path):
    platform = Platform(platform_binary, tmp_path)
    try:
        response = platform.request("POST", "/v1/signals", {"symbol": "RELIANCE", "unknown": 1})
        assert response["status"] == 400
        portfolio = platform.request("GET", "/v1/portfolio")["body"]
        assert not portfolio["open_positions"]
        assert portfolio["cash"] == 1_000_000
    finally:
        platform.close()


def test_execution_authentication_crosses_transport_boundary(platform_binary, tmp_path):
    platform = Platform(platform_binary, tmp_path)
    try:
        for token in ("", "invalid"):
            response = platform.request("POST", "/v1/signals", {}, token=token)
            assert response["status"] == 401
        portfolio = platform.request("GET", "/v1/portfolio")["body"]
        assert portfolio["cash"] == 1_000_000 and not portfolio["open_positions"]
    finally:
        platform.close()
