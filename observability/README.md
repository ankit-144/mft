# Observability

Prometheus and Grafana for the MFT platform. Two containers, no changes to the
trading services, and nothing to run from Docker that you cannot already run
with `make dev`.

| Asset | What it is |
| :--- | :--- |
| `docker-compose.observability.yml` | Prometheus + Grafana, wired to the four service metrics ports |
| `prometheus/prometheus.yml` | Scrape config for ingestion `:9090`, execution `:9091`, jobs `:9092`, inference `:8000` |
| `grafana/dashboards/mft-overview.json` | The v1 dashboard, provisioned automatically |
| `grafana/provisioning/` | Datasource and dashboard providers, so there is nothing to click through |
| `METRICS.md` | The metrics catalogue: names, types, labels, and the rules |

## Bring it up

```sh
make dev                                                     # terminal 1
docker compose -f observability/docker-compose.observability.yml up -d
```

| | |
| :--- | :--- |
| Grafana | <http://localhost:3000> — `admin` / `admin` |
| Prometheus | <http://localhost:9099> — **not** 9090, which is ingestion's metrics port |

Tear down with `docker compose -f observability/docker-compose.observability.yml down`.
Add `-v` to discard the 15 days of history. Both ports are bound to `127.0.0.1`
on purpose: this is a local platform, and the dashboards should not be reachable
from the network.

`make status` prints the five service ports; it is a port check, not a health
check. For real health, use each service's metrics port:

```sh
curl -s localhost:9090/healthz | jq   # liveness  — the process is up
curl -s localhost:9091/readyz  | jq   # readiness — every dependency probe passed (503 if not)
```

## The trading services are not in the compose file

They run on the host, because the north star is local-first and `make dev`
already starts them with the right config. Prometheus reaches them through
`host.docker.internal`, which the compose file maps onto the host gateway — that
mapping is why it works on Linux, where the name does not exist by default.

If you are running the services in Docker (`docker compose up -d` at the repo
root), change the four `host.docker.internal:` targets in
`prometheus/prometheus.yml` to the service names (`ingestion:9090`,
`execution:9090`, `jobs:9090`) — note that inside the compose network every Go
service exposes its metrics on `:9090` regardless of the host port it is
published on.

## Which panels matter

The dashboard is one screen, four rows, ordered by the question you are actually
asking.

1. **Ingestion — is the data flowing?** Tick rate and candle writes per symbol.
   Tick rate at zero during market hours (09:15–15:30 IST, weekdays) is the
   single most important signal on the page: the WebSocket is down, and nothing
   downstream will tell you. Candle writes should step once per minute per
   instrument.
2. **Execution — is it trading, and why not?** Orders placed, rejections split
   by reason code, and order latency p50/p95/p99. The reason codes are the point:
   `RISK_MAX_DRAWDOWN` climbing is a risk event, while `RISK_DEBOUNCED` climbing
   means the model is repeating the same call and something upstream is wrong.
   Latency p99 above ~2s means Kite is rate-limiting or the session is close to
   expiry.
3. **Risk headroom — how close to the limits?** Drawdown, open positions,
   realised PnL and largest position, each with the configured limit as its
   threshold. These are the four numbers that stop a bad day from becoming a
   bad week, and they are all read from the risk engine's own view — the risk
   checks in `docs/contracts.md` §6 are the same numbers.
4. **Platform — what is actually running?** Build info per service, uptime, and
   HTTP status codes. When a metric surprises you, the first question is which
   commit produced it; the second is whether the service just restarted.

Panels for metrics that no component has registered yet are empty rather than
wrong. `METRICS.md` marks them ⧗ and names the owner. That is deliberate: a
missing panel is a visible gap, a fabricated zero is a lie you will act on.

## Adding a panel

Use a `mft_`-prefixed metric and follow the naming rules in `METRICS.md`. Then
add a row to `METRICS.md` as well, so the next person can find it. Grafana
reloads the file every 30 seconds, so save the JSON and the panel is there.

## Version pinning

The images are pinned (`prom/prometheus:v3.0.1`, `grafana/grafana:11.3.0`).
Prometheus 3 is required for exemplars: it negotiates OpenMetrics by default,
which is how the correlation id from `log.Middleware` reaches the latency
histogram. Bump the tags in the compose file and the dashboard JSON is
unaffected.
