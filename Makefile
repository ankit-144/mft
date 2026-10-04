.PHONY: \
	build test vet fmt lint tidy \
	setup dev stop down status \
	run-ingestion run-execution run-jobs run-inference \
	venv inference-deps tabfm-weights \
	backtest \
	wt wt-list wt-remove wt-prune \
	docker-build docker-up docker-down docker-logs

MODULES := core services/ingestion services/execution services/jobs
VENV    := services/inference/.venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
ROOT    := $(shell pwd)

# --- build & quality -------------------------------------------------------

# The repo root is not itself a Go module; go.work stitches the modules
# together. `go build ./...` at the root therefore fails, so every module is
# built individually.
build:
	@for m in $(MODULES); do \
		echo "== build $$m =="; \
		( cd $$m && go build ./... ) || exit 1; \
	done

test:
	@for m in $(MODULES); do \
		echo "== test $$m =="; \
		( cd $$m && go test ./... ) || exit 1; \
	done

vet:
	@for m in $(MODULES); do \
		echo "== vet $$m =="; \
		( cd $$m && go vet ./... ) || exit 1; \
	done

fmt:
	@for m in $(MODULES); do \
		( cd $$m && gofmt -l -w . ) || exit 1; \
	done

# lint is the gate CI and agents should run before pushing.
lint: vet
	@out=$$(for m in $(MODULES); do ( cd $$m && gofmt -l . ); done); \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	@echo "lint ok"

tidy:
	@for m in $(MODULES); do \
		( cd $$m && go mod tidy ) || exit 1; \
	done
	go work sync

# --- run -------------------------------------------------------------------

run-ingestion:
	cd $(ROOT) && go run ./services/ingestion/cmd/server

run-execution:
	cd $(ROOT) && go run ./services/execution/cmd/server

run-jobs:
	cd $(ROOT) && go run ./services/jobs/cmd/server

run-inference:
	cd $(ROOT)/services/inference && ../inference/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# --- python / tabfm --------------------------------------------------------

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PIP) install --upgrade pip --quiet

inference-deps: venv
	@$(PIP) install -r services/inference/requirements.txt

# TabFM weights download from Hugging Face on first use and are cached.
# They are licensed for non-commercial use only — see Plan.md §5.
tabfm-weights:
	@$(PY) -c "import tabfm, sys; print('tabfm import ok')" || \
		echo "tabfm not installed yet — run 'make inference-deps'"

setup: build inference-deps
	@cp -n configs/config.example.yaml configs/config.yaml 2>/dev/null || true
	@echo "setup complete. Edit configs/config.yaml with your Kite credentials."

# --- dev loop --------------------------------------------------------------

# dev starts every service in the background with logs under .dev/.
# It is deliberately not Docker-based: the north star is local-first.
dev: build
	@mkdir -p .dev
	@echo "starting services... (logs in .dev/, stop with 'make stop')"
	@$(MAKE) --no-print-directory stop
	@MFT_CONFIG=configs/config.yaml $(MAKE) --no-print-directory run-ingestion > .dev/ingestion.log 2>&1 &
	@MFT_CONFIG=configs/config.yaml $(MAKE) --no-print-directory run-execution > .dev/execution.log 2>&1 &
	@MFT_CONFIG=configs/config.yaml $(MAKE) --no-print-directory run-jobs      > .dev/jobs.log 2>&1 &
	@sleep 3
	@$(MAKE) --no-print-directory status

stop:
	@-pkill -f 'exe/(server|ingestion|execution|jobs)' 2>/dev/null || true
	@-pkill -f 'go run ./services' 2>/dev/null || true
	@-pkill -f 'uvicorn app.main:app' 2>/dev/null || true

down: stop

status:
	@for p in 9090 8080 9091 9092 8000; do \
		if ss -ltn 2>/dev/null | grep -q ":$$p "; then \
			echo "  :$$p  UP"; \
		else \
			echo "  :$$p  down"; \
		fi; \
	done

# backtest replays historical candles through the model without placing orders.
backtest:
	@test -d $(VENV) || { echo "run 'make setup' first"; exit 1; }
	@cd $(ROOT)/services/inference && ../inference/.venv/bin/python -m app.backtest $(ARGS)

# --- worktrees -------------------------------------------------------------
# Each component is developed in an isolated worktree under .worktrees/.
# These live inside the repo so that all file access stays within the
# project directory.

wt:
	@scripts/worktree.sh add "$(NAME)" "$(BRANCH)"

wt-list:
	@git worktree list

wt-remove:
	@scripts/worktree.sh remove "$(NAME)"

wt-prune:
	@scripts/worktree.sh prune

# --- docker ----------------------------------------------------------------

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
