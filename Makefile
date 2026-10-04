.PHONY: \
	build test vet fmt lint tidy \
	setup dev stop down status test-python \
	run-ingestion run-execution run-jobs run-inference \
	venv inference-deps tabfm-weights \
	backtest research integration benchmarks \
	wt wt-list wt-remove wt-prune \
	docker-build docker-up docker-down docker-logs

MODULES := core services/ingestion services/execution services/jobs
VENV    := services/inference/.venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
ROOT    := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
GO_JOBS ?= 2
export GOMAXPROCS ?= 2
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export GOCACHE ?= /tmp/mft-go-cache

build:
	@for m in $(MODULES); do \
		echo "== build $$m =="; \
		( cd $$m && go build -p $(GO_JOBS) ./... ) || exit 1; \
	done
	@mkdir -p $(ROOT)/bin
	@for service in ingestion execution jobs; do go build -p $(GO_JOBS) -o $(ROOT)/bin/$$service ./services/$$service/cmd/server || exit 1; done

test:
	@for m in $(MODULES); do \
		echo "== test $$m =="; \
		( cd $$m && go test -race -p $(GO_JOBS) -timeout 90s ./... ) || exit 1; \
	done

PYTEST_PERMITTED := model/tests/test_base.py model/tests/test_heuristic_model.py

test-python:
	@test -x $(ROOT)/$(VENV)/bin/python || { echo "run 'make setup' first"; exit 1; }
	cd $(ROOT)/services/inference && PYTHONPATH=. $(ROOT)/$(VENV)/bin/python -m pytest app/tests $(PYTEST_PERMITTED) -q

vet:
	@for m in $(MODULES); do \
		echo "== vet $$m =="; \
		( cd $$m && go vet -p $(GO_JOBS) ./... ) || exit 1; \
	done

fmt:
	@for m in $(MODULES); do \
		( cd $$m && gofmt -l -w . ) || exit 1; \
	done

lint: vet
	@out=$$(for m in $(MODULES); do ( cd $$m && gofmt -l . ); done); \
	if [ -n "$$out" ]; then echo "gofmt needed:"; echo "$$out"; exit 1; fi
	@echo "lint ok"

tidy:
	@for m in $(MODULES); do \
		( cd $$m && go mod tidy ) || exit 1; \
	done
	go work sync

run-ingestion:
	cd $(ROOT) && go run ./services/ingestion/cmd/server

run-execution:
	cd $(ROOT) && go run ./services/execution/cmd/server

run-jobs:
	cd $(ROOT) && go run ./services/jobs/cmd/server

run-inference:
	cd $(ROOT)/services/inference && MFT_INFERENCE_MODEL=$${MFT_INFERENCE_MODEL:-heuristic} $(ROOT)/$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port 8000

venv:
	services/inference/venv.sh

inference-deps: venv

setup: build inference-deps
	@test -f configs/config.yaml || cp configs/config.example.yaml configs/config.yaml

dev: build
	python3 scripts/dev.py

stop:
	python3 scripts/dev.py --stop

down: stop

status:
	python3 scripts/dev.py --status

ARGS ?=
backtest:
	cd $(ROOT)/services/inference && PYTHONPATH=. $(ROOT)/$(PY) -m app.backtest $(if $(strip $(ARGS)),$(ARGS),--synthetic --model heuristic --bars 200 --feature-mode incremental)

integration: build
	cd $(ROOT)/services/inference && PYTHONPATH=. $(ROOT)/$(PY) -m pytest app/tests/test_platform_integration.py -q

benchmarks:
	python3 scripts/benchmark_runtime.py

research:
	cd $(ROOT)/services/inference && PYTHONPATH=. $(ROOT)/$(PY) -m app.research $(if $(strip $(ARGS)),$(ARGS),--source synthetic --symbols RELIANCE --bars 1000)

wt:
	scripts/worktree.sh add "$(NAME)" "$(BRANCH)"

wt-list:
	git worktree list

wt-remove:
	scripts/worktree.sh remove "$(NAME)"

wt-prune:
	scripts/worktree.sh prune

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
