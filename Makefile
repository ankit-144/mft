.PHONY: build test vet fmt tidy \
	run-ingestion run-execution run-jobs \
	docker-build docker-up docker-down docker-logs

MODULES := core services/ingestion services/execution services/jobs

build:
	go build ./...

test:
	@for m in $(MODULES); do \
		echo "== test $$m =="; \
		(cd $$m && go test ./...); \
	done

vet:
	@for m in $(MODULES); do \
		echo "== vet $$m =="; \
		(cd $$m && go vet ./...); \
	done

fmt:
	gofmt -l -w $(shell find core services -name '*.go')

tidy:
	cd core && go mod tidy
	cd services/ingestion && go mod tidy
	cd services/execution && go mod tidy
	cd services/jobs && go mod tidy
	go work sync

run-ingestion:
	go run ./services/ingestion/cmd/server

run-execution:
	go run ./services/execution/cmd/server

run-jobs:
	go run ./services/jobs/cmd/server

# --- Docker ---

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f
