.PHONY: build run-ingestion run-execution run-jobs tidy vet fmt

build:
	go build ./...

run-ingestion:
	go run ./services/ingestion/cmd/server

run-execution:
	go run ./services/execution/cmd/server

run-jobs:
	go run ./services/jobs/cmd/server

tidy:
	cd core && go mod tidy
	cd services/ingestion && go mod tidy
	cd services/execution && go mod tidy
	cd services/jobs && go mod tidy
	go work sync

vet:
	go vet ./...

fmt:
	gofmt -l -w $(shell find core services -name '*.go')
