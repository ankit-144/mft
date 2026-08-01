# syntax=docker/dockerfile:1

# Build stage: compile the ingestion service binary.
FROM golang:1.25-alpine AS build
WORKDIR /src

# Go workspace modules.
COPY go.work go.work.sum ./
COPY core/ core/
COPY services/ services/

# Build only the ingestion server.
RUN cd services/ingestion && \
    CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
      -o /out/ingestion ./cmd/server

# Runtime stage: minimal image with the binary and config dir.
FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app
COPY --from=build /out/ingestion /usr/local/bin/ingestion
# Use the example config (empty credentials) as the in-image default.
COPY configs/config.example.yaml configs/config.yaml

EXPOSE 9090
ENV MFT_CONFIG=/app/configs/config.yaml
CMD ["ingestion"]
