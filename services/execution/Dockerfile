# syntax=docker/dockerfile:1

# Build stage: compile the execution service binary.
FROM golang:1.25-alpine AS build
WORKDIR /src

COPY go.work go.work.sum ./
COPY core/ core/
COPY services/ services/

RUN cd services/execution && \
    CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" \
      -o /out/execution ./cmd/server

# Runtime stage.
FROM alpine:3.20
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app
COPY --from=build /out/execution /usr/local/bin/execution
# Use the example config (empty credentials) as the in-image default.
COPY configs/config.example.yaml configs/config.yaml

EXPOSE 8080 9090
ENV MFT_CONFIG=/app/configs/config.yaml
CMD ["execution"]
