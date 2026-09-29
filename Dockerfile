# Stage 1: Build binary
FROM rust:1.88-slim AS builder
WORKDIR /app
COPY . .
ENV CARGO_RESOLVER_INCOMPATIBLE_RUST_VERSIONS=fallback
RUN cargo generate-lockfile && cargo build --locked --release -p gen-zero-cli

# Stage 2: Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /app/target/release/gen-zero /usr/local/bin/gen-zero
COPY LICENSE NOTICE /app/

ENV PORT=8999
EXPOSE 8999

# Built-in container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8999/health || exit 1

ENTRYPOINT ["gen-zero"]
CMD ["serve", "--host", "0.0.0.0", "--mode", "sse", "--port", "8999"]
