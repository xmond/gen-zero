.PHONY: all test-rust test-python test-all fmt clippy check clean build-rust install-python

all: check test-rust

# Rust Commands
build-rust:
	cargo build --workspace --release

test-rust:
	cargo test --workspace

fmt:
	cargo fmt --all

clippy:
	cargo clippy --workspace --all-targets --all-features -- -D warnings

check:
	cargo check --workspace

# Python Commands
install-python:
	cd python && pip install -e .

test-python:
	@if command -v pytest >/dev/null 2>&1; then \
		pytest python/gen_zero/tests; \
	else \
		python3 -m unittest discover -s python/gen_zero/tests -p "test_*.py"; \
	fi

# Benchmark Commands
benchmark:
	python3 benchmarks/run_benchmark.py

# Combined Commands
test-all: test-rust test-python

clean:
	cargo clean
	rm -rf dist/
	find python -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find python -type f -name "*.pyc" -delete 2>/dev/null || true
