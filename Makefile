.PHONY: install test test-unit test-integration run clean

install:
	uv sync

test:
	uv run pytest tests/

test-unit:
	uv run pytest tests/test_mocks.py tests/test_file_access.py -v

test-integration:
	uv run pytest tests/test_main.py -v

run:
	uv run main.py

clean:
	rm -f results.csv
	rm -f reports/test-report.html
