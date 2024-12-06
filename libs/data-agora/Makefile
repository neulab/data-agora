.PHONY: style

check_dirs := .

style:
	uv run isort $(check_dirs)
	uv run black --line-length 119 --target-version py310 $(check_dirs)


unittest:
	uv run pytest -v tests/