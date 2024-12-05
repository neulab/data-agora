.PHONY: style

check_dirs := alchemy augv2 tests

style:
	isort $(check_dirs)
	black --line-length 119 --target-version py310 $(check_dirs)


unittest:
	pytest -v --cov=alchemy tests/


test:
	# rm -f augv2/samples/output.json
	python -m augv2.instance_generation