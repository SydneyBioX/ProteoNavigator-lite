.PHONY: install-dev run test

install-dev:
	conda run -n cytogater-python python -m pip install -e .

run:
	DEBUG=false conda run -n cytogater-python chainlit run app.py

test:
	conda run -n cytogater-python pytest -q
