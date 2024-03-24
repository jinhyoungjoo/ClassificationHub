PYTHONPATH=python3

train:
	$(PYTHONPATH) train.py -c $(TRAIN_OPTIONS)

test:
	$(PYTHONPATH) -m pytest .

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

