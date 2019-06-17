IMAGE_NAME ?= nima
SRC ?= nima tests setup.py
LINE_LENGTH=120


.PHONY: lint
lint:
	black --check --line-length=$(LINE_LENGTH) $(SRC)
	flake8 $(SRC)
	isort -rc -c $(SRC)

.PHONY: format
format:
	isort -rc $(SRC)
	black --line-length=$(LINE_LENGTH) $(SRC)


.PHONY: test_integration
test_integration:
	pytest -vv -ss --cov-config=setup.cfg --cov nima tests/integration

.PHONY: test_unit
test_unit:
	pytest -vv -ss --cov-config=setup.cfg --cov nima tests/unit

.PHONY: build_docker
build_docker:
	docker build -t $(IMAGE_NAME):latest .

.PHONY: push_docker
push_docker: build_docker
	docker tag $(IMAGE_NAME):latest truskovskyi/$(IMAGE_NAME):latest
	docker login -u truskovskyi -p $(DOCKER_PASS)
	docker push truskovskyi/$(IMAGE_NAME):latest


.PHONY: run_lint
run_lint: build_docker
	docker run -it $(IMAGE_NAME):latest make lint

.PHONY: run_unit
run_unit: build_docker
	docker run -it $(IMAGE_NAME):latest make test_unit

.PHONY: run_integration
run_integration: build_docker
	docker run -it $(IMAGE_NAME):latest make test_integration
