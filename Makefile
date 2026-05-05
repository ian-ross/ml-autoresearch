.PHONY: runner-image

RUNNER_IMAGE ?= ml-autoresearch-runner:local

runner-image:
	docker build -t $(RUNNER_IMAGE) .
