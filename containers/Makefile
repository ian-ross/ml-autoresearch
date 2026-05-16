.PHONY: runner-image

RUNNER_IMAGE ?= ml-autoresearch-runner:local
ENCLAVE_IMAGE ?= ml-autoresearch-enclave:local

all: gondolin-ml-autoresearch runner-image

gondolin-ml-autoresearch: enclave-image
	gondolin build --config gondolin-build-config.json --output gondolin-ml-autoresearch

enclave-image: Dockerfile.enclave
	docker build -f Dockerfile.enclave -t $(ENCLAVE_IMAGE) .

runner-image: Dockerfile.runner
	docker build -f Dockerfile.runner -t $(RUNNER_IMAGE) .
