.PHONY: all gondolin-ml-autoresearch enclave-image runner-image

RUNNER_IMAGE ?= ml-autoresearch-runner:local
ENCLAVE_IMAGE ?= ml-autoresearch-enclave:local

MAKEFILE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/..)

all: gondolin-ml-autoresearch runner-image

gondolin-ml-autoresearch: enclave-image
	gondolin build --config $(MAKEFILE_DIR)/gondolin-build-config.json --output $(MAKEFILE_DIR)/gondolin-ml-autoresearch

enclave-image: $(MAKEFILE_DIR)/Dockerfile.enclave
	docker build -f $(MAKEFILE_DIR)/Dockerfile.enclave -t $(ENCLAVE_IMAGE) $(PROJECT_ROOT)

runner-image: $(MAKEFILE_DIR)/Dockerfile.runner
	docker build -f $(MAKEFILE_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) $(PROJECT_ROOT)
