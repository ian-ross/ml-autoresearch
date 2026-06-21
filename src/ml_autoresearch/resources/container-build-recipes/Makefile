.PHONY: all ml-autoresearch-agent agent-image runner-image enclave-image

RUNNER_IMAGE ?= ml-autoresearch-runner:local
AGENT_IMAGE ?= ml-autoresearch-agent:local
ENCLAVE_IMAGE ?= ml-autoresearch-enclave:local

MAKEFILE_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PROJECT_ROOT := $(abspath $(MAKEFILE_DIR)/..)

all: ml-autoresearch-agent runner-image

ml-autoresearch-agent: agent-image
	gondolin build --config $(MAKEFILE_DIR)/gondolin-build-config.json --output $(MAKEFILE_DIR)/ml-autoresearch-agent

agent-image: $(MAKEFILE_DIR)/Dockerfile.agent
	docker build -f $(MAKEFILE_DIR)/Dockerfile.agent -t $(AGENT_IMAGE) $(PROJECT_ROOT)

runner-image: $(MAKEFILE_DIR)/Dockerfile.runner
	docker build -f $(MAKEFILE_DIR)/Dockerfile.runner -t $(RUNNER_IMAGE) $(PROJECT_ROOT)

enclave-image: $(MAKEFILE_DIR)/Dockerfile.enclave
	docker build -f $(MAKEFILE_DIR)/Dockerfile.enclave -t $(ENCLAVE_IMAGE) $(PROJECT_ROOT)
