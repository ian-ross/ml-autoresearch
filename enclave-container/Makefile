# Make targets:
#   all / build       Build the gondolin output directory
#   docker-image      Build the Docker image used by gondolin
#   clean             Remove make stamps and gondolin output
#   clean-stamps      Remove only make stamps

.DEFAULT_GOAL := all
.DELETE_ON_ERROR:

# gondolin needs tools from /usr/sbin. Export once instead of repeating PATH=...
export PATH := /usr/sbin:$(PATH)

BUILD_CONFIG  ?= build-config.json
# Default the Docker tag to the image name gondolin will consume.
IMAGE_TAG     ?= $(shell python3 -c 'import json; print(json.load(open("$(BUILD_CONFIG)"))["oci"]["image"])' 2>/dev/null || printf gondolin-debian)
DOCKERFILE    ?= Dockerfile
DOCKER_CONTEXT?= .
OUTPUT_DIR    ?= gondolin-debian
STAMP_DIR     ?= .make

DOCKER_STAMP  := $(STAMP_DIR)/docker-image.$(subst /,_,$(subst :,_,$(IMAGE_TAG))).stamp
BUILD_STAMP   := $(OUTPUT_DIR)/.gondolin-build.stamp

.PHONY: all build docker-image clean clean-stamps rebuild

all: build

build: $(BUILD_STAMP)

docker-image: $(DOCKER_STAMP)

$(STAMP_DIR):
	mkdir -p $@

# Docker images are not filesystem artifacts, so represent the completed build
# with a stamp file. If Dockerfile changes, the image is rebuilt.
$(DOCKER_STAMP): $(DOCKERFILE) | $(STAMP_DIR)
	docker build -t $(IMAGE_TAG) -f $(DOCKERFILE) $(DOCKER_CONTEXT)
	touch $@

# The gondolin output is a directory, so track the build with a stamp inside the
# output directory. If the directory is deleted, the stamp disappears too.
$(BUILD_STAMP): $(DOCKER_STAMP) $(BUILD_CONFIG)
	gondolin build --config $(BUILD_CONFIG) --output $(OUTPUT_DIR)
	touch $@

rebuild: clean all

clean-stamps:
	rm -rf $(STAMP_DIR)
	rm -f $(BUILD_STAMP)

clean: clean-stamps
	rm -rf $(OUTPUT_DIR)
