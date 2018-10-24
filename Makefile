.PHONY: image

IMAGE_NAME ?= codeclimate/feature-or-not

image:
	docker build --tag $(IMAGE_NAME) .
