.PHONY: build run

IMAGE = sstubs_miner

build:
	docker build \
	--tag=${IMAGE} \
	.

run:
	docker run --rm -ti \
	--volume="$(shell pwd)/src:/home" \
	${IMAGE} \
	/home/index.js
