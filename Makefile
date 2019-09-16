SHELL := /bin/bash

help:
	@echo "init - initialize the project"
	@echo "test - setup pyenv and pipenv"
	@echo "publishmajor - publish major version"
	@echo "publisminor - publish minor version"
	@echo "publishpatch - publish patch version"

init:
	bash bin/init.sh

.PHONY: test
test:
	bash bin/test.sh

publishmajor:
	bash bin/publishmajor.sh

publishminor:
	bash bin/publishminor.sh

publishpatch:
	bash bin/publishpatch.sh