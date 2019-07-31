SHELL := /bin/bash

help:
	@echo "init - initialize the project"
	@echo "test - setup pyenv and pipenv"
	@echo "publishmajor - publish major version"
	@echo "publisminor - publish minor version"
	@echo "publishpatch - publish patch version"

init:
	bash bin/init.sh
	pipenv shell

.PHONY: test
test:
	bash bin/test.sh
	pylint logsensei --reports=y

publishmajor:
	bash bin/publishmajor.sh

publishminor:
	bash bin/publishminor.sh

publishpatch:
	bash bin/publishpatch.sh