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
	pipenv run pylint logsensei --reports=y
	bash bin/test.sh

publishmajor:
	bash bin/publishmajor.sh

publishminor:
	bash bin/publishminor.sh

publishpatch:
	bash bin/publishpatch.sh