SHELL = /bin/bash

.PHONY : test

default :
	@echo -e "INFO targets:\n\
	    make test	runs tests\n\
	    make todo	show all TODOs in project"

test : test/run-tests.sh
	$(SHELL) $<

todo :
	grep -nir todo * --color
