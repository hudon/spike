SHELL = /bin/bash

.PHONY : test clean

default :
	@echo -e "INFO targets:\n\
	    make test [hosts_file=filename.txt]		runs tests\n\
	    make todo					show all TODOs in project"

test : test/run-tests.sh
	if [ -n "$(hosts_file)" ]; then \
		$(SHELL) $< `pwd`/$(hosts_file) ; \
	else \
		$(SHELL) $< ; \
	fi

clean :
	find . -name "*.pyc" | xargs rm

todo :
	grep -nir todo * --color
