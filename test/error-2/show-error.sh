#!/bin/bash          

PYTHON_COMMAND=$(command -v python || command -v python2 || command -v python2.7)
THIS_SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
command="${PYTHON_COMMAND} -m ${THIS_SCRIPT_DIRECTORY}/func_test ${THIS_SCRIPT_DIRECTORY}/../../src/distribute-proto"
#print the command
echo ${command}

#execute the command
${command}

