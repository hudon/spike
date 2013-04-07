#!/bin/bash          

THIS_SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
command="python -m ${THIS_SCRIPT_DIRECTORY}/matrix_multiplication ${THIS_SCRIPT_DIRECTORY}/../../src/distribute-proto"
#print the command
echo ${command}

#execute the command
${command}

