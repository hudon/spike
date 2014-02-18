#! /bin/bash

if [ `ps axf | grep distribution | grep -v grep | wc -l` -ne 0 ]
then
    echo "Killing previous instances of daemon:"
    cmds=$(ps axf | grep distribution | grep -v grep | awk '{print "kill " $1}')
    echo "${cmds}"
    ${cmds}
else
    echo "No previous instances of daemon found."
fi
