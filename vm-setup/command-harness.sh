#!/bin/bash
#  This script will execute a set of commands and test them for success.
#  The script will stop after it the first command with an invalid return code

if [ $# == 0 ]; then
        echo "The number of parmeters was 0"
        echo ""
        echo "Expected syntax:"
        echo "this-script file-of-newline-delimited-commands"
        echo ""
        exit
fi

Escape="\033"
RedF="${Escape}[31m"
GreenF="${Escape}[32m"
Reset="${Escape}[0m"

#  The file of commands which are delimited by newlines
COMMANDS=$1

echo "-- Executing commands from file $COMMANDS"
cat $COMMANDS | while IFS='' read -r line || [ -n "$line" ] ; do
        #make sure that this is not an empty line
        [[ ! $line ]] && continue
        printf "${GreenF}Executing command ${line}${Reset}\n"
        #execute our command
        eval "$line"
        #store the return value to see if was successful
        RTNVAL=$?
        if [ $RTNVAL -gt 0 ]; then
                printf "${RedF}ERROR: failed to execute command\n${line}\nwith error code ${RTNVAL}${Reset}\n"
                exit 1
        fi
done
echo "-- Successfuly finished executing commands from file $COMMANDS"
