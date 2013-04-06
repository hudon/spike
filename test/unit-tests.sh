#!/bin/bash          

THIS_SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#  All file locations are relative to this file
CODE_TO_TEST_PATH="../src/distribute-proto"
WORKING_CODE_PATH="../examples/theano"

TEST_SCRIPTS=(
	'matrix_multiplication'
	'cleanup_test'
	'array_test'
	'func_test'
);

compareOutput(){
	observed_output_command="python -m ${THIS_SCRIPT_DIRECTORY}/${1} ${THIS_SCRIPT_DIRECTORY}/${CODE_TO_TEST_PATH}"
	goal_output_command="python -m ${THIS_SCRIPT_DIRECTORY}/${1} ${THIS_SCRIPT_DIRECTORY}/${WORKING_CODE_PATH}"

	start=$(date +"%s")
	echo -ne "Computing output from command '${observed_output_command}'..."
	observed_output=$(${observed_output_command})
	end=$(date +"%s")
	diff=$(($end-$start))
	echo "Took ${diff} seconds."

	start=$(date +"%s")
	echo -ne "Computing output from command '${goal_output_command}'..."
	goal_output=$(${goal_output_command})
	end=$(date +"%s")
	diff=$(($end-$start))
	echo "Took ${diff} seconds."

	diff=$(diff <(echo "$output") <(echo "$2"))

	#TODO don't just compare output, look at return code
	diff=$(diff <(echo "${observed_output}") <(echo "${goal_output}"))

	if [ ${#diff} -ne 0 ]; then
		echo "Unit Test Failure:"
		echo "Output from '${observed_output_command}' and '${goal_output_command}' does not match."
		echo -e "Diff was:"
		echo "$diff"
		exit 1
	else
		echo "Outputs match exactly. Test Passed."
	fi
}

for test_script in "${TEST_SCRIPTS[@]}"
do 
	compareOutput $test_script
done

