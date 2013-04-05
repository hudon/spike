#!/bin/bash          

CODE_TO_TEST_PATH="../src/distribute-proto"
WORKING_CODE_PATH="../examples/theano"

TEST_SCRIPTS=(
	'matrix_multiplication.py'
	'cleanup_test.py'
	'array_test.py'
	'func_test.py'
);

compareOutput(){
	realpath=$(echo $(cd $(dirname ${1}); pwd)/$(basename ${1}))
	observed_output_command="python ${realpath} ${CODE_TO_TEST_PATH}"
	goal_output_command="python ${realpath} ${WORKING_CODE_PATH}"

	echo "Computing output from command '${observed_output_command}'..."
	observed_output=$(${observed_output_command})

	echo "Computing output from command '${goal_output_command}'..."
	goal_output=$(${goal_output_command})

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

