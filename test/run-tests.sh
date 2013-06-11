#!/bin/bash

PYTHON=$(command -v python2 || command -v python2.7 || command -v python)
THIS_SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#  All file locations are relative to this file
CODE_TO_TEST_PATH="../src"
WORKING_CODE_PATH="../examples/new-theano"
NENGO_TESTS_DIR="nengo_tests"

TEST_SCRIPTS=(
  "${NENGO_TESTS_DIR}/test_array.py"
#"${NENGO_TESTS_DIR}/test_cleanup.py"
#"${NENGO_TESTS_DIR}/test_decoded_weight_matrix.py"
);

compareOutput(){
  ACTUAL_OUT_CMD="${PYTHON} ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${CODE_TO_TEST_PATH}"
  EXPECTED_OUT_CMD="${PYTHON}  ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${WORKING_CODE_PATH}"

  start=$(date +"%s")
  echo -ne "Computing output from command '${ACTUAL_OUT_CMD}'..."
  ACTUAL_OUT=$(${ACTUAL_OUT_CMD})
  end=$(date +"%s")
  diff=$(($end - $start))
  echo "Took ${diff} seconds."

  start=$(date +"%s")
  echo -ne "Computing output from command '${EXPECTED_OUT_CMD}'..."
  EXPECTED_OUT=$(${EXPECTED_OUT_CMD})
  end=$(date +"%s")
  diff=$(($end-$start))
  echo "Took ${diff} seconds."

  #diff=$(diff <(echo "$output") <(echo "$2"))

  #TODO don't just compare output, look at return code (it must match and must
  #be 0)
  diff=$(diff <(echo "${ACTUAL_OUT}") <(echo "${EXPECTED_OUT}"))

  if [ ${#diff} -ne 0 ]; then
    echo "ERROR:"
    echo "Output from '${ACTUAL_OUT_CMD}' and '${EXPECTED_OUT_CMD}' does not match."
    echo -e "Diff was:"
    echo "$diff"
    exit 1
  else
    echo "Outputs match exactly. Test Passed."
  fi
}

for test_script in "${TEST_SCRIPTS[@]}" ;
do
  compareOutput $test_script ;
done

