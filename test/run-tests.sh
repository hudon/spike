#!/bin/bash

RED_TEXT="\e[1;31m"
GREEN_TEXT="\e[1;32m"
NORMAL_TEXT="\e[0m"
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

  #  This will compare the output that goes to standard out.  At the moment, we don't
  #  check the output on standard error.
  start=$(date +"%s")
  echo -ne "Computing output from command '${ACTUAL_OUT_CMD}'..."
  ACTUAL_OUT=$(${ACTUAL_OUT_CMD})
  ACTUAL_RETURN_CODE=$?
  end=$(date +"%s")
  diff=$(($end - $start))
  echo "Took ${diff} seconds."

  start=$(date +"%s")
  echo -ne "Computing output from command '${EXPECTED_OUT_CMD}'..."
  EXPECTED_OUT=$(${EXPECTED_OUT_CMD})
  EXPECTED_RETURN_CODE=$?
  end=$(date +"%s")
  diff=$(($end-$start))
  echo "Took ${diff} seconds."

  diff=$(diff <(echo "${ACTUAL_OUT}") <(echo "${EXPECTED_OUT}"))

  if [ ${#diff} -ne 0 ]; then
    echo "Output from '${ACTUAL_OUT_CMD}' and '${EXPECTED_OUT_CMD}' does not match."
    echo -e "Diff was:"
    echo "$diff"
    echo -e ${RED_TEXT}"FAILURE:"${NORMAL_TEXT}
    exit 1
  elif [ ${ACTUAL_RETURN_CODE} -ne ${EXPECTED_RETURN_CODE} ]; then
    echo -e ${RED_TEXT}"FAILURE:"${NORMAL_TEXT}
    echo "Return code of ${ACTUAL_RETURN_CODE} from '${ACTUAL_OUT_CMD}' and ${EXPECTED_RETURN_CODE} from '${EXPECTED_OUT_CMD}' does not match."
    echo "Test failure."
    exit 1
  else
    echo -e ${GREEN_TEXT}"Outputs match exactly. Test Passed."${NORMAL_TEXT}
  fi
}

for test_script in "${TEST_SCRIPTS[@]}" ;
do
  compareOutput $test_script ;
done
