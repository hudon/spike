#!/bin/bash

RED_TEXT="\e[1;31m"
GREEN_TEXT="\e[1;32m"
NORMAL_TEXT="\e[0m"
PYTHON=$(command -v python2 || command -v python2.7 || command -v python)
THIS_SCRIPT_DIRECTORY="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

#  All file locations are relative to this file
TARGET_DIR="../src"
SOURCE_DIR="../examples/new-theano"
NENGO_TESTS_DIR="nengo_tests"
HOSTS_FILE=$1

TEST_SCRIPTS=(
  "${NENGO_TESTS_DIR}/test_array.py"
  "${NENGO_TESTS_DIR}/test_basal_ganglia.py"
  "${NENGO_TESTS_DIR}/test_decoded_weight_matrix.py"
  "${NENGO_TESTS_DIR}/test_direct.py"
  "${NENGO_TESTS_DIR}/test_enc.py"
  "${NENGO_TESTS_DIR}/test_eval_points.py"
  "${NENGO_TESTS_DIR}/test_fixed_seed.py"
  "${NENGO_TESTS_DIR}/test_func.py"
  "${NENGO_TESTS_DIR}/test_noise.py"
  "${NENGO_TESTS_DIR}/test_radius.py"
  "${NENGO_TESTS_DIR}/test_runtime.py"
  # "${NENGO_TESTS_DIR}/test_simplenode.py" ## uses instance methods (cannot pickle)
  "${NENGO_TESTS_DIR}/test_subnetwork.py"
  "${NENGO_TESTS_DIR}/test_transform.py"
  "${NENGO_TESTS_DIR}/test_weight_index_pre_post.py"
  # "${NENGO_TESTS_DIR}/test_writeout.py" # Note: requires extra libraries to function
  "matrix_multiplication_distributed.py"
);

compareOutput(){
  ACTUAL_OUT_CMD="${PYTHON} ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${TARGET_DIR} ${HOSTS_FILE}"

  EXPECTED_OUT_CMD="${PYTHON}  ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${SOURCE_DIR}"

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

  echo "$ACTUAL_OUT" > /tmp/1
  echo "$EXPECTED_OUT" > /tmp/2

  diff=$(diff <(echo "${ACTUAL_OUT}") <(echo "${EXPECTED_OUT}"))

  if [ ${#diff} -ne 0 ]; then
    echo -e "ERROR: Diff was:"
    echo "$diff"
    echo -e ${RED_TEXT}"ERROR: Output from '${ACTUAL_OUT_CMD}' and \
      '${EXPECTED_OUT_CMD}' does not match."${NORMAL_TEXT}
    exit 1
  #  Always test it against 0 because we arn't testing failure cases and we want
  #  to catch any import errors which will make both tests fail.
  elif [ ${ACTUAL_RETURN_CODE} -ne 0 ]; then
    echo -e ${RED_TEXT}"ERROR:"${NORMAL_TEXT}
    echo "Return code of ${ACTUAL_RETURN_CODE} from '${ACTUAL_OUT_CMD} was not 0.' \
      Expected was ${EXPECTED_RETURN_CODE} from '${EXPECTED_OUT_CMD}'"
    exit 1
  else
    echo -e ${GREEN_TEXT}"INFO: Program outputs are identical."${NORMAL_TEXT}
  fi
}

killtree() {
    local _pid=$1
    local _sig=${2:-TERM}
    kill -stop ${_pid}
    for _child in $(ps -o pid --no-headers --ppid ${_pid}); do
        killtree ${_child} ${_sig}
    done
    kill -${_sig} ${_pid}
}

function handle_sigint()
{
    #  Kill sub processes (the daemon) on ctrl-c
    for proc in `jobs -p`
    do
        killtree $proc
    done
    exit 0
}
trap handle_sigint SIGINT


PROGRAM="${PYTHON} ${THIS_SCRIPT_DIRECTORY}/${TARGET_DIR}/distributiond.py"
$PROGRAM > /dev/null &
PID=$!

echo "Started daemon with pid "${PID}"."

for test_script in "${TEST_SCRIPTS[@]}" ;
do
  compareOutput $test_script ;
done

kill ${PID}
echo "Killed daemon with pid "${PID}"."
