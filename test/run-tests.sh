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
  # "${NENGO_TESTS_DIR}/test_eval_points.py"
  # "${NENGO_TESTS_DIR}/test_fixed_seed.py"
  "${NENGO_TESTS_DIR}/test_func.py"
  "${NENGO_TESTS_DIR}/test_noise.py"
  # "${NENGO_TESTS_DIR}/test_radius.py"
  # "${NENGO_TESTS_DIR}/test_runtime.py"
  # # "${NENGO_TESTS_DIR}/test_simplenode.py" ## uses instance methods (cannot pickle)
  # "${NENGO_TESTS_DIR}/test_subnetwork.py"
  # "${NENGO_TESTS_DIR}/test_transform.py"
  # "${NENGO_TESTS_DIR}/test_weight_index_pre_post.py"
  # # "${NENGO_TESTS_DIR}/test_writeout.py" # Note: requires extra libraries to function
  # "matrix_multiplication_distributed.py"

  "${NENGO_TESTS_DIR}/test_array_subs.py"
);

compareOutput(){
  ACTUAL_OUT_CMD="${PYTHON} ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${TARGET_DIR} --hosts=${THIS_SCRIPT_DIRECTORY}/remote_hosts.txt -s"

  EXPECTED_OUT_CMD="${PYTHON}  ${THIS_SCRIPT_DIRECTORY}/${1}\
    ${THIS_SCRIPT_DIRECTORY}/${SOURCE_DIR}"

  #  This will compare the output that goes to standard out.  At the moment, we don't
  #  check the output on standard error.
  start=$(date +"%s")
  echo -ne "Computing output from command \n${ACTUAL_OUT_CMD}\n"
  ACTUAL_OUT=$(${ACTUAL_OUT_CMD})
  ACTUAL_RETURN_CODE=$?
  end=$(date +"%s")
  diff=$(($end - $start))
  echo "Took ${diff} seconds."

  start=$(date +"%s")
  echo -ne "Computing output from command \n${EXPECTED_OUT_CMD}\n"
  EXPECTED_OUT=$(${EXPECTED_OUT_CMD})
  EXPECTED_RETURN_CODE=$?
  end=$(date +"%s")
  diff=$(($end-$start))
  echo "Took ${diff} seconds."

  echo "$ACTUAL_OUT" > /tmp/1
  echo "$EXPECTED_OUT" > /tmp/2

  diff=$(diff <(echo "${ACTUAL_OUT}") <(echo "${EXPECTED_OUT}"))

  DECIMAL_PLACES=8
  AWK_ARG="{ printf \"%0.${DECIMAL_PLACES}f\\n\", \$1}"
  EGREP_ARG="[0-9]+.[0-9]+"
  SED_ARG="s/([0-9]+.[0-9]+)/\\n\1\\n/g"

  if [ ${#diff} -ne 0 ]; then
    #  Get all the numbers our, round em, diff em
    ACTUAL_OUT=`echo "${ACTUAL_OUT}" | sed -r "${SED_ARG}" | egrep "${EGREP_ARG}" | awk "${AWK_ARG}"`
    EXPECTED_OUT=`echo "${EXPECTED_OUT}" | sed -r "${SED_ARG}" | egrep "${EGREP_ARG}" | awk "${AWK_ARG}"`
    diff=$(diff <(echo "${ACTUAL_OUT}") <(echo "${EXPECTED_OUT}"))
    if [ ${#diff} -ne 0 ]; then
      echo -e "ERROR: Diff was:"
      echo "$diff"
      echo -e ${RED_TEXT}"ERROR: Output from '${ACTUAL_OUT_CMD}' and \
        '${EXPECTED_OUT_CMD}' does not match even when rounding all numbers to ${DECIMAL_PLACES} decimal places."${NORMAL_TEXT}
      exit 1
    else
      echo -e ${GREEN_TEXT}"INFO: Program outputs are different, but identical when rounded to ${DECIMAL_PLACES} decimal places."${NORMAL_TEXT}
    fi
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
