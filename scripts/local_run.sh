#!/bin/bash
# Script that launches VALAN jobs locally with multiple workers and run
# asycronously with the learner.
#
# NOTE: running this scirpt requires specific packages (e.g., seed_rl and
# protobuf). Thus it is better to run inside a docker container using
# `launch_locally_with_docker.sh`.
#
# Usage:
# ./local_run.sh PROBLEM TRAIN_FILE EVAL_FILE NUM_ACTORS NUM_EVAL_ACTORS
#
# Example:
# ./local_run.sh R2R R2R_3scans R2R_3scans 3 3
#

set -e
set -o pipefail
set -o nounset

if [[ "$#" -ne 5 ]]; then
    echo "Number of input args must be 5." && exit 1
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

PROBLEM=$1
TRAIN_DATA_SOURCE=$2
EVAL_DATA_SOURCE=$3
NUM_ACTORS=$4
NUM_EVAL_ACTORS=$5
shift 5

export PYTHONPATH="/"

ACTOR_BINARY="CUDA_VISIBLE_DEVICES='' python ../r2r/actor_main.py"
LEARNER_BINARY="python ../r2r/learner_main.py"
AGGREGATOR_BINARY="CUDA_VISIBLE_DEVICES='' python ../r2r/eval_aggregator_main.py"

SCAN_BASE_DIR="/tmp/valan/testdata/"
DATA_BASE_DIR="/tmp/valan/testdata/"
VOCAB_DIR="/tmp/valan/testdata/"
IMAGE_FEATURES_DIR="/tmp/valan/testdata/image_features_efficientnet/"

SERVER_ADDRESS="unix:/tmp/foo"
AGG_SERVER_ADDRESS="${SERVER_ADDRESS}_agg"
LOG_DIR="/tmp/agent"

echo "$(pwd)"

# Copy data to /tmp/testdata for local access.
mkdir -p /tmp/valan/testdata/
cp -R -f ../r2r/testdata/R2R_3scans.json /tmp/valan/testdata/
cp -R -f ../r2r/testdata/vocab.txt /tmp/valan/testdata/
cp -R -f ../r2r/testdata/scans /tmp/valan/testdata/
cp -R -f ../r2r/testdata/connections /tmp/valan/testdata/
cp -R -f ../r2r/testdata/image_features_efficientnet /tmp/valan/testdata/

rm "${LOG_DIR}" -Rf
mkdir -p "${LOG_DIR}"

# Start tmux session and spawn a window for each worker.
tmux new-session -d -t valan  # Create a session and group it as `valan`.
cat >"${LOG_DIR}"/instructions <<EOF
********************************************************************************

Welcome to the VALAN local training of problem "${PROBLEM}" with "${NUM_ACTORS}"
train actors and "${NUM_EVAL_ACTORS}" eval actors. VALAN uses tmux for easy
navigation between the learner and different actors and the eval_aggregator, all
of which are running asyncronously during training.

To switch to a specific task, press CTRL+b, then [tab id],
  e.g., CTRL+b then 0 for the learner, CTRL+b then 1 for actor_1, etc.


To monitor the training and eval progress, connect to Tensorboard using the link
below. Note that the eval results will show up after accumulating 3 checkpoints.
So please be patient.
    http://localhost:6006/


To stop:
- If running on a local machine, you can stop training at any time by executing:
    \`./stop_local.sh\`

- If running in a Docker container, you can stop and remove the
container by typing CTRL+b then d. Note that this will completely remove the
container.

********************************************************************************
EOF
tmux send-keys clear
tmux send-keys KPEnter
tmux send-keys "python3 check_gpu.py 2> /dev/null"
tmux send-keys KPEnter
tmux send-keys "cat ${LOG_DIR}/instructions"
tmux send-keys KPEnter

# Launch learner.
tmux new-window -d -n learner

COMMAND="${LEARNER_BINARY} --logtostderr --pdb_post_mortem "\
"--problem=${PROBLEM} --server_address=${SERVER_ADDRESS} "\
"--scan_base_dir=${SCAN_BASE_DIR} --data_base_dir=${DATA_BASE_DIR} "\
"--vocab_dir=${VOCAB_DIR} --image_features_dir=${IMAGE_FEATURES_DIR}  $@"
tmux send-keys -t "learner" "${COMMAND}" ENTER

# Launch actors for training.
for (( id=0; id<${NUM_ACTORS}; id++ )); do
    tmux new-window -d -n "actor_${id}"
    COMMAND="${ACTOR_BINARY} --data_source=${TRAIN_DATA_SOURCE} --logtostderr "\
"--pdb_post_mortem --task=${id} --num_tasks=${NUM_ACTORS} "\
"--problem=${PROBLEM} --server_address=${SERVER_ADDRESS} "\
"--scan_base_dir=${SCAN_BASE_DIR} --data_base_dir=${DATA_BASE_DIR} "\
"--vocab_dir=${VOCAB_DIR} --image_features_dir=${IMAGE_FEATURES_DIR}  $@"
    tmux send-keys -t "actor_${id}" "${COMMAND}" ENTER
done

# Launch eval actors.
for (( id=0; id<${NUM_EVAL_ACTORS}; id++ )); do
    tmux new-window -d -n "evaler_${id}"
    COMMAND="${ACTOR_BINARY} --data_source=${EVAL_DATA_SOURCE} --mode=eval"\
" --logtostderr --pdb_post_mortem --task=${id} --num_tasks=${NUM_EVAL_ACTORS} "\
"--problem=${PROBLEM} --server_address=${AGG_SERVER_ADDRESS} "\
"--scan_base_dir=${SCAN_BASE_DIR} --data_base_dir=${DATA_BASE_DIR} "\
"--vocab_dir=${VOCAB_DIR} --image_features_dir=${IMAGE_FEATURES_DIR}  $@"
    tmux send-keys -t "evaler_${id}" "${COMMAND}" ENTER
done

# Launch eval aggregator.
tmux new-window -d -n "aggregator"
COMMAND="${AGGREGATOR_BINARY} --logdir=${LOG_DIR} --logtostderr "\
"--server_address=${AGG_SERVER_ADDRESS} --aggregator_prefix=eval_agg"
tmux send-keys -t "aggregator" "${COMMAND}" ENTER

# Launch Tensorboard.
tmux new-window -d -n "tensorboard"
COMMAND="tensorboard --logdir=${LOG_DIR} --port=6006 --bind_all"
tmux send-keys -t "tensorboard" "${COMMAND}" ENTER

tmux attach -t valan
