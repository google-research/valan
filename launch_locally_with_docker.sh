#!/bin/bash
# Launches a VALAN training job locally with docker container.
#
# Prerequisite: docker.
#
# Usage:
# launch_locally_with_docker.sh PROBLEM TRAIN_DATA EVAL_DATA NUM_ACTORS \
#   NUM_EVAL_ACTORS
#
# Example:
# launch_locally_with_docker.sh R2R R2R_3scans R2R_3scans 3 3
#

set -o pipefail
set -o nounset

die()
{
  echo "$1" >&2
  exit 1
}


PROBLEMS="R2R|NDH|R2R+NDH"

[[ "$#" -eq 5 ]] || die "Usage: launch_locally_with_docker.sh ["${PROBLEMS}"] \
 [train_dataset] [eval_dataset] [Num train actors] [Num eval actors]"
echo $1 | grep -E -q "${PROBLEMS}" || die "Supported problems: ${PROBLEMS}"

export PROBLEM=$1
export TRAIN_DATA=$2
export EVAL_DATA=$3
export NUM_ACTORS=$4
export NUM_EVAL_ACTORS=$5

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $DIR

bash ./docker/build.sh

docker_version=$(docker version --format '{{.Server.Version}}')
# Launch jobs with all available GPUs.
if [[ "${docker_version}" > "19.03" ]]; then
  docker run --gpus all --entrypoint ./scripts/local_run.sh -it -p 6006:6006 \
   --name "valan_${PROBLEM}_local_run_gpu" --rm \
    valan "${PROBLEM}" "${TRAIN_DATA}" "${EVAL_DATA}" "${NUM_ACTORS}" \
    "${NUM_EVAL_ACTORS}"
fi

# Fall back to CPU training if the above job fails or docker version is too old.
if [[ "$?" > "0" ]] || [[ "${docker_version}" < "19.03" ]]; then
  echo ""
  echo "No valid GPUs found. Use CPUs only!!"; sleep 3
  docker run --entrypoint ./scripts/local_run.sh -it -p 6006:6006 \
   --name "valan_${PROBLEM}_local_run" --rm \
    valan "${PROBLEM}" "${TRAIN_DATA}" "${EVAL_DATA}" "${NUM_ACTORS}" \
    "${NUM_EVAL_ACTORS}"
fi
