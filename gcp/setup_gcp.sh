#!/bin/bash
# Sets up GCP project, storage bucket, region, and job submission function.

set -e
set -o nounset

# Project specific config.
PROJECT_ID="$(gcloud config get-value project)"
REGION="us-central1"
CONFIG_FILE="/tmp/config.yaml"
export IMAGE_URI="gcr.io/${PROJECT_ID}/valan"


# Function to submit training job to AI Platform.
submit_training_job () {
  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
  bash $DIR/../docker/build.sh
  bash $DIR/../docker/push.sh

  # Check GCS bucket.
  gsutil ls "${GS_BUCKET}" || (echo "GCS bucket ${GS_BUCKET} does not exist!" && exit 1)

  JOB_NAME="VALAN_${PROBLEM}_$(date +"%Y%b%d_%H%M%S")"
  # Assign job dir if not defined.
  if [[ "${JOB_DIR}" = "" ]]; then
    JOB_DIR="${JOB_NAME}"
  fi

  # Submit to AI platform and start training.
  gcloud beta ai-platform jobs submit training ${JOB_NAME} \
    --project="${PROJECT_ID}" \
    --job-dir="${GS_BUCKET}/${JOB_DIR}" \
    --region="${REGION}" \
    --config="${CONFIG_FILE}" \
    --stream-logs -- \
    --scan_base_dir="${SCAN_BASE_DIR}" \
    --data_base_dir="${DATA_BASE_DIR}" \
    --vocab_dir="${VOCAB_DIR}" \
    --vocab_file="${VOCAB_FILE}" \
    --image_features_dir="${IMAGE_FEATURES_DIR}" \
    --agent_type="${AGENT_TYPE}" \
    --problem="${PROBLEM}" \
    --data_source="${TRAIN_DATA_SOURCE}" \
    --eval_data_source="${EVAL_DATA_SOURCE}" \
    --num_eval_workers="${NUM_EVAL_WORKERS}" \
    --num_train_workers=${NUM_TRAIN_WORKERS} \
    --actors_per_eval_worker=${ACTORS_PER_EVAL_WORKER} \
    --actors_per_train_worker=${ACTORS_PER_TRAIN_WORKER} \
    --logdir="${GS_BUCKET}/${JOB_DIR}" \
    -- \
    --agent_device="GPU" \
    --save_checkpoint_secs=600 \
    --max_ckpt_to_keep=100
}
    # Additional config params or hyperparams can be set below.
    # multiple sets of eval data, defined by `multi_string` and `multi_integer`
