#!/bin/bash
# Config training params and hyperparams and launch a toy R2R training job.
#
# Prerequisite: the full R2R dataset including the pre-extracted image features,
# scan, connections, and vocab data must exist in the GCS bucket defined below.
#
# Docker image, GCP project info, and the actual job submission function are
# configurated in `setup_gcp.sh`.
#
# To launch the job:
# `bash r2r_train_3scans.sh`

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Data dirs on GCS.
export GS_BUCKET="gs://valan"
export SCAN_BASE_DIR="${GS_BUCKET}/testdata"
export DATA_BASE_DIR="${GS_BUCKET}/testdata"
export VOCAB_DIR="${GS_BUCKET}/testdata"
export VOCAB_FILE="vocab.txt"
export IMAGE_FEATURES_DIR="${GS_BUCKET}/testdata/image_features_efficientnet"

# If JOB_DIR="" then it will be assigned automatically. The job will continue
# training from the latest checkpoint if JOB_DIR is the same as a previous job.
export JOB_DIR=""

export PROBLEM=R2R
export AGENT_TYPE=R2R  # The same problem can have different agents e.g. R2R, MT

export TRAIN_DATA_SOURCE="R2R_3scans"
export NUM_TRAIN_WORKERS=3  # 3 workers for 3 scans.
export ACTORS_PER_TRAIN_WORKER=4  # Each worker runs 4 concurrent threads.

export ACTORS_PER_EVAL_WORKER=1
export EVAL_DATA_SOURCE="R2R_3scans_eval R2R_3scans_test"
export NUM_EVAL_WORKERS="3 3"  # 3 workers for 3 scans each.
NUM_TOTAL_EVAL_WORKERS=$((3 + 3 + 2))  # Add 2 for the aggregators.


# Copy toy data to gcs bucket.
gsutil ls "${GS_BUCKET}" >/dev/null 2>&1 || gsutil mb "${GS_BUCKET}"
(gsutil ls "${SCAN_BASE_DIR}" >/dev/null 2>&1  &&  gsutil ls "${IMAGE_FEATURES_DIR}" >/dev/null 2>&1) || gsutil -m cp -r ../r2r/testdata "${GS_BUCKET}"


# Set up GCP project and launch function.
source $DIR/setup_gcp.sh

# Create the training config yaml file with hyperparams.
cat > /tmp/config.yaml <<EOF
---
trainingInput:
  scaleTier: CUSTOM
  masterType: n1-standard-4  # Use small instance for toy dataset
  masterConfig:
    imageUri: "${IMAGE_URI}"
    acceleratorConfig:
      count: 1
      type: NVIDIA_TESLA_K80
  workerCount: ${NUM_TRAIN_WORKERS}
  workerType: n1-standard-4
  workerConfig:
    imageUri: "${IMAGE_URI}"
  evaluatorCount: ${NUM_TOTAL_EVAL_WORKERS}
  evaluatorType: n1-highmem-2
  evaluatorConfig:
    imageUri: "${IMAGE_URI}"
  parameterServerCount: 0
  useChiefInTfConfig: true
  hyperparameters:
    goal: MAXIMIZE
    hyperparameterMetricTag: eval/sdtw  # Should exactly match what's shown in TensorBoard
    maxTrials: 1
    maxParallelTrials: 1
    enableTrialEarlyStopping: true
    # Hyperparams are passed as command line FLAGS for python launcher.
    params:
    - parameterName: batch_size
      type: INTEGER
      minValue: 128
      maxValue: 128
      scaleType: NONE
    - parameterName: unroll_length
      type: INTEGER
      minValue: 10
      maxValue: 10
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: sync_agent_every_n_steps
      type: INTEGER
      minValue: 1
      maxValue: 1
      scaleType: NONE
    - parameterName: learning_rate
      type: DOUBLE
      minValue: 0.0001
      maxValue: 0.0001
      scaleType: UNIT_LOG_SCALE
    - parameterName: discounting
      type: DOUBLE
      minValue: 0.95
      maxValue: 0.95
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: lr_decay_rate
      type: DOUBLE
      minValue: 0.8
      maxValue: 0.8
      scaleType: UNIT_LINEAR_SCALE
    - parameterName: lr_decay_steps
      type: DISCRETE
      discreteValues:
      - 5000.
    - parameterName: entropy_cost
      type: DOUBLE
      minValue: 0.0001
      maxValue: 0.0001
      scaleType: UNIT_LOG_SCALE
EOF

submit_training_job
