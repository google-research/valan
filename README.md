# VALAN: Vision and Language Agent Navigation

VALAN, short for Vision and Language Agent Navigation is a lightweight and
scalable software framework for deep reinforcement learning based on the
[SEED RL](https://arxiv.org/abs/1910.06591) architecture. The framework
facilitates the development and evaluation of embodied agents for solving
grounded language understanding tasks, such as Vision-and-Language Navigation
and Vision-and-Dialog Navigation, in photo-realistic environments, such as
[Matterport3D](https://niessner.github.io/Matterport/) and
[StreetLearn](https://sites.google.com/corp/view/streetlearn/dataset). Such
tasks require agents to interpret natural language instructions/dialog to
navigate in photo-realistic environments in order to achieve prescribed
navigation goals. We have added a minimal set of abstractions on top of SEED RL
allowing us to generalize the architecture to solve a variety of other RL
problems.

This package contains the implementations of the following problems:

*   VLN task on R2R dataset in Matterport3D environment
    ([paper](https://arxiv.org/abs/1711.07280))
*   NDH task on CVDN dataset in Matterport3D environment
    ([paper](https://arxiv.org/abs/1907.04957))
*   SDR and VLN tasks on Touchdown dataset in StreetLearn environment
    ([paper](https://arxiv.org/abs/1811.12354))

See [Mehta et al.](https://arxiv.org/abs/2001.03671) for details about our
implementation for Touchdown and the data supporting it.

For a detailed description of the architecture please read
[Lansing et al](https://arxiv.org/abs/1912.03241). Please cite the paper if you
use the code from this repository in your work.

### Bibtex

```
@article{lansing2019valan,
    title={VALAN: Vision and Language Agent Navigation},
    author={Larry Lansing and Vihan Jain and Harsh Mehta and Haoshuo Huang and Eugene Ie},
    year={2019},
    eprint={1912.03241},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

## Prerequisites

Before getting started, we need two main packages to run VALAN:
docker and git.

+ Docker
    1. Install docker by following the links below. Note that
      GPU support is only available for docker version 19.03 and later
      (after Aug 2019).
        + For Ubuntu: https://docs.docker.com/engine/install/ubuntu
        + For Debian: https://docs.docker.com/engine/install/debian
    2. Make sure docker works as non-root user (i.e., without `sudo`) by
      following [this instruction](https://docs.docker.com/install/linux/linux-postinstall)
      and section "**Manage Docker as a non-root user**" inside.
        + Now the following command should work without `sudo`:

            ```
            docker run hello-world
            ```

+ Git
    1. Install git use the following:

        ```
        apt-get update && apt-get install git
        ```

    2. Clone the VALAN git repository to your directory:

        ```
        git clone https://github.com/google-research/valan.git
        cd valan
        ```

+ Optional: To launch distributed training on the GCP AI-Platform you also need
  to sign up for a GCP account and install the Cloud SDK.
    - Follow https://cloud.google.com/ai-platform


## Usage

### Running on local machine

+ Quick Start

    For a quick start, run the following toy example that works out of the box.
    It  launches a training job inside a docker container, and uses Tmux to
    manage the `learner` with 3 `train actors`, 3 `eval actors`, and an
    `eval aggregator`, each of which runs separately in an asyncronous manner.
    The toy dataset contains only 3 R2R scans (houses) and
    a dozen paths, and will be copied to `/tmp/valan/testdata/`.
    Make sure you are inside directory `valan/` before start. Then run the
    following:

    ```
    sh launch_locally_with_docker.sh R2R R2R_3scans R2R_3scans 3 3
    ```

    To stop an individual worker, type CTRL+C in the worker's Tmux window. To
    stop training, switch to the learner's window by typing CTRL+b then 0, then
    type CTRL+C to kill the learner. To terminate and quit
    the docker container, type CTRL+b then d, which will detach the container,
    kill all tasks inside, and remove the container.


+ Custom Training

    You can run the job with your own R2R dataset as follows:

    ```
    sh launch_locally_with_docker.sh PROBLEM TRAIN_DATA EVAL_DATA NUM_ACTORS NUM_EVAL_ACTORS
    ```

    Note that the full R2R dataset is quite large and has 56 scans with 14k
    paths. Although the training job can run with only a few actors on a single
    machine, it can be very slow to do so. Thus it is recommended to run the
    full R2R dataset or other data of similar size on a distributed platform,
    e.g., GCP.


### Running on distributed environment (e.g., GCP)

We provide the tool and a concrete example to run VALAN on GCP with distributed learning.
Note that training the AI Platform requires signing up for a GCP account and will will
incurr charges for using the compute resources. See: https://cloud.google.com/ai-platform

+ Set up GCP SDK and project

    - To get started, install the GCP Cloud SDK following this instruction
      https://cloud.google.com/sdk/docs/quickstart and set up your GCP project.
    - Enable billing for your project.
    - Set up your Cloud Storage bucket for you AI Platform account following this
      instruction: https://cloud.google.com/ai-platform/training/docs/working-with-cloud-storage.
      Note that all of your input and training data will be stored in Cloud buckets.
    - Authenticate your `gcloud` account and set your default project:

      ```
      gcloud auth login
      gcloud config set project [YOUR_PROJECT]
      ```

+ Optional: prepare your Matterport 3D data and image features.

    TODO

+ Launch training job on GCP

    Each full training job consists of a learner (the main training node), a group
    of train actors that process and enqueue data for the learner concurrently, as
    well as several sets of eval actors for the evaluation job, all of which run
    asynchronously with the learner.  The evaluation job can have multiple data
    sources, for instance, "train", "val_seen", and "val_unseen" for the training
    set, the "val_seen" set, and "val_unseen" set respectively. Each of them has its
    own group of actors plus one eval aggregator that dequeues and aggregates
    evaluated examples from all eval actors within this group.

    The learner always runs on the "master" node with an accelerator. The train
    actors run on the "worker" nodes and use multi-threading to concurrently load
    and process the input data. All eval actors and eval aggregators run on the
    "evaluator" nodes. The learner runs on accelerator and all actors run on
    CPUs.

    - To train a VLN agent using a toy dataset (containing only 3 scans), run the
      following:

      ```
      bash gcp/r2r_train_3scans.sh
      ```
      This script will first make a copy of `./r2r/testdata`  to a GCS bucket
      `gs://valan/testdata`, then launch a training example with 1 GPU, 12
      training actors (3 workers each runs 4 threads), 4 eval actors (including
      1 eval aggregator) and 4 test actors (including 1 test aggregator).

      To monitor the training status using TensorBoard:
      https://cloud.google.com/ai-platform/docs/getting-started-tensorflow-estimator#tensorboard-local

      ```
      tensorboard --logdir=gs://valan/$JOB_DIR
      ```

      To monitor the job details and logs: http://console.cloud.google.com/ai-platform

      More details on job monitoring can be found here:
      https://cloud.google.com/ai-platform/training/docs/monitor-training#console


    - To train a VLN agent using the full set of R2R data, make sure you data is
      copied to your GCS bucket, then change the dir paths and configurations
      defined in the following script accordingly. Then:

      ```
      bash gcp/r2r_train.sh
      ```

      Similarly, the training status can be monitored using TensorBoard and the
      AI-platform console as aforementioned.


## Disclaimer

This is not an official Google product.
