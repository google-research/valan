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

TODO

## Usage

### Running on local machine

TODO

### Running on distributed environment (e.g., GCP)

TODO

## Disclaimer

This is not an official Google product.
