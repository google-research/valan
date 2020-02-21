# Touchdown

This and the related [streeview_common subpackage](https://github.com/google-research/valan/tree/master/streetview_common) together contain code to pre-process and train the TensorFlow baseline model for Vision-and-Language Navigation (VLN) and Spatial Description Resolution (SDR) tasks defined in the [Touchdown dataset](https://github.com/lil-lab/touchdown).

Details for the original release of the dataset can be found in the accompanying paper: [Touchdown: Natural Language Navigation and Spatial Reasoning in Visual Street Environments, 2019](http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_TOUCHDOWN_Natural_Language_Navigation_and_Spatial_Reasoning_in_Visual_Street_CVPR_2019_paper.pdf).

The StreetView panorama images needed to support the Touchdown tasks are now available in the Streetlearn dataset. Details about the process and results based on these panoramas and the implementations provided in [Retouchdown: Adding Touchdown to StreetLearn as a Shareable Resource for Language Grounding Tasks in Street View, 2020](https://arxiv.org/abs/2001.03671).

## Data

1. Panoramas: please follow the instructions mentioned at the [Streetlearn](https://sites.google.com/corp/view/streetlearn/touchdown) project website to obtain the Street View panoramas.

2. Instructions: for both VLN and SDR tasks can be obtained from the Touchdown [github repository](https://github.com/lil-lab/touchdown).

## Preprocessing

*Stay tuned!*

## Training the model

*Stay tuned!*

## Reference

If you use or discuss this dataset in your work, please cite the following papers correspondingly:

```
@inproceedings{chen2019touchdown,
  title={Touchdown: Natural language navigation and spatial reasoning in visual street environments},
  author={Chen, Howard and Suhr, Alane and Misra, Dipendra and Snavely, Noah and Artzi, Yoav},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12538--12547},
  year={2019}
}

@article{mehta2020retouchdown,
  title={Retouchdown: Adding Touchdown to StreetLearn as a Shareable Resource for Language Grounding Tasks in Street View},
  author={Harsh Mehta and Yoav Artzi and Jason Baldridge and Eugene Ie and Piotr Mirowski},
  year={2020},
  eprint={2001.03671},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Contact

If you have a technical question regarding the dataset, codebase or publication, please create an issue in this repository.

