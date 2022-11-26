# Few-Shot Audio-Visual Learning of Environment Acoustics
This repository contains the PyTorch implementation of our **Neurips 2022 paper** and the associated datasets: 

[Few-Shot Audio-Visual Learning of Environment Acoustics](https://vision.cs.utexas.edu/projects/fs_rir/)<br />
Sagnik Majumder, Changan Chen*, Ziad Al-Halah*, Kristen Grauman<br />
The University of Texas at Austin, Facebook AI Research  
\*Equal contribution

Project website: [https://vision.cs.utexas.edu/projects/fs_rir/](https://vision.cs.utexas.edu/projects/fs_rir/)

## Abstract
Room impulse response (RIR) functions capture how the surrounding physical environment transforms the sounds heard by a listener, with implications for various applications in AR, VR, and robotics. Whereas traditional methods to estimate RIRs assume dense geometry and/or sound measurements throughout the environment, we explore how to infer RIRs based on a sparse set of images and echoes observed in the space. Towards that goal, we introduce a transformer-based method that uses self-attention to build a rich acoustic context, then predicts RIRs of arbitrary query source-receiver locations through cross-attention. Additionally, we design a novel training objective that improves the match in the acoustic signature between the RIR predictions and the targets. In experiments using a state-of-the-art audio-visual simulator for 3D environments, we demonstrate that our method successfully generates arbitrary RIRs, outperforming state-of-the-art methods and---in a major departure from traditional methods---generalizing to novel environments in a few-shot manner.

## Dependencies
This code has been tested with ```python 3.6.13```, ```habitat-api 0.1.4```, ```habitat-sim 0.1.4``` and ```torch 1.4.0```. Additional python package requirements are available in ```requirements.txt```.   
  
First, install the required versions of [habitat-api](https://github.com/facebookresearch/habitat-lab), [habitat-sim](https://github.com/facebookresearch/habitat-sim) and [torch](https://pytorch.org/) inside a [conda](https://www.anaconda.com/) environment. 

Next, install the remaining dependencies either by 
```
pip3 install -r requirements.txt
``` 
or by parsing ```requirements.txt``` to get the names and versions of individual dependencies and install them individually.



## Citation
```
@inproceedings{
majumder2022fewshot,
title={Few-Shot Audio-Visual Learning of Environment Acoustics},
author={Sagnik Majumder and Changan Chen and Ziad Al-Halah and Kristen Grauman},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=PIXGY1WgU-S}
}
```

# License
This project is released under the MIT license, as found in the LICENSE file.
