# LiverMatch - Learning Feature Descriptors for Pre- and Intra-operative Point Cloud Matching for Laparoscopic Liver Registration

### Introduction

This work presents, for the first time, compelling evidence that learning-based descriptors and correspondences can effectively support initial rigid registration in laparoscopic liver surgery (LLS), paving the way for their practical application in real-world surgical settings.

### Dataset

The original simulated dataset uses the 3D-IRCADb-01 dataset under the license CC BY-NC-ND 4.0. In this license, we should follow the "NoDerivatives" policy.

In our following work, we released a larger dataset in [P2P](https://github.com/zixinyang9109/P2P) under the license CC BY-SA 4.0, which allows modifications. Also, datasets included in [BCF-FEM](https://github.com/zixinyang9109/BCF-FEM) could be helpful.


### Install
```
conda create -n match python==3.8
conda activate match
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pytorch
pip install PyYAML
conda install scipy
pip install easydict
pip install tensorboardX
pip install tqdm
pip install -U scikit-learn
pip install mayavi
pip install PyQt5
pip install open3d
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
### Run

Please change the paths in the following files and run:

```
python train.py configs/liver.yaml
```

```
python eval.py
```

```
python demos/PBSM-inSilicoData_demo.py # The weight is included in the snapshot. 
```




### Important note to train and test on other datasets

In our following works, we found that to get correspondence-based methods using KPConv to work correctly on other datasets, attentions are needed to:

1. Normalize and voxelize the point clouds properly.
2. Adjust the hyperparameter "init_sampling" required for KPConv to allow downsampling for different scales.
3. If RANSAC ICP implemented in open3D is used to estimate the transformation, the hyperparameter "max_correspondence_distance" should be appropriately set. Otherwise, SVD is suggested.

This is an exploratory work, and the method does have limitations. However, this simple method should work reasonably well if the above attention has been paid, even compared to more sophisticated techniques, as shown in [P2P](https://github.com/zixinyang9109/P2P) 

Please feel free to send an email to yy8898@rit.edu for questions.

## Citation

```bibtex
@article{yang2023learning,
  title={Learning feature descriptors for pre-and intra-operative point cloud matching for laparoscopic liver registration},
  author={Yang, Zixin and Simon, Richard and Linte, Cristian A},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  pages={1--8},
  year={2023},
  publisher={Springer}
}
```



## Acknowledgements

- [Lepard](https://github.com/rabbityl/lepard) 
- [PREDATOR](https://github.com/prs-eth/OverlapPredator)
- [V2S-Net](https://gitlab.com/nct_tso_public/Volume2SurfaceCNN) Deformation simulation.
- Liver segmentation 3D-IRCADb-01 https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/
