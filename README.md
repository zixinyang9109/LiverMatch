# LiverMatch - Learning Feature Descriptors for Pre- and Intra-operative Point Cloud Matching for Laparoscopic Liver Registration

### Introduction
In this project, we show promising results of using learning-based descriptors for laparoscopic liver registration (LLS).

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


```
python train.py configs/liver.yaml
```


```
python eval.py
```

```
python demos/PBSM-inSilicoData_demo.py
```

### Dataset

The simulated dataset uses the 3D-IRCADb-01 dataset under the license CC BY-NC-ND 4.0. In this license, we should follow "NoDerivatives".

We will release a larger dataset using Medical Segmentation Decathlon (http://medicaldecathlon.com) under the license CC BY-SA 4.0, which allows modifications.

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
- [V2S-Net] (https://gitlab.com/nct_tso_public/Volume2SurfaceCNN)
- Liver registration dataset https://opencas.webarchiv.kit.edu/?q=PhysicsBasedShapeMatching
- Liver segmentation 3D-IRCADb-01 https://www.ircad.fr/research/data-sets/liver-segmentation-3d-ircadb-01/
