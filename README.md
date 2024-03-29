# PTree-Net

This repository is an official PyTorch implementation of the paper **"Diagnose Like A Pathologist: Weakly-Supervised Pathologist-Tree Network for Slide-Level Immunohistochemical Scoring"** [[paper](https://www.researchgate.net/publication/350089153_Diagnose_Like_A_Pathologist_Weakly-Supervised_Pathologist-Tree_Network_for_Slide-Level_Immunohistochemical_Scoring)] from **AAAI 2021**.

<div align=center><img width="700" src=/figs/framework.png></div>

## Dependencies
* Python 3.6
* PyTorch >= 1.5.0
* torch-geometric
* numpy
* sklearn
* openslide

## Quickstart 
* Train the PTree-Net with your HER2 WSI dataset:
```python
python ./train.py 
```

## Cite
If you find our work useful in your research or publication, please cite our work:
```
@inproceedings{chen2021diagnose,
  title={Diagnose Like A Pathologist: Weakly-Supervised Pathologist-Tree Network for Slide-Level Immunohistochemical Scoring},
  author={Chen, Zhen and Zhang, Jun and Che, Shuanlong and Huang, Junzhou and Han, Xiao and Yuan, Yixuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## Acknowledgements
* GCN implementation with [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric).
