# TFNet
---
### Notice

- This project is implemented in Pytorch (1.11.0+cu113). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- This project runs in pycharm, so you need to install pycharm

- The SLR is the main function.

---
### Data Preparation

1. Download the CE-CSL Dataset [[download link]]https://pan.baidu.com/s/1qCzGHqHd1UzgVMumdZ9OHA
   extraction code：0000
   
2. After finishing dataset download, extract it.

---
### Data Process
1. Run python CE-CSLDataPreProcess.py to convert video data to image data.

2. SRL.py is the main function

3. Remember to change the path to the dataset before running it.

---
### Inference
|            | WER on Dev | WER on Test |
|  --------  | ---------- | ----------- |
|   TFNet    | 42.1       | 41.9%       |

---
```
---
### Relevant paper

Continuous Sign Language Recognition via Temporal Super-Resolution Network. [[paper]](https://arxiv.org/pdf/2207.00928.pdf)

```latex
@article{zhu2022continuous,
  title={Continuous Sign Language Recognition via Temporal Super-Resolution Network},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2207.00928},
  year={2022}
}
```

Temporal superimposed crossover module for effective continuous sign language. [[paper]](https://arxiv.org/pdf/2211.03387.pdf)
```latex
@article{zhu2022temporal,
  title={Temporal superimposed crossover module for effective continuous sign language},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2211.03387},
  year={2022}
}
```

Continuous sign language recognition based on cross-resolution knowledge distillation. [[paper]](https://arxiv.org/pdf/2303.06820.pdf)
```latex
@article{zhu2023continuous,
  title={Continuous sign language recognition based on cross-resolution knowledge distillation},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2303.06820},
  year={2023}
}
```

Continuous Sign Language Recognition Based on Motor attention mechanism and frame-level Self-distillation. [[paper]](https://arxiv.org/pdf/2402.19118.pdf)
```latex
@article{zhu2024continuous,
  title={Continuous Sign Language Recognition Based on Motor attention mechanism and frame-level Self-distillation},
  author={Zhu, Qidan and Li, Jing and Yuan, Fei and Gan, Quan},
  journal={arXiv preprint arXiv:2402.19118},
  year={2024}
}
```
