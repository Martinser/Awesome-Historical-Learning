<div align="center">
<h1>A Survey of Historical Learning: Learning Models with Learning History</h1>

[Ge Wu](https://github.com/Martinser), [Lingfeng Yang](https://scholar.google.com/citations?user=RLhH0jwAAAAJ&hl=zh-CN), [Borui Zhao](https://scholar.google.com.hk/citations?user=DzRfzYwAAAAJ&hl=zh-CN&oi=sra), [Renjie Song](https://scholar.google.com.hk/citations?user=-EgH8oIAAAAJ&hl=zh-CN&oi=sra), [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN), [Xiang Li∗](https://scholar.google.com/citations?user=oamjJdYAAAAJ&hl=zh-CN)

∗ Corresponding author
</div>

This repo is the official paper list of [A Survey of Historical Learning: Learning Models with Learning History]().
It mainly includes papers related to historical learning, which are classified according to different aspects of historical types.

If you have any questions about the contents of the [paper]() or [list](), please contact us through opening issues or [email](gewu.nku@gmail.com).

## Update
* **[February, 2023]** The repo 'Awesome-Historical-Learning' is public at github.

## Overview

* [Prediction](#prediction)
* [Intermediate Feature Representation](#intermediate-feature-representation)
    * [Record the instance-level feature representations](#record-the-instance-level-feature-representations)
    * [Memorize the feature statistics](#memorize-the-feature-statistics)
* [Model Parameter](#model-parameter)
    * [Constructing the teachers from past models](#constructing-the-teachers-from-past-models)
    * [Directly exploiting ensemble results in inference](#directly-exploiting-ensemble-results-in-inference)
    * [Building unitary ensemble architecture in inference](#building-unitary-ensemble-architecture-in-inference)
* [Gradient](#gradient)
    * [The gradients of the model parameters](#the-gradients-of-the-model-parameters)
    * [The gradients of the all-level features](#the-gradients-of-the-all-level-features)
* [Loss Values](#loss-values)

---
## Prediction
* **[CVPR2019 CCN]** Mutual learning of complementary networks via residual correction for improving semi-supervised classification [[Paper](https://arxiv.org/abs/2012.12556)] 
* **[Arxiv2019 SELF]** Self: Learning to filter noisy labels with self-ensembling [[Paper](https://arxiv.org/abs/2201.08683)]
* **[AAAI2020 D2CNN]** Deep discriminative CNN with temporal ensembling for ambiguously-labeled image classification [[Paper](https://arxiv.org/abs/2204.07356)]
* **[ICCV2021 PS-KD]** Self-knowledge distillation with progressive refinement of targets [[Paper](https://arxiv.org/abs/2204.07356)]
* **[CVPR2022 DLB]** Self-Distillation from the Last Mini-Batch for Consistency Regularization [[Paper](https://arxiv.org/abs/2204.07356)]


## Intermediate Feature Representation
### Record the instance-level feature representations
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
### Memorize the feature statistics
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]


## Model Parameter
### Constructing the teachers from past models
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
### Directly exploiting ensemble results in inference
* **[MICCAI2017]** Automatic Segmentation and Disease Classification Using Cardiac Cine MR Images [[Paper]()]
* **[TSG2018]** Short-term load forecasting with deep residual networks [[Paper]()]
* **[ECCV2018]** Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[Paper]()]
* **[CVPRWs2019 DIDN]** Deep iterative down-up cnn for image denoising [[Paper]()]
* **[PR2020]** Multi-model ensemble with rich spatial information for object detection [[Paper]()]
* **[CVPR2020 Self]** On the uncertainty of self-supervised monocular depth estimation [[Paper]()]
* **[NeurIPS2020 PAS]** On the Loss Landscape of Adversarial Training: Identifying Challenges and How to Overcome Them [[Paper]()]
### Building unitary ensemble architecture in inference
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]


## Gradient
### The gradients of the model parameters
  * **[JMLR2011 Adagrad]** Adaptive subgradient methods for online learning and stochastic optimization [[Paper]()]
  * **[ICML2013 SGD]** On the importance of initialization and momentum in deep learning [[Paper]()]
  * **[ICLR2015 Adam]** Adam: A method for stochastic optimization [[Paper]()]
  * **[ICLR2016 Nadam]** Incorporating nesterov momentum into adam [[Paper]()]
  * **[ICCV2017  AdaMod]** An Adaptive and Momental Bound Method for Stochastic Learning [[Paper]()]
  * **[ICLR2019 AdamW]** Decoupled weight decay regularization [[Paper]()]
  * **[IJCAI2020 Padam]** Closing the generalization gap of adaptive gradient methods in training deep neural networks [[Paper]()]
  * **[ICLR2020 Radam]** On the variance of the adaptive learning rate and beyond [[Paper]()]
  * **[ICLR2021 Adamp]** Adamp: Slowing down the slowdown for momentum optimizers on scale-invariant weights [[Paper]()]
  * **[NeurIPS2022 Adan]** Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models [[Paper]()]
### The gradients of the all-level features
  * **[ECCV2020 DLB]** Backpropgated gradient representations for anomaly detection [[Paper]()]
  * **[CVPR2021 Eqlv2]** Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection [[Paper]()]
  * **[Arxiv2022 DLB]** Equalized Focal Loss for Dense Long-Tailed Object Detection [[Paper]()]


## Loss Values
* **[CVPR2019 O2u-net]** O2u-net: A simple noisy label detection approach for deep neural networks [[Paper]()]
* **[ECCV2020 DLB]** Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [[Paper]()]
* **[ICCV2021 iLPC]** Iterative label cleaning for transductive and semi-supervised few-shot learning [[Paper]()]
* **[ECML PKDD2021 DLB]** Small-vote sample selection for label-noise learning [[Paper]()]
* **[AAAI2022 DLB]** Delving into sample loss curve to embrace noisy and imbalanced data [[Paper]()]
* **[Arxiv2022 CNLCU]** Ctrl: Clustering training losses for label error detection [[Paper]()]


---
## Citation
If you find this repository useful, please consider citing this list:
```

```

---
## Acknowledgement
The our project are based on [Ultimate-Awesome-Transformer-Attention](https://github.com/cmhungsteve/Awesome-Transformer-Attention) and [UM-MAE](https://github.com/implus/UM-MAE). Thanks for their wonderful work.


## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

