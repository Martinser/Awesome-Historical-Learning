<div align="center">
<h1>A Survey of Historical Learning: Learning Models with Learning History</h1>

[Ge Wu](https://github.com/Martinser), [Lingfeng Yang](https://scholar.google.com/citations?user=RLhH0jwAAAAJ&hl=zh-CN), [Wenhai Wang](https://scholar.google.com/citations?user=WM0OglcAAAAJ&hl=zh-CN), [Renjie Song](https://scholar.google.com.hk/citations?user=-EgH8oIAAAAJ&hl=zh-CN&oi=sra), [Jian Yang](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN), [Xiang Li∗](https://scholar.google.com/citations?user=oamjJdYAAAAJ&hl=zh-CN)

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
* **[CVPR2019 CCN]** Mutual learning of complementary networks via residual correction for improving semi-supervised classification [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Mutual_Learning_of_Complementary_Networks_via_Residual_Correction_for_Improving_CVPR_2019_paper.pdf)] 
* **[Arxiv2019 SELF]** Self: Learning to filter noisy labels with self-ensembling [[Paper](https://arxiv.org/pdf/1910.01842)]
* **[AAAI2020 D2CNN]** Deep discriminative CNN with temporal ensembling for ambiguously-labeled image classification [[Paper](https://gcatnjust.github.io/ChenGong/paper/yao_aaai20.pdf)]
* **[ICCV2021 PS-KD]** Self-knowledge distillation with progressive refinement of targets [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Kim_Self-Knowledge_Distillation_With_Progressive_Refinement_of_Targets_ICCV_2021_paper.pdf)]
* **[CVPR2022 DLB]** Self-Distillation from the Last Mini-Batch for Consistency Regularization [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Shen_Self-Distillation_From_the_Last_Mini-Batch_for_Consistency_Regularization_CVPR_2022_paper.pdf)]
* **[NeurIPS2022 RecursiveMix]** RecursiveMix: Mixed Learning with History [[Paper](https://arxiv.org/pdf/2203.06844.pdf)]


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
* **[MICCAI2017]** Automatic Segmentation and Disease Classification Using Cardiac Cine MR Images [[Paper](https://arxiv.org/pdf/1708.01141.pdf)]
* **[TSG2018]** Short-term load forecasting with deep residual networks [[Paper](https://arxiv.org/pdf/1805.11956.pdf)]
* **[ECCV2018]** Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Uncertainty_Estimates_and_ECCV_2018_paper.pdf)]
* **[CVPRWs2019 DIDN]** Deep iterative down-up cnn for image denoising [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)]
* **[PR2020]** Multi-model ensemble with rich spatial information for object detection [[Paper](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/2E/E1/BD/2EE1BDAF6F0A0CC67ECE5E3A35376440.pdf)]
* **[CVPR2020 Self]** On the uncertainty of self-supervised monocular depth estimation [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf)]
* **[NeurIPS2020 PAS]** On the Loss Landscape of Adversarial Training: Identifying Challenges and How to Overcome Them [[Paper](https://proceedings.neurips.cc/paper/2020/file/f56d8183992b6c54c92c16a8519a6e2b-Paper.pdf)]
### Building unitary ensemble architecture in inference
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]
* **[CVPR2022 DLB]**  [[Paper]()]


## Gradient
### The gradients of the model parameters
  * **[JMLR2011 Adagrad]** Adaptive subgradient methods for online learning and stochastic optimization [[Paper](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)]
  * **[ICML2013 SGD]** On the importance of initialization and momentum in deep learning [[Paper](http://proceedings.mlr.press/v28/sutskever13.pdf)]
  * **[ICLR2015 Adam]** Adam: A method for stochastic optimization [[Paper](https://arxiv.org/pdf/1412.6980.pdf%5D)]
  * **[ICLR2016 Nadam]** Incorporating nesterov momentum into adam [[Paper](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ)]
  * **[ICCV2017  AdaMod]** An Adaptive and Momental Bound Method for Stochastic Learning [[Paper](https://arxiv.org/pdf/1910.12249.pdf)]
  * **[ICLR2019 AdamW]** Decoupled weight decay regularization [[Paper](https://arxiv.org/pdf/1711.05101.pdf)]
  * **[IJCAI2020 Padam]** Closing the generalization gap of adaptive gradient methods in training deep neural networks [[Paper](https://arxiv.org/pdf/1806.06763.pdf)]
  * **[ICLR2020 Radam]** On the variance of the adaptive learning rate and beyond [[Paper](https://arxiv.org/pdf/1908.03265.pdf)]
  * **[ICLR2021 Adamp]** Adamp: Slowing down the slowdown for momentum optimizers on scale-invariant weights [[Paper](https://arxiv.org/pdf/2006.08217.pdf)]
  * **[NeurIPS2022 Adan]** Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models [[Paper](https://arxiv.org/pdf/2208.06677.pdf)]
### The gradients of the all-level features
  * **[ECCV2020 GradCon]** Backpropgated gradient representations for anomaly detection [[Paper](https://arxiv.org/pdf/2007.09507.pdf)]
  * **[CVPR2021 Eqlv2]** Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Tan_Equalization_Loss_v2_A_New_Gradient_Balance_Approach_for_Long-Tailed_CVPR_2021_paper.pdf)]
  * **[Arxiv2022 EFL]** Equalized Focal Loss for Dense Long-Tailed Object Detection [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Equalized_Focal_Loss_for_Dense_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf)]


## Loss Values
* **[CVPR2019 O2u-net]** O2u-net: A simple noisy label detection approach for deep neural networks [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)]
* **[ECCV2020 DLB]** Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [[Paper](https://www.ri.cmu.edu/app/uploads/2020/05/WenHsuan_MSR_Thesis-1.pdf)]
* **[ICCV2021 iLPC]** Iterative label cleaning for transductive and semi-supervised few-shot learning [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lazarou_Iterative_Label_Cleaning_for_Transductive_and_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)]
* **[ECML PKDD2021 DLB]** Small-vote sample selection for label-noise learning [[Paper](https://discovery.ucl.ac.uk/id/eprint/10135634/1/YouzeXu-ECMLPKDD2021-accepted.pdf)]
* **[AAAI2022 DLB]** Delving into sample loss curve to embrace noisy and imbalanced data [[Paper](https://arxiv.org/pdf/2201.00849.pdf)]
* **[Arxiv2022 CNLCU]** Ctrl: Clustering training losses for label error detection [[Paper](https://arxiv.org/pdf/2208.08464.pdf)]


---
## Citation
If you find this repository useful, please consider citing this list:
```

```

---
## Acknowledgement
The project are based on [Ultimate-Awesome-Transformer-Attention](https://github.com/cmhungsteve/Awesome-Transformer-Attention) and [UM-MAE](https://github.com/implus/UM-MAE). Thanks for their wonderful work.


## License
This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

