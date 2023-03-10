<div align="center">
<h1>A Survey of Historical Learning: Learning Models with Learning History</h1>

[Xiang Li*](https://scholar.google.com/citations?user=oamjJdYAAAAJ&hl=zh-CN), [Ge Wu*](https://github.com/Martinser), [Lingfeng Yang](https://scholar.google.com/citations?user=RLhH0jwAAAAJ&hl=zh-CN), [Wenhai Wang](https://scholar.google.com/citations?user=WM0OglcAAAAJ&hl=zh-CN), [Renjie Song](https://scholar.google.com.hk/citations?user=-EgH8oIAAAAJ&hl=zh-CN&oi=sra), [Jian Yang#](https://scholar.google.com/citations?user=6CIDtZQAAAAJ&hl=zh-CN)

  $*$ Equal contribution. # Corresponding author.
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
* **[AAAI2022 MBJ]** Memory-Based Jitter: Improving Visual Recognition on Long Tailed Data with Diversity in Memory [[Paper](https://arxiv.org/pdf/2008.09809.pdf)]
* **[AAAI2022 MeCoQ]** Contrastive quantization with code memory for unsupervised image retrieval [[Paper](https://arxiv.org/pdf/2109.05205.pdf)]
* **[AAAI2022 InsCLR]** InsCLR: Improving instance retrieval with self-supervision [[Paper](https://arxiv.org/pdf/2112.01390.pdf)]
* **[CVPR2022 MAUM]** Learning Memory-Augmented Unidirectional Metrics for Cross-Modality Person Re-Identification [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Learning_Memory-Augmented_Unidirectional_Metrics_for_Cross-Modality_Person_Re-Identification_CVPR_2022_paper.pdf)]
* **[CVPR2022 QB-Norm]** Cross Modal Retrieval with Querybank Normalisation [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.pdf)]
* **[CVPR2022 CIRKD]** Cross-Image Relational Knowledge Distillation for Semantic Segmentation [[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Cross-Image_Relational_Knowledge_Distillation_for_Semantic_Segmentation_CVPR_2022_paper.pdf)]
* **[ECCV2022 DAS]** DAS: Densely-Anchored Sampling for Deep Metric Learning [[Paper](https://arxiv.org/pdf/2208.00119.pdf)]
* **[Arxiv2022 Memorizing transformers]** Memorizing transformers [[Paper](https://arxiv.org/pdf/2203.08913.pdf)]
* **[Arxiv2022 ]** Learning Equivariant Segmentation with Instance-Unique Querying [[Paper](https://arxiv.org/pdf/2210.00911.pdf)]
* **[Arxiv2022 MCL]** Online Knowledge Distillation via Mutual Contrastive Learning for Visual Recognition [[Paper](https://arxiv.org/pdf/2207.11518.pdf)]
* **[AAAI2021 IM-CFB]** Instance mining with class feature banks for weakly supervised object detection [[Paper](https://ojs.aaai.org/index.php/AAAI/article/view/16429/16236)]
* **[CVPR2021 LOCE]** Exploring classification equilibrium in long-tailed object detection [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Exploring_Classification_Equilibrium_in_Long-Tailed_Object_Detection_ICCV_2021_paper.pdf)]
* **[CVPR2021 VPL]** Variational prototype learning for deep face recognition [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Deng_Variational_Prototype_Learning_for_Deep_Face_Recognition_CVPR_2021_paper.pdf)]
* **[CVPR2021 GLT]** Group-aware label transfer for domain adaptive person re-identification [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Group-aware_Label_Transfer_for_Domain_Adaptive_Person_Re-identification_CVPR_2021_paper.pdf)]
* **[CVPR2021 MCIBI]** Mining contextual information beyond image for semantic segmentation [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_Mining_Contextual_Information_Beyond_Image_for_Semantic_Segmentation_ICCV_2021_paper.pdf)]
* **[CVPR2021 SAN]** Spatial assembly networks for image representation learning [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Spatial_Assembly_Networks_for_Image_Representation_Learning_CVPR_2021_paper.pdf)]
* **[CVPR2021 PRISM]** Noise-resistant deep metric learning with ranking-based instance selection [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Noise-Resistant_Deep_Metric_Learning_With_Ranking-Based_Instance_Selection_CVPR_2021_paper.pdf)]
* **[ICCV2021 LOCE]** Exploring classification equilibrium in long-tailed object detection [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Exploring_Classification_Equilibrium_in_Long-Tailed_Object_Detection_ICCV_2021_paper.pdf)]
* **[NeurIPS2021 CBA-MR]** When False Positive is Intolerant: End-to-End Optimization with Low FPR for Multipartite Ranking [[Paper]()]
* **[TIP2021 Dual-Refinement]** Dual-refinement: Joint label and feature refinement for unsupervised domain adaptive person re-identification [[Paper]()]
* **[CVPR2020 ]** Cross-Batch Memory for Embedding Learning [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.pdf)]
* **[CVPR2020 PIRL]** Self-supervised learning of pretext-invariant representations [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.pdf)]
* **[ECCV2020 TAC-CCL]** Unsupervised Deep Metric Learning with Transformed Attention Consistency and Contrastive Clustering Loss [[Paper](https://arxiv.org/pdf/2008.04378.pdf)]
* **[ECCV2020 CMC]** Contrastive multiview coding [[Paper](https://arxiv.org/pdf/1906.05849.pdf)]
* **[CVPR2019 ECN]** Invariance Matters: Exemplar Memory for Domain AdaptivePerson Re-identification [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Invariance_Matters_Exemplar_Memory_for_Domain_Adaptive_Person_Re-Identification_CVPR_2019_paper.pdf)]
* **[CVPR2018 NCE]** Unsupervised Feature Learning via Non-Parametric Instance Discrimination [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_Unsupervised_Feature_Learning_CVPR_2018_paper.pdf)]
* **[ECCV2018 NCA]** Improving Generalization via Scalable Neighborhood Component Analysis [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhirong_Wu_Improving_Embedding_Generalization_ECCV_2018_paper.pdf)]
* **[ICLR2018 MbPA]** Memory-based Parameter Adaptation [[Paper](https://arxiv.org/pdf/1802.10542.pdf)]
* **[CVPR2017 OIM]** Joint detection and identification feature learning for person search [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Xiao_Joint_Detection_and_CVPR_2017_paper.pdf)]
### Memorize the feature statistics
* **[ECCV2016 center loss]** A discriminative feature learning approach for deep face recognition [[Paper](https://kpzhang93.github.io/papers/eccv2016.pdf)]
* **[CVPR2018 TCL]** Triplet-Center Loss for Multi-View 3D Object Retrieval [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/He_Triplet-Center_Loss_for_CVPR_2018_paper.pdf)]
* **[AAAI2019 ATCL]** Angular triplet-center loss for multi-view 3d shape retrieval [[Paper](https://ojs.aaai.org/index.php/AAAI/article/download/4890/4763)]
* **[ICCV2019 3C-Net]** 3C-Net: Category Count and Center Loss for Weakly-Supervised Action Localization [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Narayan_3C-Net_Category_Count_and_Center_Loss_for_Weakly-Supervised_Action_Localization_ICCV_2019_paper.pdf)]
* **[ECCV2020 A2CL-PT]** Adversarial Background-Aware Loss for Weakly-supervised Temporal Activity Localization [[Paper](https://arxiv.org/pdf/2007.06643.pdf)]
* **[Transactions on Multimedia2020 HCTL]** Deep fusion feature representation learning with hard mining center-triplet loss for person re-identification [[Paper](https://iip.tongji.edu.cn/2020TMM_ZCR.pdf.pdf)]
* **[Neurocomputing2020 HC loss]** Hetero-center loss for cross-modality person re-identification [[Paper](https://arxiv.org/ftp/arxiv/papers/1910/1910.09830.pdf)]
* **[CVPR2020 OBTL]** Generalized zero-shot learning via over-complete distribution [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Keshari_Generalized_Zero-Shot_Learning_via_Over-Complete_Distribution_CVPR_2020_paper.pdf)]
* **[ICCV2021 SAMC-loss]** FREE: Feature Refinement for Generalized Zero-Shot Learning [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_FREE_Feature_Refinement_for_Generalized_Zero-Shot_Learning_ICCV_2021_paper.pdf)]
* **[CVPR2021 SCL]** Frequency-aware Discriminative Feature Learning Supervised by Single-Center Loss for Face Forgery Detection [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Frequency-Aware_Discriminative_Feature_Learning_Supervised_by_Single-Center_Loss_for_Face_CVPR_2021_paper.pdf)]
* **[CVPR2021 Cross-modal center Loss]** Cross-Modal Center Loss for 3D Cross-Modal Retrieval [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Jing_Cross-Modal_Center_Loss_for_3D_Cross-Modal_Retrieval_CVPR_2021_paper.pdf)]
* **[ICML2015 Batch Normalization]** Batch normalization: Accelerating deep network training by reducing internal covariate shift [[Paper](http://proceedings.mlr.press/v37/ioffe15.pdf)]
* **[NeurIPS2016 Weight Normalization]** Weight normalization: A simple reparameterization to accelerate training of deep neural networks [[Paper](https://proceedings.neurips.cc/paper/2016/file/ed265bc903a5a097f61d3ec064d96d2e-Paper.pdf)]
* **[ICML2016 Normalization Propagation]** Normalization prop agation: A parametric technique for removing internal covariate shift in deep networks [[Paper](http://proceedings.mlr.press/v48/arpitb16.pdf)]
* **[NeurIPS2017 Batch Renormalization]** Batch renormalization: Towards reducing minibatch depen dence in batch-normalized models [[Paper](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf)]
* **[ICLR2017 AdaBN]** Revisiting batch normaliza tion for practical domain adaptation [[Paper](https://arxiv.org/pdf/1603.04779.pdf)]


## Model Parameter
### Constructing the teachers from past models
* **[NeurIPS2020 BYOL]** Bootstrap your own latent: A new approach to self-supervised learning [[Paper](https://proceedings.neurips.cc/paper/2020/file/f3ada80d5c4ee70142b17b8192b2958e-Paper.pdf)]
* **[CVPR2020 MoCo]** Momentum contrast for unsupervised visual representation learning [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)]
* **[Arxiv2020 MoCo v2]** Improved baselines with momentum contrastive learning [[Paper](https://arxiv.org/pdf/2003.04297.pdf)]
* **[ICCV2021 MoCo v3]** An empirical study of training self supervised visual transformers [[Paper]()]
* **[ICCV2021 TKC]** Temporal knowledge consistency for unsupervised visual representation learning [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Feng_Temporal_Knowledge_Consistency_for_Unsupervised_Visual_Representation_Learning_ICCV_2021_paper.pdf)]
* **[ICCV2021 DINO]** Emerging properties in self-supervised vision transformers [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)]
* **[ICML2018 BANs]** Born again neural networks [[Paper](http://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf)]
* **[CVPR2019 MLNT]** Learning to learn from noisy labeled data [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Learning_to_Learn_From_Noisy_Labeled_Data_CVPR_2019_paper.pdf)]
* **[CVPR2019 SD]** Snapshot distillation: Teacher student optimization in one generation [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yang_Snapshot_Distillation_Teacher-Student_Optimization_in_One_Generation_CVPR_2019_paper.pdf)]
* **[CVPR2020 Tf-KD]** Revisiting knowledge distillation via label smoothing regularization [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Revisiting_Knowledge_Distillation_via_Label_Smoothing_Regularization_CVPR_2020_paper.pdf)]
* **[NeurIPS2022 CheckpointKD]** Efficient knowledge distillation from model checkpoints [[Paper](https://arxiv.org/pdf/2210.06458.pdf)]
* **[Arxiv2022 EEKD]** Learn from the past: Experience ensemble knowledge distillation [[Paper](https://arxiv.org/pdf/2202.12488.pdf)]
* **[Arxiv2022 SEAT]** Self-ensemble adversarial training for improved robustness [[Paper](https://arxiv.org/pdf/2203.09678.pdf)]
* **[NeurIPS2017 Mean teacher]** Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results [[Paper](https://proceedings.neurips.cc/paper/2017/file/68053af2923e00204c3ca7c6a3150cf7-Paper.pdf)]
* **[ECCV2018 TSSDL]** Transductive semi-supervised deep learning using min-max features [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Weiwei_Shi_Transductive_Semi-Supervised_Deep_ECCV_2018_paper.pdf)]
* **[CVPR2021 EMAN]** Exponential moving average normalization for self-supervised and semi-supervised learning [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Cai_Exponential_Moving_Average_Normalization_for_Self-Supervised_and_Semi-Supervised_Learning_CVPR_2021_paper.pdf)]
* **[ECCV2022]** Unsupervised selective labeling for more effective semi-supervised learning [[Paper](https://arxiv.org/pdf/2110.03006.pdf)]
* **[Arxiv2022 Semi-ViT]** Semi-supervised vision transformers at scale [[Paper](https://arxiv.org/pdf/2208.05688.pdf)]
* **[Arxiv2023 AEMA]** Robust domain adaptive object detection with unified multi-granularity alignment [[Paper](https://arxiv.org/pdf/2301.00371.pdf)]
### Directly exploiting ensemble results in inference
* **[MICCAI2017]** Automatic Segmentation and Disease Classification Using Cardiac Cine MR Images [[Paper](https://arxiv.org/pdf/1708.01141.pdf)]
* **[TSG2018]** Short-term load forecasting with deep residual networks [[Paper](https://arxiv.org/pdf/1805.11956.pdf)]
* **[ECCV2018]** Uncertainty Estimates and Multi-Hypotheses Networks for Optical Flow [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Eddy_Ilg_Uncertainty_Estimates_and_ECCV_2018_paper.pdf)]
* **[CVPRWs2019 DIDN]** Deep iterative down-up cnn for image denoising [[Paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Yu_Deep_Iterative_Down-Up_CNN_for_Image_Denoising_CVPRW_2019_paper.pdf)]
* **[PR2020]** Multi-model ensemble with rich spatial information for object detection [[Paper](https://cz5waila03cyo0tux1owpyofgoryroob.oss-cn-beijing.aliyuncs.com/2E/E1/BD/2EE1BDAF6F0A0CC67ECE5E3A35376440.pdf)]
* **[CVPR2020 Self]** On the uncertainty of self-supervised monocular depth estimation [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Poggi_On_the_Uncertainty_of_Self-Supervised_Monocular_Depth_Estimation_CVPR_2020_paper.pdf)]
* **[NeurIPS2020 PAS]** On the Loss Landscape of Adversarial Training: Identifying Challenges and How to Overcome Them [[Paper](https://proceedings.neurips.cc/paper/2020/file/f56d8183992b6c54c92c16a8519a6e2b-Paper.pdf)]
### Building unitary ensemble architecture in inference
* **[UAI2018 SWA]** Averaging weights leads to wider optima and better generalization [[Paper](https://arxiv.org/pdf/1803.05407.pdf%20%20https://github.com/timgaripov/swa)]
* **[NeurIPS2018 FGE]** Loss surfaces, mode connectivity, and fast ensembling of dnns [[Paper](https://proceedings.neurips.cc/paper/2018/file/be3087e74e9100d4bc4c6268cdbe8456-Paper.pdf)]
* **[ICLR2019 fast-SWA]** There are many consistent explanations of unlabeled data: Why you should average [[Paper](https://arxiv.org/pdf/1806.05594.pdf)]
* **[NeurIPS2019 SWAG]** A simple baseline for bayesian uncertainty in deep learning [[Paper](https://proceedings.neurips.cc/paper/2019/file/118921efba23fc329e6560b27861f0c2-Paper.pdf)]
* **[ICML2019 SWALP]** SWALP: Stochastic Weight Averaging in Low-Precision Training [[Paper](http://proceedings.mlr.press/v97/yang19d/yang19d.pdf)]
* **[Arxiv2020 SWA Object Detection]** Swa object detection [[Paper](https://arxiv.org/pdf/2012.12645.pdf)]
* **[ICML2021 late-phase weights]** Neural networks with late-phase weights [[Paper](https://arxiv.org/pdf/2007.12927.pdf)]
* **[NeurIPS2021 SWAD]** SWAD: Domain generalization by seeking flat minima [[Paper](https://proceedings.neurips.cc/paper/2021/file/bcb41ccdc4363c6848a1d760f26c28a0-Paper.pdf)]
* **[Arxiv2022 PSWA]** Stochastic Weight Averaging Revisited [[Paper](https://arxiv.org/pdf/2201.00519.pdf)]


## Gradient
### The gradients of the model parameters
  * **[JMLR2011 Adagrad]** Adaptive subgradient methods for online learning and stochastic optimization [[Paper](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)]
  * **[ICML2013 SGD]** On the importance of initialization and momentum in deep learning [[Paper](http://proceedings.mlr.press/v28/sutskever13.pdf)]
  * **[ICLR2015 Adam]** Adam: A method for stochastic optimization [[Paper](https://arxiv.org/pdf/1412.6980.pdf%5D)]
  * **[ICLR2016 Nadam]** Incorporating nesterov momentum into adam [[Paper](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ)]
  * **[Arxiv2019 AdaMod]** An Adaptive and Momental Bound Method for Stochastic Learning [[Paper](https://arxiv.org/pdf/1910.12249.pdf)]
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
* **[ECCV2020 ]** Neural batch sampling with reinforcement learning for semi-supervised anomaly detection [[Paper](https://www.ri.cmu.edu/app/uploads/2020/05/WenHsuan_MSR_Thesis-1.pdf)]
* **[ICCV2021 iLPC]** Iterative label cleaning for transductive and semi-supervised few-shot learning [[Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Lazarou_Iterative_Label_Cleaning_for_Transductive_and_Semi-Supervised_Few-Shot_Learning_ICCV_2021_paper.pdf)]
* **[ECML PKDD2021 HVS]** Small-vote sample selection for label-noise learning [[Paper](https://discovery.ucl.ac.uk/id/eprint/10135634/1/YouzeXu-ECMLPKDD2021-accepted.pdf)]
* **[Arxiv2021 CNLCU]** Sample selection with uncertainty of losses for learning with noisy labels [[Paper](https://discovery.ucl.ac.uk/id/eprint/10135634/1/YouzeXu-ECMLPKDD2021-accepted.pdf)]
* **[AAAI2022 ]** Delving into sample loss curve to embrace noisy and imbalanced data [[Paper](https://arxiv.org/pdf/2201.00849.pdf)]
* **[Arxiv2022 CTRL]** Ctrl: Clustering training losses for label error detection [[Paper](https://arxiv.org/pdf/2208.08464.pdf)]


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

