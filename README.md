# (Neural Networks) Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness

This repository provides a Official PyTorch implementation of our Neural Networks paper ***Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness***

- Neural Networks (will be published soon)
- [arXiv version](https://arxiv.org/abs/2403.15517)


## Abstract

Class Incremental Learning (CIL) constitutes a pivotal subfield within continual learning, aimed at enabling models to progressively learn new classification tasks while retaining knowledge obtained from prior tasks. Although previous studies have predominantly focused on backward compatible approaches to mitigate catastrophic forgetting, recent investigations have introduced forward compatible methods to enhance performance on novel tasks and complement existing backward compatible methods. In this study, we introduce an effective-Rank based Feature Richness enhancement (RFR) method, designed for improving forward compatibility. Specifically, this method increases the effective rank of representations during the base session, thereby facilitating the incorporation of more informative features pertinent to unseen novel tasks. Consequently, RFR achieves dual objectives in backward and forward compatibility: minimizing feature extractor modifications and enhancing novel task performance, respectively. To validate the efficacy of our approach, we establish a theoretical connection between effective rank and the Shannon entropy of representations. Subsequently, we conduct comprehensive experiments by integrating RFR into eleven well-known CIL methods. Our results demonstrate the effectiveness of our approach in enhancing novel-task performance while mitigating catastrophic forgetting. Furthermore, our method notably improves the average incremental accuracy across all eleven cases examined.


## Examples

We provide RFR with [UCIR](https://openaccess.thecvf.com/content_CVPR_2019/papers/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.pdf).

~~~
bash ucir_w_rfc.sh              # script for ucir w/ rfr
bash ucir_wo_rfc.sh             # script for ucir w/o rfr
~~~

## Citation

Please consider citing our work if you find our repository/paper useful.

~~~
@article{kim2024improving,
  title={Improving Forward Compatibility in Class Incremental Learning by Increasing Representation Rank and Feature Richness},
  author={Kim, Jaeill and Lee, Wonseok and Eo, Moonjung and Rhee, Wonjong},
  journal={arXiv preprint arXiv:2403.15517},
  year={2024}
}
~~~

## Acknowledgements

This code is based on the open-source codebase provided by [FACIL](https://github.com/mmasana/FACIL).