# Stereographic Spherical Sliced Wasserstein Distances

### [[ Project Page ]](https://abi-kothapalli.github.io/s3w)

This repository contains the accompanying code/experiments for the paper [Stereographic Spherical Sliced Wasserstein Distances](https://arxiv.org/abs/2402.02345). 

## Abstract

Comparing spherical probability distributions is of great interest in various fields, including geology, medical domains, computer vision, and deep representation learning. The utility of optimal transport-based distances, such as the Wasserstein distance, for comparing probability measures has spurred active research in developing computationally efficient variations of these distances for spherical probability measures. This paper introduces a high-speed and highly parallelizable distance for comparing spherical measures using the stereographic projection and the generalized Radon transform, which we refer to as the Stereographic Spherical Sliced Wasserstein (S3W) distance. We carefully address the distance distortion caused by the stereographic projection and provide an extensive theoretical analysis of our proposed metric and its rotationally invariant variation. Finally, we evaluate the performance of the proposed metrics and compare them with recent baselines in terms of both speed and accuracy through a wide range of numerical studies, including gradient flows and self-supervised learning.

## Citation

```
@inproceedings{tran2024stereographic,
      title={Stereographic Spherical Sliced Wasserstein Distances}, 
      author={Huy Tran and Yikun Bai and Abihith Kothapalli and Ashkan Shahbazi and Xinran Liu and Rocio Diaz Martin and Soheil Kolouri},
      year={2024},
      booktitle={International Conference on Machine Learning},
}
```

