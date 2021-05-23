#!/usr/bin/env python
# coding: utf-8

# # Contrastive Learning

# Contrastive learning is a self-supervised learning method that learns feature representations to distinguish between similar and different samples.
# 
# ![image](https://lh4.googleusercontent.com/NFYj5oT4MCnqcMSu_asvwqAb1A8rmzZr0jB2SZjMJiOGEz_my3VWcZq_jOGMogc_fwnuFGmXbaHjAa6eugaeauio1uvgR5W9be7tn_Nxo0jXiWGIb-eB9x6km22ZM6kOqIA0YAyr)
# 
# Source: [https://analyticsindiamag.com/contrastive-learning-self-supervised-ml/](https://analyticsindiamag.com/contrastive-learning-self-supervised-ml/)
# 
# 
# Contrastive learning can be formulated as a score:
# 
# $$ score(f(x),f(x^+))>score(f(x),f(x^-)) $$
# 
# where $x^+$ and $x^-$ refers to a sample similar and different to $x$, respectively.

# ## What is SimCLR?
# 
# [SimCLR](https://arxiv.org/abs/2002.05709) is a contrastive learning framework introduced by Chen et. al that outperforms the previously SOTA supervised learning method on ImageNet when scaling up the architecture.
# 
# ![https://amitness.com/images/simclr-general-architecture.png](https://amitness.com/images/simclr-general-architecture.png)
# 
# Source: [https://amitness.com/2020/03/illustrated-simclr/](https://amitness.com/2020/03/illustrated-simclr/)
# 
# Essentially, SimCLR applies data augmentation on a single image to obtain 2 augmented images, and then optimises by maximising the similarity between the representations of both augmented images.

# ## How is similarity computed?
# 
# SimCLR applies cosine similarity between the embeddings of images, $z_i, z_j$:
# 
# $$ s_{i,j}=\frac{z_i^Tz_j}{(\tau||z_i||||z_j||)} $$
# 
# where \tau is the temperature that scales the range of cosine similarity.

# ## What loss does SimCLR optimise?
# 
# SimCLR uses NT-Xent loss (Normalized Temperature-scaled Cross-entropy loss).
# 
# First, obtain the Noise Contrastive Estimation (NCE) loss:
# 
# $$ l(i,j)=-log\frac{exp(s_{i,j})}{\sum_{k=1}^{2N}l_{[k!=i]}exp(s_{i,k})} $$
# 
# Then compute average loss over all pairs in the batch of size N=2:
# 
# $$ L=\frac{1}{2N}\sum_{k=1}^N[l(2k-1,2k)+l(2k,2k-1)] $$

# ## So what?
# 
# The encoder representations of the image (not the projection head representations) can be used for downstream tasks such as ImageNet classification.
# 
# ![https://amitness.com/images/simclr-performance.png](https://amitness.com/images/simclr-performance.png)
# 
# Source: [https://arxiv.org/abs/2002.05709](https://arxiv.org/abs/2002.05709)
# 
# It is shown that on ImageNet ILSVRC-2012, it achieves 76.5% top-1 accuracy, a 7% improvement over the previous SOTA self-supervised method, [Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272), and comparable with supervised method ResNet50.
