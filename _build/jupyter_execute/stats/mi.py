#!/usr/bin/env python
# coding: utf-8

# # Mutual Information
# 
# Mutual information is a measure of the reduction in uncertainty for one variable given a known value of another variable:
# 
# $$I(X,Y)=H(X)-H(X|Y)$$
# 
# where $I(X,Y)$ is the mutual information for $X$ and $Y$, $H(X)$ is the entropy for $X$, $H(X|Y)$ is the conditional entropy for $X$ given $Y$.
# 
# Mutual information is symmetric:
# 
# $$I(X,Y)=I(Y,X)$$
# 
# Mutual information can be calculated as the KL divergence between the joint distribution and the product of marginal probabilities for each variable:
# 
# $$I(X,Y)=KL(P(X,Y)||P(X)\times P(Y))$$
# 
# Mutual information and information gain are computed equivalently, and therefore MI is sometimes used as as a synonym for information gain:
# 
# $$IG(S,a)=H(S)-H(S|a)$$
