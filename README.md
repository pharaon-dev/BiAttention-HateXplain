# BiAtt-HateXplain
This algorithm is derived from the BiRNN-HateXplain algorithm, and this project is based on the HateXplain project https://arxiv.org/pdf/2012.10289
The results of our studies can be found in the ```models_and_results/BiAtt_BiRNN_max_2``` folder.

## Objective:
The objective of this project is to improve the results of the BiRNN-HateXplain and BERT-HateXplain algorithms in terms of detection performance, unintentional bias, and explainability.

## Problem with current approaches:
In current algorithms such as BiRNN-HateXplain, we observe a large variation in the estimated attention when it should be constant.

## Proposal:
Our hypothesis is that considering the sequential aspect of input data in HateXplain models could improve explainability and, consequently, also improve classification performance and unintentional biases related to communities indexed in hate speech.

## Results:
The results show that the proposed approach improves explainability, prediction performance, and metrics that measure unintentional biases of the model.

## Installation:
It is recommended to use a tool like conda to create a virtual environment and facilitate conflict management.
