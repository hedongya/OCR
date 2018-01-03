# Optical Character Recognition

OCR for Baidu competition

Environment requirement: tensorflow 1.1.0, python 2.7

This work is based on model from Peiwen Wang whose excellent work (https://github.com/ypwhs/baiduyun_deeplearning_competition) is carried out on keras. Thanks to him sincerely.  

# Contents
* Train data
* Model
* Loss & accuracy
* Features learned by different level layers
* Weights and biases 

# Training datａ
Train data has 100,000 pictures including characters from 0123456789+-*() with different length as below picture shows.The label format, such as to the first training picture, is (7+5)+4 16

 <img src="https://github.com/hedongya/OCR/blob/master/results/image.png" width = "600">

# Model

The neural network incluing convolution network,rnn(GRU) and CTC (Connectionist Temporal Classifier) as picture below shows.
There are three convolution modules, every modules has two convolution layers and a max_pool layer. [3,3] kernel and [1,1] stride are used behind every convolution layers. [2,2] kernel and [2,2] stride are used by max_pool layer. What's more, learning features by different conv modules are outputted.

 <img src="https://github.com/hedongya/OCR/blob/master/results/Graph.png" width = "600">



# Loss & accuracy
As we can see, after 2.5h, there is a sharp ｄrop of CTCloss and the training and validation accuracy turn to be around 1.

<img src="https://github.com/hedongya/OCR/blob/master/results/CTCloss.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/acc.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/seqPredic.png" width = "600">





# Features learned by different layers

<img src="https://github.com/hedongya/OCR/blob/master/results/featureLayer1.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/featureLayer2.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/featureLayer3.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/fc1.png" width = "600">



Weights and biases distribution
================================
<img src="https://github.com/hedongya/OCR/blob/master/results/distributions.png" width = "600">
<img src="https://github.com/hedongya/OCR/blob/master/results/history.png" width = "600">








