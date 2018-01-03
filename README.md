# OCR
OCR for Baidu competition
Requirement:
tensorflow 1.1.0
python 2.7

This work is based on model from Peiwen Wang whose excellent work (https://github.com/ypwhs/baiduyun_deeplearning_competition) is carried out on keras. Thanks to him sincerely.  

Training datａ
=============



Train data has 100,000 pictures including characters from 0123456789+-*() with different length as below picture shows.
![image](https://github.com/hedongya/OCR/blob/master/results/image.png)
The label to the first training picture is (7+5)+4 16

The neural network incluing convolution network,rnn(GRU) and CTC (Connectionist Temporal Classifier) as picture below shows.
There are three convolution modules, every modules has two convolution layers and a max_pool layer. 3*3 kernel and 1*1 stride are used behind every convolution layers. 2*2 kernel and 2*2 stride are used by max_pool layer. What's more, learning features by different conv modules are outputted.
![image](https://github.com/hedongya/OCR/blob/master/results/Graph.png)


Following pictures are the training results.
![image](https://github.com/hedongya/OCR/blob/master/results/CTCloss.png)
![image](https://github.com/hedongya/OCR/blob/master/results/acc.png)
![image](https://github.com/hedongya/OCR/blob/master/results/seqPredic.png)

As we can see, after 2.5h, there is a sharp ｄrop of CTCloss and the training and validation accuracy turn to be around 1.

Features learned by different layers.
![image](https://github.com/hedongya/OCR/blob/master/results/featureLayer1.png)
![image](https://github.com/hedongya/OCR/blob/master/results/featureLayer2.png)
![image](https://github.com/hedongya/OCR/blob/master/results/featureLayer3.png)
![image](https://github.com/hedongya/OCR/blob/master/results/fc1.png)

Weights and biases distribution
![image](https://github.com/hedongya/OCR/blob/master/results/distributions.png)
![image](https://github.com/hedongya/OCR/blob/master/results/history.png)






