# OCR
OCR for Baidu competition
Requirement:
tensorflow 1.1.0
python 2.7

Training data:
Train data has 100,000 pictures including characters from 0123456789+-*() with different length as below picture shows.
![image](https://github.com/hedongya/OCR/blob/master/results/image.png)
The label to the first training picture is (7+5)+4 16

The neural network incluing convolution network,rnn(GRU) and CTC (Connectionist Temporal Classifier) as picture below shows.
![image](https://github.com/hedongya/OCR/blob/master/results/Graph.png)
