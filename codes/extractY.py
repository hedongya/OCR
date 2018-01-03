
import numpy as np
def extractY(data_dir):
    y=[]
    if data_dir == "train":
       with open("./train_labels.txt") as f:
            for line in f.readlines():
                line=line.split(" ")[0]
                x=[]
                tmp=[s for s in line]
                for i in range(np.shape(tmp)[0]):
                    x.append(tmp[i])
                y.append(x)
       return y
    elif data_dir=="val":
       with open("./val_labels.txt") as f:
            for line in f.readlines():
                line=line.split(" ")[0]
                x=[]
                tmp=[s for s in line]
                for i in range(np.shape(tmp)[0]):
                    x.append(tmp[i])
                y.append(x)
       return y  
    else:
       print ("Wrong directary")
   
