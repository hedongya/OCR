import os,sys
#import config
import numpy as np
import tensorflow as tf
import random
import cv2,time
from tensorflow.python.client import device_lib
import extractY
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_width=160
image_height=60
SPACE_INDEX=0
SPACE_TOKEN=''

maxPrintLen =5 
tf.app.flags.DEFINE_boolean('restore', False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_integer('num_epochs', 300, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 128, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 100, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps', 100, 'the step to validation')

tf.app.flags.DEFINE_float('decay_rate', 1, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 1, 'the lr decay_step for optimizer')

tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

tf.app.flags.DEFINE_string('log_dir', './log/', 'the logging dir')

FLAGS=tf.app.flags.FLAGS


charset = '0123456789+-*)('
encode_maps={}
decode_maps={}
for i,char in enumerate(charset,1):
    encode_maps[char]=i
    decode_maps[i]=char
encode_maps[SPACE_TOKEN]=SPACE_INDEX
decode_maps[SPACE_INDEX]=SPACE_TOKEN

class DataIterator:
    def __init__(self, data_dir):
        files=str(data_dir).split('/')[1]
        self.labels=[]
        temp=extractY.extractY(files)
        for i in np.arange(np.shape(temp)[0]):
            code=[encode_maps[c] for c in temp[i]]
            self.labels.append(code)
        lb=np.zeros((np.shape(self.labels)[0],7))
        self.labels1=np.array(self.labels)
        for i in range(len(self.labels)):
            for j in range(np.shape(self.labels1[i])[0]):
                    lb[i][j]=self.labels1[i][j]
        self.labels=lb
        self.image_names = []
        self.image = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for i in range(len(file_list)):
                file_list[i]=file_list[i].split('.')
                file_list[i][0]=int(file_list[i][0])
            file_list.sort()
            for i in range(len(file_list)):
                file_list[i][0]=str(file_list[i][0])
                file_list[i]=file_list[i][0]+'.'+file_list[i][1]
            for file_path in file_list:
                image_name = os.path.join(root,file_path)
                self.image_names.append(image_name)
                im = cv2.imread(image_name,0).astype(np.float32)/255.
                #resize to same height, different width will consume time on padding
                im = cv2.resize(im,(image_width,image_height))

                # transpose to (160*60) and the step shall be 160
                # in this way, each row is a feature vector
                im = im.swapaxes(0,1)
                self.image.append(np.array(im))

    @property
    def size(self):
        return len(self.labels)

    def the_label(self,indexs):
        labels=[]
        for i in indexs:
            labels.append(self.labels[i])
        return labels


    def input_index_generate_batch(self,index=None):
        if index:
            image_batch=[self.image[i] for i in index]
            label_batch=[self.labels[i] for i in index]
        else:
            image_batch=self.image
            label_batch=self.labels
        def get_input_lens(sequences):
            lengths = np.asarray([17 for s in sequences], dtype=np.int64)
            return sequences,lengths
        batch_inputs,batch_seq_len = get_input_lens(np.array(image_batch))  #64,160
        batch_labels = sparse_tuple_from_label(label_batch)
        return batch_inputs,batch_seq_len,batch_labels,label_batch

def accuracy_calculation(original_seq,decoded_seq,ignore_value=-1,isPrint = True):
    if  len(original_seq)!=len(decoded_seq):
        print('original lengths is different from the decoded_seq,please check again')
        return 0
    count = 0
    for i,origin_label in enumerate(original_seq):
        decoded_label  = [j for j in decoded_seq[i] if j!=ignore_value]
        if isPrint and i<maxPrintLen:
            print('seq{0:4d}: origin: {1} decoded:{2}'.format(i,origin_label,decoded_label))
        n=0
        if len(decoded_label)==7:
            for i in range(7):
                if (origin_label[i]-decoded_label[i])==0.0:
                    n=n+1
        if n==7:
                count+=1
    return count*1.0/len(original_seq)

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values,dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)
    return indices, values, shape

