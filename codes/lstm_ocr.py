# coding: utf-8
import os,sys
import numpy as np
import tensorflow as tf
import random
import cv2,time
import logging,datetime
from tensorflow.python.client import device_lib
from tensorflow.python.client import timeline
import utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 

FLAGS=utils.FLAGS

width=160
hight=60
num_features=hight*1



# output learned feature
def output_features(pool1):
    for i in [1,7,13,19,25,31]:
        test1 = tf.transpose(pool1, (0, 2, 1,3))[:,:,:,i:i+1]
        tf.summary.image('image%d'%i,tf.transpose(pool1, (0, 2, 1,3))[:,:,:,i:i+1], max_outputs=1)
class Graph(object):
    def __init__(self):
        def variable_summaries(var,name):
            with tf.name_scope('summaries'):
                tf.summary.histogram(name,var)
                mean=tf.reduce_mean(var)
                tf.summary.scalar('mean/'+name,mean)
                stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
                tf.summary.scalar('stddev/'+name,stddev)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, [None, None, num_features])
            # Here we use sparse_placeholder that will generate a
            # SparseTensor required by ctc_loss op.
            self.labels = tf.sparse_placeholder(tf.int32)
            self.inputs2=tf.reshape(self.inputs,[-1,width,num_features,1])
            # 1d array of size [batch_size]
            self.seq_len = tf.placeholder(tf.int32, [None])
            
            #define first layer of conv
            with tf.variable_scope('layer1'):
                    weights1 = tf.Variable(tf.truncated_normal([3,3,1,32],stddev=0.1,dtype=tf.float32),name='W')
                    biases1 = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[32]))
                    conv1=tf.nn.conv2d(self.inputs2,weights1,strides=[1,1,1,1],padding='VALID')
                    relu1=tf.nn.relu(tf.nn.bias_add(conv1,biases1))
                
                    weights2 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.1,dtype=tf.float32),name='W')
                    biases2 = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[32]))
                    conv2=tf.nn.conv2d(relu1,weights2,strides=[1,1,1,1],padding='VALID')
                    relu2=tf.nn.relu(tf.nn.bias_add(conv2,biases2))

                    pool1=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
                    
                    # output learned feature
                    output_features(pool1)
            
            #define the last two conv
            for i in [2,3]:
                with tf.variable_scope('layer%d'%i):
                    weights1 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.1,dtype=tf.float32),name='W')
                    biases1 = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[32]))
                    conv1=tf.nn.conv2d(pool1,weights1,strides=[1,1,1,1],padding='VALID')
                    relu1=tf.nn.relu(tf.nn.bias_add(conv1,biases1))

                    weights2 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.1,dtype=tf.float32),name='W')
                    biases2 = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[32]))
                    conv2=tf.nn.conv2d(relu1,weights2,strides=[1,1,1,1],padding='VALID')
                    relu2=tf.nn.relu(tf.nn.bias_add(conv2,biases2))

                    pool1=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
                    # output learned feature
                    output_features(pool1)

            shape2=tf.shape(pool1) 
            result=tf.reshape(pool1,[-1,shape2[1],shape2[2]*shape2[3]])
            
            #define first fully connected layer
            with tf.variable_scope('fc-1'):
                shape=tf.shape(result) 
                rshape=tf.reshape(result,[-1,shape[2]])
                weights = tf.Variable(tf.truncated_normal([128,128],stddev=0.1,dtype=tf.float32),name='W')
                biases = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[128]))
                fc1=tf.matmul(rshape,weights)+biases
                result=tf.reshape(fc1,[-1,shape[1],128])
                test1=tf.reshape(result,[-1,shape[1],128,1])
                # output learned feature
                tf.summary.image('image1',test1, max_outputs=1)

            # input feature to rnn
            with tf.variable_scope('gru1'):
                stack = tf.contrib.rnn.GRUCell(num_hidden)
            # The second output is the last state and we will no use that
                gru1, _ = tf.nn.dynamic_rnn(stack,result, self.seq_len, dtype=tf.float32)
            with tf.variable_scope('gru1_b'):
                stack = tf.contrib.rnn.GRUCell(num_hidden)
                gru1_b, _ = tf.nn.dynamic_rnn(stack, result, self.seq_len, dtype=tf.float32)
            gru=tf.add(gru1,gru1_b)            

            with tf.variable_scope('gru2'):
                stack = tf.contrib.rnn.GRUCell(num_hidden)
                gru2, _ = tf.nn.dynamic_rnn(stack, gru, self.seq_len, dtype=tf.float32)
            with tf.variable_scope('gru2_b'):
                stack = tf.contrib.rnn.GRUCell(num_hidden)
                gru2_b, _ = tf.nn.dynamic_rnn(stack, gru, self.seq_len, dtype=tf.float32)
            gru=tf.concat([gru2,gru2_b],2)            
            gru=tf.nn.dropout(gru,0.25)
            #define second fully connected layer
            with tf.variable_scope('fc-2'):
                shape=tf.shape(gru) 
                rshape=tf.reshape(gru,[-1,shape[2]])
                weights = tf.Variable(tf.truncated_normal([256,17],stddev=0.1,dtype=tf.float32),name='W')
                biases = tf.Variable(tf.constant(0.0, dtype = tf.float32,shape=[17]))
                fc2=tf.matmul(rshape,weights)+biases
            logits=tf.reshape(fc2,[-1,shape[1],17])
           
           # Time major
            logits = tf.transpose(logits, (1, 0, 2))
            self.global_step = tf.Variable(0,trainable=False)
            
            #Compute CTC loss        
            self.loss = tf.nn.ctc_loss(labels=self.labels,inputs=logits, sequence_length=self.seq_len)
            self.CTCloss = tf.reduce_mean(self.loss)
            self.learning_rate=tf.train.exponential_decay(initial_learning_rate,
                    self.global_step, 
                    FLAGS.decay_steps,
                    FLAGS.decay_rate,staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=initial_learning_rate,beta1=FLAGS.beta1,beta2=FLAGS.beta2).minimize(self.CTCloss,global_step=self.global_step)
           
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(logits, self.seq_len,merge_repeated=True)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=0)[:,:7]
            self.lengthOferr = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels, normalize=True))
            tf.summary.scalar('CTCloss',self.CTCloss)
            tf.summary.scalar('lengthOferr',self.lengthOferr)
            self.merged_summay = tf.summary.merge_all()

def train(train_dir=None,val_dir=None,initial_learning_rate=None,num_hidden=None,num_classes=None,hparam=None):
    g = Graph()
    print('loading train data, please wait---------------------')
    train_feeder=utils.DataIterator(data_dir=train_dir)
    print('get image: ',train_feeder.size)

    print('loading validation data, please wait---------------------')
    val_feeder=utils.DataIterator(data_dir=val_dir)
    print('get image: ',val_feeder.size)

    num_train_samples = train_feeder.size 
    num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size) 

    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100) #持久化
        g.graph.finalize() #???
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+hparam,sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        val_inputs,val_seq_len,val_labels,_=val_feeder.input_index_generate_batch()  # seq_len 时序长度120
        
        val_feed={g.inputs: val_inputs,
                  g.labels: val_labels,
                 g.seq_len: val_seq_len}
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = train_err=0
            start_time = time.time()
            batch_time = time.time()
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels,label_batch=train_feeder.input_index_generate_batch(indexs)
                feed={g.inputs: batch_inputs,
                        g.labels:batch_labels,
                        g.seq_len:batch_seq_len}

                train_dense_decoded,summary_str, batch_cost,step,_ = sess.run([g.dense_decoded,g.merged_summay,g.CTCloss,g.global_step,g.optimizer],feed)
            #check computing cell propeties
#                print "||||||||||||||||||||"
         #       print(sess.run(g.dense_decoded,feed))
    #            print(np.shape(sess.run(g.pool1,feed)))
                train_cost+=batch_cost*FLAGS.batch_size
                train_writer.add_summary(summary_str,step)
                # save the checkpoint
                if step%FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)
                if step%FLAGS.validation_steps == 0:
                    dense_decoded,validation_length_err,learningRate = sess.run([g.dense_decoded,g.lengthOferr,
                        g.learning_rate],val_feed)
                    valid_acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
                    train_acc = utils.accuracy_calculation(label_batch,train_dense_decoded,ignore_value=-1,isPrint=True)
                    avg_train_cost=train_cost/((cur_batch+1)*FLAGS.batch_size)
                    now = datetime.datetime.now()
                    log = "*{}/{} {}:{}:{} Epoch {}/{}, accOfvalidation = {:.3f},train_accuracy={:.3f}, time = {:.3f},learningRate={:.8f}"
                    print(log.format(now.month,now.day,now.hour,now.minute,now.second,cur_epoch+1,FLAGS.num_epochs,valid_acc,train_acc,time.time()-start_time,learningRate)) 
def make_hparam_string(initial_learning_rate,num_hidden,num_classes):
    return "lr_%.0E,%s,%s"%(initial_learning_rate,num_hidden,num_classes)

if __name__ == '__main__':
    for initial_learning_rate in [0.001]:
        for num_hidden in [128]:
            for num_classes in [17]:
                hparam=make_hparam_string(initial_learning_rate,num_hidden,num_classes)
                print ('Starting run for %s'%hparam)
                train(train_dir='./train',val_dir='./val',initial_learning_rate=initial_learning_rate,num_hidden=num_hidden,num_classes=num_classes,hparam=hparam)

