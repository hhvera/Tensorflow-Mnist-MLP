# Tensorflow-Mnist-MLP
Tensorflow  Mnist  /MLP/BP/CNN

# Tnesorflow 1.7 用3层隐藏层实现手写体识别
import  tensorflow as tf
import  numpy as np
import  matplotlib.pyplot as plt
import  os
# 解决CPU不匹配的 忽略码问题 ‘1’，‘2’，‘3’
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#  加载MNIST数据集
# 从官网直接下载
# 从本地加载数据
mnist_folder="D:\\datamist\\MNIST"
from tensorflow.examples.tutorials.mnist import  input_data
mnist = input_data.read_data_sets(mnist_folder,one_hot=True)

# 定义模型参数
n_input = 784
n_labels = 10
n_hidden1 = 128
n_hidden2 = 256

# 定义模型 预测初始化的占位符 x  y
X = tf.placeholder(tf.float32,[None,n_input])
Y = tf.placeholder(tf.float32,[None,n_labels])

# 构建模型
#  权值 偏置  数据字典 键值对方式 去获取权值  偏置参数
weights ={
        'h1': tf.Variable(tf.truncated_normal([n_input,n_hidden1],stddev=0.1)),
        'h2': tf.Variable(tf.truncated_normal([n_hidden1,n_hidden2],stddev=0.1)),
        'out':tf.Variable(tf.truncated_normal([n_hidden2,n_labels],stddev=0.1))
}
biases= {
        'h1':tf.Variable(tf.random_normal([n_hidden1])),
        'h2':tf.Variable(tf.random_normal([n_hidden2])),
        'out':tf.Variable(tf.random_normal([n_labels]))
}

def mlpmodel(X,weights,biases):
    layer1 = tf.nn.relu(tf.add(tf.matmul(X,weights['h1']),biases['h1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,weights['h2']),biases['h2']))
    outlayer = tf.add(tf.matmul(layer2,weights['out']),biases['out'])
    return  outlayer

# 前向计算网络模型输出
pred = mlpmodel(X,weights,biases)
# 反向传播 交叉熵
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=Y))
# 优化器
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# 模型预测 模型训练
# 准确率 最大似然估计
correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

epoches =1000
display_step=100
batch_size = 50
avg_loss = 0.0
# 模型训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epoches):
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range (total_batch):
            batch_x ,batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer,feed_dict={X: batch_x,Y:batch_y})
        loss,acc = sess.run([cost,accuracy],feed_dict={X:mnist.train.images,Y:mnist.train.labels})
        avg_loss += loss/total_batch
        if(epoch+1) % display_step ==0:
            print('epoch:',epoch+1,'cost=','{:.6f}'.format(avg_loss),'accuracy=','{:.6f}'.format(acc))
    print("train finished")


