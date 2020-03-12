import os
import struct
import math
import numpy as np

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels.idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images.idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

#从训练集中随机生成mini-batch列表
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    :param X: input data
    :param Y: the labels of X
    :param mini_batch_size: size of the mini-batches, integer
    :param seed: seed
    :return: mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)  #设置随机种子
    m = X.shape[1]  #训练样本数量
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):

        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:

        mini_batch_X = shuffled_X[:, -(m - mini_batch_size * num_complete_minibatches):]
        mini_batch_Y = shuffled_Y[:, -(m - mini_batch_size * num_complete_minibatches):]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    :param parameters: python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    :return: v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2 #神经网络的层数
    v = {}
    
    # Initialize velocity
    for l in range(L):   
        v["dW" + str(l+1)] = np.zeros((parameters['W'+str(l+1)]).shape)
        v["db" + str(l+1)] = np.zeros((parameters['b'+str(l+1)]).shape)      
    return v


def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
            - keys: "dW1", "db1", ..., "dWL", "dbL" 
            - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    :param parameters: python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl
    :return:  
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...
    """
    L = len(parameters) // 2 #神经网络的层数
    v = {}
    u = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
    
        v["dW" + str(l+1)] = np.zeros(np.shape(parameters['W'+str(l+1)]))
        v["db" + str(l+1)] = np.zeros(np.shape(parameters['b'+str(l+1)]))
        u["dW" + str(l+1)] = np.zeros(np.shape(parameters['W'+str(l+1)]))
        u["db" + str(l+1)] = np.zeros(np.shape(parameters['b'+str(l+1)]))
            
    return v, u
