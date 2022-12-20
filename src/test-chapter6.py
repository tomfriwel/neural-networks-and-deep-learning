# http://neuralnetworksanddeeplearning.com/chap6.html#convolutional_neural_networks_in_practice
# we'll use network3.py as a library to build convolutional networks.
# [Install Theano](https://theano-pymc.readthedocs.io/en/latest/install.html)
# pip install theano
# git clone https://github.com/Theano/libgpuarray.git
# https://github.com/Theano/libgpuarray/blob/master/doc/installation.rst#

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
training_data, validation_data, test_data = network3.load_data_shared()
mini_batch_size = 10
net = Network([
        FullyConnectedLayer(n_in=784, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(training_data, 60, mini_batch_size, 0.1, 
            validation_data, test_data)