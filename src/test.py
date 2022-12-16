import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print('load')

import network
# layer: input, hidden, output
# step 1
# net = network.Network([784, 30, 10])
# step 2
# net = network.Network([784, 100, 10])
# training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=None
# net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
# step 3
# net.SGD(list(training_data), 30, 10, 0.001, test_data=list(test_data))

# step 4
# net = network.Network([784, 30, 10])
# net.SGD(list(training_data), 30, 10, 100.0, test_data=list(test_data))

