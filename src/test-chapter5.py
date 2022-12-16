import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
print('load')

import network2

# step1
# a three-layer network, with the first layer containing 784 neurons, the second layer 30 neurons, and the third layer 10 neuron.
# net = network2.Network([784, 30, 10])
# net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)

# step2 add another 30-neuron hidden layer
# a four-layer network, with the first layer containing 784 neurons, the second layer 30 neurons, the third layer 30 neuron, and the fourth layer 10 neuron.
# net = network2.Network([784, 30, 30, 10])
# net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)

# step3 another 30-neuron
net = network2.Network([784, 30, 30, 30, 10])
net.SGD(list(training_data), 30, 10, 0.1, lmbda=5.0, evaluation_data=list(validation_data), monitor_evaluation_accuracy=True)

# step1 output - 9646 / 10000
# step2 output - 9650 / 10000
# step3 output - 9648 / 10000
# not better, even worse

# step1 output - 9646 / 10000
# Epoch 0 training complete
# Accuracy on evaluation data: 9250 / 10000
# Epoch 1 training complete
# Accuracy on evaluation data: 9393 / 10000
# Epoch 2 training complete
# Accuracy on evaluation data: 9440 / 10000
# Epoch 3 training complete
# Accuracy on evaluation data: 9485 / 10000
# Epoch 4 training complete
# Accuracy on evaluation data: 9527 / 10000
# Epoch 5 training complete
# Accuracy on evaluation data: 9547 / 10000
# Epoch 6 training complete
# Accuracy on evaluation data: 9589 / 10000
# Epoch 7 training complete
# Accuracy on evaluation data: 9575 / 10000
# Epoch 8 training complete
# Accuracy on evaluation data: 9591 / 10000
# Epoch 9 training complete
# Accuracy on evaluation data: 9606 / 10000
# Epoch 10 training complete
# Accuracy on evaluation data: 9607 / 10000
# Epoch 11 training complete
# Accuracy on evaluation data: 9601 / 10000
# Epoch 12 training complete
# Accuracy on evaluation data: 9602 / 10000
# Epoch 13 training complete
# Accuracy on evaluation data: 9632 / 10000
# Epoch 14 training complete
# Accuracy on evaluation data: 9622 / 10000
# Epoch 15 training complete
# Accuracy on evaluation data: 9625 / 10000
# Epoch 16 training complete
# Accuracy on evaluation data: 9642 / 10000
# Epoch 17 training complete
# Accuracy on evaluation data: 9639 / 10000
# Epoch 18 training complete
# Accuracy on evaluation data: 9642 / 10000
# Epoch 19 training complete
# Accuracy on evaluation data: 9628 / 10000
# Epoch 20 training complete
# Accuracy on evaluation data: 9637 / 10000
# Epoch 21 training complete
# Accuracy on evaluation data: 9641 / 10000
# Epoch 22 training complete
# Accuracy on evaluation data: 9648 / 10000
# Epoch 23 training complete
# Accuracy on evaluation data: 9654 / 10000
# Epoch 24 training complete
# Accuracy on evaluation data: 9664 / 10000
# Epoch 25 training complete
# Accuracy on evaluation data: 9654 / 10000
# Epoch 26 training complete
# Accuracy on evaluation data: 9658 / 10000
# Epoch 27 training complete
# Accuracy on evaluation data: 9650 / 10000
# Epoch 28 training complete
# Accuracy on evaluation data: 9647 / 10000
# Epoch 29 training complete
# Accuracy on evaluation data: 9646 / 10000


# step2 output - 9650 / 10000
# Epoch 0 training complete
# Accuracy on evaluation data: 9220 / 10000
# Epoch 1 training complete
# Accuracy on evaluation data: 9437 / 10000
# Epoch 2 training complete
# Accuracy on evaluation data: 9532 / 10000
# Epoch 3 training complete
# Accuracy on evaluation data: 9541 / 10000
# Epoch 4 training complete
# Accuracy on evaluation data: 9576 / 10000
# Epoch 5 training complete
# Accuracy on evaluation data: 9588 / 10000
# Epoch 6 training complete
# Accuracy on evaluation data: 9618 / 10000
# Epoch 7 training complete
# Accuracy on evaluation data: 9599 / 10000
# Epoch 8 training complete
# Accuracy on evaluation data: 9630 / 10000
# Epoch 9 training complete
# Accuracy on evaluation data: 9626 / 10000
# Epoch 10 training complete
# Accuracy on evaluation data: 9637 / 10000
# Epoch 11 training complete
# Accuracy on evaluation data: 9639 / 10000
# Epoch 12 training complete
# Accuracy on evaluation data: 9643 / 10000
# Epoch 13 training complete
# Accuracy on evaluation data: 9647 / 10000
# Epoch 14 training complete
# Accuracy on evaluation data: 9633 / 10000
# Epoch 15 training complete
# Accuracy on evaluation data: 9648 / 10000
# Epoch 16 training complete
# Accuracy on evaluation data: 9639 / 10000
# Epoch 17 training complete
# Accuracy on evaluation data: 9644 / 10000
# Epoch 18 training complete
# Accuracy on evaluation data: 9651 / 10000
# Epoch 19 training complete
# Accuracy on evaluation data: 9631 / 10000
# Epoch 20 training complete
# Accuracy on evaluation data: 9639 / 10000
# Epoch 21 training complete
# Accuracy on evaluation data: 9658 / 10000
# Epoch 22 training complete
# Accuracy on evaluation data: 9646 / 10000
# Epoch 23 training complete
# Accuracy on evaluation data: 9635 / 10000
# Epoch 24 training complete
# Accuracy on evaluation data: 9646 / 10000
# Epoch 25 training complete
# Accuracy on evaluation data: 9627 / 10000
# Epoch 26 training complete
# Accuracy on evaluation data: 9666 / 10000
# Epoch 27 training complete
# Accuracy on evaluation data: 9641 / 10000
# Epoch 28 training complete
# Accuracy on evaluation data: 9672 / 10000
# Epoch 29 training complete
# Accuracy on evaluation data: 9650 / 10000

# step3 output - 9648 / 10000
# Epoch 0 training complete
# Accuracy on evaluation data: 8794 / 10000
# Epoch 1 training complete
# Accuracy on evaluation data: 9269 / 10000
# Epoch 2 training complete
# Accuracy on evaluation data: 9470 / 10000
# Epoch 3 training complete
# Accuracy on evaluation data: 9528 / 10000
# Epoch 4 training complete
# Accuracy on evaluation data: 9540 / 10000
# Epoch 5 training complete
# Accuracy on evaluation data: 9581 / 10000
# Epoch 6 training complete
# Accuracy on evaluation data: 9543 / 10000
# Epoch 7 training complete
# Accuracy on evaluation data: 9582 / 10000
# Epoch 8 training complete
# Accuracy on evaluation data: 9556 / 10000
# Epoch 9 training complete
# Accuracy on evaluation data: 9620 / 10000
# Epoch 10 training complete
# Accuracy on evaluation data: 9619 / 10000
# Epoch 11 training complete
# Accuracy on evaluation data: 9548 / 10000
# Epoch 12 training complete
# Accuracy on evaluation data: 9611 / 10000
# Epoch 13 training complete
# Accuracy on evaluation data: 9609 / 10000
# Epoch 14 training complete
# Accuracy on evaluation data: 9586 / 10000
# Epoch 15 training complete
# Accuracy on evaluation data: 9634 / 10000
# Epoch 16 training complete
# Accuracy on evaluation data: 9600 / 10000
# Epoch 17 training complete
# Accuracy on evaluation data: 9636 / 10000
# Epoch 18 training complete
# Accuracy on evaluation data: 9598 / 10000
# Epoch 19 training complete
# Accuracy on evaluation data: 9621 / 10000
# Epoch 20 training complete
# Accuracy on evaluation data: 9595 / 10000
# Epoch 21 training complete
# Accuracy on evaluation data: 9616 / 10000
# Epoch 22 training complete
# Accuracy on evaluation data: 9571 / 10000
# Epoch 23 training complete
# Accuracy on evaluation data: 9628 / 10000
# Epoch 24 training complete
# Accuracy on evaluation data: 9617 / 10000
# Epoch 25 training complete
# Accuracy on evaluation data: 9640 / 10000
# Epoch 26 training complete
# Accuracy on evaluation data: 9592 / 10000
# Epoch 27 training complete
# Accuracy on evaluation data: 9641 / 10000
# Epoch 28 training complete
# Accuracy on evaluation data: 9644 / 10000
# Epoch 29 training complete
# Accuracy on evaluation data: 9648 / 10000