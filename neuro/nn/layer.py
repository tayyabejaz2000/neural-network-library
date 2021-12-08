from typing import List

import tensorflow as tf
from neuro.nn.module import Module


class Dense(Module):
    __input: int
    __output: int

    def __init__(self, input: int, output: int, trainable: bool = True) -> None:
        super().__init__(trainable)
        self.__input = input
        self.__output = output
        self.__init_weights()

    def __init_weights(self) -> None:
        self.weights = [
            tf.Variable(tf.random.normal((self.__input, self.__output), stddev=0.03)),
            tf.Variable(tf.random.normal((self.__output,), stddev=0.03)),
        ]

    def forward(self, x: tf.Variable) -> tf.Variable:
        output = tf.matmul(x, self.weights[0])
        output = tf.add(output, self.weights[1])
        self._cachedTensor = x
        return output

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        gradients = [
            tf.matmul(grad, tf.transpose(self.weights[0])),  # da[l-1]
            tf.matmul(tf.transpose(self._cachedTensor), grad)
            / float(grad.shape[0]),  # dW
            tf.reduce_mean(grad, axis=0),  # dB
        ]
        return gradients


class Flatten(Module):
    def __init__(self, trainable: bool = True) -> None:
        super().__init__(trainable)
        self.weights = []

    def forward(self, x: tf.Variable) -> tf.Variable:
        flattened = tf.reshape(x, [x.shape[0], tf.reduce_prod(x.shape[1:])])
        return flattened

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [grad]


class Dropout(Module):
    __dropout_prob: float

    def __init__(self, p: float, trainable: bool = True) -> None:
        super().__init__(trainable)
        self.__dropout_prob = p
        self.weights = []

    def forward(self, x: tf.Variable) -> tf.Variable:
        output = x
        if self._training:
            keep_prob = 1 - self.__dropout_prob
            dropout_mask = tf.random.uniform(x.shape)
            output = tf.where(dropout_mask < keep_prob, output, 0.0) / keep_prob
            self._cachedTensor = dropout_mask
        return output

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [self._cachedTensor * grad]
