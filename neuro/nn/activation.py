from typing import List

import tensorflow as tf
from neuro.nn.module import Module


class Sigmoid(Module):
    def forward(self, x: tf.Variable) -> tf.Variable:
        forward_pass = 1 / (1 + tf.exp(-x))
        self._cachedTensor = forward_pass
        return forward_pass

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [(self._cachedTensor * (1 - self._cachedTensor)) * grad]


class Softmax(Module):
    def forward(self, x: tf.Variable) -> tf.Variable:
        forward_pass_1 = tf.exp(x)
        forward_pass = forward_pass_1 / tf.reduce_sum(
            forward_pass_1, axis=1, keepdims=True
        )
        self._cachedTensor = forward_pass
        return forward_pass

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        y = tf.reshape(self._cachedTensor, (*self._cachedTensor.shape, 1))
        y_T = tf.transpose(y, perm=[0, 2, 1])
        localGradient = tf.linalg.diag(self._cachedTensor) - tf.matmul(y, y_T)
        gradient = tf.matmul(localGradient, tf.reshape(grad, (*grad.shape, 1)))
        return [tf.reshape(gradient, gradient.shape[:-1])]


class Tanh(Module):
    def forward(self, x: tf.Variable) -> tf.Variable:
        e_x = tf.exp(x)
        e_neg_x = tf.exp(-x)
        forward_pass = (e_x - e_neg_x) / (e_x + e_neg_x)
        self._cachedTensor = forward_pass
        return forward_pass

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [(1 - tf.square(self._cachedTensor)) * grad]


class ReLU(Module):
    def forward(self, x: tf.Variable) -> tf.Variable:
        forward_pass = tf.maximum(0, x)
        self._cachedTensor = x
        return forward_pass

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [tf.where(self._cachedTensor >= 0, grad, 0.0)]


class LeakyReLU(Module):
    def __init__(self, neg_slope: float = 0.01) -> None:
        super().__init__()
        self.alpha = neg_slope

    def forward(self, x: tf.Variable) -> tf.Variable:
        forward_pass = tf.maximum(self.alpha * x, x)
        self._cachedTensor = x
        return forward_pass

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        localGradient = tf.where(self._cachedTensor < 0, self.alpha, 1)
        return [localGradient * grad]
