from abc import abstractclassmethod
from typing import List

import tensorflow as tf


class Module:
    weights: List[tf.Variable]
    _training: bool
    _cachedTensor: tf.Variable

    def __init__(self, trainable: bool = True) -> None:
        self._training = trainable
        self.weights = []
        self._cachedTensor = None

    def __call__(self, x: tf.Variable) -> tf.Variable:
        return self.forward(x)

    def SetTrainable(self, trainable: bool) -> None:
        self._training = trainable

    @abstractclassmethod
    def forward(self, x: tf.Variable) -> tf.Variable:
        pass

    @abstractclassmethod
    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        pass
