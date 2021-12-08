from abc import abstractclassmethod
from typing import List

import tensorflow as tf
from neuro.nn.layer import Module
from neuro.nn.losses import Loss


class Model:
    layers: List[Module]
    weights: List[List[tf.Variable]]

    def __init__(self) -> None:
        self.layers = []
        self.weights = []

    @abstractclassmethod
    def forward(self, X: tf.Variable) -> tf.Variable:
        pass

    @abstractclassmethod
    def backward(self, loss: tf.Variable) -> List[List[tf.Variable]]:
        pass

    @abstractclassmethod
    def updateWeights(self, weights: List[List[tf.Variable]]) -> None:
        pass

    def __call__(self, X: tf.Variable) -> tf.Variable:
        return self.forward(X)

    def __get_trainable(self) -> bool:
        return all([layer._training for layer in self.layers])

    def __set_trainable(self, train: bool) -> None:
        for layer in self.layers:
            layer.SetTrainable(train)

    trainable: bool = property(__get_trainable, __set_trainable)


class Sequential(Model):
    def __init__(self, *args: Module) -> None:
        super().__init__()
        self.layers = list(args)
        for layer in self.layers:
            if layer.weights is not None and len(layer.weights) != 0:
                self.weights.append(layer.weights)

    def forward(self, X: tf.Variable) -> tf.Variable:
        output = tf.cast(X, dtype="float")
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss: Loss) -> List[List[tf.Variable]]:
        gradients = [loss.gradient()]
        learning_gradients = []
        for layer in reversed(self.layers):
            gradients = layer.backprop(gradients[0])
            if len(gradients) > 1:
                learning_gradients.append(gradients[1:])
        return list(reversed(learning_gradients))

    def updateWeights(self, weights: List[List[tf.Variable]]) -> None:
        for i, layer_weights in enumerate(weights):
            for j, weight in enumerate(layer_weights):
                self.weights[i][j] = weight
