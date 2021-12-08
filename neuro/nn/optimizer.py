from abc import abstractclassmethod
from typing import List, Tuple

import tensorflow as tf
from neuro.nn.losses import Loss
from neuro.nn.models import Model


class Optimizer:
    @abstractclassmethod
    def applyGradients(self, gradients: List[List[tf.Variable]], weights: List[List[tf.Variable]]) -> List[List[tf.Variable]]:
        pass

    def __call__(self, model: Model, loss: Loss) -> None:
        gradients = model.backward(loss)
        updatedWeights = self.applyGradients(gradients, model.weights)
        model.updateWeights(updatedWeights)


class SGD(Optimizer):
    learning_rate: float

    def __init__(self, lr: float = 1e-3) -> None:
        self.learning_rate = lr

    def applyGradients(self, gradients: List[List[tf.Variable]], weights: List[List[tf.Variable]]) -> List[List[tf.Variable]]:
        updatedWeights = weights
        for i, (layer, layer_gradients) in enumerate(zip(weights, gradients)):
            for j, (weight, gradient) in enumerate(zip(layer, layer_gradients)):
                updatedWeights[i][j] = weight - self.learning_rate * gradient

        return updatedWeights


class AdaGrad(Optimizer):
    learning_rate: float
    __delta: List[float]

    def __init__(self, lr: float = 1e-3) -> None:
        self.learning_rate = lr
        self.__delta = None

    def applyGradients(self, gradients: List[List[tf.Variable]], weights: List[List[tf.Variable]]) -> List[List[tf.Variable]]:
        if self.__delta is None:
            self.__delta = [0.0] * \
                (len(weights) * sum([len(i) for i in weights]))

        updatedWeights = weights
        a = 0
        for i, (layer, layer_gradients) in enumerate(zip(weights, gradients)):
            for j, (weight, gradient) in enumerate(zip(layer, layer_gradients)):
                self.__delta[a] += tf.square(gradient)
                updatedWeights[i][j] = weight - self.learning_rate * \
                    (gradient / (tf.sqrt(self.__delta[a]) + 1e-6))
                a += 1
        return updatedWeights


class RMSProp(Optimizer):
    learning_rate: float
    decay_rate: float
    __delta: List[float]

    def __init__(self, lr: float = 1e-3, decay_rate: float = 0.9) -> None:
        self.learning_rate = lr
        self.decay_rate = decay_rate
        self.__delta = None

    def applyGradients(self, gradients: List[List[tf.Variable]], weights: List[List[tf.Variable]]) -> List[List[tf.Variable]]:
        if self.__delta is None or len(self.__delta) != len(weights) * sum([len(i) for i in weights]):
            self.__delta = [0.0] * len(weights) * \
                sum([len(i) for i in weights])

        updatedWeights = weights
        a = 0
        for i, (layer, layer_gradients) in enumerate(zip(weights, gradients)):
            for j, (weight, gradient) in enumerate(zip(layer, layer_gradients)):
                self.__delta[a] = self.decay_rate * self.__delta[a] + \
                    (1 - self.decay_rate) * tf.square(gradient)
                updatedWeights[i][j] = weight - self.learning_rate * \
                    (gradient / (tf.sqrt(self.__delta[a]) + 1e-6))
                a += 1

        return updatedWeights


class Adam(Optimizer):
    learning_rate: float
    beta_rates: Tuple[float, float]
    __m: List[float]
    __v: List[float]
    __t: float

    def __init__(self, lr: float = 1e-3, beta_1: float = 0.9, beta_2: float = 0.999) -> None:
        self.learning_rate = lr
        self.beta_rates = (beta_1, beta_2)
        self.__t = 1

        self.__m = None
        self.__v = None

    def applyGradients(self, gradients: List[List[tf.Variable]], weights: List[List[tf.Variable]]) -> List[List[tf.Variable]]:
        if self.__m is None or len(self.__m) != len(weights) * sum([len(i) for i in weights]):
            self.__m = [0.0] * (len(weights) * sum([len(i) for i in weights]))
        if self.__v is None or len(self.__v) != len(weights) * sum([len(i) for i in weights]):
            self.__v = [0.0] * (len(weights) * sum([len(i) for i in weights]))

        updatedWeights = weights
        a = 0
        for i, (layer, layer_gradients) in enumerate(zip(weights, gradients)):
            for j, (weight, gradient) in enumerate(zip(layer, layer_gradients)):
                self.__m[a] = self.beta_rates[0] * self.__m[a] + \
                    (1-self.beta_rates[0]) * gradient
                self.__v[a] = self.beta_rates[1] * self.__v[a] + \
                    (1-self.beta_rates[1]) * tf.square(gradient)
                m_corrected = self.__m[a] / \
                    (tf.pow(1-self.beta_rates[0], self.__t))
                v_corrected = self.__v[a] / \
                    (tf.pow(1-self.beta_rates[1], self.__t))
                a += 1

                updatedWeights[i][j] = weight - self.learning_rate * \
                    m_corrected / (tf.sqrt(v_corrected) + 1e-6)
        return updatedWeights
