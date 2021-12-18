from abc import abstractclassmethod

import tensorflow as tf


class Loss:
    _localGradients: tf.Variable

    @abstractclassmethod
    def __call__(cls, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        pass

    @abstractclassmethod
    def gradient(cls) -> tf.Variable:
        pass


class MeanSquaredError(Loss):
    def mean_squared_error(
        self, y_pred: tf.Variable, y_true: tf.Variable
    ) -> tf.Variable:
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.square(y_pred - y_true), axis=1), keepdims=True
        )
        self._localGradients = -2 * (y_true - y_pred) * loss
        return loss

    def gradient(self) -> tf.Variable:
        return self._localGradients

    def __call__(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        return self.mean_squared_error(y_pred, y_true)


class HingeLoss(Loss):
    def hinge_loss(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        mul = y_true * y_pred
        loss = tf.reduce_sum(tf.maximum(-mul, 0))
        self._localGradients = tf.where(
            mul < 1, -y_true / float(y_pred.shape[0]), 0)
        return loss

    def gradient(self) -> tf.Variable:
        return self._localGradients

    def __call__(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        return self.hinge_loss(y_pred, y_true)


class BinaryCrossentropy(Loss):
    def binary_crossentropy_loss(
        self, y_pred: tf.Variable, y_true: tf.Variable
    ) -> tf.Variable:
        y_pred_neg_1 = 1 - y_pred
        y_true_neg_1 = 1 - y_true
        loss = -tf.reduce_mean(
            tf.multiply(y_true, tf.math.log(y_pred))
            + tf.multiply(y_true_neg_1, tf.math.log(y_pred_neg_1))
        )
        self._localGradients = (y_pred - y_true) / (y_pred * y_pred_neg_1)
        return loss

    def gradient(self) -> tf.Variable:
        return self._localGradients

    def __call__(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        return self.binary_crossentropy_loss(y_pred, y_true)


class CategoricalCrossentropy(Loss):
    def categorical_crossentropy_loss(
        self, y_pred: tf.Variable, y_true: tf.Variable
    ) -> tf.Variable:
        self._localGradients = y_pred - y_true
        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))

    def gradient(self) -> tf.Variable:
        return self._localGradients

    def __call__(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        return self.categorical_crossentropy_loss(y_pred, y_true)


class SparseCategoricalCrossentropy(Loss):
    def sparse_categorical_crossentropy_loss(
        self, y_pred: tf.Variable, y_true: tf.Variable
    ) -> tf.Variable:
        one_hot_labels = tf.one_hot(y_true, tf.unique(y_true)[0].shape[0])
        self._localGradients = y_pred - one_hot_labels
        return tf.reduce_mean(
            -tf.reduce_sum(one_hot_labels * tf.math.log(y_pred), axis=1)
        )

    def gradient(self) -> tf.Variable:
        return self._localGradients

    def __call__(self, y_pred: tf.Variable, y_true: tf.Variable) -> tf.Variable:
        return self.sparse_categorical_crossentropy_loss(y_pred, y_true)
