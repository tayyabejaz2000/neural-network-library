from ctypes import ArgumentError
from tkinter import Variable
from typing import List, Tuple, Union

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
            tf.Variable(tf.random.normal(
                (self.__input, self.__output), stddev=0.03)),
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
            tf.matmul(tf.transpose(self._cachedTensor),
                      grad) / float(grad.shape[0]),  # dW
            tf.reduce_mean(grad, axis=0),  # dB
        ]
        return gradients


class Flatten(Module):
    def __init__(self, trainable: bool = True) -> None:
        super().__init__(trainable)
        self.weights = []

    def forward(self, x: tf.Variable) -> tf.Variable:
        flattened = tf.reshape(x, [x.shape[0], tf.reduce_prod(x.shape[1:])])
        self._cachedTensor = tf.shape(x)
        return flattened

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [tf.reshape(grad, self._cachedTensor)]


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
            dropout_mask = tf.random.uniform((1, *x.shape[1:]))
            output = (
                tf.cast(dropout_mask < keep_prob, dtype="float") * output
            ) / keep_prob
            self._cachedTensor = dropout_mask
        return output

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        return [self._cachedTensor * grad]


class Conv2D(Module):
    __num_filters: int
    __filter_size: Tuple[int, int, int, int]
    __prev_channels: int

    __stride: Tuple[int, int]
    __padding: Tuple[int, int]

    def __init__(
        self,
        input_filters: int,
        filters: int,
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, str] = 0,
        trainable: bool = True,
    ) -> None:
        super().__init__(trainable)
        self.__prev_channels = input_filters
        self.__num_filters = filters
        self.__filter_size = (self.__num_filters,
                              self.__prev_channels, *kernel_size)
        if isinstance(stride, int):
            self.__stride = (stride, stride)
        else:
            self.__stride = stride

        if isinstance(padding, int):
            self.__padding = (padding, padding)
        elif isinstance(padding, str):
            pass
        else:
            raise ArgumentError(
                "Padding should have integer value or str values ['SAME', 'VALID']"
            )

        self.__init_weights()

    def __init_weights(self) -> None:
        self.weights = [
            tf.random.normal(self.__filter_size, stddev=0.03),
            tf.random.normal((self.__num_filters,), stddev=0.03),
        ]

    def forward(self, x: tf.Variable) -> tf.Variable:
        # Padding
        x_padded = tf.pad(
            x,
            [
                [0, 0],  # Batches
                [0, 0],  # Channel
                [self.__padding[0], self.__padding[0]],  # Width
                [self.__padding[1], self.__padding[1]],  # Height
            ],
        )
        output_shape = [
            int(x.shape[0]),
            int(self.__num_filters),
            int(
                (
                    int(x.shape[2] - self.__filter_size[2] +
                        2 * self.__padding[0])
                    // self.__stride[0]
                )
                + 1
            ),
            int(
                (
                    int(x.shape[3] - self.__filter_size[3] +
                        2 * self.__padding[1])
                    // self.__stride[1]
                )
                + 1
            ),
        ]

        output = []
        for i in range(0, x_padded.shape[2] - self.__filter_size[2] + 1, self.__stride[0]):
            window_col = x_padded[:, :, i: i + self.__filter_size[2]]
            for j in range(0, x_padded.shape[3] - self.__filter_size[3] + 1, self.__stride[1]):
                window = window_col[..., j: j + self.__filter_size[3]]
                output.append(tf.reduce_sum(window[:, tf.newaxis, ...] *
                                            self.weights[0][tf.newaxis, ...], axis=[2, 3, 4]) + self.weights[1][tf.newaxis, ...])
        self._cachedTensor = x
        return tf.reshape(tf.transpose(output, perm=[1, 2, 0]), output_shape)

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        input = self._cachedTensor
        gradients = tf.Variable(tf.zeros(input.shape))
        dW = tf.Variable(tf.zeros(self.__filter_size))
        i, x = 0, 0
        while i < input.shape[2] - self.__filter_size[2] + 1:
            j, y = 0, 0
            while j < input.shape[3] - self.__filter_size[3] + 1:
                dW = dW + tf.reduce_mean(
                    grad[..., tf.newaxis, x:x+1, y:y+1] * input[
                        :,
                        tf.newaxis,
                        :,
                        i: i + self.__filter_size[2],
                        j: j + self.__filter_size[3]
                    ],
                    axis=0
                )
                grad_window = gradients[
                    ...,
                    i: i + self.__filter_size[2],
                    j: j + self.__filter_size[3]
                ]
                gradients = grad_window.assign(
                    grad_window + tf.reduce_sum(
                        grad[:, :, x:x+1, y:y+1, tf.newaxis] *
                        self.weights[0][tf.newaxis, ...],
                        axis=1
                    )
                )
                # m, f, 3, 3
                # f,
                j, y = j + self.__stride[1], y + 1
            i, x = i + self.__stride[0], x + 1
        dB = tf.reduce_mean(tf.reduce_sum(grad, axis=[-1, -2]), axis=0)
        return [
            gradients,
            dW,
            dB,
        ]


class MaxPool2D(Module):
    __kernel_size: Tuple[int, int]
    __stride: Tuple[int, int]
    __padding: Tuple[int, int]

    def __init__(
        self,
        kernel_size: Tuple[int, int],
        stride: Union[int, Tuple[int, int]] = None,
        padding: Union[str, int, Tuple[int, int]] = None,
        trainable: bool = True
    ) -> None:
        super().__init__(trainable=trainable)
        self.__kernel_size = kernel_size

        if stride is None:
            self.__stride = kernel_size
        elif isinstance(stride, int):
            self.__stride = (stride, stride)
        elif isinstance(stride, tuple):
            self.__stride = stride

        if padding is None:
            self.__padding = (0, 0)
        elif isinstance(padding, int):
            self.__padding = (padding, padding)
        elif isinstance(padding, tuple):
            self.__padding = padding
        elif isinstance(padding, str):
            pass

    def forward(self, x: tf.Variable) -> tf.Variable:
        x_padded: tf.Tensor = tf.pad(
            x,
            [
                [0, 0],  # Batches
                [0, 0],  # Channel
                [self.__padding[0], self.__padding[0]],  # Width
                [self.__padding[1], self.__padding[1]],  # Height
            ],
        )
        output_shape = [
            int(x.shape[0]),
            int(x.shape[1]),
            int(
                (
                    int(x.shape[2] - self.__kernel_size[0] +
                        2 * self.__padding[0])
                    // self.__stride[0]
                )
                + 1
            ),
            int(
                (
                    int(x.shape[3] - self.__kernel_size[1] +
                        2 * self.__padding[1])
                    // self.__stride[1]
                )
                + 1
            ),
        ]

        output = []
        for i in range(0, x_padded.shape[2] - self.__kernel_size[0] + 1, self.__stride[0]):
            window_col = x_padded[:, :, i: i + self.__kernel_size[0]]
            for j in range(0, x_padded.shape[3] - self.__kernel_size[1] + 1, self.__stride[1]):
                window = window_col[..., j: j + self.__kernel_size[1]]
                output.append(tf.reduce_max(window, axis=[-1, -2]))
        self._cachedTensor = x
        return tf.reshape(tf.transpose(output, perm=[1, 2, 0]), output_shape)

    def backprop(self, grad: tf.Variable) -> List[tf.Variable]:
        input = self._cachedTensor
        gradient = tf.Variable(tf.zeros(tf.shape(input)))
        x = 0
        for i in range(0, input.shape[2] - self.__kernel_size[0] + 1, self.__stride[0]):
            window_col = input[:, :, i: i + self.__kernel_size[0]]
            y = 0
            for j in range(0, input.shape[3] - self.__kernel_size[1] + 1, self.__stride[1]):
                window = window_col[..., j: j + self.__kernel_size[1]]
                grad_value = grad[..., x:x+1, y:y+1]
                grad_window = gradient[
                    ...,
                    i:i + self.__kernel_size[0],
                    j:j + self.__kernel_size[1]
                ]
                gradient = grad_window.assign(
                    grad_window + (
                        tf.where(
                            window == tf.reduce_max(
                                window, axis=[-1, -2], keepdims=True
                            ),
                            grad_value, 0.0
                        )
                    )
                )
                y += 1
            x += 1
        return [gradient]
