"""
A class for dense layer transposed from another dense layer. Used for 
tied-weights autoencoder. Inherits keras Layer class.

Usage:
	- constructor

November 5, 2026

*** This implementation of DenseTranspose edits some things in the original version: ***
    - Pass in **kwargs to super().__init__()

:authors: Georgia Doing, Mukundan Thanigaivelan
"""

import tensorflow as tf
from tensorflow import keras

class DenseTranspose(keras.layers.Layer):
    """
    DenseTranspose is a class for linked decoders. It inherits
    the keras Layer class.

    Attributes:
        - dense (keras.layers.Dense): The encoding layer to transpose
        - activation (str): The decoder activation function 
        
    Methods:
        - build(batch_input_shape): sets biases and shape
        - call(inputs) : Transpose encoding weights, returns activations
        - get_config()

    To Do:
        * from_config
    """

    def __init__(self, dense, activation = "tanh", **kwargs):
        super().__init__(**kwargs)

        self.dense = dense
        self.activation_name = activation
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        output_dim = int(self.dense.kernel.shape[0])

        self.biases = self.add_weight(
            name = "bias",
            shape = (output_dim,),
            initializer = "zeros",
            trainable = True
        )
        super().build(input_shape)

    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.kernel, transpose_b = True)
        return self.activation(z + self.biases)

    def get_config(self):
        config = super().get_config()
        
        config.update({
            "dense": keras.layers.serialize(self.dense),
            "activation": keras.activations.serialize(self.activation),
            }
        )
        return config
