"""
A class for a tied-weight autoencoder. Inherits keras.Model class 
and uses DenseTranspose.

Usage:
    - constructor

November 5, 2026

*** This implementation of LinkedAE edits some things in the original version: ***
    - Inherits keras.Model, not keras.Sequential (for more versatile architectures)
    - Parameter additions:
        - Configurable encoder activation
        - Required input dimension
        - Configurable tied/untied weights
        - **kwargs
    - Weights initialized to 'glorot_uniform', not 'tanh'
    - Defined instance variables
    - * Important: Activation #1 is applied to encoding layer
    - * Important: Tied-weights implemented --> if tied == True, transposed Dense layer,
    else if tied == False, conventional keras.layers.Dense layer
    - Custom encode() and decode() methods

:authors: Georgia Doing, Mukundan Thanigaivelan
"""

from tensorflow import keras
from tensorflow.keras import regularizers
from m_adage import DenseTranspose


class LinkedAE(keras.Model):
    """
    LinkedAE is a class for a linked, tied-weights autoencoder. It inherits
    the keras Model class.

    Attributes:
        - act1 (str): The encoder activation function
        - dense_1 (keras.layers.Dense): The encoding layer
        - d (keras.layers.Dropout): A dropout layer
        - act2 (str): The decoder activation function
        - decoder (DenseTranspose): The decoding layer
        
    Methods:
        call(inputs) : Initialized the LinkedAE with an input dataset

    To Do:
        * implement pre-training option
    """
    
    def __init__(self, input_dim, encoding_dim = 10, act1 = "tanh", act2 = "tanh", dropout = 0.0,
                 kl1 = 0.0, kl2 = 0.0, init = "glorot_uniform", tied = True, **kwargs
                ):
        """
        Tied weight-autoencoder class.
        """
        super().__init__(**kwargs)

        self.encoding_dim = encoding_dim
        self.act1 = act1
        self.act2 = act2
        self.dropout_rate = dropout
        self.kl1 = kl1
        self.kl2 = kl2
        self.init = init
        self.tied = tied

        regularizer = regularizers.l1_l2(
            l1 = kl1,
            l2 = kl2,
        )

        self.dropout_layer = keras.layers.Dropout(
            dropout,
            name = "input_dropout",
        )

        self.encoder = keras.layers.Dense(
            encoding_dim,
            activation = act1,
            kernel_regularizer = regularizer,
            kernel_initializer = init,
            name = "encoder",
        )

        if tied:
            self.decoder = DenseTranspose.DenseTranspose(
                self.encoder,
                activation = act2,
                name = "decoder",
            )
        else:
            self.decoder = keras.layers.Dense(
                units = input_dim,
                activation = act2,
                kernel_regularizer = regularizer,
                kernel_initializer = init,
                name = "decoder",
            )

    # if tied == True --> decoder.units == None : False AND True
    # if tied == False --> decoder.units == input_dim != None : True and False
    # So the conditional in build never runs?
    # Then what is the purpose of build()?
    # We comment this for now!
  
    # def build(self, input_shape):
    #     """
    #     For an untied autoencoder, the decoder output dimension must equal
    #     the input dimension, which isn't known until build time.
    #     """
    #     input_dim = int(input_shape[-1])

    #     if not self.tied and self.decoder.units is None:
    #         regularizer = regularizers.l1_l2(
    #             l1 = self.kl1,
    #             l2 = self.kl2,
    #         )

    #         self.decoder = keras.layers.Dense(
    #             input_dim,
    #             activation = self.act2,
    #             kernel_regularizer = regularizer,
    #             kernel_initializer = self.init,
    #             name = "decoder",
    #         )

    #     super().build(input_shape)

    def encode(self, inputs, training = False):
        """
        Return hidden-layer activities using the actual encoder. Dropout is only 
        applied during training.
        """
        x = self.dropout_layer(inputs, training = training)
        return self.encoder(x)

    def decode(self, encoded, training = False):
        return self.decoder(encoded, training = training)

    def call(self, inputs, training = False):
        encoded = self.encode(inputs, training = training)
        return self.decode(encoded, training = training)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "encoding_dim": self.encoding_dim,
                "act1": self.act1,
                "act2": self.act2,
                "dropout": self.dropout_rate,
                "kl1": self.kl1,
                "kl2": self.kl2,
                "init": self.init,
                "tied": self.tied,
            }
        )

        return config
