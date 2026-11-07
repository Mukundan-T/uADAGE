"""
A class for a tied-weight autoencoder that can be used with Keras Tuners.

Usage:
	- constructor

November 5, 2026

*** This implementation of AdageHyperModel edits some things in the original version: ***
    - Removed many unused imports
    - Removed preT and init_weights parameters since unused
    - Added super().__init__()
    - * Important: Retrieval of hyperparameters from SeqAdage.tune_model() rather than
    defining a new search space
    - Pass 'tied' and 'act1' to LinkedAE

:authors: Georgia Doing, Mukundan Thanigaivelan
"""

import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras import optimizers
from m_adage import LinkedAE


class AdageHyperModel(kt.HyperModel):
    """
    Keras Tuner HyperModel for ADAGE. Hyperparameter definitions should be supplied 
    by SeqAdage.tune_model().
    """

    def __init__(self, input_shape):
        super().__init__()
        self.input_shape = input_shape

    def build(self, hp):
        enc_dim = hp.get("units")
        act1 = hp.get("act1")
        act2 = hp.get("act2")
        tied = hp.get("tied")
        dropout = hp.get("dropout")
        init = hp.get("init")
        kl1 = hp.get("kl1")
        kl2 = hp.get("kl2")
        lr = hp.get("lr")
        mm = hp.get("mm")

        model = LinkedAE.LinkedAE(
            input_dim = self.input_shape,
            encoding_dim = enc_dim,
            act1 = act1,
            act2 = act2,
            dropout = dropout,
            kl1 = kl1,
            kl2 = kl2,
            init = init,
            tied = tied
        )

        model.build((None, self.input_shape))
        optimizer = optimizers.SGD(learning_rate = lr, momentum = mm)
        
        model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.MeanSquaredError()
        )

        return model

    def fit(self, hp, model, *args, **kwargs):
        batch_size = hp.get("bs")
        shuffle = hp.get("shuffle")

        return model.fit(
            *args,
            batch_size = batch_size,
            shuffle = shuffle,
            **kwargs
        )
