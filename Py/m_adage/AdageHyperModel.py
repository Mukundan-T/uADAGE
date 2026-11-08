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

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

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
        u1 = hp.get("u1")
        u2 = hp.get("u2")

        # Define the correct reg_type
        if kl1 != 0:
            reg_type = "l1"
        elif kl2 != 0:
            reg_type = "l2"
        elif u1 != 0:
            reg_type = "u1"
        elif u2 != 0:
            reg_type = "u2"
        else:
            reg_type = None
        
        model = LinkedAE.LinkedAE(
            input_dim = self.input_dim,
            encoding_dim = enc_dim,
            act1 = act1,
            act2 = act2,
            dropout = dropout,
            kl1 = kl1,
            kl2 = kl2,
            init = init,
            tied = tied,
            u1 = u1,
            u2 = u2,
            reg_type = reg_type
        )

        model.build((None, self.input_dim))
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
