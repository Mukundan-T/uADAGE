"""
A class for an RNA-Seq based ADAGE model. Inherits Adage class.

Usage:
    - constructor
    - SeqAdage.train_model()

November 5, 2026

*** This implementation of SeqAdage edits some things in the original version: ***
    - Sets a configurable constant NOISE_FACTOR
    - * Important: Adds 'act1', 'tied', 'shuffle' for encoding, passed to LinkedAE
    - Defined _set_seed() to get NumPy, Python, and TensorFlow seeds for reproducibility
    - Explicitly did validation split through _split_train_validation()
    - Pre-training now changed to train for at least one epoch
    - ** Super important: x_train_noisy and x_train are swapped to correct order in 
    fine_tune_model() TWICE
    - More robust save_weights() method
    - * Important: 'tied' is tuned in tune_model()

:authors: Georgia Doing, Mukundan Thanigaivelan
"""

import os
import random

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import optimizers, losses

import keras_tuner

from m_adage import Adage, LinkedAE, AdageHyperModel


class SeqAdage(Adage.Adage):
    """
    Adage instance with associated linked denoising autoencoder (LinkedAE), 
    training data, and hyperparameter settings. Weight matrix used to 
    assign Adage atributes, namely high weight genes and gene set 
    enrichment results.

    Attributes:
        autoencoder (LinkedAE): The tied-weights denoising autoencoder
        all_comp (pd.df): Pandas DataFrame of training data
        gene_num (int): Number of genes (deatures, dims) in input data
        enc_dim (int): Encoding dimension, number of nodes in hidden layer
        epochs (int): Number of epochs for training
        seed (int): Random seed
        batch_size (int): Number of samples per training batch
        mm (flt): Momentum for fitting, 0-1
        lr (flt): Learning rate, 0-1
        v (int): Verbose, 1-True, 0-False
        u1 (flt): Amount of U1 (ultrametric) regularization, if any
        u2 (flt): Amount of U2 (ultrametric) regularization, if any

    Methods:
        prep_data() : prepares data by introducing noise
        train_model(): fits autoencoder to training data
        tune_model(): hyperparameter search
        save_weights(out_dir, out_file): write weights, bias and loss to csvs

    To Do:
        * check autoencoder weights update after training/fitting
        * reconcile with Adage constructor, super().__init__()
        * prep E. coli KEGG, GO, operon and regulon gene sets
        * implement transfer models
    """

    # Set a changable noise factor for injecting noise
    NOISE_FACTOR = 0.1

    def __init__(self, input_file, seed = 100, enc_dim = 10, kl1 = 0.0, kl2 = 0.0,
                 act1 = "tanh", act2 = "tanh", tied = True, epochs = 50,
                 init = "glorot_uniform", batch_size = 10, dropout = 0.0, mm = 0.0,
                 lr = 0.01, v = 1, u1 = 0.0, u2 = 0.0, shuffle = True
                ):
        """
        Constructor for SeqAdage with hyperparameters.

        This class inherits the Adage class, instantiating a linked
        denoising autoencoder with a set of hyperparameter values, fitting
        it to a training dataset and saving weight and bias matrices to files.

        Args:
            input_file (str): Name of CSV with training data
            seed (int): Random seed
            enc_dim (int): Encoding dimension, number of nodes in hidden layer
            kl1 (flt): Amount of L1 regularization, 0-1
            kl2 (flt): Amount of L2 regularization, 0-1
            act1 (str): Activation function during encoding
            act2 (str): Activation function during decoding
            tied (bool): Whether encoder and decoder weights are tied
            epochs (int): Number of epochs for training
            init (str): Inititalization function for random weight distribution
            batch_size (int): Number of samples per training batch
            dropout (flt): Proportion of input values randomly dropped
            mm (flt): Momentum for fitting, 0-1
            lr (flt): Learning rate, 0-1
            v (int): Verbose, 1-True, 0-False
            u1 (flt): Amount of U1 (ultrametric) regularization, if any
            u2 (flt): Amount of U2 (ultrametric) regularization, if any
            shuffle: If the batches should be shuffled or not
        """
        self.input_file = input_file

        self.seed = seed
        self.encoding_dim = enc_dim

        self.act1 = act1
        self.act2 = act2

        self.tied = tied

        self.epochs = epochs
        self.init = init
        self.batch_size = batch_size

        self.dropout = dropout
        self.mm = mm
        self.lr = lr

        self.kl1 = kl1
        self.kl2 = kl2
        self.u1 = u1
        self.u2 = u2

        self.v = v
        self.shuffle = shuffle

        if kl1 != 0:
            self.reg_type = "l1"
        elif kl2 != 0:
            self.reg_type = "l2"
        elif u1 != 0:
            self.reg_type = "u1"
        elif u2 != 0:
            self.reg_type = "u2"
        else:
            self.reg_type = None

        self.all_comp = pd.read_csv(
            input_file,
            index_col = 0,
        )

        self.gene_num = self.all_comp.shape[0]
        self.autoencoder = self._build_autoencoder()
        self.history = None

    def _set_seed(self, seed = None):
        """
        Set NumPy, Python, and TensorFlow seeds.
        """
        if seed is None:
            seed = self.seed

        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)

    def _build_autoencoder(self):
        """
        Construct an ADAGE autoencoder using current hyperparameters.
        """
        model = LinkedAE.LinkedAE(
            input_dim = self.gene_num,
            encoding_dim = self.encoding_dim,
            act1 = self.act1,
            act2 = self.act2,
            dropout = self.dropout,
            kl1 = self.kl1,
            kl2 = self.kl2,
            init = self.init,
            tied = self.tied,
            u1 = self.u1,
            u2 = self.u2,
            reg_type = self.reg_type
        )
        return model

    def get_act(self):
        """
        Return encoder activation function.
        """
        return self.act1

    def prep_data(self):
        """
        Prepare clean and noisy training data.
        """
        self._set_seed()
        all_comp_np = self.all_comp.values.astype(np.float32)

        # Transpose to samples × genes
        x_clean = all_comp_np.T

        noise = self.NOISE_FACTOR * np.random.normal(
            loc = 0.0,
            scale = 1.0,
            size = x_clean.shape,
        )

        x_noisy = x_clean + noise
        x_noisy = np.clip(x_noisy, 0.0, 1.0)

        return (x_clean, x_noisy)

    def _split_train_validation(self, x_clean, x_noisy, 
                                validation_fraction = 0.1,
                                seed = None
                               ):
        """
        Create a random training/validation split.
        """
        if seed is None:
            seed = self.seed

        n_samples = x_clean.shape[0]

        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)

        n_val = max(1, int(round(n_samples * validation_fraction)))
        n_val = min(n_val, n_samples - 1)

        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        return (
            x_noisy[train_idx],
            x_clean[train_idx],
            x_noisy[val_idx],
            x_clean[val_idx]
        )

    def train_model_variable(self, num_epochs):
        """
        Train the autoencoder for a specified number of epochs.
        """
        x_clean, x_noisy = self.prep_data()
        
        (
            x_train_noisy,
            x_train_clean,
            x_val_noisy,
            x_val_clean
        ) = self._split_train_validation(
            x_clean,
            x_noisy
        )

        optimizer = optimizers.SGD(
            learning_rate = self.lr,
            momentum = self.mm,
        )

        self.autoencoder.compile(
            optimizer = optimizer,
            loss = losses.MeanSquaredError()
        )

        self.history = self.autoencoder.fit(
            x = x_train_noisy,
            y = x_train_clean,
            validation_data = (
                x_val_noisy,
                x_val_clean
            ),
            epochs = num_epochs,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            verbose = self.v,
        )

        print("Fitted")
        return self.autoencoder

    def train_model(self):
        """
        Train for self.epochs.
        """
        return self.train_model_variable(self.epochs)

    def pre_train_model(self):
        """
        Train for approximately one tenth of configured epochs
        (or at least 1 if less).

        Always trains for at least one epoch.
        """
        pretrain_epochs = max(1, self.epochs // 10)
        return self.train_model_variable(pretrain_epochs)

    def fine_tune_model(self, map_file, new_data, kl1 = None, kl2 = None,
                        act1 = None, act2 = None, epochs = None,
                        batch_size = None, dropout = None, mm = None,
                        lr = None, init = None
                       ):
        """
        Fits the autoencoder model to new training set.
        """
        # Update hps
        if act1 is not None:
            self.act1 = act1
        if act2 is not None:
            self.act2 = act2
        if epochs is not None:
            self.epochs = epochs
        if batch_size is not None:
            self.batch_size = batch_size
        if dropout is not None:
            self.dropout = dropout
        if mm is not None:
            self.mm = mm
        if lr is not None:
            self.lr = lr
        if kl1 is not None:
            self.kl1 = kl1
        if kl2 is not None:
            self.kl2 = kl2
        if init is not None:
            self.init = init
        
        if self.kl1 != 0:
            self.reg_type = "l1"
        elif self.kl2 != 0:
            self.reg_type = "l2"
        
        # Get weights after pre-training
        weights_prev, b_weights_prev, d_weights_prev = self.autoencoder.get_weights()[0:3]

        # Load mapping file between pre-training and fine-tuning domains
        mapper = pd.read_csv(map_file, index_col = 0)
        new_weights = weights_prev.T.dot(mapper).T
        new_d_weights = d_weights_prev.T.dot(mapper).T

        # Load new data
        np.random.seed(self.seed)
        self.all_comp = pd.read_csv(new_data, index_col = 0)
        self.gene_num = self.all_comp.shape[0]
        
        x_train, x_train_noisy = self.prep_data()

        # Initialize a new data with new input shape
        autoencoder2 = self._build_autoencoder()
        
        optim = optimizers.SGD(
            learning_rate = self.lr, 
            momentum = self.mm
        )
        autoencoder2.compile(
            optimizer = optim, 
            loss = losses.MeanSquaredError()
        )
        
        history = autoencoder2.fit(
            x = x_train_noisy, 
            y = x_train, 
            epochs = 1,
            batch_size = self.batch_size,
            validation_split = 0.1,
            verbose = self.v,
            shuffle = self.shuffle
        )
        
        # Get initialized weights as background distribution
        weights_tmp, b_weights_tmp, d_weights_tmp = autoencoder2.get_weights()[0:3]
        new_weights[new_weights == 0] = weights_tmp[new_weights == 0]
        new_d_weights[new_d_weights == 0] = d_weights_tmp[new_d_weights == 0]

        # Initialize to weights from mapping and background
        autoencoder2.set_weights([new_weights, b_weights_prev, new_d_weights])

        # Fine-tune training
        history = autoencoder2.fit(
          x = x_train_noisy, 
          y = x_train, 
          epochs = self.epochs,
          batch_size = self.batch_size,
          validation_split = 0.1,
          verbose = self.v
        )
        
        print("fine-tuned")
        self.autoencoder = autoencoder2
        self.history = history
        
        return autoencoder2

    def save_weights(self, out_dir, out_file):
        """
        Save encoder weights, encoder bias, and training history.
        """

        if self.history is None:
            raise RuntimeError(
                "Cannot save training history before training."
            )

        weights_dir = os.path.join(out_dir, "weights")
        bias_dir = os.path.join(out_dir, "bias")
        loss_dir = os.path.join(out_dir, "loss")
        val_loss_dir = os.path.join(out_dir, "val_loss")

        for directory in (
            weights_dir,
            bias_dir,
            loss_dir,
            val_loss_dir
        ):
            os.makedirs(directory, exist_ok = True)

        weights = (
            self.autoencoder.encoder.get_weights()[0]
        )
        bias = (
            self.autoencoder.encoder.get_weights()[1]
        )

        np.savetxt(
            os.path.join(
                weights_dir,
                f"{out_file}_ew_da.csv",
            ),
            weights,
            fmt = "%s",
            delimiter = ",",
        )
        np.savetxt(
            os.path.join(
                bias_dir,
                f"{out_file}_eb_da.csv",
            ),
            bias,
            fmt = "%s",
            delimiter = ",",
        )
        np.savetxt(
            os.path.join(
                loss_dir,
                f"{out_file}_l_da.csv",
            ),
            np.asarray(
                self.history.history.get(
                    "loss",
                    [],
                )
            ),
            fmt = "%s",
            delimiter = ",",
        )
        np.savetxt(
            os.path.join(
                val_loss_dir,
                f"{out_file}_vl_da.csv",
            ),
            np.asarray(
                self.history.history.get(
                    "val_loss",
                    [],
                )
            ),
            fmt = "%s",
            delimiter = ",",
        )

    def tune_model(self, seed = None, u = [50, 150, 10], d = [0.0, 0.9, 0.1],
                   a1 = ["sigmoid", "tanh", "relu", "celu"],
                   a2 = ["sigmoid", "tanh", "relu", "celu"],
                   i = ["glorot_uniform", "glorot_normal"], 
                   k1 = [0.0, 0.15, 0.005], k2 = [0.0, 0.9, 0.05], 
                   l = [0.001, 0.1, 0.01], b = [5, 100, 5], m = [0.0, 0.9, 0.1], 
                   sh = [False, True], tied = [True, False], u1 = [0.0, 0.15, 0.005],
                   u2 = [0.0, 0.9, 0.05]
                  ):
        """
        Tune ADAGE hyperparameters.
        """
        if seed is None:
            seed = self.seed

        self._set_seed(seed)
        hp = keras_tuner.HyperParameters()

        hp.Int(
            name = "units",
            min_value = u[0],
            max_value = u[1],
            step = u[2]
        )
        hp.Float(
            name = "dropout",
            min_value = d[0],
            max_value = d[1],
            step = d[2]
        )
        
        hp.Choice("act1", values = a1)
        hp.Choice("act2", values = a2)

        hp.Choice("tied", values = tied)
        hp.Choice("init", values = i)

        hp.Float(
            "kl1",
            min_value = k1[0],
            max_value = k1[1],
            step = k1[2]
        )
        hp.Float(
            "kl2",
            min_value = k2[0],
            max_value = k2[1],
            step = k2[2]
        )

        hp.Float(
            "lr",
            min_value = l[0],
            max_value = l[1],
            step = l[2]
        )

        hp.Int(
            "bs",
            min_value = b[0],
            max_value = b[1],
            step = b[2]
        )
        hp.Float(
            "mm",
            min_value = m[0],
            max_value = m[1],
            step = m[2]
        )
        hp.Choice("shuffle", values = sh)

        hp.Float(
          "u1",
          min_value = u1[0],
          max_value = u1[1],
          step = u1[2]
        )
        hp.Float(
          "u2",
          min_value = u2[0],
          max_value = u2[1],
          step = u2[2]
        )
        
        tuner = keras_tuner.Hyperband(
            hypermodel = AdageHyperModel.AdageHyperModel(input_dim = self.gene_num),
            hyperparameters = hp,
            objective = "val_loss",
            max_epochs = self.epochs,
            overwrite = True,
            directory = "/scratch/group/p.cis262240.000/mukundan/uADAGE/Py/muk_uADAGE/data/hyperband_search_dir",
            project_name = (
                os.path.basename(self.input_file).removesuffix(".csv")
                + "_"
                + str(seed)
            )
        )

        x_clean, x_noisy = self.prep_data()

        (
            x_train_noisy,
            x_train_clean,
            x_val_noisy,
            x_val_clean
        ) = self._split_train_validation(
            x_clean,
            x_noisy,
            seed = seed
        )

        tuner.search(
            x_train_noisy,
            x_train_clean,
            validation_data = (
                x_val_noisy,
                x_val_clean,
            )
        )

        tuner.results_summary()
        num_trials = len(tuner.oracle.trials.values())
        best_hps = tuner.get_best_hyperparameters(num_trials)
        
        model = tuner.hypermodel.build(best_hps[0])
        hist = tuner.hypermodel.fit(
    	    best_hps[0], 
    		model,
    	    x = x_train_noisy, 
    		y = x_train_clean
        )

        return best_hps, tuner
