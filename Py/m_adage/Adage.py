"""
A class for a tied-weight autoencoder.

Usage:
    - constructor

November 5, 2026

*** This implementation of Adage edits some things in the original version: ***
    - Initializes weights to 0, then retrieves them through refresh_analysis_state()
    - Defines refresh_analysis_state() to retrieve weights and losses
    - Error checking on history attribute
    - High/low weight genes moved to refresh_hwg_masks()
    - self.activities is only calculated when set_act() is called
    - calc_act() now takes (genes, samples), not (samples, genes)
    - calc_act() now uses the actual encoder if possible, not just apply a linear activation
    - Applied Benjamini-Hochberg multiple-testing correction

:authors: Georgia Doing, Mukundan Thanigaivelan
"""

import numpy as np
import pandas as pd

from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


class Adage(object):
    """
    Adage (Analysis using Denoising Autoencoder of Gene Expression) class
    with the DAE, training data and biological enrichments of 'high weight'
    gene sets extracted from the hidden layer. Designed to facilitate 
    biological assessment as done in Tan and Doing et al, 2017, Cell Systems
    (<https://www.cell.com/cell-systems/fulltext/S2405-4712(17)30231-4>)

    Attributes:
        - autoencoder (keras.Model): The tied-weights denoising autoencoder
        - history (keras.History): Training loss over epochs
        - weights (np.array): Encoding weight matrix
        - loss (array): Training loss from history
        - val_loss (array): Validation loss from history
        - hw_genes (array): Arrays of high weight gene sets
        - hw_genes_all (array): Arrays of high and low weight gene sets
        - hw_genes_down (array): Arrays of low weight gene sets
        - compendium (np.array): Training compendium
        - activities (array): Encoded training compendium
        - kegg_ps (list): KEGG enrichment p-values
        - go_ps (list): GO term enrichment p-values
        - regs_ps (list): Regulon enrichment p-values
        - ops_ps (list): Operon enrichment p-values
    """
    
    def __init__(self, autoencoder, history, train_comp):
        """
        Class for calculating enrichment of biological gene sets in high
        and low weight gene sets from a denoising autoencoder trained on 
        gene expression data.

        Args:
            - autoencoder (keras.Model): The tied-weights denoising autoencoder
            - history (keras.History): Training loss over epochs
            - train_comp (np.array): Training compendium
        """
        self.autoencoder = autoencoder
        self.history = history
        self.compendium = train_comp

        self.hwg_cutoff = 2.5

        self.weights = None
        self.loss = None
        self.val_loss = None

        self.hw_genes = None
        self.hw_genes_down = None
        self.hw_genes_all = None

        self.activities = None

        self.kegg_ps = []
        self.go_ps = []
        self.regs_ps = []
        self.ops_ps = []

        self.refresh_analysis_state()

    def refresh_analysis_state(self):
        """
        Refresh encoder weights, training history, and high/low-weight masks.
        """
        self.weights = self._get_encoder_weights()

        if self.history is not None:
            self.loss = self.history.history.get("loss", [])
            self.val_loss = self.history.history.get("val_loss", [])
        else:
            self.loss = []
            self.val_loss = []

        self._refresh_hwg_masks()

    def _get_encoder_weights(self):
        """
        Retrieve encoder kernel.

        Supports the refactored LinkedAE architecture as well as
        the original weight-list fallback.
        """
        if hasattr(self.autoencoder, "encoder"):
            return self.autoencoder.encoder.get_weights()[0]

        return self.autoencoder.get_weights()[0]

    def _refresh_hwg_masks(self):
        """
        Calculate high- and low-weight genes independently for each
        hidden node.

        High:
            weight > cutoff * standard deviation

        Low:
            weight < -cutoff * standard deviation
        """
        stds = np.std(
            self.weights,
            axis = 0,
            keepdims = True,
        )

        upper = self.hwg_cutoff * stds
        lower = self.hwg_cutoff * stds

        self.hw_genes = self.weights > upper
        self.hw_genes_down = self.weights < lower

        self.hw_genes_all = np.concatenate(
            (
                self.hw_genes,
                self.hw_genes_down,
            ),
            axis = 1
        )

    def set_hwg_cutoff(self, x):
        """
        Recalculate high/low-weight gene sets using x standard deviations.
        """
        self.hwg_cutoff = x
        self.refresh_analysis_state()
        return self.hw_genes_all

    def _validate_gene_expression(self, gene_exp):
        """
        Validate genes × samples input.
        """
        gene_exp = np.asarray(gene_exp)
        expected_genes = self.weights.shape[0]
        
        if gene_exp.shape[0] != expected_genes:
            raise ValueError(
                f"Expected {expected_genes} genes, got {gene_exp.shape[0]}."
            )
        
        return gene_exp

    def calc_act(self, gene_exp):
        """
        Calculate ADAGE node activities.
        """
        gene_exp = self._validate_gene_expression(gene_exp)
        model_input = gene_exp.T

        if hasattr(self.autoencoder, "encode"):
            activities = self.autoencoder.encode(
                model_input,
                training = False
            )
            return activities.numpy()

        # Fallback for linear models
        return np.dot(model_input, self.weights)

    def set_act(self, comp):
        """
        Calculate and store encoded activities. Input must be 
        genes × samples.
        """
        self.activities = self.calc_act(comp)
        return self.activities

    def calc_enrich(self, path_file, all_sigs = True):
        """
        Calculate enrichment of gene sets for ADAGE high/low-weight
        signatures.

        For each ADAGE signature:

        1. Calculate a hypergeometric enrichment p-value for every gene set.
        2. Apply Benjamini-Hochberg FDR correction across all tested gene sets.
        3. Return -log10(adjusted p-value).
        """
        self.refresh_analysis_state()
        hw_temp = self.hw_genes_all

        gene_sets = pd.read_csv(path_file, header = None, sep = "\t")

        if gene_sets.shape[1] < 3:
            raise ValueError(
                "Gene-set file must contain at least three columns: "
                "gene-set name and a gene list in column 2."
            )

        gene_set_names = gene_sets.iloc[:, 0].astype(str).tolist()
        all_enrichment_scores = []

        population_size = self.weights.shape[0]

        for signature_idx in range(hw_temp.shape[1]):
            signature_mask = hw_temp[:, signature_idx]
            sig_genes = set(
                self.compendium.index[
                    np.where(signature_mask)[0]
                ]
            )
            signature_size = len(sig_genes)
          
            raw_p_values = []
            for gene_set_idx in range(gene_sets.shape[0]):
                path_genes = set(
                    str(gene_sets.iloc[gene_set_idx, 2]).split(";")
                )
                overlap = len(
                    sig_genes & path_genes
                )
                gene_set_size = len(path_genes)

                if (
                    signature_size == 0
                    or gene_set_size == 0
                    or overlap == 0
                ):
                    p_value = 1.0
                else:
                    p_value = hypergeom.sf(
                        overlap - 1,
                        population_size,
                        signature_size,
                        gene_set_size,
                    )
                    if not np.isfinite(p_value):
                        p_value = 1.0

                raw_p_values.append(p_value)

            raw_p_values = np.asarray(raw_p_values, dtype = float)

            # Benjamini-Hochberg correction
            _, adjusted_p_values, _, _ = multipletests(
                raw_p_values,
                alpha = 0.05,
                method = "fdr_bh"
            )

            safe_adjusted_p = np.clip(
                adjusted_p_values,
                np.finfo(float).tiny,
                1.0
            )
            enrichment_scores = -np.log10(safe_adjusted_p)
            all_enrichment_scores.append(enrichment_scores)

        return pd.DataFrame(
            all_enrichment_scores,
            columns = gene_set_names,
        )

    def set_kegg(self, path_file, all = True):
        """
        Calculate and store KEGG enrichment scores.
        """
        self.kegg_ps = self.calc_enrich(
            path_file,
            all_sigs = all,
        )
        return self.kegg_ps

    def set_go(self, path_file, all = True):
        """
        Calculate and store GO enrichment scores.
        """
        self.go_ps = self.calc_enrich(
            path_file,
            all_sigs = all,
        )
        return self.go_ps

    def set_reg(self, path_file, all = True):
        """
        Calculate and store Regulon enrichment scores.
        """
        self.regs_ps = self.calc_enrich(
            path_file,
            all_sigs = all,
        )
        return self.regs_ps

    def set_op(self, path_file, all = True):
        """
        Calculate and store Operon enrichment scores.
        """
        self.ops_ps = self.calc_enrich(
            path_file,
            all_sigs = all,
        )
        return self.ops_ps
