"""
Mukundan Thanigaivelan 
UltraRegularizer class

October 18, 2026

A class to define a custom loss function penalty using ultrametric p-adic topology. Inherits
the Keras Regularizer class.

:author: Mukundan Thanigaivelan
"""

import tensorflow as tf
from keras.regularizers import Regularizer

class UltraRegularizer(Regularizer):
    def __init__(self, u1 = 0.0, u2 = 0.0, num_triplets = 256):
        self.u1 = float(u1)
        self.u2 = float(u2)
        self.num_triplets = num_triplets
    
    def __call__(self, weight_matrix):
        # Transpose to get nodes x genes  
        node_vectors = tf.transpose(weight_matrix)
        num_nodes = tf.shape(node_vectors)[0]
        
        # Create a random noise matrix of shape (256, num_nodes)
        shape_tensor = tf.stack([self.num_triplets, num_nodes])
        random_scores = tf.random.uniform(shape_tensor)
        
        # Get array of triplets where each triplet has different values
        _, unique_indices = tf.math.top_k(random_scores, k = 3)
        
        # Generates 256 random indices between 0 and num_nodes 
        # (repeats thrice for A, B, and C in calculation)
        idx_A = unique_indices[:, 0]
        idx_B = unique_indices[:, 1]
        idx_C = unique_indices[:, 2]
        
        # Pull 256 x 3 = 768 nodes as 256 triplets
        A = tf.gather(node_vectors, idx_A)
        B = tf.gather(node_vectors, idx_B)
        C = tf.gather(node_vectors, idx_C)
        
        # Compute pairwise distances --> each one is a vector of 256 distances
        d_AB = tf.sqrt(tf.reduce_sum(tf.square(A - B), axis = 1) + 1e-8)
        d_BC = tf.sqrt(tf.reduce_sum(tf.square(B - C), axis = 1) + 1e-8)
        d_AC = tf.sqrt(tf.reduce_sum(tf.square(A - C), axis = 1) + 1e-8)
        
        # Sort distances in descending order
        distances = tf.stack([d_AB, d_BC, d_AC], axis = 1)
        sorted_distances = tf.sort(distances, axis = 1, direction = "DESCENDING")
        
        # Get longest and 2nd-longest sides
        d1 = sorted_distances[:, 0]
        d2 = sorted_distances[:, 1]
        
        # Calculate and return weighted penalty
        if self.u1 != 0:
          return self.u1 * tf.reduce_mean(tf.abs(d1 - d2))
        
        return self.u2 * tf.reduce_mean(tf.square(d1 - d2))
    
    def get_config(self):
        return {
            "u1": self.u1,
            "u2": self.u2,
            "num_triplets": self.num_triplets
        }
