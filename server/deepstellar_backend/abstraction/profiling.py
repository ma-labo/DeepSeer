"""
Name   : profiling.py
Author : Zhijie Wang
Time   : 2021/7/7
"""

import numpy as np
import torch
import os
from tqdm import tqdm
from utils import PCAReduction, GMM, KMeans


class DeepStellar(object):
    def __init__(self, pca_dimension, abstract_state, state_vec, class_num=2, method='GMM'):
        """

        :param pca_dimension: reduced dimension
        :param abstract_state: # of abstract states
        :param state_vec: e.g. state_vec = [np.array(text_length, pca_dimension), ...]
        :param class_num: # of classes when classification
        """
        self.pca_dimesion = pca_dimension
        self.abstract_state = abstract_state
        self.pca = PCAReduction(pca_dimension)
        pca_data, _, _ = self.pca.create_pca(state_vec)
        if method == 'GMM':
            self.ast_model = GMM([pca_data], abstract_state, class_num)
        elif method == 'KMeans':
            self.ast_model = KMeans([pca_data], abstract_state, class_num)
        else:
            raise NotImplementedError('Unknown clustering method!')

    def get_trace(self, pca_data):
        """

        :param pca_data: pca_data = self.pca.do_reduction(state_vec)
        :return: trace: e.g. tr = [(1,2,3), (4,2,1,5,7), ...]
        """
        return self.ast_model.get_trace(pca_data)