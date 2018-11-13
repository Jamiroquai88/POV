#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
from time import sleep

import numpy as np
from scipy import spatial


def get_eer(tar, non):
    """ Obtain Equal Error Rate.

    Args:
        tar (np.array|List[float]): target scores as 1 dimensional vector
        non (np.array|List[float]): non-target scores as 1 dimensional vector

    Returns:
        tuple: EER, EER threshold, EER misses, EER false alarms
    """
    target_scores = sorted(np.array(tar).flatten())
    imposter_scores = sorted(np.array(non).flatten())
    min_tar_score, max_imp_score = min(target_scores), max(imposter_scores)
    # system did not make any mistake, EER algorithm would return bullsense
    # return all zeros and calculate the threshold as unweighted average of min target score and max impostor score
    if max_imp_score < min_tar_score:
        return 0.0, (max_imp_score + min_tar_score) / 2, 0, 0
    target_count, imposter_count = len(target_scores), len(imposter_scores)
    eer = 0.0
    eer_dif = float('inf')
    imp_index = 0
    tar_index = 0
    eer_thr = None
    eer_miss, eer_fa = None, None
    if target_scores:
        threshold = float(target_scores[0])
    elif imposter_scores:
        threshold = float(imposter_scores[0])
    else:
        raise ValueError('Cannot evaluate two empty lists.')

    while True:
        while imp_index < imposter_count and threshold >= imposter_scores[imp_index]:
            imp_index = imp_index + 1
        while tar_index < target_count and threshold >= target_scores[tar_index]:
            tar_index = tar_index + 1
        miss = tar_index
        fa = imposter_count - imp_index
        if target_count > 0:
            frr = 1.0 * miss / target_count if target_count else 0.0
        else:
            frr = 0.0
        far = 1.0 * fa / imposter_count if imposter_count else 0.0
        if abs(frr - far) < eer_dif:
            eer_dif = abs(frr - far)
            eer = (frr + far) / 2 * 100
            eer_thr = threshold
            eer_miss, eer_fa = miss, fa
        if imp_index < imposter_count and tar_index < target_count:
            threshold = float(min(imposter_scores[imp_index], target_scores[tar_index]))
        else:
            break
    if eer_thr is not None and not isinstance(eer_thr, float):
        raise ValueError('Invalid threshold value %s (type %s), float or None expected.' % (eer_thr, type(eer_thr)))
    if not isinstance(eer, float):
        raise ValueError('Invalid threshold value %s (type %s), float expected.' % (eer, type(eer)))
    return eer, eer_thr, eer_miss, eer_fa


def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def cosine_similarity(v1, v2):
    """Compute similarity distance of v1 to v2: (v1 dot v2)/(||v1||*||v2||).

    Args:
        v1 (np.array): first vector
        v2 (np.array): second vector

    Returns:
        float: cosine similarity
    """
    return 1 - spatial.distance.cosine(v1, v2)


def safe_mkdir(path):
    """Behaviour similar to mkdir -p. If folder already exist, silently pass.
    Fail only if the folder doesn't exist and cannot be created.

    Args:
        path (string_types): target path

    Raises:
        OSError
    """
    # avoid race condition
    while True:
        try:
            if os.path.isdir(path):
                return
            os.makedirs(path)
            break
        except FileExistsError:
            sleep(0.1)
