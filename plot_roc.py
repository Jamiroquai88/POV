#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import argparse

import numpy as np

from train_model import plot_roc


COLORS = ['b', 'r', 'g', 'black']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list-paths', nargs='+', required=True)
    parser.add_argument('-n', '--list-names', nargs='+', required=True)

    args = parser.parse_args()
    assert len(args.list_paths) == len(args.list_names)

    casia, lfw = {}, {}
    for path_idx, path in enumerate(args.list_paths):
        if os.path.exists(os.path.join(path, 'tar.npy')) and os.path.exists(os.path.join(path, 'non.npy')):
            casia[path] = (list(np.load(os.path.join(path, 'tar.npy'))), list(np.load(os.path.join(path, 'non.npy'))))
            plot_roc(casia[path][0], casia[path][1], label=args.list_names[path_idx],
                     color=COLORS[path_idx], output_path='roc.png')
        # if os.path.exists(os.path.join(path, 'tar_lfw.npy')) and os.path.exists(os.path.join(path, 'non_lfw.npy')):
        #     lfw[path] = (
        #         list(np.load(os.path.join(path, 'tar_lfw.npy'))), list(np.load(os.path.join(path, 'non_lfw.npy'))))
        #     plot_roc(lfw[path][0], lfw[path][1], label=args.list_names[path_idx],
        #              color=COLORS[path_idx], output_path='roc_lfw.png')
