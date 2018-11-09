#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import os
import pickle
import logging
import argparse

from keras.utils import to_categorical
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

from keras_model import create_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# set seed
np.random.seed(42)


def plot_curves():
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', required=True,
                        help='path to the input directory in format Person1/image1.jpg, Person4/image3.jpg,'
                             'expects already prepared data in pickle format')
    parser.add_argument('--imgs4validation', required=False, default=5, type=int,
                        help='number of images used for cross-validation')
    parser.add_argument('--num-persons', required=False, default=0, type=int, help='number of persons for training')

    args = parser.parse_args()

    persons = os.listdir(args.input_dir)
    persons_dict = {}
    for idx, person in enumerate(persons):
        pickle_file_path = os.path.join(args.input_dir, '{}.pkl'.format(person))
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                try:
                    persons_dict[person] = pickle.load(f)
                except pickle.UnpicklingError:
                    logger.warning('pickle data was truncated for file `{}`.'.format(pickle_file_path))
                    continue

    train_data, test_data, train_labels, test_labels, num_persons = [], [], [], [], 0
    for person in persons_dict:
        num_imgs = len(persons_dict[person])
        if num_imgs > 3 * args.imgs4validation:
            if args.num_persons != 0:
                if num_persons > args.num_persons:
                    break
            np.random.shuffle(persons_dict[person])
            test_data.extend([x[0] for x in persons_dict[person][:args.imgs4validation]])
            train_data.extend([x[0] for x in persons_dict[person][args.imgs4validation:]])
            test_labels.extend([person for x in range(args.imgs4validation)])
            train_labels.extend([person for x in range(num_imgs - args.imgs4validation)])

    label_encoder = LabelEncoder().fit(test_labels)
    test_encoded_labels = label_encoder.transform(test_labels)
    train_encoded_labels = label_encoder.transform(train_labels)

    assert len(test_encoded_labels) == len(test_data)
    assert len(train_encoded_labels) == len(train_data)
    print(test_data[0].shape)

    # model = create_model(input_shape=train_data[0].shape, num_classes=len(set(test_labels)))
    from resnet_model import resnet_v2
    model = resnet_v2(input_shape=train_data[0].shape, depth=56, num_classes=len(set(test_labels)))
    batch_size = 128
    epochs = 50
    model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    train_encoded_labels = to_categorical(train_encoded_labels)
    test_encoded_labels = to_categorical(test_encoded_labels)
    print(train_data.shape)
    print(train_encoded_labels.shape)
    history = model.fit(train_data, train_encoded_labels, batch_size=batch_size, epochs=epochs,
                        verbose=1, validation_data=(test_data, test_encoded_labels))

    model.evaluate(test_data, test_encoded_labels)






