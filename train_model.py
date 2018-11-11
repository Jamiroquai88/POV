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
import subprocess

from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder

from utils import get_eer, l2_norm, cosine_similarity
from resnet_model import resnet_v2
from keras_model import create_model

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', level=logging.INFO)

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
    plt.savefig('loss_curves.png')

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig('acc_curves.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', required=True,
                        help='path to the input directory in format Person1/image1.jpg, Person4/image3.jpg,'
                             'expects already prepared data in pickle format')
    parser.add_argument('--imgs4validation', required=False, default=5, type=int,
                        help='number of images used for cross-validation')
    parser.add_argument('--num-persons', required=False, default=0, type=int, help='number of persons for training')
    parser.add_argument('--num-ver-persons', required=False, default=0, type=int,
                        help='number of persons for verification')
    parser.add_argument('--learning-rate', required=False, default=0.0001, type=float, help='learning rate')
    parser.add_argument('--no-use-gpu', required=False, default=False, action='store_true',
                        help='use GPU for training')
    parser.add_argument('--no-train-model', required=False, default=False, action='store_true')
    
    args = parser.parse_args()
    if args.no_use_gpu is not True:
        try:
            command = 'nvidia-smi --query-gpu=memory.free,memory.total --format=csv |tail -n+2| ' \
                    'awk \'BEGIN{FS=" "}{if ($1/$3 > 0.98) print NR-1}\''
            gpu_idx = subprocess.check_output(command, shell=True).rsplit(b'\n')[0].decode('utf-8')
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
            logger.info('Using GPU {}.'.format(gpu_idx))
        except subprocess.CalledProcessError:
            raise ValueError('No GPUs seems to be available.')
        
    persons = os.listdir(args.input_dir)
    persons_dict = {}
    for idx, person in enumerate(persons):
        pickle_file_path = os.path.join(args.input_dir, '{}.pkl'.format(person))
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                try:
                    persons_dict[person] = pickle.load(f)

                    def show_data():
                        import cv2
                        print(person)
                        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                        for x in persons_dict[person]:
                            cv2.imshow('image', x[0])
                            cv2.waitKey(0)
                        cv2.destroyAllWindows()

                except pickle.UnpicklingError:
                    logger.warning('pickle data was truncated for file `{}`.'.format(pickle_file_path))
                    continue

    train_data, test_data, train_labels, test_labels, num_persons = [], [], [], [], 0
    ver_test_data, ver_test_labels, num_ver_persons = [], [], 0
    for person in persons_dict:
        num_imgs = len(persons_dict[person])
        if num_imgs > 3 * args.imgs4validation:
            if args.num_persons != 0:
                if num_persons > args.num_persons:
                    continue
            np.random.shuffle(persons_dict[person])
            test_data.extend([x[0] for x in persons_dict[person][:args.imgs4validation]])
            train_data.extend([x[0] for x in persons_dict[person][args.imgs4validation:]])
            test_labels.extend([person for x in range(args.imgs4validation)])
            train_labels.extend([person for x in range(num_imgs - args.imgs4validation)])
            num_persons += 1
        else:
            # we are gonna use this data for face verification
            if args.num_ver_persons != 0:
                if num_ver_persons == args.num_ver_persons:
                    continue
            ver_test_data.extend([x[0] for x in persons_dict[person]])
            ver_test_labels.extend([person for x in range(num_imgs)])
            num_ver_persons += 1

    if args.no_train_model is not True:
        label_encoder = LabelEncoder().fit(test_labels)
        test_encoded_labels = label_encoder.transform(test_labels)
        train_encoded_labels = label_encoder.transform(train_labels)

        assert len(test_encoded_labels) == len(test_data)
        assert len(train_encoded_labels) == len(train_data)
        logger.info('Using {} images for training with {} classes.'.format(len(train_data), len(set(train_labels))))
        logger.info('Using {} images for testing with {} classes.'.format(len(test_data), len(set(test_labels))))

        # choose model
        # model = create_model(input_shape=train_data[0].shape, num_classes=len(set(test_labels)))
        model = resnet_v2(input_shape=train_data[0].shape, depth=20, num_classes=len(set(test_labels)))

        batch_size = 128
        epochs = 50
        lr = args.learning_rate
        logger.info('Using batch size: {}, number of epochs: {}, learning rate: {}.'.format(batch_size, epochs, lr))
        model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_encoded_labels = to_categorical(train_encoded_labels)
        test_encoded_labels = to_categorical(test_encoded_labels)
        history = model.fit(train_data, train_encoded_labels, batch_size=batch_size, epochs=epochs,
                            verbose=1, validation_data=(test_data, test_encoded_labels))

        model.evaluate(test_data, test_encoded_labels)
        plot_curves()
        model.save('model.h5')
    else:
        model = load_model('model.h5')
        model.summary()

        ver_num_persons, ver_num_faces = len(set(ver_test_labels)), len(ver_test_labels)
        logger.info('Using {} persons with {} images for verification.'.format(ver_num_persons, ver_num_faces))

        ver_test_data = np.array(ver_test_data)
        # print(model.predict(np.array(ver_test_data[:2])).shape)

        output_layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
        face_prints = l2_norm(output_layer_model.predict(np.array(ver_test_data)))
        print(face_prints[0])
        print(face_prints.shape)
        assert ver_num_faces == face_prints.shape[0]

        scores_matrix = cosine_similarity(face_prints, face_prints)
        tar_scores, non_scores = [], []
        for x in range(ver_num_faces):
            for y in range(x):
                distance = cosine_similarity(face_prints[x], face_prints[y])
                print(x, y, distance)
                if ver_test_labels[x] == ver_test_labels[y]:
                    tar_scores.append(distance)
                else:
                    non_scores.append(distance)

        print('EER: {}'.format(get_eer(tar_scores, non_scores)))




