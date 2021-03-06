#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import csv
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
from keras_preprocessing.image import ImageDataGenerator
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from utils import get_eer, l2_norm, cosine_similarity, safe_mkdir
from resnet_model import resnet_v2
from keras_model import create_model
from keras_model2 import create_model as create_model2
from keras.applications.resnet50 import ResNet50

logger = logging.getLogger()
logging.basicConfig(format='%(asctime)s %(levelname)s %(name)s %(message)s', level=logging.INFO)

# set seed
np.random.seed(42)


def plot_curves(output_dir):
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))

    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'acc_curves.png'))


def plot_roc(tar_scores, non_scores, label, color, output_path):
    scores = np.array(tar_scores + non_scores)
    labels = np.concatenate((np.ones((len(tar_scores))), np.zeros(len(non_scores))))

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    import matplotlib.pyplot as plt

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color, label='%s AUC = %0.2f' % (label, roc_auc * 100))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(output_path)


def show_data():
    import cv2
    print(person)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    for x in persons_dict[person]:
        cv2.imshow('image', x[0])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', required=True,
                        help='path to the input directory in format Person1/image1.jpg, Person4/image3.jpg,'
                             'expects already prepared data in pickle format')
    parser.add_argument('-D', '--output-dir', required=True,
                        help='path to the output directory for storing figures and models')
    parser.add_argument('--lfw-path', required=False, default='../data/lfw', help='path to LFW dataset')
    parser.add_argument('--lfw-trials', required=False, default='../data/lfw_pairs.txt', help='path to lfw pairs')
    parser.add_argument('--imgs4validation', required=False, default=5, type=int,
                        help='number of images used for cross-validation')
    parser.add_argument('--num-persons', required=False, default=0, type=int, help='number of persons for training')
    parser.add_argument('--num-ver-persons', required=False, default=0, type=int,
                        help='number of persons for verification')
    parser.add_argument('--learning-rate', required=False, default=0.0001, type=float, help='learning rate')
    parser.add_argument('--no-use-gpu', required=False, default=False, action='store_true',
                        help='use GPU for training')
    parser.add_argument('--no-train-model', required=False, default=False, action='store_true')
    parser.add_argument('--continue-training', required=False, type=str, help='continue training of model from file')
    
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

    safe_mkdir(args.output_dir)
        
    persons = os.listdir(args.input_dir)
    persons_dict = {}
    for idx, person in enumerate(persons):
        # TODO fix
        if idx == 1000:
            break
        pickle_file_path = os.path.join(args.input_dir, '{}.pkl'.format(person))
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                try:
                    persons_dict[person] = pickle.load(f)
                except pickle.UnpicklingError:
                    logger.warning('pickle data was truncated for file `{}`.'.format(pickle_file_path))
                    continue

    train_data, test_data, train_labels, test_labels, num_persons = [], [], [], [], 0
    ver_test_data, ver_test_labels, num_ver_persons = [], [], 0
    class_data, class_labels = [], []
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
                    # class_data.extend([x[0] for x in persons_dict[person]])
                    # class_labels.extend([person for x in range(num_imgs)])
                    # num_ver_persons -= 1
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

        datagen = ImageDataGenerator(rotation_range=30,  # randomly rotate images
                                     width_shift_range=0.1,  # randomly shift images horizontally
                                     height_shift_range=0.1,  # randomly shift images vertically
                                     horizontal_flip=True,  # randomly flip images
                                     vertical_flip=False)  # randomly flip images
        if not args.continue_training:

            # choose model

            # basic model
            # model = create_model(input_shape=train_data[0].shape, num_classes=len(set(test_labels)))

            # model = create_model2(input_shape=train_data[0].shape, num_classes=len(set(test_labels)))

            # resnet
            model = resnet_v2(input_shape=train_data[0].shape, depth=11, num_classes=len(set(test_labels)))

            # resnet50
            # model = ResNet50(include_top=False, pooling='max',
            #                  input_shape=train_data[0].shape, classes=len(set(test_labels)))
        else:
            model = load_model(args.continue_training)

        batch_size = 256
        epochs = 50
        lr = args.learning_rate
        logger.info('Using batch size: {}, number of epochs: {}, learning rate: {}.'.format(batch_size, epochs, lr))
        model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

        print('Number of layers: {}.'.format(len(model.layers)))
        model.summary()
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_encoded_labels = to_categorical(train_encoded_labels)
        test_encoded_labels = to_categorical(test_encoded_labels)

        history = model.fit_generator(datagen.flow(train_data, train_encoded_labels, batch_size=batch_size),
                                      steps_per_epoch=int(np.ceil(train_data.shape[0] / float(batch_size))),
                                      epochs=epochs, workers=4, verbose=1,
                                      validation_data=(test_data, test_encoded_labels))

        model.evaluate(test_data, test_encoded_labels)
        plot_curves(args.output_dir)
        model.save(os.path.join(args.output_dir, 'model.h5'))
    else:
        model = load_model(os.path.join(args.output_dir, 'model.h5'))
        print('Number of layers: {}.'.format(len(model.layers)))
        model.summary()

        ver_num_persons, ver_num_faces = len(set(ver_test_labels)), len(ver_test_labels)
        logger.info('Using {} persons with {} images for verification.'.format(ver_num_persons, ver_num_faces))

        ver_test_data = np.array(ver_test_data)

        output_layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten_1').output)
        face_prints = output_layer_model.predict(np.array(ver_test_data))

        face_prints = l2_norm(face_prints)
        assert ver_num_faces == face_prints.shape[0]

        tar_scores, non_scores = [], []
        for x in range(ver_num_faces):
            for y in range(x):
                distance = cosine_similarity(face_prints[x], face_prints[y])
                if ver_test_labels[x] == ver_test_labels[y]:
                    tar_scores.append(distance)
                else:
                    non_scores.append(distance)

        np.save(os.path.join(args.output_dir, 'tar.npy'), tar_scores)
        np.save(os.path.join(args.output_dir, 'non.npy'), non_scores)
        plot_roc(tar_scores, non_scores, label='', color='b', output_path=os.path.join(args.output_dir, 'roc.png'))
        print('EER: {}'.format(get_eer(tar_scores, non_scores)))

        # lfw
        persons = os.listdir(args.lfw_path)
        persons_dict = {}
        for idx, person in enumerate(persons):
            if os.path.isdir(os.path.join(args.lfw_path, person)):
                num_images = len(os.listdir(os.path.join(args.lfw_path, person)))
                pickle_file_path = os.path.join(args.lfw_path, '{}.pkl'.format(person))
                if os.path.exists(pickle_file_path):
                    with open(pickle_file_path, 'rb') as f:
                        try:
                            images = pickle.load(f)
                            if len(images) == num_images:
                                persons_dict[person] = images
                        except pickle.UnpicklingError:
                            logger.warning('pickle data was truncated for file `{}`.'.format(pickle_file_path))
                            continue

        lfw_data = {}
        trials = []
        with open(args.lfw_trials) as f:
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                if len(row) == 3:
                    # same person comparison
                    person, enroll_idx, test_idx = row
                    if person in persons_dict:
                        if person not in lfw_data:
                            lfw_data[person] = output_layer_model.predict(
                                np.array([x[0] for x in persons_dict[person]]))
                    else:
                        logger.warning('Ignoring trial {}.'.format(' '.join(row)))
                        continue
                    trials.append((person, int(enroll_idx) - 1, person, int(test_idx) - 1))
                elif len(row) == 4:
                    # different persons
                    enroll_person, enroll_idx, test_person, test_idx = row
                    if enroll_person in persons_dict and test_person in persons_dict:
                        if enroll_person not in lfw_data:
                            lfw_data[enroll_person] = output_layer_model.predict(
                                np.array([x[0] for x in persons_dict[enroll_person]]))
                        if test_person not in lfw_data:
                            lfw_data[test_person] = output_layer_model.predict(
                                np.array([x[0] for x in persons_dict[test_person]]))
                    else:
                        logger.warning('Ignoring trial {}.'.format(' '.join(row)))
                        continue
                    trials.append((enroll_person, int(enroll_idx) - 1, test_person, int(test_idx) - 1))
                else:
                    raise ValueError('Unexpected row {}.'.format(' '.join(row)))

        tar_scores, non_scores = [], []
        logger.info('Computing scores for {} trials.'.format(len(trials)))
        for trial in trials:
            enroll_person, enroll_idx, test_person, test_idx = trial
            enroll_emb, test_emb = lfw_data[enroll_person][enroll_idx], lfw_data[test_person][test_idx]
            distance = cosine_similarity(l2_norm(enroll_emb), l2_norm(test_emb))
            if enroll_person == test_person:
                tar_scores.append(distance)
            else:
                non_scores.append(distance)

        np.save(os.path.join(args.output_dir, 'tar_lfw.npy'), tar_scores)
        np.save(os.path.join(args.output_dir, 'non_lfw.npy'), non_scores)
        plot_roc(tar_scores, non_scores, label='', color='b', output_path=os.path.join(args.output_dir, 'roc_lfw.png'))
        print('EER: {}'.format(get_eer(tar_scores, non_scores)))
