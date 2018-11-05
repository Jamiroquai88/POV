#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2018 Phonexia
# Author: Jan Profant <jan.profant@phonexia.com>
# All Rights Reserved

import argparse
import pickle
import logging
import os

import cv2

import openface


logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--input-dir', required=True,
                        help='path to the input directory in format Person1/image1.jpg, Person4/image3.jpg')
    parser.add_argument('--face-predictor', help='path to dlib face predictor', required=False,
                        default=os.path.join('..', 'openface', 'models', 'dlib',
                                             'shape_predictor_68_face_landmarks.dat'))
    parser.add_argument('--img-dim', required=False, default=96)

    args = parser.parse_args()

    persons = os.listdir(args.input_dir)
    num_classes = len(persons)
    logger.info('Found {} persons in input directory `{}`.'.format(num_classes, args.input_dir))
    for person in persons:
        aligned_examples = []
        for input_image in os.listdir(os.path.join(args.input_dir, person)):
            input_image = os.path.join(args.input_dir, person, input_image)
            image = cv2.imread(input_image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            align = openface.AlignDlib(args.face_predictor)
            bbs = align.getAllFaceBoundingBoxes(rgb_image)
            if len(bbs) != 1:
                logger.warning('Detected {} faces in image `{}`, expecting only 1.'.format(len(bbs), input_image))
                continue
            boundary_box = bbs[0]
            aligned_face = align.align(args.img_dim, rgb_image, boundary_box,
                                       landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                logger.warning('Failed to align face in file `{}`.'.format(input_image))
                continue

            aligned_examples.append((aligned_face, person))

        with open(os.path.join(args.input_dir, '{}.pkl'.format(person)), 'w') as f:
            pickle.dump(aligned_examples, f, pickle.HIGHEST_PROTOCOL)
