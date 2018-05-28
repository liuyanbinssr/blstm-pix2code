#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

import sys

from classes.dataset.Generator import *
from classes.model.pix2code import *




def run(input_path, output_path, is_memory_intensive=False, pretrained_model=None):
    np.random.seed(1234)

    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)
    print("dataset.size",dataset.size)

    
    dataset.convert_arrays()

    input_shape = dataset.input_shape
    output_size = dataset.output_size

    print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
    print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    
    maxlengh =  len(dataset.input_images)
    i=0
    batch_size = 1000
    input_images , dataset.partial_sequences, next_words = minconvert_arrays(i,batch_size)
    model = pix2code(input_shape, output_size, output_path)


   
    model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)

   

if __name__ == "__main__":
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Error: not enough argument supplied:")
        print("train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>")
        exit(0)
    else:
        input_path = argv[0]
        output_path = argv[1]
        use_generator = False if len(argv) < 3 else True if int(argv[2]) == 1 else False
        pretrained_weigths = None if len(argv) < 4 else argv[3]

    run(input_path, output_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)