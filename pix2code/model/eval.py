from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys
from classes.dataset.Generator import *
from classes.Sampler import *
from classes.model.pix2code import *

argv = sys.argv[1:]
def alltarn(Y):
    length = len(Y)
    for i in range(length):
        print("i:",i)
        a = Y[i]
        arr = a.T
        arr = np.reshape(arr,(1,19,48))
        if(i==0):
            Yarr = arr
        else:
            Yarr = np.concatenate((Yarr,arr),axis=0)
    return Yarr        

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("generate.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>")
    exit(0)
else:
    trained_weights_path = argv[0]
    trained_model_name = argv[1]
    input_path = argv[2]


meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path))
input_shape = meta_dataset[0]
output_size = meta_dataset[1]
model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

dataset = Dataset()
dataset.load(input_path, generate_binary_sequences=True)



model.compile()
score = model.minevaluate(dataset)
print("accuracy:",score)
#print("size:",size)
#print(Y.shape)

#dataset2 = Dataset()
#dataset2.load(input_path, generate_binary_sequences=True)
#dataset2.convert_arrays()

#predictions = dataset2.partial_sequences
#predictions = alltarn(predictions)

#print(Y.shape)
#print(predictions.shape)
#a = np.dot(Y[0],predictions[0].T)
#print(a)
#print ('Accuracy: %d' % float((np.dot(Y,predictions) + np.dot(1-Y,1-predictions))/float(size)*100) + '%')




