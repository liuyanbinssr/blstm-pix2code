from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten, BatchNormalization, ZeroPadding2D, Activation, AveragePooling2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers import Bidirectional
from keras import *
from keras.initializers import glorot_uniform
from keras import initializers
from keras.layers.core import Dropout
from keras.callbacks import TensorBoard
from .Config import *
from .AModel import *
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

weight_init="glorot_uniform"
class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"
        print("input_shape-model",input_shape)
        X_input = Input(input_shape)

    
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
    
        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
        X = Dropout(0.2)(X)
        ### START CODE HERE ###

        # Stage 3 (≈4 lines)
        X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
        X = Dropout(0.2)(X)
        # Stage 4 (≈6 lines)
        X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
        X = Dropout(0.2)(X)
        # Stage 5 (≈3 lines)
        X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
        X = MaxPooling2D((2, 2),name="avg_pool")(X)
    
        ### END CODE HERE ###

        # output layer
        X = Flatten()(X)
        X = Dense(1024, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
        X = Dropout(0.3)(X)
        X = Dense(1024, activation='relu', kernel_initializer = glorot_uniform(seed=0))(X)
        X = Dropout(0.3)(X)
        X = RepeatVector(CONTEXT_LENGTH)(X)
        image_model = Model(inputs = X_input, outputs = X, name='ResNet50')
        #image_model.add(RepeatVector(CONTEXT_LENGTH))
       
        #image_model.summary()
        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        #language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        #language_model.add(LSTM(128, return_sequences=True))
        language_model.add(Bidirectional(LSTM(128, return_sequences=True,kernel_initializer=weight_init), input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(Bidirectional(LSTM(128, return_sequences=True,kernel_initializer=weight_init)))
       

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        #decoder = LSTM(512, return_sequences=True)(decoder)
        #decoder = Dropout(0.25)(decoder)
        #decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Bidirectional(LSTM(512, return_sequences=True,kernel_initializer=weight_init))(decoder)
        #decoder = Dropout(0.2)(decoder)
        decoder = Bidirectional(LSTM(512, return_sequences=False,kernel_initializer=weight_init))(decoder)
        decoder = Dense(output_size, activation='softmax',kernel_initializer=weight_init)(decoder)
        

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)
        
        optimizer = RMSprop(lr=0.0002, clipvalue=1.0)
        #optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])

    def fit(self, images, partial_captions, next_words):
        print("partial_captions",partial_captions.shape)
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def fit_p(self,dataset):
        dataleng =  len(dataset.input_images)
        batch_size = 1024
        step = dataleng//batch_size + 1
        for j in range(10):
            print("epoch:",j)
            print("##############")
            for i in range(step):
                images , partial_captions, next_words = dataset.minconvert_arrays(i,batch_size)
                #print(i,"step ","totalstep:",step,"size:",len(images),images.shape, partial_captions.shape,next_words.shape)
                self.model.fit([images, partial_captions], next_words, shuffle=False, batch_size=32, verbose=1, callbacks=[TensorBoard(log_dir='logdir/log-adma')])

        self.save() 
    def compile(self):
        optimizer = RMSprop(lr=0.0002, clipvalue=1.0)
        #optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def evaluate(self,images , partial_captions, next_words):
       score = self.model.evaluate([images, partial_captions], next_words, batch_size=16)
       return score
    def minevaluate(self,dataset):
       dataleng =  len(dataset.input_images)
       batch_size = 2048
       step = dataleng//batch_size + 1
       sumscore = 0.0
       sumloss = 0.0
       for i in range(step):
            images , partial_captions, next_words = dataset.minconvert_arrays(i,batch_size)
            loss,score = self.model.evaluate([images, partial_captions], next_words, batch_size=64)
            sumscore = sumscore + score
            sumloss = sumloss + loss
            print("loss:",loss,"accuracy:",score)
       avscore = sumscore/step
       avloss = sumloss/step
       return avscore,score

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
    
def identity_block(X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
    
       # Retrieve Filters
        F1, F2, F3 = filters
    
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X
        # First component of main path
        X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
        X = Activation('relu')(X)
        ### START CODE HERE ###
    
        # Second component of main path (≈3 lines)
        X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path (≈2 lines)
        X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
        # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
        X = layers.Add()([X,X_shortcut])
        X = Activation('relu')(X)
    
        ### END CODE HERE ###
    
        return X



def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X


    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, kernel_size = (1, 1), strides = (s,s), name = conv_name_base + '2a', padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, kernel_size = (f, f), strides = (1,1), name = conv_name_base + '2b', padding = 'same',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, kernel_size = (1, 1), strides = (1,1), name = conv_name_base + '2c', padding = 'valid',kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, kernel_size = (1, 1), strides = (s,s), name = conv_name_base + '1', padding = 'valid')(X)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X,X_shortcut])
    X = Activation('relu')(X)
    
    ### END CODE HERE ###
    
    return X
