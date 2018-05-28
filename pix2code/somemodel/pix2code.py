from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras import *
from .Config import *
from .AModel import *


class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"
        print("input_shape-model",input_shape)
        image_model = Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
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
        for j in range(20):
            print("epoch:",j)
            print("##############")
            for i in range(step):
                images , partial_captions, next_words = dataset.minconvert_arrays(i,batch_size)
                print(i,"step ","totalstep:",step,"size:",len(images),images.shape, partial_captions.shape,next_words.shape)
                self.model.fit([images, partial_captions], next_words, shuffle=False, batch_size=64, verbose=1)

        self.save()
        print("保存成功") 
    def compile(self):
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
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
            print(i,"step ","totalstep:",step,"size:",len(images),images.shape, partial_captions.shape,next_words.shape)
            loss,score = self.model.evaluate([images, partial_captions], next_words, batch_size=64)
            sumscore = sumscore + score
            sumloss = sumloss + loss
            print("loss:",loss,"accuracy:",score)
       avscore = sumscore/step
       avloss = sumloss/step
       return avloss,avscore,score

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)
