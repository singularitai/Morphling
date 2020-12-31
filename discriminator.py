import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, Conv3D, BatchNormalization, Activation, \
						Concatenate, AvgPool2D, Input, MaxPool2D, UpSampling2D, Add, \
						ZeroPadding2D, ZeroPadding3D, Lambda, Reshape, Flatten, LeakyReLU
from keras_contrib.layers import InstanceNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import keras
import cv2
import os
import librosa
import scipy
from tensorflow.keras.utils import plot_model

from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K

class ModelMGPU(Model):
    def __init__(self, ser_model, gpus):
        pmodel = multi_gpu_model(ser_model, gpus)
        self.__dict__.update(pmodel.__dict__)
        self._smodel = ser_model

    def __getattribute__(self, attrname):
        '''Override load and save methods to be used from the serial-model. The
        serial-model holds references to the weights in the multi-gpu model.
        '''
        # return Model.__getattribute__(self, attrname)
        if 'load' in attrname or 'save' in attrname:
            return getattr(self._smodel, attrname)

        return super(ModelMGPU, self).__getattribute__(attrname)

def contrastive_loss(y_true, y_pred):
	margin = 1.
	loss = (1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(0., margin - y_pred))
	return K.mean(loss)

def conv_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
	x = Conv2D(filters=num_filters, kernel_size= kernel_size, 
					strides=strides, padding=padding)(x)
	x = InstanceNormalization()(x)
	x = LeakyReLU(alpha=.2)(x)
	return x

def create_model(args, mel_step_size):
	############# encoder for face/identity
	input_face = Input(shape=(args.img_size, args.img_size, 3), name="input_face_disc")

	x = conv_block(input_face, 64, 7)
	x = conv_block(x, 128, 5)
	x = conv_block(x, 256, 3)
	x = conv_block(x, 512, 3)
	x = conv_block(x, 512, 3)
	x = Conv2D(filters=512, kernel_size=3, strides=1, padding="valid")(x)
	face_embedding = Flatten() (x)

	############# encoder for audio
	input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

	x = conv_block(input_audio, 32, strides=1)
	x = conv_block(x, 64, strides=3)	#27X9
	x = conv_block(x, 128, strides=(3, 1)) 		#9X9
	x = conv_block(x, 256, strides=3)	#3X3
	x = conv_block(x, 512, strides=1, padding='valid')	#1X1
	x = conv_block(x, 512, 1, strides=1)

	audio_embedding = Flatten() (x)

	# L2-normalize before taking L2 distance
	l2_normalize = Lambda(lambda x: K.l2_normalize(x, axis=1)) 
	face_embedding = l2_normalize(face_embedding)
	audio_embedding = l2_normalize(audio_embedding)

	d = Lambda(lambda x: K.sqrt(K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True))) ([face_embedding,
																		audio_embedding])

	model = Model(inputs=[input_face, input_audio], outputs=[d])



	if args.n_gpu > 1:
		model = ModelMGPU(model , args.n_gpu)
		
	model.compile(loss=contrastive_loss, optimizer=Adam(lr=args.lr)) 
	
	return model

if __name__ == '__main__':
	model = create_model()
	#plot_model(model, to_file='model.png', show_shapes=True)
