# import os 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

# def load_model() -> Sequential: 
#     model = Sequential()

#     model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(256, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(Conv3D(75, 3, padding='same'))
#     model.add(Activation('relu'))
#     model.add(MaxPool3D((1,2,2)))

#     model.add(TimeDistributed(Reshape((-1,))))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
#     model.add(Dropout(.5))

#     # model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))
#     vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
#     import tensorflow as tf
#     char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
#     model.add(Dense(char_to_num.vocabulary_size()+1, kernel_initializer='he_normal', activation='softmax'))

#     model.load_weights(os.path.join('..','models','checkpoint'))

#     return model

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, TimeDistributed, Flatten
)

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(TimeDistributed(Flatten()))

    # Bidirectional LSTM layers
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(os.path.join('..', 'models', 'checkpoint'))
    return model