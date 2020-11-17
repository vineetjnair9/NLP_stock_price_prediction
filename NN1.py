import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pandas
from keras.models import load_model
from os.path import isfile
import os
from keras.optimizers import Adam

#%%
def train(data, model_name, day='tomorrow'):
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(model_name, monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)
    
# data consists of features and target vector (stock price)

    (X_train, y_train), (X_val, y_val) = data
    
    # truncate and pad input sequences
    max_review_length = 100
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32 # k from Word2Vec / TF-IDF
    model = Sequential()
#    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    if isfile(model_name) and False:
        print('Checkpoint Loaded')
        model = load_model(model_name)
    print(model.summary())

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences = True))
    model.add(Dropout(drop_out))
    model.add(LSTM(neurons,return_sequences = True))
    model.add(LSTM(neurons,return_sequences =False))
    model.add(Dropout(drop_out))
    model.add(Dense(dense_output, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error',
                    optimizer='adam')
    # Fit the model
    model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size)
    
    
    model.fit(X_train, y_train, validation_split=0.02, epochs=100, batch_size=64, callbacks=[checkpoint, earlyStopping, reduceLR])
    # Final evaluation of the model
    model = load_model(model_name)
    scores = model.evaluate(X_test, y_test, verbose=0)
    
    