from module_neural_net_v3 import *

def tcn_lstm(X_train, y_train, X_val, y_val, X_test, y_test):
    # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')
    # checkpoint = ModelCheckpoint(monitor='val_acc', mode='auto', verbose=1, save_best_only=True)
    # earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1)

    verbose = 1
    epochs = 20
    batch_size = 32

    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[1]
    
    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)
    X_test = np.expand_dims(X_test, axis=2)

    # create the model
    model = Sequential()
    
    # INPUT LAYER
    # 1D temporal convolution = Causal convolutional networks
    model.add(Conv1D(filters=5, kernel_size=3, padding='causal', activation='relu',input_shape = (n_features,1))) 
    model.add(MaxPooling1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    
    # LSTM layer with 128 internal units
    model.add(LSTM(128,input_dim = len(X_train.columns),return_sequences=True))

    model.add(Dense(32, activation='relu')) 
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
#    model.add(Flatten())
    
    # OUTPUT LAYER (scalar)
    model.add(Dense(n_outputs, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=100, batch_size=64, callbacks=[checkpoint, earlyStopping, reduceLR],verbose=1)
    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)     

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)

    return history, scores