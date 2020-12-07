from module_neural_net_v3 import *

def tcn_functional(X_train, y_train, X_val, y_val, X_test, y_test):

    verbose = 1
    epochs = 50
    batch_size = 32

    n_timesteps, n_features, n_outputs = X_train.shape[0], X_train.shape[1], y_train.shape[1]

    input_shape = (batch_size,n_timesteps,n_features)
    # input_shape = (n_timesteps,n_features)
    input_data = Input(input_shape)

    x = Dense(32, activation="relu")(input_data)
    x = Dropout(0.3)(x)
    # x = Conv1D(filters=10, kernel_size=5, padding='causal', activation='relu',input_shape = input_shape[1:])(input_data)
    x = Conv1D(filters=10, kernel_size=5, padding='causal', activation='relu',input_shape = input_shape)(x)
    x = Conv1D(filters=20, kernel_size=3, activation="relu")(x)
    # x = MaxPooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    stock_return = layers.Dense(n_outputs, activation='softmax')(x)

    model = keras.Model(inputs=input_data, outputs=stock_return, name="tcn_func")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)     

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=verbose)

    return history, scores