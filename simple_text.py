from module_neural_net_v3 import *

def simple_text(X_train, y_train, X_val, y_val, X_test, y_test):

    # create the model
    model = Sequential()

    model.add(Dense(8, activation='relu', input_dim = X_train.shape[1])) 
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    # OUTPUT LAYER (scalar)
    model.add(Dense(y_train.shape[1], activation='softmax'))
    loss_func = 'categorical_crossentropy'
    model.compile(loss=loss_func, optimizer=Adam(), metrics=['accuracy'])
    print(model.summary())

    history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=50, batch_size=32, verbose=1)     

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)

    return history, scores