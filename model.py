from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers import Dense
from keras.optimizers import Adam

def design_model(features, learning_rate):
    model = Sequential(name = "my_first_model")
    inputLayer = InputLayer(input_shape=(features.shape[1],)) 
    model.add(inputLayer) 
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1)) 
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
    return model
