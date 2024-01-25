from keras.callbacks import EarlyStopping

from model import design_model

def fit_model(features_train, labels_train, learning_rate, epochs, batch_size, validation_split):
    model = design_model(features_train, learning_rate)
    stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=20)
    history = model.fit(features_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=validation_split, callbacks=[stop])
    return history
