from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

def build_model(input_dim):
    """
    Builds and compiles a simple feedforward neural network for binary classification.
    """
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
