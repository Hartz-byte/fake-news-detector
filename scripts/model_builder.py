from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, regularizers
import tensorflow as tf

def build_model(input_dim):
    model = Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)),
        layers.Dropout(0.6),
        layers.Dense(1, activation='sigmoid')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
