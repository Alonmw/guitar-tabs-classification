import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import train_test_split


def build_model(input_shape, num_classes):
    """
    builds a base CNN model optimized for a small dataset.
    """
    model = models.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=(3,3), activation='relu', 
                            input_shape=input_shape, padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Flatten())
    model.add(layers.Dense(units=64, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
