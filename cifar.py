import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

BATCH_SIZE = 100
NUM_CLASSES = 100
EPOCHS = 165000
DROPOUT_RATE = 0.5
MOMENTUM_RATE = 0.9
LEARNING_RATE = 0.01
L2_DECAY_RATE = 0.0005
CROP_SIZE = 32
LOG_DIR = './logs'
MODEL_PATH = './models/keras_cifar100_model.h5'

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0


model = keras.models.Sequential()
model.add(keras.layers.ZeroPadding2D(4, input_shape=x_train.shape[1:]))
model.add(keras.layers.Conv2D(384, (3, 3), padding='same', kernel_regularizer=l2(0.01)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(384, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(384, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(640, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(640, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(768, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(768, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(896, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(896, (3, 3), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(1024, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(1024, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Conv2D(1152, (2, 2), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Conv2D(1152, (1, 1), padding='same', kernel_regularizer=l2(L2_DECAY_RATE)))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(keras.layers.Dropout(DROPOUT_RATE))

model.add(keras.layers.Flatten())
model.add(keras.layers.keras.layers.Dense(NUM_CLASSES))
model.add(keras.layers.Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer= keras.optimizers.SGD(lr=LEARNING_RATE, momentum=MOMENTUM_RATE),
             metrics=['accuracy'])

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10.,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.,
    zoom_range=1.,
    validation_split=0.2,
    samplewise_center=True,
    horizontal_flip=True,
    vertical_flip=True)

history = model.fit_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              steps_per_epoch=len(x_train) / BATCH_SIZE,
                              epochs=EPOCHS,
                              validation_data=datagen.flow(x_train,y_train, batch_size=BATCH_SIZE),
                              validation_steps=len(x_test) / BATCH_SIZE)
result = model.evaluate_generator(datagen.flow(x_train,y_train, batch_size=BATCH_SIZE), steps=len(x_test) / BATCH_SIZE )

print('Test loss: ' + str(results[0]))
print('Accuracy: ' + str(results[1]))



      
