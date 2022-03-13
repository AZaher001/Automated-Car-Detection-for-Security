import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras import optimizers


###### Create CNN MODEL TRAINING #######

# Increasing your images in real time as you train your model
train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)

# Increasing Feature
train_generator = train_datagen.flow_from_directory('data/train',  # this is the target directory
    target_size=(28,28),  # all images will be resized to 28x28
    batch_size=1,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory( 'data/val',  # this is the target directory
    target_size=(28,28), batch_size=1,
    class_mode='categorical')

# building a linear stack of layers with the sequential model
model = Sequential()
# convolutional layer  
model.add(Conv2D(32, (24,24), input_shape=( 28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (20,20), input_shape=(28, 28, 3), activation='relu', padding='same'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))
model.add(Flatten())
# Fully Connection
# hidden layer
model.add(Dense(128, activation='relu'))
# output layer
model.add(Dense(36, activation='softmax'))
# compiling the sequential model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=0.00001),
              metrics=['accuracy'])

    
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    
batch_size = 1
callbacks = [tensorboard_callback,tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]

# training the model
history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 20, callbacks=callbacks)


model.save('F:/Work/Poject/Training_CNN')
