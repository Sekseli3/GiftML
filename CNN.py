import os
import numpy as np
from collections import Counter
import cv2 as cv
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.layers import  Flatten, Dense
from sklearn.utils import class_weight
from keras.optimizers.legacy import RMSprop
from keras.utils import to_categorical
#Prepare data
input_dir = '/Users/akselituominen/Desktop/giftWrapML'
categories = ['books','bottles','letters','posters','shirts','socks','sportsStuff','tech','treats','toys']

data = []
labels = []
print('Loading images...')



# Create an ImageDataGenerator object
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


# For each image
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        if not file.endswith('.DS_Store'):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            if img.shape[2] == 4:  # If the image has 4 channels
                img = rgba2rgb(img)  # Convert to RGB
            img = resize(img,(100,100))
            data.append(img)  # Append the processed image to data
            labels.append(category_idx) 

data = np.asarray(data)
labels = np.asarray(labels)
# Train / test split

#In this code, we first load a pre-trained VGG16 model without the top layer 
#(which includes the final classification layers). We then freeze the layers of this base model, 
#so their weights will not be updated during training. We add our own layers on top of the base model, 
#including a Flatten layer to convert the feature maps to a 1D vector, a Dense layer with ReLU activation, 
#a Dropout layer for regularization, and a final Dense layer with softmax activation for the classification. 
#We then compile and train the model on our data.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, shuffle=True, stratify=labels)

# Convert labels to categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Data augmentation
train_X = []
train_y = []
for im, l in zip(x_train, y_train):
  train_X.append(im)
  train_X.append(cv.flip(im, 0))
  train_X.append(cv.flip(im, 1))
  train_X.append(cv.flip(im, -1))
  train_y += [l] * 4

X_train = np.array(train_X)
y_train = np.array(train_y)

# Load pre-trained VGG16 model without the top layer (which includes the final classification layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add your own layers
x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(y_train.shape[1], activation='softmax')(x)  # y_train.shape[1] gives the number of classes

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Convert one-hot encoded targets to labels
labels = np.argmax(y_train, axis=-1)

# Calculate class weights
sample_weights = class_weight.compute_sample_weight('balanced', labels)

# Calculate class weights
counter = Counter(labels)  # count the number of instances per class
class_weights = {cls: weight for cls, weight in zip(counter.keys(), sample_weights)}

# Define the optimizer and compile the model
optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model on the new data for a few epochs
model.fit(X_train, y_train, batch_size=32, epochs=30, validation_data=(x_test, y_test), callbacks=[early_stopping], class_weight=class_weights)
# Test performance
loss, accuracy = model.evaluate(x_test, y_test)
print('{}% of samples were correctly classified'.format(str(accuracy*100)))

# Save the model
#model.save('./model.h5')