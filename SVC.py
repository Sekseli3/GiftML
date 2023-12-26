import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

#Prepare data
input_dir = '/Users/akselituominen/Desktop/giftWrapML'
categories = ['books','bottles','letters','posters','shirts','socks','sportsStuff','tech','treats','toys']

data = []
labels = []
print('Loading images...')


datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# For each image
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        if not file.endswith('.DS_Store'):
            img_path = os.path.join(input_dir, category, file)
            img = imread(img_path)
            if img.shape[2] == 4:  # If the image has 4 channels
                img = rgba2rgb(img)  # Convert to RGB
            img = resize(img,(200,200))
            data.append(img)  # Append the processed image to data
            labels.append(category_idx) 
data = np.asarray(data)
labels = np.asarray(labels)

#train / test split
x_train,x_test, y_train, y_test = train_test_split(data, labels, test_size=0.15,shuffle= True, stratify = labels)

# Flatten the images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Train the classifier
print('Training classifier...')
classifier = SVC(probability=True)
parameters = [{'gamma':[0.01, 0.001, 0.0001],'C':[1,10,100,1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train_flat, y_train)

# Test performance
best_estimator = grid_search.best_estimator_
y_prediction = best_estimator.predict(x_test_flat)
score = accuracy_score(y_prediction, y_test)
print('{}% of samples were correctly classified'.format(str(score*100)))

pickle.dump(best_estimator,open('./model.p','wb'))


