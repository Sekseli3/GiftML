import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgba2rgb
import numpy as np

def predict_category(image_path, model, categories):
    # Load and preprocess the image
    img = imread(image_path)
    if img.shape[2] == 4:  # If the image has 4 channels
        img = rgba2rgb(img)  # Convert to RGB
    img = resize(img, (200, 200))  # Resize to match the model's expected input size
    img_flat = img.reshape(1, -1)  # Flatten the image

    # Use the model to predict the category
    prediction = model.predict(img_flat)

    # Get the confidence scores
    scores = model.decision_function(img_flat)
    scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize to [0, 1]
    # Get the probabilities
    probabilities = model.predict_proba(img_flat)[0]
    # Display the image
    plt.imshow(img)
    plt.axis('off')

    # Print the confidence scores on the image
    plt.text(100,220,'SVC', color='black')
    plt.text(-75, -20, 'Confidence scores:', color='black')
    for i, (category, score) in enumerate(zip(categories, scores[0])):
        plt.text(-75, i*20, f'{category}: {score * 100:.2f}%', color='black')
    # Print the probabilities on the image
    plt.text(205, -20, 'Probabilities:', color='black')
    for i, (category, probability) in enumerate(zip(categories, probabilities)):
        plt.text(205, i*20, f'{category}: {probability * 100:.2f}%', color='black')

    plt.text(60, -20, f'Prediction: {categories[prediction[0]]}', color='black')
    plt.show()

    # Return the predicted category
    return categories[prediction[0]]


# Load the trained model from the pickle file
model = pickle.load(open('./model.p', 'rb'))

# Define the categories
categories = ['books','bottles','letters','posters','shirts','socks','sportsStuff','tech','treats','toys']

# Use the function to predict the category of a new image
image_path = '/Users/akselituominen/Desktop/Screenshot 2023-12-20 at 10.07.33.png'
predicted_category = predict_category(image_path, model, categories)


def predict_category_cnn(image_path, model_path, categories):
    # Load the image
    img = image.load_img(image_path, target_size=(100, 100))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Load the trained model from the h5 file
    model = load_model(model_path)

    # Use the model to predict the category of the image
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)

    # Print the probabilities on the image
    plt.imshow(img)
    plt.axis('off')
    plt.text(100, -10, 'Probabilities:', color='black')
    for i, (category, probability) in enumerate(zip(categories, probabilities)):
        plt.text(100, i*10, f'{category}: {probability * 100:.2f}%', color='black')

    plt.text(30, -10, f'Prediction: {categories[prediction]}', color='black')
    plt.show()
    # Use the model to predict the category of the image
    probabilities = model.predict(img_array)[0]
    prediction = np.argmax(probabilities)
    confidence_score = probabilities[prediction]

    # Print the confidence score
    print(f'Confidence score: {confidence_score * 100:.2f}%')

    # Return the predicted category
    return categories[prediction]

# Define the categories

# Use the function to predict the category of a new image
model_path = './model.h5'
predicted_category = predict_category_cnn(image_path, model_path, categories)