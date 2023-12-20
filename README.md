# Christmas Gift Predictor

This project uses Support Vector Classifier (SVC) and Convolutional Neural Network (CNN) models to predict what's inside Christmas presents based on images. The pre determined catagories of christmas presents are: books,bottles,letters,posters,shirts,socks,sportsStuff,tech,treats,toys

## Data

The data for this project was collected manually due to the lack of sufficient online resources. The images were obtained by print screening presents, so the data quality may vary.

## Models

Two types of models are used in this project:

1. **SVC Model**: A custom SVC model was created for this project. SVC is a type of supervised learning model that analyzes data and recognizes patterns.

2. **CNN Model**: A pre-trained VGG16 model was used for the CNN. The top layers of the VGG16 model were frozen to prevent them from being updated during training. This approach was found to yield the best results for this project.

## Usage

To use this project, run the SVC and CNN files to get the models. After that take a photo of a christmas present and replace the image_path in use.py. The `predict_category_cnn` function in `use.py` can be used to predict the category of the gift with CNN.'predict_category' on the other hand uses SVC. The function takes the path to the image, the path to the model, and the categories as arguments.


Note
This project is experimental and the predictions are not always be accurate due to the quality and diversity of the data.

