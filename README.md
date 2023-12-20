# Christmas Gift Predictor

This project uses Support Vector Classifier (SVC) and Convolutional Neural Network (CNN) models to predict what's inside Christmas presents based on images. The pre determined catagories of christmas presents are: books,bottles,letters,posters,shirts,socks,sportsStuff,tech,treats,toys.
The problem with this project is that almost all presents are cubes so even to achieve 50-60% success rate would be good.

## Data

The data for this project was collected manually due to the lack of sufficient online resources. The images were obtained by print screening presents, so the data quality may vary.

## Models

Two types of models are used in this project:

1. **SVC Model**: A custom SVC model was created for this project. SVC is a type of supervised learning model that analyzes data and recognizes patterns.

2. **CNN Model**: A pre-trained VGG16 model was used for the CNN. The top layers of the VGG16 model were frozen to prevent them from being updated during training. This approach was found to yield the best results for this project.

## Usage

To use this project, run the SVC and CNN files to get the models. After that take a photo of a christmas present and replace the image_path in use.py. The `predict_category_cnn` function in `use.py` can be used to predict the category of the gift with CNN.'predict_category' on the other hand uses SVC. The function takes the path to the image, the path to the model, and the categories as arguments.

## Results
The SVC models gives consistatly bad resuslts all the time getting about 35% of the presents correct.
The CNN model is probably overfitting or something because the probabilities of each category are almost always either close to 0 or 100.
That being said the CNN model gives more correct answers than the SVC with success rate of maybe 50%, ofcourse depending on the photo.
![Alt text](https://github.com/Sekseli3/GiftML/blob/main/photo2.png)
![Alt text](https://github.com/Sekseli3/GiftML/blob/main/photo1.png)
### Note
This project is experimental and the predictions are not always be accurate due to the quality and diversity of the data.

