import cv2
import os
import pickle
import glob
import numpy as np
from tensorflow.keras.applications import ResNet50, VGG19, EfficientNetB5
# from tensorflow.keras.utils import load_img For tf >= 2.9
from tensorflow.keras.preprocessing.image import load_img, img_to_array


# Load images in list
image_files = glob.glob("image_path_list")

# Path to trained model
model_file = "log_reg.pickle"

# Image shape
img_shape = (224,224)


# Extract image features.
def prepare_image(img_path, backbone, tgt_size=(224, 224)):
    """
        Function takes a path to an image, the model to be used for feature extraction and
        the image input shape to the feature extractor.
        It returns a numpy array of extracted features.
    """
    image = load_img(img_path, target_size=tgt_size) # Default target size assumes VGG19 input. Edit as required.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    features = backbone.predict(image) # Extract features using backbone
    return features.flatten().reshape(1, -1) # Reshape to prepare input for model


# Mealiness prediction
def predict_mealiness(X, estimator):
    """ Takes an array and a trained model as parameters.
        Returns an array of predictions for mealiness.
    """
    return estimator.predict(X)



# This section runs the code

# Load mealiness prediction model
with open(model_file, "rb") as p_file:
    model = pickle.load(p_file)


# Using EfficientNetB5 as feature extractor
feature_extractor = EfficientNetB5(weights="imagenet", include_top=False)

# Get features for a single image
single_image_features = prepare_image(image_files[1], feature_extractor)

# Optional for extracting and predicting for a batch of images (uncomment to run)
# multi_image_features = np.array([prepare_image(i, feature_extractor) for i in image_files])

# Predictions is an array of float values for the mealiness
mealiness = predict_mealiness(single_image_features, model)
# mealiness = predict_mealiness(multi_image_features, model)

print(mealiness)
# print(mealiness[0]) # Get value inside array

