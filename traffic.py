import cv2
import numpy as np
import os
import sys
import tensorflow as tf
import keras

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    print(f"images: {images}")
    print(f"labels: {labels}")

    # Split data into training and testing sets
    labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=str(2))

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    # ([[]], [labels])
    images = []
    labels = []
    for i in range(NUM_CATEGORIES):
        folder_path = os.path.join(data_dir, f"{i}") # Ex. gtsrb/0
        entries = os.listdir(folder_path)
        for j, entry in enumerate(entries):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path):
                try:
                    #has_reader = cv2.haveImageReader(entry)
                    #print(f"has reader: {has_reader}")

                    #if not has_reader:
                    #img_from_buff = cv2.imdecode()
                
                    img = cv2.imread(os.path.join(folder_path, entry))
                    if img is not None:

                        if i == 0 and j == 0:
                            print(img.shape)

                            img.resize((30, 30, 3))
                        
                        if i == 0 and j == 0:
                            print(img.shape)
                    
                        images.append(img / 255)
                        labels.append(i)
                except:
                    raise Exception("an error ocurred while trying to read the image")
    
    return (images, labels)
        

    
def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=16, kernel_size=(3, 3), input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    model.compile(
        optimizer=str(keras.optimizers.Adam),
        loss=keras.losses.BinaryCrossentropy, 
        metrics=[keras.metrics.Accuracy]
    )

    return model 


if __name__ == "__main__":
    main()
