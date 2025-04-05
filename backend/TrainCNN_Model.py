import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import os

# Define paths
BASE_PATH = r"D:\GitHub\Google-Meet-Sign-Language-Convertor\backend"
DATA_PATH = os.path.join(BASE_PATH,"SplitDataSet")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "best_model.h5")
CLASSES_SAVE_PATH = os.path.join(BASE_PATH, "classes.npy")

# Use GPU
with tf.device('/gpu:0'):
    img_width = 150
    img_height = 150
    input_shape = (img_width, img_height, 3)

    # Define model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(35, activation='softmax')
    ])

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Data augmentation
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    valid_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    print("Train:")
    train_generator = train_datagen.flow_from_directory(
        directory=os.path.join(DATA_PATH, "train"),
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    print("Validation:")
    valid_generator = valid_datagen.flow_from_directory(
        directory=os.path.join(DATA_PATH, "val"),
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    print("Test:")
    test_generator = test_datagen.flow_from_directory(
        directory=os.path.join(DATA_PATH, "test"),
        target_size=(150, 150),
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size

    print("Step size:")
    print(f"--- Train: {STEP_SIZE_TRAIN}")
    print(f"--- Validation: {STEP_SIZE_VALID}")
    
    model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN, validation_data=valid_generator,
              validation_steps=STEP_SIZE_VALID, epochs=1)

    model.evaluate(valid_generator)
    test_generator.reset()

    print("\n\n Prediction:")
    pred = model.predict(test_generator, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    np.save(CLASSES_SAVE_PATH, labels)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    print("\n\nGenerating CSV 'results.csv'")
    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
    results.to_csv(os.path.join(BASE_PATH, "results.csv"), index=False)

    model.save(MODEL_SAVE_PATH)

print("\n\nExiting program\n\n")
