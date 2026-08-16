import numpy as np
import pickle
import cv2
import os
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam

EPOCHS = 10
INIT_LR = 1e-3
BS = 32
default_image_size = (256, 256)
width, height, depth = 256, 256, 3

# Candidate paths to search for dataset
candidate_paths = [
    '../input/plantvillage/',
    './plantvillage',
    './PlantVillage',
    './dataset',
    './data',
    './sample_images'
]

directory_root = None
for path in candidate_paths:
    if os.path.exists(path):
        directory_root = path
        break

if directory_root is None:
    directory_root = '../input/plantvillage/'

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else:
            return None
    except Exception as e:
        print(f"Error reading image '{image_dir}': {e}")
        return None

image_list, label_list = [], []

print(f"[INFO] Looking for dataset in: '{directory_root}'...")

if os.path.exists(directory_root):
    # Walk directory to support any nesting level (1-level, 2-level, or flat sample directory)
    for root, dirs, files in os.walk(directory_root):
        valid_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")]
        if valid_files:
            # Determine class label from folder name or file prefix
            folder_name = os.path.basename(root)
            if folder_name and folder_name != os.path.basename(directory_root):
                label = folder_name
            else:
                label = "Sample_Class"

            for file in valid_files[:200]: # limit to 200 per folder for speed
                img_path = os.path.join(root, file)
                if label == "Sample_Class":
                    # Derive label from filename (e.g. Tomato_Healthy_Sample.jpg -> Tomato_Healthy)
                    file_label = file.rsplit("_Sample", 1)[0].rsplit(".", 1)[0]
                    curr_label = file_label
                else:
                    curr_label = label
                    
                arr = convert_image_to_array(img_path)
                if arr is not None:
                    image_list.append(arr)
                    label_list.append(curr_label)

    print(f"[INFO] Loaded {len(image_list)} images across {len(set(label_list))} classes.")
else:
    print(f"[WARNING] Dataset directory '{directory_root}' not found.")
    print("[TIP] To train on your full dataset, download the PlantVillage dataset from Kaggle and place it in './plantvillage' or update 'directory_root' in PlantDiseaseDetection.py.")

if len(image_list) > 0:
    # Target Encoding
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    
    # If 2 classes or 1 class, fit_transform returns (N, 1) or (N, 0), reshape to 2D one-hot
    if len(label_binarizer.classes_) == 2:
        image_labels = np.hstack((1 - image_labels, image_labels))
    elif len(label_binarizer.classes_) == 1:
        image_labels = np.ones((len(label_list), 1))

    pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
    n_classes = len(label_binarizer.classes_)

    print(f"[INFO] Detected {n_classes} classes: {list(label_binarizer.classes_)}")

    np_image_list = np.array(image_list, dtype=np.float32) / 255.0

    print("[INFO] Splitting dataset into train and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

    actual_bs = min(BS, max(1, len(x_train)))

    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, 
        zoom_range=0.2, horizontal_flip=True, 
        fill_mode="nearest")

    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1
    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(n_classes))
    model.add(Activation("softmax"))

    opt = Adam(learning_rate=INIT_LR)
    loss_fn = "categorical_crossentropy" if n_classes > 1 else "binary_crossentropy"
    model.compile(loss=loss_fn, optimizer=opt, metrics=["accuracy"])
    
    print("[INFO] Model Architecture Summary:")
    model.summary()

    print("[INFO] Training CNN network...")
    steps_per_epoch = max(1, len(x_train) // actual_bs)
    
    history = model.fit(
        aug.flow(x_train, y_train, batch_size=actual_bs),
        validation_data=(x_test, y_test),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, 
        verbose=1
    )

    acc = history.history.get('accuracy', history.history.get('acc', []))
    val_acc = history.history.get('val_accuracy', history.history.get('val_acc', []))
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, 'b', label='Training Accuracy')
    plt.plot(epochs_range, val_acc, 'r', label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, 'b', label='Training Loss')
    plt.plot(epochs_range, val_loss, 'r', label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plot.png')
    plt.close()

    print("[INFO] Evaluating model performance...")
    scores = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {scores[1]*100:.2f}%")

    print("[INFO] Saving trained model and binarizer...")
    model.save('plant_disease_model.h5')
    pickle.dump(model, open('cnn_model.pkl', 'wb'))
    print("[SUCCESS] Model successfully saved to 'plant_disease_model.h5' and 'cnn_model.pkl'.")