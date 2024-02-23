import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

# Define image dimensions for resizing
img_width, img_height = 224, 224

# Download the dataset from Kaggle (replace with your download method)
# !kaggle datasets download shubhamgoel27/dermnet

# Define data paths
data_path = 'path/to/downloaded/dataset/dermnet'  # Replace with actual path

# Define training, validation, and test set directories
train_dir = data_path + '/train'
val_dir = data_path + '/validation'
test_dir = data_path + '/test'

# Create ImageDataGenerators for data augmentation (optional)
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation data using the generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'  # Assuming binary classification for lupus vs. other conditions
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)

# Load test data using the test generator
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary'
)


# Define CNN Model
def create_cnn_model():
    cnn_model = Sequential()

    # Add convolutional layers
    cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))
    cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D((2, 2)))

    # Flatten the output
    cnn_model.add(Flatten())

    # Add dense layers
    cnn_model.add(Dense(64, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1, activation='sigmoid'))

    # Compile the CNN model
    cnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return cnn_model


# Define VGG19 Model
def create_vgg19_model():
    # Load the pre-trained VGG19 model
    vgg19_model = VGG19(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

    # Freeze convolutional layers
    for layer in vgg19_model.layers[:-1]:
        layer.trainable = False

    # Add Global Average Pooling layer
    vgg19_model.add(GlobalAveragePooling2D())

    # Add dense layer for classification
    vgg19_model.add(Dense(1, activation='sigmoid'))

    # Compile the VGG19 model
    vgg19_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return vgg19_model


# Train and Evaluate Models
def train_and_evaluate(model):
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

    test_loss, test_acc = model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    # Optional: Visualize training and validation curves
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

# Create a model instance
model = create_cnn_model()  # Or create_vgg19_model()

# Train and evaluate the model
train_and_evaluate(model)