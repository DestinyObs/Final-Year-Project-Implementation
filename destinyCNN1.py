from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_curve,auc
from sklearn.metrics import confusion_matrix
import seaborn as sns


warnings.filterwarnings("ignore") # suppress any python warning message



mysize = (30, 30)
sub = ["Acne_Rosacea", "Actinic_Keratosis","Atopic_Dermatitis", "Bullous_Disease", "Cellulitis_Impetigo", 
       "Eczema", "Exanthems", "Hair_Loss_Alopecia", "Herpes_HPV", "Light_Diseases", 
       "Lupus", "Melanoma", "Nail_Fungus", "Poison_Ivy_Dermatitis", "Psoriasis_Lichen_Planus", 
       "Scabies_Lyme", "Seborrheic_Keratoses", "Systemic_Disease", 
       "Tinea_Ringworm_Candidiasis", "Urticaria_Hives", "Vascular_Tumors", "Vasculitis", "Warts_Molluscum"]


def read_images_from_folder(folder_path):
    image_list = []
    label_list = []

    subfolders = os.listdir(folder_path)
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            images = os.listdir(subfolder_path)
            for image_file in images:
                image_path = os.path.join(subfolder_path, image_file)
                if os.path.isfile(image_path):
                    # Read the image using PIL
                    image = Image.open(image_path).convert('L')
                    image = image.resize(mysize)
                    image = np.array(image)
                    image = image.astype('float32')
                    image /= 255
                    if image is not None:
                        image_list.append(image)
                        label_list.append(subfolder)

    return image_list, label_list





folder_path = 'D:\\datasets\\lupusDisease'

train_folder_path = os.path.join(folder_path, 'train')
train_images, train_labels = read_images_from_folder(train_folder_path)

# Read images and labels from the test subfolder
test_folder_path = os.path.join(folder_path, 'test')
test_images, test_labels = read_images_from_folder(test_folder_path)

trainX = np.array(train_images)
trainY = np.array(train_labels)

testX = np.array(test_images)
testY = np.array(test_labels)

# Initialize the label encoder
label_encoder = LabelEncoder()

# Fit and transform the training labels
trainY_encoded = label_encoder.fit_transform(trainY)

# Transform the test labels
testY_encoded = label_encoder.transform(testY)



trainY_categorical = to_categorical(trainY_encoded, num_classes=23)
testY_categorical = to_categorical(testY_encoded, num_classes=23)


# num_classes = len(np.unique(trainY_encoded))
# trainY_categorical = to_categorical(trainY_encoded, num_classes=num_classes)
# testY_categorical = to_categorical(testY_encoded, num_classes=num_classes)




# trainY_categorical = to_categorical(trainY, num_classes=4)
# testY_categorical = to_categorical(testY, num_classes=4)



model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(30, 30, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

# Adding dilated convolutional layer
# model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', dilation_rate=2))

# without dilation-ordinary CNN
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))




model.add(Flatten())

# model.add(Dense(64, activation='relu'))
model.add(Dense(23, activation='softmax'))


# ===========================Compiling the model==========================================================
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===========================Training the model==========================================================
hs=model.fit(trainX, trainY_categorical, epochs=250, validation_data=(testX, testY_categorical))

model.save("lupusDiseasePredictionModel.h5")

# ===================After training, evaluate the model and calculate the metrics=========================


# Make predictions on the test set
predictions = model.predict(testX)
predicted_classes = np.argmax(predictions, axis=1)

# Convert one-hot encoded labels back to categorical labels
true_classes = np.argmax(testY_categorical, axis=1)

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Plotting Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Acne_Rosacea", "Actinic_Keratosis","Atopic_Dermatitis", "Bullous_Disease", 
                         "Cellulitis_Impetigo","Eczema", "Exanthems", "Hair_Loss_Alopecia", "Herpes_HPV", 
                         "Light_Diseases","Lupus", "Melanoma", "Nail_Fungus", "Poison_Ivy_Dermatitis", 
                         "Psoriasis_Lichen_Planus","Scabies_Lyme", "Seborrheic_Keratoses", "Systemic_Disease",
                         "Tinea_Ringworm_Candidiasis", "Urticaria_Hives", "Vascular_Tumors", "Vasculitis", 
                         "Warts_Molluscum"],

             yticklabels=["Acne_Rosacea", "Actinic_Keratosis","Atopic_Dermatitis", "Bullous_Disease", 
                         "Cellulitis_Impetigo","Eczema", "Exanthems", "Hair_Loss_Alopecia", "Herpes_HPV", 
                         "Light_Diseases","Lupus", "Melanoma", "Nail_Fungus", "Poison_Ivy_Dermatitis", 
                         "Psoriasis_Lichen_Planus","Scabies_Lyme", "Seborrheic_Keratoses", "Systemic_Disease",
                         "Tinea_Ringworm_Candidiasis", "Urticaria_Hives", "Vascular_Tumors", "Vasculitis", "Warts_Molluscum"])

    
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Lupus Disease Analysis')
plt.show()



# y_pred = model.predict(testX)
# y_pred_classes = np.argmax(y_pred, axis=1)

accuracy = accuracy_score(testY_encoded, predicted_classes)
precision = precision_score(testY_encoded, predicted_classes, average='weighted')
recall = recall_score(testY_encoded, predicted_classes, average='weighted')
f1 = f1_score(testY_encoded, predicted_classes, average='weighted')


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



# predictions = model.predict(testX)
# predicted_classes = np.argmax(predictions, axis=1)

# y_pred_prob = model.predict(testX)
fpr, tpr, thresholds = roc_curve(testY_categorical[:, 1], predictions[:, 1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Plot the training loss and validation loss
plt.plot(hs.history['loss'], label='Training Loss')
plt.plot(hs.history['val_loss'], label='Validation Loss')
plt.title('Traning  and Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Plot the training accuracy and validation accuracy
plt.plot(hs.history['accuracy'], label='Training Accuracy')
plt.plot(hs.history['val_accuracy'], label='Validation Accuracy')
plt.title('Traning  and Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()