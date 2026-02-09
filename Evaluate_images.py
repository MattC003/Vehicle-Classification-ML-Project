import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive'
labels = ['Van', 'SUV', 'Pickup', 'Convertible', '4dr', '2dr']

batch_size = 32
num_classes = len(labels)
img_size = (224, 224)

model_file = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive/final_model_ResNet50_improved_run_v3.keras'
model = load_model(model_file)

test_images = []
test_targets = []
label_mapping = {label: idx for idx, label in enumerate(labels)}
print("Label Mapping:", label_mapping)

for label in labels:
    test_folder = os.path.join(data_dir, label, f"{label}_Test")
    if not os.path.exists(test_folder):
        print(f"Folder not found error")
        continue
    files = os.listdir(test_folder)
    print(f"Found {len(files)} files for label '{label}'.")
    for file_name in files:
        img_path = os.path.join(test_folder, file_name)
        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            test_images.append(img_array)
            test_targets.append(label_mapping[label])
        except Exception as error:
            print(f"Image loading error '{img_path}': {error}")

X_test = np.array(test_images)
y_test = np.array(test_targets)

print(f"Loaded {len(y_test)} test samples.")

y_test_encoded = to_categorical(y_test, num_classes=num_classes)
results = model.evaluate(X_test, y_test_encoded, batch_size=batch_size, verbose=1)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f}")

y_pred_probs = model.predict(X_test, batch_size=batch_size, verbose=1)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels, target_names=labels))

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred_labels)
print(conf_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix_plot.png')
plt.show()
