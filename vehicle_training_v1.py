import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--session_name', type=str, default='custom_session', help='Name for the current session.')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimization.')
parser.add_argument('--max_epochs', type=int, default=100, help='Number of training epochs.')
parser.add_argument('--layers_to_train', type=int, default=50, help='Number of layers to unfreeze.')
args = parser.parse_args()

session_name = args.session_name
lr = args.lr
max_epochs = args.max_epochs
layers_to_train = args.layers_to_train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_directory = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive'
classes = ['Van', 'SUV', 'Pickup', 'Convertible', '4dr', '2dr']

batch_size = 32
num_classes = len(classes)
image_dims = (224, 224)

def prepare_data():
    train_data = []
    val_data = []
    total_train = 0
    total_val = 0

    for index, class_name in enumerate(classes):
        path = os.path.join(data_directory, class_name)

        train_augment = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            brightness_range=(0.8, 1.2),
            channel_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_gen = train_augment.flow_from_directory(
            directory=path,
            target_size=image_dims,
            batch_size=batch_size,
            classes=[f"{class_name}_Train"],
            class_mode=None,
            shuffle=True
        )
        if train_gen.samples > 0:
            train_data.append((train_gen, index))
            total_train += train_gen.samples

        val_augment = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_gen = val_augment.flow_from_directory(
            directory=path,
            target_size=image_dims,
            batch_size=batch_size,
            classes=[f"{class_name}_Test"],
            class_mode=None,
            shuffle=False
        )
        if val_gen.samples > 0:
            val_data.append((val_gen, index))
            total_val += val_gen.samples

    if total_train == 0:
        raise ValueError("Training data loading error")

    return train_data, val_data, total_train, total_val

def calculate_class_weights(train_data):
    labels = np.concatenate([
        np.full(data.samples, idx, dtype=int)
        for (data, idx) in train_data
    ])

    unique_labels = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    return dict(zip(unique_labels, weights))

def merge_generators(data_sources, label_count, weights_map):
    while True:
        data_batches = []
        label_batches = []
        weight_batches = []
        for (source, label_index) in data_sources:
            try:
                batch = next(source)
            except StopIteration:
                continue
            if batch.shape[0] == 0:
                continue
            data_batches.append(batch)
            batch_labels = np.full((batch.shape[0],), label_index, dtype=int)
            batch_labels = to_categorical(batch_labels, num_classes=label_count)
            label_batches.append(batch_labels)
            sample_weights = np.full((batch.shape[0],), weights_map[label_index])
            weight_batches.append(sample_weights)
        if not data_batches:
            continue
        yield (
            np.concatenate(data_batches),
            np.concatenate(label_batches),
            np.concatenate(weight_batches)
        )

train_data, val_data, train_count, val_count = prepare_data()
class_weights = calculate_class_weights(train_data)

train_gen = merge_generators(train_data, num_classes, class_weights)
val_gen = merge_generators(val_data, num_classes, class_weights)
train_steps = sum(data.samples // batch_size for data, _ in train_data)
val_steps = sum(data.samples // batch_size for data, _ in val_data)

model_name = 'ResNet50'
print(f"Training {model_name} with session '{session_name}'")
input_dims = image_dims + (3,)

K.clear_session()

best_model_file = f'best_model_{model_name}_{session_name}.keras'
final_model_file = f'final_model_{model_name}_{session_name}.keras'
history_file = f'history_{model_name}_{session_name}.pkl'
log_file = f'training_{model_name}_{session_name}.csv'

for file_path in [best_model_file, final_model_file, history_file, log_file]:
    if os.path.exists(file_path):
        os.remove(file_path)

print("Beginning training.")

base_net = ResNet50(weights='imagenet', include_top=False, input_shape=input_dims)
if layers_to_train > 0:
    for layer in base_net.layers[:-layers_to_train]:
        layer.trainable = False
    for layer in base_net.layers[-layers_to_train:]:
        layer.trainable = True
else:
    base_net.trainable = False

model = Sequential([
    base_net,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=SGD(learning_rate=lr, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(best_model_file, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    CSVLogger(log_file, append=False),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
]

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=max_epochs,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=callbacks
)

model.save(final_model_file)
print(f"Training for {model_name} complete.\n")

log_data = pd.read_csv(log_file)
history_data = log_data.to_dict(orient='list')
with open(history_file, 'wb') as file:
    pickle.dump(history_data, file)

