import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tensorflow.keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--session_name', type=str, default='default_session', help='Name for this session.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate.')
parser.add_argument('--num_epochs', type=int, default=50, help='Total training epochs.')
parser.add_argument('--trainable_layers', type=int, default=10, help='Layers to unfreeze.')
args = parser.parse_args()

session_name = args.session_name
lr = args.lr
epochs = args.num_epochs
trainable_layers = args.trainable_layers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_dir = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive'
vehicle_types = ['Van', 'SUV', 'Pickup', 'Convertible', '4dr', '2dr']

batch_size = 32
num_labels = len(vehicle_types)
image_dims = (224, 224)

def setup_data():
    train_sets = []
    validation_sets = []
    total_train = 0
    total_validation = 0

    for idx, vehicle in enumerate(vehicle_types):
        path = os.path.join(data_dir, vehicle)

        train_aug = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )
        train_data = train_aug.flow_from_directory(
            directory=path,
            target_size=image_dims,
            batch_size=batch_size,
            classes=[f"{vehicle}_Train"],
            class_mode=None,
            shuffle=True
        )
        if train_data.samples > 0:
            train_sets.append((train_data, idx))
            total_train += train_data.samples

        validation_aug = ImageDataGenerator(preprocessing_function=preprocess_input)
        validation_data = validation_aug.flow_from_directory(
            directory=path,
            target_size=image_dims,
            batch_size=batch_size,
            classes=[f"{vehicle}_Test"],
            class_mode=None,
            shuffle=False
        )
        if validation_data.samples > 0:
            validation_sets.append((validation_data, idx))
            total_validation += validation_data.samples

    if total_train == 0:
        raise ValueError("Training data loading error")

    return train_sets, validation_sets, total_train, total_validation

def calculate_weights(train_sets):
    train_classes = np.concatenate([
        np.full(data.samples, label, dtype=int)
        for (data, label) in train_sets
    ])

    unique_classes = np.unique(train_classes)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_classes,
        y=train_classes
    )
    return dict(zip(unique_classes, class_weights))

def data_merger(data_sources, labels_count, weights_map):
    while True:
        data_batches = []
        label_batches = []
        weight_batches = []
        for (source, idx) in data_sources:
            try:
                batch = next(source)
            except StopIteration:
                continue
            if batch.shape[0] == 0:
                continue
            data_batches.append(batch)
            batch_labels = np.full((batch.shape[0],), idx, dtype=int)
            batch_labels = to_categorical(batch_labels, num_classes=labels_count)
            label_batches.append(batch_labels)
            sample_weights = np.full((batch.shape[0],), weights_map[idx])
            weight_batches.append(sample_weights)
        if not data_batches:
            continue
        yield (
            np.concatenate(data_batches),
            np.concatenate(label_batches),
            np.concatenate(weight_batches)
        )

train_sets, validation_sets, train_total, validation_total = setup_data()
weights_map = calculate_weights(train_sets)

train_data = data_merger(train_sets, num_labels, weights_map)
validation_data = data_merger(validation_sets, num_labels, weights_map)
train_steps = sum(data.samples // batch_size for data, _ in train_sets)
validation_steps = sum(data.samples // batch_size for data, _ in validation_sets)

model_type = 'ResNet50'
print(f"Training {model_type} with session '{session_name}'")
input_size = image_dims + (3,)

K.clear_session()

best_model_file = f'best_model_{model_type}_{session_name}.keras'
final_model_file = f'final_model_{model_type}_{session_name}.keras'
history_file_name = f'history_{model_type}_{session_name}.pkl'
log_file_name = f'training_{model_type}_{session_name}.csv'

for file in [best_model_file, final_model_file, history_file_name, log_file_name]:
    if os.path.exists(file):
        os.remove(file)

print("Beginning training")

resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_size)
if trainable_layers > 0:
    for layer in resnet_base.layers[:-trainable_layers]:
        layer.trainable = False
else:
    resnet_base.trainable = False

model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_labels, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

training_callbacks = [
    ModelCheckpoint(best_model_file, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    CSVLogger(log_file_name, append=False),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
]

history = model.fit(
    train_data,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_data,
    validation_steps=validation_steps,
    callbacks=training_callbacks
)

model.save(final_model_file)
print(f"Training finished for {model_type}.\n")

training_data = pd.read_csv(log_file_name)
history_records = training_data.to_dict(orient='list')
with open(history_file_name, 'wb') as file:
    pickle.dump(history_records, file)


