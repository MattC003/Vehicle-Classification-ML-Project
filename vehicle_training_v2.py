import os
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2

parser = argparse.ArgumentParser()
parser.add_argument('--session_name', type=str, default='custom_run_v2', help='Name for the current session.')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for the optimizer.')
parser.add_argument('--max_epochs', type=int, default=50, help='Number of training epochs.')
parser.add_argument('--layers_to_train', type=int, default=15, help='Number of layers to unfreeze for training.')
args = parser.parse_args()

session_name = args.session_name
lr = args.lr
max_epochs = args.max_epochs
layers_to_train = args.layers_to_train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_directory = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive'
class_labels = ['Van', 'SUV', 'Pickup', 'Convertible', '4dr', '2dr']

batch_size = 32
num_classes = len(class_labels)
image_dimensions = (224, 224)

def setup_data():
    train_sources = []
    val_sources = []
    total_train = 0
    total_val = 0

    for index, label in enumerate(class_labels):
        path = os.path.join(data_directory, label)

        train_augment = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        train_data = train_augment.flow_from_directory(
            directory=path,
            target_size=image_dimensions,
            batch_size=batch_size,
            classes=[f"{label}_Train"],
            class_mode=None,
            shuffle=True
        )
        if train_data.samples > 0:
            train_sources.append((train_data, index))
            total_train += train_data.samples

        val_augment = ImageDataGenerator(preprocessing_function=preprocess_input)
        val_data = val_augment.flow_from_directory(
            directory=path,
            target_size=image_dimensions,
            batch_size=batch_size,
            classes=[f"{label}_Test"],
            class_mode=None,
            shuffle=False
        )
        if val_data.samples > 0:
            val_sources.append((val_data, index))
            total_val += val_data.samples

    if total_train == 0:
        raise ValueError("Training data loading error")

    return train_sources, val_sources, total_train, total_val

def calculate_weights(train_sources):
    train_labels = np.concatenate([
        np.full(data.samples, idx, dtype=int)
        for (data, idx) in train_sources
    ])
    unique_labels = np.unique(train_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=train_labels
    )
    return dict(zip(unique_labels, weights))

def data_merger(data_sources, label_count):
    while True:
        batch_images = []
        batch_targets = []
        for (data, label_idx) in data_sources:
            try:
                batch = next(data)
            except StopIteration:
                continue
            if batch.shape[0] == 0:
                continue
            batch_images.append(batch)
            targets = np.full((batch.shape[0],), label_idx, dtype=int)
            targets = to_categorical(targets, num_classes=label_count)
            batch_targets.append(targets)
        if not batch_images:
            continue
        yield (
            np.concatenate(batch_images),
            np.concatenate(batch_targets)
        )

train_sources, val_sources, train_total, val_total = setup_data()
class_weights = calculate_weights(train_sources)

train_gen = data_merger(train_sources, num_classes)
val_gen = data_merger(val_sources, num_classes)
train_steps = sum(data.samples // batch_size for data, _ in train_sources)
val_steps = sum(data.samples // batch_size for data, _ in val_sources)

model_name = 'ResNet50'
print(f"Training {model_name} model in session '{session_name}'")
input_shape = image_dimensions + (3,)

K.clear_session()

best_model_file = f'best_model_{model_name}_{session_name}.keras'
final_model_file = f'final_model_{model_name}_{session_name}.keras'
history_file = f'history_{model_name}_{session_name}.pkl'
log_file = f'training_{model_name}_{session_name}.csv'

for path in [best_model_file, final_model_file, history_file, log_file]:
    if os.path.exists(path):
        os.remove(path)

print("Beginning training")

resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

if layers_to_train > 0:
    for layer in resnet_base.layers[:-layers_to_train]:
        layer.trainable = False
    for layer in resnet_base.layers[-layers_to_train:]:
        layer.trainable = True
else:
    resnet_base.trainable = False

model = Sequential([
    resnet_base,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    BatchNormalization(),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=lr),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    ModelCheckpoint(best_model_file, save_best_only=True, monitor='val_loss', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
    CSVLogger(log_file, append=False),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=max_epochs,
    validation_data=val_gen,
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=callbacks
)

model.save(final_model_file)
print(f"Training for {model_name} model completed.\n")

log_data = pd.read_csv(log_file)
training_records = log_data.to_dict(orient='list')
with open(history_file, 'wb') as file:
    pickle.dump(training_records, file)

