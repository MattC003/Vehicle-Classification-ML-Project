import os
import pickle
import matplotlib.pyplot as plt

history_files = {
    'ResNet50': '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive/history_ResNet50_default_run.pkl',
    'MobileNet': '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive/history_MobileNet.pkl',
    'Sequential': '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive/history_Sequential.pkl'
}

histories = {model_name: pickle.load(open(file_path, 'rb'))
             for model_name, file_path in history_files.items() if os.path.exists(file_path)}

if not histories:
    exit()

plt.figure(figsize=(10, 6))
for model_name, history in histories.items():
    epochs = range(1, len(history['accuracy']) + 1)
    plt.plot(epochs, history['accuracy'], label=f'{model_name} Training Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_accuracy_comparison_all_models.png')
plt.show()

plt.figure(figsize=(10, 6))
for model_name, history in histories.items():
    epochs = range(1, len(history['loss']) + 1)
    plt.plot(epochs, history['loss'], label=f'{model_name} Training Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_loss_comparison_all_models.png')
plt.show()
