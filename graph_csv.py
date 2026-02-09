import pandas as pd
import matplotlib.pyplot as plt

csv_files = [
    r"C:\Users\Yarid\Desktop\archive\ResNet50 Files\training_ResNet50_default_run.csv",
    r"C:\Users\Yarid\Desktop\archive\ResNet50 Files\training_ResNet50_improved_run.csv",
    r"C:\Users\Yarid\Desktop\archive\ResNet50 Files\training_ResNet50_improved_run_v2.csv",
    r"C:\Users\Yarid\Desktop\archive\ResNet50 Files\training_ResNet50_improved_run_v3.csv"
]

run_labels = [
    "Default Run",
    "Improved Run",
    "Improved Run v2",
    "Improved Run v3"
]

output_dir = r"C:\Users\Yarid\Desktop\archive\Python Code Files"

plt.figure(figsize=(10, 6))
for file, label in zip(csv_files, run_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy'], label=label)
plt.title("Training Accuracy For ResNet50 Models")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/training_accuracy_ResNet50.png")
plt.close()

plt.figure(figsize=(10, 6))
for file, label in zip(csv_files, run_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['val_accuracy'], label=label)
plt.title("Validation Accuracy For ResNet50 Models")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/validation_accuracy_ResNet50.png")
plt.close()

plt.figure(figsize=(10, 6))
for file, label in zip(csv_files, run_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['loss'], label=label)
plt.title("Training Loss For ResNet50 Models")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/training_loss_ResNet50.png")
plt.close()

plt.figure(figsize=(10, 6))
for file, label in zip(csv_files, run_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['val_loss'], label=label)
plt.title("Validation Loss For ResNet50 Models")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(f"{output_dir}/validation_loss_ResNet50.png")
plt.close()