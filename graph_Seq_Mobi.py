import re
import matplotlib.pyplot as plt
import pickle

def performance_metrics(log_file, model_list):
    model_data = {name: {'epoch': [], 'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []} for name in model_list}
    active_model = None
    with open(log_file, 'r') as log:
        for line in log:
            line = line.strip().replace('\b', '').replace('\r', '')
            
            start_model = re.match(r'Training (\w+) model\.\.\.', line)
            if start_model:
                model_name = start_model.group(1)
                if model_name in model_list:
                    active_model = model_name
                    print(f"Processing {active_model}...")
                else:
                    active_model = None
                continue
            
            end_model = re.match(r'Finished training (\w+) model\.', line)
            if end_model:
                active_model = None
                continue
            
            if active_model:
                epoch_match = re.match(r'Epoch (\d+)/(\d+)', line)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                    if not model_data[active_model]['epoch'] or model_data[active_model]['epoch'][-1] != epoch_num:
                        model_data[active_model]['epoch'].append(epoch_num)
                    continue
                
                metrics_match = re.match(
                    r'.* - loss: ([\d\.e\-]+) - accuracy: ([\d\.e\-]+) - val_loss: ([\d\.e\-]+) - val_accuracy: ([\d\.e\-]+)',
                    line
                )
                if metrics_match:
                    loss, acc, val_loss, val_acc = map(float, metrics_match.groups())
                    model_data[active_model]['loss'].append(loss)
                    model_data[active_model]['accuracy'].append(acc)
                    model_data[active_model]['val_loss'].append(val_loss)
                    model_data[active_model]['val_accuracy'].append(val_acc)
                    continue

    return model_data

def create_graph(data, models):
    plt.figure(figsize=(10, 6))
    for model in models:
        epochs = range(1, len(data[model]['accuracy']) + 1)
        plt.plot(epochs, data[model]['accuracy'], label=f'{model} Train Accuracy')
        plt.plot(epochs, data[model]['val_accuracy'], label=f'{model} Val Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('model_accuracy_comparison.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    for model in models:
        epochs = range(1, len(data[model]['loss']) + 1)
        plt.plot(epochs, data[model]['loss'], label=f'{model} Train Loss')
        plt.plot(epochs, data[model]['val_loss'], label=f'{model} Val Loss')
    plt.title('Model Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('model_loss_comparison.png')
    plt.show()

log_path = '/home/yat20006/BigDataProjectAttempt2/Vehicles/predicting_vehicles/archive/vehicle_output.log'
model_list = ['Sequential', 'MobileNet']

data = performance_metrics(log_path, model_list)

for model in model_list:
    with open(f'history_{model}.pkl', 'wb') as file:
        pickle.dump(data[model], file)

create_graph(data, model_list)
