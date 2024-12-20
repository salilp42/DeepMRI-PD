import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ..data_loading import load_ppmi_data, load_taowu_data, load_neurocon_data
from ..models import CNN3D, ConvKAN3D, GCN3D
from ..utils import calculate_metrics, calculate_ci
import numpy as np
import joblib

def run_3d_cross_dataset_experiment(config):
    datasets = {
        'PPMI': load_ppmi_data,
        'Tao Wu': load_taowu_data,
        'NEUROCON': load_neurocon_data
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_data = {}
    for dataset_name, load_func in datasets.items():
        volumes, labels, _ = load_func(config['data_dir'])
        all_data[dataset_name] = (volumes, labels)

    models = {
        'CNN3D': CNN3D(),
        'ConvKAN3D': ConvKAN3D(input_channels=1, hidden_dim=config['hidden_dim'], output_dim=2),
        'GCN3D': GCN3D(num_node_features=5, hidden_channels=64, num_classes=2)
    }

    results = {}

    for test_dataset_name in datasets:
        print(f"\nUsing {test_dataset_name} as the test dataset")

        test_data, test_labels = all_data[test_dataset_name]
        train_data, train_labels = [], []

        for train_dataset_name in datasets:
            if train_dataset_name != test_dataset_name:
                data, labels = all_data[train_dataset_name]
                train_data.append(data)
                train_labels.append(labels)

        train_data = np.concatenate(train_data)
        train_labels = np.concatenate(train_labels)

        # Convert to PyTorch tensors
        train_data = torch.FloatTensor(train_data)
        train_labels = torch.LongTensor(train_labels)
        test_data = torch.FloatTensor(test_data)
        test_labels = torch.LongTensor(test_labels)

        # Create data loaders
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])

        for model_name, model in models.items():
            print(f"\nTraining and evaluating {model_name}")
            model.to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

            # Training
            for epoch in range(config['num_epochs']):
                model.train()
                for batch_data, batch_labels in train_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()

            # Evaluation
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                    outputs = model(batch_data)
                    preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels.cpu().numpy())

            # Calculate metrics
            metrics = calculate_metrics(all_labels, all_preds)
            results[f"{model_name}_{test_dataset_name}"] = metrics

    # Save results
    joblib.dump(results, f'results/3d_cross_dataset.joblib')

    return results

if __name__ == "__main__":
    config = {
        'data_dir': '/path/to/data',
        'batch_size': 8,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'num_epochs': 100
    }
    run_3d_cross_dataset_experiment(config)
