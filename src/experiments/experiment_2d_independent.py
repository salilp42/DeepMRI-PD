import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from ..data_loading import load_ppmi_data, load_taowu_data, load_neurocon_data
from ..models import CNN2D, ConvKAN2D, GCN2D
from ..utils import calculate_metrics, calculate_ci
import numpy as np
import joblib
import time

def run_2d_independent_experiment(config):
    datasets = {
        'PPMI': load_ppmi_data,
        'Tao Wu': load_taowu_data,
        'NEUROCON': load_neurocon_data
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name, load_func in datasets.items():
        print(f"\nProcessing {dataset_name} dataset")
        
        # Load data
        slices, labels, _ = load_func(config['data_dir'])
        
        # Initialize models
        models = {
            'CNN2D': CNN2D(),
            'ConvKAN2D': ConvKAN2D(input_channels=1, hidden_dim=config['hidden_dim'], output_dim=2),
            'GCN2D': GCN2D(num_node_features=2, hidden_channels=64, num_classes=2)
        }

        results = {}

        skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)

        for model_name, model in models.items():
            print(f"\nTraining and evaluating {model_name}")
            model.to(device)

            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(slices, labels), 1):
                print(f"\nFold {fold}")

                # Prepare data loaders
                train_loader = DataLoader(list(zip(slices[train_idx], labels[train_idx])), 
                                          batch_size=config['batch_size'], shuffle=True)
                val_loader = DataLoader(list(zip(slices[val_idx], labels[val_idx])), 
                                        batch_size=config['batch_size'])

                # Training
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

                for epoch in range(config['num_epochs']):
                    model.train()
                    for batch in train_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        optimizer.step()

                # Evaluation
                model.eval()
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch in val_loader:
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(targets.cpu().numpy())

                # Calculate metrics
                fold_metrics = calculate_metrics(all_labels, all_preds)
                fold_results.append(fold_metrics)

            # Calculate average metrics and confidence intervals
            avg_metrics = {metric: np.mean([fold[metric] for fold in fold_results]) for metric in fold_results[0]}
            ci_metrics = {metric: calculate_ci([fold[metric] for fold in fold_results]) for metric in fold_results[0]}

            results[model_name] = {
                'avg_metrics': avg_metrics,
                'ci_metrics': ci_metrics,
                'fold_results': fold_results
            }

        # Save results
        joblib.dump(results, f'results/2d_independent_{dataset_name}.joblib')

    return results

if __name__ == "__main__":
    config = {
        'data_dir': '/path/to/data',
        'n_splits': 5,
        'batch_size': 32,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'num_epochs': 100
    }
    run_2d_independent_experiment(config)
