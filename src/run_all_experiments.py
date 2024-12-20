from experiments import (
    run_2d_independent_experiment,
    run_3d_independent_experiment,
    run_2d_cross_dataset_experiment,
    run_3d_cross_dataset_experiment
)
import argparse
import json

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Running 2D Independent Experiments")
    run_2d_independent_experiment(config)

    print("Running 3D Independent Experiments")
    run_3d_independent_experiment(config)

    print("Running 2D Cross-Dataset Experiments")
    run_2d_cross_dataset_experiment(config)

    print("Running 3D Cross-Dataset Experiments")
    run_3d_cross_dataset_experiment(config)

    print("All experiments completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all PD classification experiments")
    parser.add_argument("config", help="Path to the configuration JSON file")
    args = parser.parse_args()
    
    main(args.config)
