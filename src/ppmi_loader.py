import os
import numpy as np
import pandas as pd
from typing import Tuple, List
from utils.preprocessing import resize_volume

def load_ppmi_data(processed_dir: str, info_csv: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess PPMI dataset.
    
    Args:
        processed_dir (str): Directory containing processed MRI scans.
        info_csv (str): Path to CSV file with scan information.
    
    Returns:
        Tuple containing:
        - np.ndarray: Preprocessed MRI data
        - np.ndarray: Labels (0 for control, 1 for PD)
        - List[str]: Subject identifiers
    """
    info_df = pd.read_csv(info_csv)
    mri_data = []
    labels = []
    identifiers = []

    for _, row in info_df.iterrows():
        subject_folder = row['ID_and_Label']
        npy_file = os.path.join(processed_dir, subject_folder, 'processed_scan.npy')
        if os.path.exists(npy_file):
            scan = np.load(npy_file)
            resized_scan = resize_volume(scan)
            mri_data.append(resized_scan)
            labels.append(1 if row['Group'] == 'PD' else 0)
            identifiers.append(subject_folder)
        else:
            print(f"Warning: File not found - {npy_file}")

    if len(mri_data) == 0:
        raise ValueError("No preprocessed MRI scans found. Please check the file paths and ensure the preprocessed scans exist.")

    return np.array(mri_data), np.array(labels), identifiers
