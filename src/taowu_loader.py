import os
import numpy as np
import nibabel as nib
from typing import Tuple, List
from tqdm import tqdm
from ..utils.preprocessing import resize_volume

def load_taowu_data(main_folder: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess Tao Wu dataset.
    
    Args:
        main_folder (str): Main directory containing the Tao Wu dataset.
    
    Returns:
        Tuple containing:
        - np.ndarray: Preprocessed MRI data
        - np.ndarray: Labels (0 for control, 1 for PD)
        - List[str]: Subject identifiers
    """
    mri_data = []
    labels = []
    identifiers = []

    subfolders = [f for f in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, f))]

    for subfolder in tqdm(subfolders, desc="Loading Tao Wu data"):
        anat_folder = os.path.join(main_folder, subfolder, 'anat')
        if os.path.exists(anat_folder):
            files = [f for f in os.listdir(anat_folder) if f.endswith('.nii.gz')]
            for file in files:
                file_path = os.path.join(anat_folder, file)
                img = nib.load(file_path)
                data = img.get_fdata()
                resized_data = resize_volume(data)
                mri_data.append(resized_data)
                labels.append(1 if 'patient' in subfolder else 0)
                identifiers.append(subfolder + '_' + file)

    return np.array(mri_data), np.array(labels), identifiers
