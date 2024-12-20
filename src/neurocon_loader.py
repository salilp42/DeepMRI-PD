import os
import numpy as np
import nibabel as nib
from typing import Tuple, List
from ..utils.preprocessing import resize_volume, normalize_volume

def load_neurocon_data(main_folder: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess NEUROCON dataset.
    
    Args:
        main_folder (str): Main directory containing the NEUROCON dataset.
    
    Returns:
        Tuple containing:
        - np.ndarray: Preprocessed MRI data
        - np.ndarray: Labels (0 for control, 1 for PD)
        - List[str]: Subject identifiers
    """
    mri_data = []
    labels = []
    identifiers = []

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            anat_folder = os.path.join(subfolder_path, 'anat')
            if os.path.exists(anat_folder):
                for file in os.listdir(anat_folder):
                    if file.endswith('.nii.gz'):
                        file_path = os.path.join(anat_folder, file)
                        img = nib.load(file_path).get_fdata()
                        img = normalize_volume(img)
                        img = resize_volume(img)
                        mri_data.append(img)
                        labels.append(1 if 'patient' in subfolder else 0)
                        identifiers.append(subfolder + '_' + file)

    return np.array(mri_data), np.array(labels), identifiers
