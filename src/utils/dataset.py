import json
import os
from torch.utils.data import Dataset
import logging

logger = logging.getLogger(__name__)

class CocoFinetuneDataset(Dataset):
    """Custom PyTorch Dataset for COCO fine-tuning data."""
    def __init__(self, json_path, data_dir):
        """
        Args:
            json_path (str): Path to dataset.json
            data_dir (str): Directory containing the 'images' subdirectory
        """
        self.data_dir = os.path.join(data_dir, "images")  # Use images/ subdirectory
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Validate and filter dataset
        valid_data = []
        for i, item in enumerate(self.data):
            if not isinstance(item, dict):
                logger.warning(f"Skipping invalid entry at index {i} (not a dict): {item}")
                continue
            if not all(key in item for key in ['image', 'prompt']):
                logger.warning(f"Skipping invalid entry at index {i} (missing keys): {item}")
                continue
            if item['prompt'] is None or not isinstance(item['prompt'], str) or not item['prompt'].strip():
                logger.warning(f"Skipping invalid entry at index {i} (invalid prompt): {item}")
                continue
            image_path = os.path.join(self.data_dir, item['image'])
            if not os.path.exists(image_path):
                logger.warning(f"Skipping invalid entry at index {i} (image not found): {image_path}")
                continue
            valid_data.append(item)

        self.data = valid_data
        if not self.data:
            raise ValueError("No valid entries found in dataset")
        logger.info(f"Loaded dataset with {len(self.data)} valid entries")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'image': item['image'],
            'prompt': item['prompt']
        }

def load_dataset(dataset_path):
    """
    Load COCO fine-tuning dataset from a JSON file.

    Args:
        dataset_path (str): Path to dataset.json

    Returns:
        CocoFinetuneDataset: PyTorch Dataset object
    """
    data_dir = os.path.dirname(dataset_path)
    return CocoFinetuneDataset(json_path=dataset_path, data_dir=data_dir) 