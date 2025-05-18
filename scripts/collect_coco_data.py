import os
import json
import logging
from zipfile import ZipFile
from PIL import Image
from pycocotools.coco import COCO
from transformers import pipeline

# Configure logging
logging.basicConfig(
    filename='/home/iris/Documents/deep_learning/src/logs/coco_data_collection.log',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s'
)
logger = logging.getLogger(__name__)

# Define therapy-related keywords
keywords = [
    "anxiety", "fear", "worry", "stress", "calm", "peaceful", "relaxed", "conversation",
    "talking", "listening", "emotion", "sad", "happy", "angry", "hope", "dream",
    "struggle", "challenge", "mental", "thinking", "reflection", "introspection",
    "relationship", "connection", "trust", "support", "existential", "meaning",
    "purpose", "identity", "self-esteem", "confidence", "growth", "acceptance",
    "mindfulness", "support", "frustration", "trauma", "family", "relationships",
    "parenting", "communication", "marriage", "friendship", "couple", "love",
    "loss", "grief", "forgiveness", "empathy", "compassion", "vulnerability",
    "boundaries", "assertiveness", "self-care", "coping", "resilience", "well-being",
    "connection", "empathy", "creativity", "empowerment", "career", "employment",
    "education", "school", "motivation", "inspiration", "self-discovery",
    "self-acceptance", "person", "people", "sitting", "talk", "feel", "mood", "meditate",
    "think", "reflect", "taking medication", "discussion", "supportive friend hug", "person crying",
    "couple arguing", "hug", "cry", "argue", "talking closely", "alone", "together",
    "sitting on a couch", "sitting on a chair", "sitting on a bed", "sitting on the floor",
    "sitting in a room", "sitting in a park", "sitting in a cafe", "sitting in a car",
    "sitting in a restaurant", "sitting in a waiting room", "sitting in a classroom"
]

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)  # Use GPU if available

def is_relevant_caption(caption, labels=keywords):
    """Check if a caption is relevant using zero-shot classification."""
    try:
        result = classifier(caption, candidate_labels=labels, multi_label=True)
        return any(score > 0.5 for score in result['scores'])
    except Exception as e:
        logger.error(f"Zero-shot classification failed for caption '{caption}': {e}")
        return False

def extract_coco_datasets(data_dir):
    """Extract COCO 2017 train, validation images, and annotations if not already extracted."""
    os.makedirs(data_dir, exist_ok=True)
    
    # Check and extract train2017
    train_zip_path = os.path.join(data_dir, "train2017.zip")
    train_dir = os.path.join(data_dir, "train2017")
    if not os.path.exists(train_dir) and os.path.exists(train_zip_path):
        logger.info("Extracting COCO train2017 images...")
        try:
            with ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("Extracted COCO train2017 images")
        except Exception as e:
            logger.error(f"Failed to extract {train_zip_path}: {e}")
            raise
    
    # Check and extract val2017
    val_zip_path = os.path.join(data_dir, "val2017.zip")
    val_dir = os.path.join(data_dir, "val2017")
    if not os.path.exists(val_dir) and os.path.exists(val_zip_path):
        logger.info("Extracting COCO val2017 images...")
        try:
            with ZipFile(val_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("Extracted COCO val2017 images")
        except Exception as e:
            logger.error(f"Failed to extract {val_zip_path}: {e}")
            raise
    
    # Check and extract annotations
    ann_zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
    ann_dir = os.path.join(data_dir, "annotations")
    if not os.path.exists(ann_dir) and os.path.exists(ann_zip_path):
        logger.info("Extracting COCO annotations...")
        try:
            with ZipFile(ann_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            logger.info("Extracted COCO annotations")
        except Exception as e:
            logger.error(f"Failed to extract {ann_zip_path}: {e}")
            raise

def filter_coco_images(coco_instances, coco_captions, keywords, output_dir, data_dir, dataset_type, max_images=200):
    """Filter COCO images based on keyword matches and zero-shot classification, prioritizing images with people."""
    os.makedirs(output_dir, exist_ok=True)
    dataset = []
    
    # Get image IDs for images containing people
    cat_ids = coco_instances.getCatIds(catNms=['person'])
    img_ids = coco_instances.getImgIds(catIds=cat_ids)
    logger.info(f"Processing {len(img_ids)} COCO {dataset_type} images containing people")
    
    for img_id in img_ids:
        # Load image metadata from captions dataset (consistent with captions)
        img_info = coco_captions.loadImgs(img_id)
        if not img_info:  # Skip if image not found in captions dataset
            logger.debug(f"Skipping image ID {img_id}: Not found in captions dataset")
            continue
        img_info = img_info[0]
        img_path = os.path.join(dataset_type, img_info['file_name'])
        
        # Load captions
        ann_ids = coco_captions.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco_captions.loadAnns(ann_ids)
        captions = [ann['caption'].lower() for ann in anns if 'caption' in ann]
        
        if not captions:  # Skip images with no captions
            logger.debug(f"Skipping image {img_info['file_name']}: No captions found")
            continue
        
        # Check for keyword matches or zero-shot relevance
        keyword_matches = sum([any(keyword in caption for caption in captions) for keyword in keywords])
        zero_shot_relevant = any(is_relevant_caption(caption) for caption in captions)
        
        if keyword_matches >= 1 or zero_shot_relevant:  # Require at least one keyword match or zero-shot relevance
            # Log captions for debugging
            logger.debug(f"Image {img_info['file_name']} captions: {captions}")
            # Select the most relevant caption
            selected_caption = max(captions, key=lambda c: sum(keyword in c for keyword in keywords))
            # Augment caption with therapy context
            prompt = f"{selected_caption}, emotional and vivid style"
            
            # Save image
            img_dir = os.path.join(output_dir, "images")
            output_img_path = os.path.join(img_dir, img_info['file_name'])
            try:
                img = Image.open(os.path.join(data_dir, img_path))
                img.save(output_img_path)
                dataset.append({"image": img_info['file_name'], "prompt": prompt})
                logger.info(f"Selected image {img_info['file_name']} with prompt: {prompt}")
            except Exception as e:
                logger.error(f"Failed to process image {img_info['file_name']}: {e}")
        
        if len(dataset) >= max_images:
            break
    
    return dataset

def main():
    """
    Output: 
    Up to 200 image-text pairs for fine-tuning, prioritizing images with people
        [
        {"image": "000000123456.jpg", "prompt": "A person sitting alone in a room, emotional and vivid style"},
        {"image": "000000123457.jpg", "prompt": "Two people talking closely, emotional and vivid style"}
        ]
    """
    data_dir = "/home/iris/Documents/deep_learning/data/finetune_dataset/coco"
    output_dir = "/home/iris/Documents/deep_learning/data/finetune_dataset/coco"
    
    # Extract existing ZIP files if not already extracted
    extract_coco_datasets(data_dir)
    
    # Process both val2017 and train2017
    dataset_types = [
        {
            "type": "val2017",
            "instances_file": "annotations/instances_val2017.json",
            "captions_file": "annotations/captions_val2017.json"
        },
        {
            "type": "train2017",
            "instances_file": "annotations/instances_train2017.json",
            "captions_file": "annotations/captions_train2017.json"
        }
    ]
    dataset = []
    max_images_per_dataset = 100  # Split 200 images across val2017 and train2017
    
    for dataset_type in dataset_types:
        instances_file = os.path.join(data_dir, dataset_type["instances_file"])
        captions_file = os.path.join(data_dir, dataset_type["captions_file"])
        
        if not os.path.exists(instances_file) or not os.path.exists(captions_file):
            logger.warning(f"Annotation files {instances_file} or {captions_file} not found. Skipping {dataset_type['type']}")
            continue
        
        # Initialize COCO API for instances and captions
        coco_instances = COCO(instances_file)
        coco_captions = COCO(captions_file)
        
        # Filter images
        dataset_subset = filter_coco_images(
            coco_instances, coco_captions, keywords, output_dir, data_dir, dataset_type["type"],
            max_images=max_images_per_dataset
        )
        dataset.extend(dataset_subset)
        logger.info(f"Collected {len(dataset_subset)} image-text pairs from {dataset_type['type']}")
        
        if len(dataset) >= 200:
            break
    
    # Save dataset as JSON
    dataset_json_path = os.path.join(output_dir, "dataset.json")
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Saved dataset with {len(dataset)} image-text pairs to {dataset_json_path}")

if __name__ == "__main__":
    main()