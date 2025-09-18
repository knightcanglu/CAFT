import torch
import torch.nn as nn
import random
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import json
import os
import numpy as np
from tqdm import tqdm

class FeatureExtractionConfig:
    clip_model_name = "openai/clip-vit-large-patch14-336"
    device = torch.device("cuda:1")
    train_file_augmented = "train_targeted_causal_neighborhood_v6.json"  # counterfactuals
    train_file_original = "train.jsonl"
    test_file = "dev.jsonl"
    image_dir = ""
    feat_save_dir = "./clip_large_features_mami"
    num_counterfactuals = 2  

os.makedirs(FeatureExtractionConfig.feat_save_dir, exist_ok=True)

processor = CLIPProcessor.from_pretrained(FeatureExtractionConfig.clip_model_name)
clip_model = CLIPModel.from_pretrained(FeatureExtractionConfig.clip_model_name)
clip_model.to(FeatureExtractionConfig.device)
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False

def load_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return [json.loads(line) for line in open(file_path, 'r', encoding='utf-8')]

def extract_image_features(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(FeatureExtractionConfig.device)
    with torch.no_grad():
        outputs = clip_model.vision_model(** inputs)
    pool_feat = outputs.pooler_output.cpu().numpy()
    seq_feat = outputs.last_hidden_state.cpu().numpy()
    return pool_feat, seq_feat

def extract_text_features(texts):
    inputs = processor(text=texts, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(FeatureExtractionConfig.device)
    with torch.no_grad():
        outputs = clip_model.text_model(**inputs)
    pool_feat = outputs.pooler_output.cpu().numpy()
    seq_feat = outputs.last_hidden_state.cpu().numpy()
    return pool_feat, seq_feat

def process_causal_pair_dataset(data_file, split_name):
    data = load_data(data_file)
    print(f"Processing {split_name} causal pair data (size: {len(data)})...")
    
    all_img_pool = []
    all_anchor_text_pool = []
    all_anchor_text_seq = []
    all_positive_text_pool = []
    all_positive_text_seq = []
    all_hardneg_text_pool = []
    all_hardneg_text_seq = []
    all_labels = []

    for item in tqdm(data):
        img_fname = item.get('img') or item.get('image') or item.get('image_path') or item.get('img_path')
        img = Image.open(os.path.join(FeatureExtractionConfig.image_dir, img_fname)).convert("RGB")
        img_pool, _ = extract_image_features([img])
        all_img_pool.append(img_pool[0])

        anchor_text = item.get('original_text') or item.get('text') or ""
        a_pool, a_seq = extract_text_features([anchor_text])
        all_anchor_text_pool.append(a_pool[0])
        all_anchor_text_seq.append(a_seq[0])

        original_label = item.get('original_label', item.get('label', 0))
        
        positive_texts = []
        if original_label == 1:
            positive_texts = [cf.get('edited_text', anchor_text) for cf in item.get('hateful_augmentations', [])]
        else:
            positive_texts = [cf.get('edited_text', anchor_text) for cf in item.get('benign_augmentations', [])]
        
        if not positive_texts:
            positive_texts = [anchor_text]
        positive_text = random.choice(positive_texts[:FeatureExtractionConfig.num_counterfactuals])
        p_pool, p_seq = extract_text_features([positive_text])
        all_positive_text_pool.append(p_pool[0])
        all_positive_text_seq.append(p_seq[0])

        hardneg_texts = []
        if original_label == 1: 
            hardneg_texts = [cf.get('edited_text', anchor_text) for cf in item.get('benign_counterfactuals', [])]
        else:  
            hardneg_texts = [cf.get('edited_text', anchor_text) for cf in item.get('hateful_counterfactuals', [])]
        
        if not hardneg_texts:
            hardneg_texts = [anchor_text]
        hardneg_text = random.choice(hardneg_texts[:FeatureExtractionConfig.num_counterfactuals])
        h_pool, h_seq = extract_text_features([hardneg_text])
        all_hardneg_text_pool.append(h_pool[0])
        all_hardneg_text_seq.append(h_seq[0])

        label = original_label
        all_labels.append(label)

    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_img_pool.npy"), np.array(all_img_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_anchor_text_pool.npy"), np.array(all_anchor_text_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_anchor_text_seq.npy"), np.array(all_anchor_text_seq))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_positive_text_pool.npy"), np.array(all_positive_text_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_positive_text_seq.npy"), np.array(all_positive_text_seq))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_hardneg_text_pool.npy"), np.array(all_hardneg_text_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_hardneg_text_seq.npy"), np.array(all_hardneg_text_seq))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_labels.npy"), np.array(all_labels))

def process_standard_dataset(data_file, split_name):
    data = load_data(data_file)
    print(f"Processing {split_name} standard data (size: {len(data)})...")
    
    all_img_pool = []
    all_text_pool = []
    all_text_seq = []
    all_labels = []

    for item in tqdm(data):
        img = Image.open(os.path.join(FeatureExtractionConfig.image_dir, item['img'])).convert("RGB")
        img_pool, _ = extract_image_features([img])
        all_img_pool.append(img_pool[0])

        text = item['text']
        t_pool, t_seq = extract_text_features([text])
        all_text_pool.append(t_pool[0])
        all_text_seq.append(t_seq[0])

        all_labels.append(item['label'])

    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_img_pool.npy"), np.array(all_img_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_text_pool.npy"), np.array(all_text_pool))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_text_seq.npy"), np.array(all_text_seq))
    np.save(os.path.join(FeatureExtractionConfig.feat_save_dir, f"{split_name}_labels.npy"), np.array(all_labels))

if __name__ == "__main__":
    process_causal_pair_dataset(FeatureExtractionConfig.train_file_augmented, "train_cda")
    process_standard_dataset(FeatureExtractionConfig.train_file_original, "train_baseline")
    process_standard_dataset(FeatureExtractionConfig.test_file, "dev")
    print("Extracting done!")