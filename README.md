# CAFT: Causal Attention Fusion Transformer for Hateful Meme Detection

---

## 📋 Project Overview
This repository implements the **CAFT (Causal Attention Fusion Transformer)** model, a specialized framework for detecting hateful memes through causal feature analysis. The codebase is streamlined to focus solely on the CAFT architecture, with all baseline models and redundant configurations removed for clarity.

---

## 📂 Project Structure.
├── config.py        # Model hyperparameters and configuration
├── dataset.py       # Data loading and preprocessing utilities
├── loss.py          # Custom loss functions for CAFT
├── model.py         # Core CAFT architecture implementation
├── train.py         # Training and evaluation pipelines
└── main.py          # Entry point for model execution
---

## 🔍 Module Details

### 1. `config.py`
Centralizes all hyperparameters and runtime settings for CAFT, including:
- Device configuration (GPU/CPU)
- Training parameters (batch size, epochs, learning rate)
- Feature dimensions (CLIP vision/text embedding sizes)
- Loss function weights and hyperparameters

*All CAFT-specific modules (multimodal fusion, gated delta mechanism, attention pooling) are enabled by default with no redundant conditional checks.*

### 2. `dataset.py`
Implements dataset classes optimized for causal contrastive learning:
- `CausalPairFeatureDataset`: Loads triplet data (anchor/positive/negative text-image pairs)
- `StandardFeatureDataset`: Loads standard single-sample data for validation
- `custom_collate_fn`: Handles batch aggregation with proper tensor formatting

### 3. `loss.py`
Defines CAFT's specialized loss functions:
- `CausalConsistencyLoss`: Enforces feature alignment between anchor and positive counterfactuals
- `SupervisedContrastiveLoss`: Enhances class-specific clustering with temperature scaling

### 4. `model.py`
Core implementation of `CausalDeltaAttentionNet` with:
- Cross-modal attention for text-image feature fusion
- Gated delta mechanism to filter non-causal feature differences
- Attention pooling to focus on critical causal tokens
- Joint classifier and projection heads for end-to-end training

### 5. `train.py`
Contains training and evaluation logic:
- `train_epoch_cda`: CAFT-specific training loop with multi-loss optimization
- `evaluate`: Computes key metrics (accuracy, F1-score, AUC) on validation data
- Utility functions for reproducibility (seed fixing)

### 6. `main.py`
Entry point that orchestrates the training pipeline:
1. Loads configuration and initializes datasets
2. Sets up model, optimizer, and learning rate scheduler
3. Executes training loop with validation checks
4. Saves the best-performing model based on validation accuracy

---

## 🚀 Usage Instructions

1. **Prepare Feature Data**  
   Organize preprocessed image/text features in the directory specified by `config.feat_dir`.

2. **Configure Parameters**  
   Adjust hyperparameters in `config.py` as needed (device, training epochs, etc.).

3. **Run Training**  
   Execute the main script to start training:
   ```bash
   python main.py
   ```

4. **Access Results**  
   - Training logs and metrics are printed to console
   - Best model is saved to `./checkpoints/` directory

---

## ⚠️ Notes
- All code is stripped of baseline models and conditional logic, focusing exclusively on CAFT
- Critical modules (attention pooling, gated fusion) are enabled by default
- Reproducibility is ensured through fixed random seeds and deterministic training settings
    
