# CAFT: Causal Attention Fusion Transformer

This is an anonymous implementation of the CAFT model for hateful meme detection. The full code will be publicly released upon acceptance to ICASSP 2026.

## Project Structure
- `config.py`: Model parameters and settings
- `dataset.py`: Data loading utilities
- `loss.py`: CAFT-specific loss functions
- `model.py`: Core CAFT architecture
- `train.py`: Training/evaluation logic
- `main.py`: Execution entry point

## Usage
1. Prepare preprocessed features in `config.feat_dir`
2. Adjust parameters in `config.py`
3. Run training:
   ```bash
   python main.py
   ```

Best model is saved to `./checkpoints/` during training.
    
