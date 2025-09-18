import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Config:
    mode = "CAFT"
    clip_model_name = "openai/clip-vit-large-patch14-336"
    feat_dir = "./clip_large_features_mami"

    use_multimodal_fusion = True
    use_gated_delta = True
    use_attention_pooling = True

    use_consistency_loss = True
    use_contrastive_loss = True
    use_strategy_weighting = False
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    batch_size = 16 
    num_epochs = 30
    head_lr = 5e-6
    weight_decay = 0.05
    max_grad_norm = 1.0
    num_warmup_steps = 1000
    lambda_consistency = 20
    lambda_contrastive = 0.5
    contrastive_temperature = 0.07