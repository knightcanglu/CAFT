import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import Config
from dataset import CausalPairFeatureDataset, StandardFeatureDataset, custom_collate_fn
from model import CausalDeltaAttentionNet
from loss import CausalConsistencyLoss, SupervisedContrastiveLoss
from train import train_epoch_cda, evaluate, set_seed

if __name__ == "__main__":
    cfg = Config()
    set_seed(2025)
    
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = f"./checkpoints/best_model_{cfg.mode}_clip_large.pth"
    
    print(f"--- Running in mode: {cfg.mode.upper()} ---")
    print("CDA-Net Arch Config:", f"MMF={cfg.use_multimodal_fusion}", f"GD={cfg.use_gated_delta}", f"AP={cfg.use_attention_pooling}")
    print("CDA-Net Loss Config:", f"Consistency={cfg.use_consistency_loss}", f"Contrastive={cfg.use_contrastive_loss}",
          f"StrategyWeighting={cfg.use_strategy_weighting}")
    model = CausalDeltaAttentionNet(
        clip_vision_dim=1024,
        clip_text_dim=768
    ).to(cfg.device)
    
    train_dataset = CausalPairFeatureDataset(cfg.feat_dir, split="train_cda")
    train_fn = train_epoch_cda
    
    criterions = {
        'cls': nn.CrossEntropyLoss(),
        'cons': CausalConsistencyLoss(use_strategy_weighting=cfg.use_strategy_weighting),
        'cont': SupervisedContrastiveLoss(
            temperature=cfg.contrastive_temperature,
            use_strategy_weighting=cfg.use_strategy_weighting
        )
    }
    
    head_params = list(model.classifier.parameters()) + list(model.cda_head.parameters())
    head_params += list(model.image_text_cross_attention.parameters()) + list(model.fusion_norm.parameters())
    head_params += list(model.delta_fusion_gate.parameters())
    head_params += list(model.attention_pooling.parameters()) + [model.pooling_query]
    optimizer = optim.AdamW(head_params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)

    dev_dataset = StandardFeatureDataset(cfg.feat_dir, split="dev")
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, 
        num_workers=4, collate_fn=custom_collate_fn
    )
    dev_dataloader = DataLoader(
        dev_dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=4, collate_fn=custom_collate_fn
    )
    
    total_steps = len(train_dataloader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.num_warmup_steps, 
        num_training_steps=total_steps
    )

    best_metric = 0.0

    print(f"Model: {model.__class__.__name__}, Feature dir: {cfg.feat_dir}")
    
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{cfg.num_epochs} ---")
        train_fn(model, train_dataloader, optimizer, scheduler, criterions, cfg)
        
        val_acc, val_f1, val_auc = evaluate(model, dev_dataloader, cfg)
        print(f"Epoch {epoch} Results: Val Acc = {val_acc:.4f}, Val Macro-F1 = {val_f1:.4f}, Val AUC = {val_auc:.4f}")

        if val_acc > best_metric:
            best_metric = val_acc
            print(f"🎉 New best model found! ACC: {best_metric:.4f}. Saving to {BEST_MODEL_PATH}")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
    print("\n--- Training Complete ---")
    print(f"🏆 Best validation ACC: {best_metric:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")