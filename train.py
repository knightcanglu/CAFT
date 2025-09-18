import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

def train_epoch_cda(model, dataloader, optimizer, scheduler, criterions, cfg):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_cons_loss = 0
    total_cont_loss = 0

    pbar = tqdm(dataloader, desc="Training CDA-Net")
    for batch in pbar:
        labels = batch['label'].to(cfg.device)
        strategy_weights = batch.get('strategy_weight', None)
        if strategy_weights is not None:
            strategy_weights = strategy_weights.to(cfg.device)
            
        optimizer.zero_grad()
        
        outputs = model(batch)
        
        feat_anchor = F.normalize(outputs['feat_anchor'], p=2, dim=1)
        feat_positive = F.normalize(outputs['feat_positive'], p=2, dim=1)
        feat_hard_negative = F.normalize(outputs['feat_hard_negative'], p=2, dim=1)
        
        loss_cls = criterions['cls'](outputs['logits'], labels)
        
        total_loss_batch = loss_cls
        total_cls_loss += loss_cls.item()
        
        loss_cons_val = 0
        loss_cont_val = 0

        loss_cons = criterions['cons'](feat_anchor, feat_positive, strategy_weights)
        total_loss_batch += cfg.lambda_consistency * loss_cons
        loss_cons_val = loss_cons.item()
        total_cons_loss += loss_cons_val

        features = torch.cat([feat_anchor, feat_positive, feat_hard_negative], dim=0)
        b_labels = torch.cat([labels, labels, labels], dim=0)
        
        if strategy_weights is not None:
            b_weights = torch.cat([strategy_weights, strategy_weights, strategy_weights], dim=0)
        else:
            b_weights = None
            
        loss_cont = criterions['cont'](features, b_labels, b_weights)
        total_loss_batch += cfg.lambda_contrastive * loss_cont
        loss_cont_val = loss_cont.item()
        total_cont_loss += loss_cont_val

        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += total_loss_batch.item()

        pbar.set_postfix({
            'Total': f"{total_loss_batch.item():.2f}",
            'CLS': f"{loss_cls.item():.2f}",
            'CONS': f"{loss_cons_val:.2f}",
            'CONT': f"{loss_cont_val:.2f}"
        })

    num_batches = len(dataloader)
    print(f"Avg CDA Training Loss: {total_loss / num_batches:.4f} | "
          f"Avg CLS: {total_cls_loss / num_batches:.4f} | "
          f"Avg CONS: {total_cons_loss / num_batches:.4f} | "
          f"Avg CONT: {total_cont_loss / num_batches:.4f}")


def evaluate(model, dataloader, cfg):
    model.eval(); all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label']
            logits = model.classifier(model.get_causal_features(
                batch["img_pool"].to(cfg.device),
                batch["text_pool"].to(cfg.device),
                batch["text_seq"].to(cfg.device),
                batch["text_pool"].to(cfg.device),
                batch["text_seq"].to(cfg.device)
            ))
            probs = F.softmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    auroc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    return acc, f1, auroc


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False