import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import json
import os
import random
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch.nn.functional as F
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==============================================================================
# 1. 配置区域（新增特征路径和反事实参数）
# ==============================================================================
class Config:
    mode = "baseline"     # cdanet or baseline
    clip_model_name = "openai/clip-vit-large-patch14-336"
    feat_dir = "./clip_large_features"  # 预提取特征路径

    # --- 模型架构控制开关 ---
    use_multimodal_fusion = False
    use_gated_delta = True
    use_attention_pooling = True

    # --- 损失函数控制开关 ---
    use_consistency_loss = True
    use_contrastive_loss = True
    # 新增：反事实相关参数
    use_strategy_weighting = False  # 是否根据反事实生成策略进行加权
    
    # --- 设备和超参数 ---
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

# ==============================================================================
# 2. 数据集定义（适配反事实样本结构）
# ==============================================================================
class CausalPairFeatureDataset(Dataset):
    """加载预提取特征的CausalPair数据集，适配反事实样本结构"""
    def __init__(self, feat_dir, split="train_cda"):
        self.feat_dir = feat_dir
        # 加载所有特征
        self.img_pool = np.load(os.path.join(feat_dir, f"{split}_img_pool.npy"))
        self.anchor_text_pool = np.load(os.path.join(feat_dir, f"{split}_anchor_text_pool.npy"))
        self.anchor_text_seq = np.load(os.path.join(feat_dir, f"{split}_anchor_text_seq.npy"))
        self.positive_text_pool = np.load(os.path.join(feat_dir, f"{split}_positive_text_pool.npy"))
        self.positive_text_seq = np.load(os.path.join(feat_dir, f"{split}_positive_text_seq.npy"))
        self.hardneg_text_pool = np.load(os.path.join(feat_dir, f"{split}_hardneg_text_pool.npy"))
        self.hardneg_text_seq = np.load(os.path.join(feat_dir, f"{split}_hardneg_text_seq.npy"))
        self.labels = np.load(os.path.join(feat_dir, f"{split}_labels.npy"))
        
        # 如果存在策略信息，加载策略权重（用于加权损失）
        strategy_weights_path = os.path.join(feat_dir, f"{split}_strategy_weights.npy")
        if os.path.exists(strategy_weights_path):
            self.strategy_weights = np.load(strategy_weights_path)
        else:
            self.strategy_weights = np.ones(len(self.labels))  # 默认权重为1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_pool": self.img_pool[idx],
            "anchor_text_pool": self.anchor_text_pool[idx],
            "anchor_text_seq": self.anchor_text_seq[idx],
            "positive_text_pool": self.positive_text_pool[idx],
            "positive_text_seq": self.positive_text_seq[idx],
            "hardneg_text_pool": self.hardneg_text_pool[idx],
            "hardneg_text_seq": self.hardneg_text_seq[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "strategy_weight": torch.tensor(self.strategy_weights[idx], dtype=torch.float32)
        }


class StandardFeatureDataset(Dataset):
    """加载预提取特征的标准数据集"""
    def __init__(self, feat_dir, split):
        self.feat_dir = feat_dir
        self.img_pool = np.load(os.path.join(feat_dir, f"{split}_img_pool.npy"))
        self.text_pool = np.load(os.path.join(feat_dir, f"{split}_text_pool.npy"))
        self.text_seq = np.load(os.path.join(feat_dir, f"{split}_text_seq.npy"))
        self.labels = np.load(os.path.join(feat_dir, f"{split}_labels.npy"))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "img_pool": self.img_pool[idx],
            "text_pool": self.text_pool[idx],
            "text_seq": self.text_seq[idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def custom_collate_fn(batch):
    """将特征转换为tensor并整理成batch，增加策略权重处理"""
    collated = {
        "img_pool": torch.tensor(np.stack([x["img_pool"] for x in batch]), dtype=torch.float32),
        "anchor_text_pool": torch.tensor(np.stack([x["anchor_text_pool"] for x in batch]), dtype=torch.float32) if "anchor_text_pool" in batch[0] else None,
        "anchor_text_seq": torch.tensor(np.stack([x["anchor_text_seq"] for x in batch]), dtype=torch.float32) if "anchor_text_seq" in batch[0] else None,
        "positive_text_pool": torch.tensor(np.stack([x["positive_text_pool"] for x in batch]), dtype=torch.float32) if "positive_text_pool" in batch[0] else None,
        "positive_text_seq": torch.tensor(np.stack([x["positive_text_seq"] for x in batch]), dtype=torch.float32) if "positive_text_seq" in batch[0] else None,
        "hardneg_text_pool": torch.tensor(np.stack([x["hardneg_text_pool"] for x in batch]), dtype=torch.float32) if "hardneg_text_pool" in batch[0] else None,
        "hardneg_text_seq": torch.tensor(np.stack([x["hardneg_text_seq"] for x in batch]), dtype=torch.float32) if "hardneg_text_seq" in batch[0] else None,
        "text_pool": torch.tensor(np.stack([x["text_pool"] for x in batch]), dtype=torch.float32) if "text_pool" in batch[0] else None,
        "text_seq": torch.tensor(np.stack([x["text_seq"] for x in batch]), dtype=torch.float32) if "text_seq" in batch[0] else None,
        "label": torch.stack([x["label"] for x in batch])
    }
    
    # 添加策略权重（如果存在）
    if "strategy_weight" in batch[0]:
        collated["strategy_weight"] = torch.stack([x["strategy_weight"] for x in batch])
        
    return collated

# ==============================================================================
# 3. 损失函数定义（增加策略加权支持）
# ==============================================================================
class CausalConsistencyLoss(nn.Module):
    def __init__(self, use_strategy_weighting=False):
        super().__init__()
        self.use_strategy_weighting = use_strategy_weighting
        
    def forward(self, feat_anchor, feat_positive, weights=None):
        cos_sim = F.cosine_similarity(feat_anchor, feat_positive)
        loss = 1 - cos_sim
        
        if self.use_strategy_weighting and weights is not None:
            loss = loss * weights
        
        return torch.mean(loss)

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, use_strategy_weighting=False):
        super().__init__()
        self.temperature = temperature
        self.use_strategy_weighting = use_strategy_weighting
        
    def forward(self, features, labels, weights=None):
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)
        
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(1, 1)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask
        
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        
        if self.use_strategy_weighting and weights is not None:
            loss = loss * weights
        
        return loss.mean()

# ==============================================================================
# 4. 模型架构定义（保持不变，已适配预提取特征）
# ==============================================================================
class CausalDeltaAttentionNet(nn.Module):
    def __init__(self,
                 clip_vision_dim=1024,  # CLIP-Large视觉特征维度
                 clip_text_dim=768,    # CLIP-Large文本特征维度
                 dropout_rate=0.4,
                 use_multimodal_fusion=True,
                 use_gated_delta=True,
                 use_attention_pooling=True):
        super().__init__()
        self.use_multimodal_fusion = use_multimodal_fusion
        self.use_gated_delta = use_gated_delta
        self.use_attention_pooling = use_attention_pooling

        # 特征维度
        self.clip_vision_dim = clip_vision_dim
        self.clip_text_dim = clip_text_dim

        # 分类器输入维度（视觉+文本+因果特征）
        new_classifier_input_dim = clip_vision_dim + clip_text_dim + clip_text_dim

        # 跨模态融合（图像序列 -> 文本序列）
        if self.use_multimodal_fusion:
            self.image_text_cross_attention = nn.MultiheadAttention(
                embed_dim=clip_text_dim,
                kdim=clip_vision_dim,
                vdim=clip_vision_dim,
                num_heads=8,
                batch_first=True
            )
            self.fusion_norm = nn.LayerNorm(clip_text_dim)

        # 门控网络（用于token级delta）
        if self.use_gated_delta:
            self.delta_fusion_gate = nn.Sequential(
                nn.Linear(clip_text_dim * 2, clip_text_dim),
                nn.Sigmoid()
            )

        # CDA头（注意力机制）
        self.cda_head = nn.MultiheadAttention(
            embed_dim=clip_text_dim,
            num_heads=8,
            batch_first=True
        )

        # 注意力池化
        if self.use_attention_pooling:
            self.pooling_query = nn.Parameter(torch.randn(1, 1, clip_text_dim))
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=clip_text_dim,
                num_heads=8,
                batch_first=True
            )

        # 分类器和投影头
        self.classifier = nn.Sequential(
            nn.Linear(new_classifier_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )
        self.proj_head = nn.Sequential(
            nn.Linear(new_classifier_input_dim, new_classifier_input_dim // 2),
            nn.ReLU(),
            nn.Linear(new_classifier_input_dim // 2, 128)
        )

    def encode_sequence(self, img_pool, text_pool, text_seq):
        """直接使用预提取的特征（替代CLIP编码）"""
        # 全局特征拼接 (B, d_v + d_t)
        global_features = torch.cat((img_pool, text_pool), dim=1)
        # 文本序列特征 (B, L_t, d_s)
        text_seq_features = text_seq
        # 图像序列特征（简化为None）
        image_seq_features = None
        return global_features, text_seq_features, image_seq_features

    def get_causal_features(self, img_pool, anchor_text_pool, anchor_text_seq, cf_text_pool, cf_text_seq, return_explain=False):
        """使用预提取特征计算因果特征"""
        # 编码锚点和反事实特征
        z_a, S_a, V_a = self.encode_sequence(img_pool, anchor_text_pool, anchor_text_seq)
        z_c, S_c, V_c = self.encode_sequence(img_pool, cf_text_pool, cf_text_seq)

        # 全局delta
        final_delta = z_a - z_c
        seq_delta = None
        cda_weights = None
        pool_weights = None

        # 多模态融合（文本序列 + 图像序列）
        S_a_fused = S_a
        S_c_fused = S_c
        if self.use_multimodal_fusion and (V_a is not None):
            try:
                attn_out, _ = self.image_text_cross_attention(query=S_a, key=V_a, value=V_a, need_weights=False)
                S_a_fused = self.fusion_norm(S_a + attn_out)
                attn_out_c, _ = self.image_text_cross_attention(query=S_c, key=V_a, value=V_a, need_weights=False)
                S_c_fused = self.fusion_norm(S_c + attn_out_c)
            except:
                pass

        # 计算序列delta
        if S_a_fused is not None and S_c_fused is not None:
            seq_delta = S_a_fused - S_c_fused

            # 门控机制
            if self.use_gated_delta:
                gate_in = torch.cat([S_a_fused, S_c_fused], dim=-1)
                gate_vals = self.delta_fusion_gate(gate_in)
                seq_delta = gate_vals * seq_delta

            # CDA头注意力
            try:
                causal_seq, cda_weights = self.cda_head(query=S_a_fused, key=seq_delta, value=seq_delta, need_weights=True)
            except TypeError:
                causal_seq, cda_weights = self.cda_head(S_a_fused, seq_delta, seq_delta, need_weights=True)

            # 池化
            if self.use_attention_pooling:
                bsz = causal_seq.size(0)
                pool_q = self.pooling_query.expand(bsz, -1, -1)
                pooled_causal, pool_weights = self.attention_pooling(query=pool_q, key=causal_seq, value=causal_seq, need_weights=True)
                pooled_causal = pooled_causal.squeeze(1)
            else:
                pooled_causal = causal_seq[:, 0, :]
        else:
            pooled_causal = torch.zeros(z_a.size(0), self.clip_text_dim, device=z_a.device)

        # 融合特征
        try:
            pooled_causal_norm = self.fusion_norm(pooled_causal) if hasattr(self, "fusion_norm") else pooled_causal
        except:
            pooled_causal_norm = pooled_causal
        fused = torch.cat([z_a, pooled_causal_norm], dim=-1)
        features = fused

        if not return_explain:
            return features

        explain = {
            "final_delta": final_delta.detach().cpu().numpy() if final_delta is not None else None,
            "seq_delta": seq_delta.detach().cpu().numpy() if seq_delta is not None else None,
            "cda_weights": cda_weights.detach().cpu().numpy() if (cda_weights is not None and isinstance(cda_weights, torch.Tensor)) else None,
            "pool_weights": pool_weights.detach().cpu().numpy() if (pool_weights is not None and isinstance(pool_weights, torch.Tensor)) else None
        }
        return features, explain

    def forward(self, batch, return_explain=False):
        """前向传播（使用预提取特征）"""
        # 从batch中获取特征
        img_pool = batch["img_pool"].to(self.classifier[0].weight.device)
        anchor_text_pool = batch["anchor_text_pool"].to(self.classifier[0].weight.device)
        anchor_text_seq = batch["anchor_text_seq"].to(self.classifier[0].weight.device)
        positive_text_pool = batch["positive_text_pool"].to(self.classifier[0].weight.device)
        positive_text_seq = batch["positive_text_seq"].to(self.classifier[0].weight.device)
        hardneg_text_pool = batch["hardneg_text_pool"].to(self.classifier[0].weight.device)
        hardneg_text_seq = batch["hardneg_text_seq"].to(self.classifier[0].weight.device)

        # 计算锚点特征
        if return_explain:
            feat_anchor, explain_anchor = self.get_causal_features(
                img_pool, anchor_text_pool, anchor_text_seq,
                hardneg_text_pool, hardneg_text_seq, return_explain=True
            )
        else:
            feat_anchor = self.get_causal_features(
                img_pool, anchor_text_pool, anchor_text_seq,
                hardneg_text_pool, hardneg_text_seq, return_explain=False
            )
            explain_anchor = None

        # 计算positive和hard-negative特征
        feat_positive = self.get_causal_features(
            img_pool, anchor_text_pool, anchor_text_seq,
            positive_text_pool, positive_text_seq, return_explain=False
        )
        feat_hard_negative = self.get_causal_features(
            img_pool, anchor_text_pool, anchor_text_seq,
            hardneg_text_pool, hardneg_text_seq, return_explain=False
        )

        # 分类和投影
        logits = self.classifier(feat_anchor)
        proj_anchor = F.normalize(self.proj_head(feat_anchor), p=2, dim=1) if hasattr(self, "proj_head") else None
        proj_positive = F.normalize(self.proj_head(feat_positive), p=2, dim=1) if hasattr(self, "proj_head") else None
        proj_hardneg = F.normalize(self.proj_head(feat_hard_negative), p=2, dim=1) if hasattr(self, "proj_head") else None

        out = {
            "logits": logits,
            "feat_anchor": feat_anchor,
            "feat_positive": feat_positive,
            "feat_hard_negative": feat_hard_negative,
            "proj_anchor": proj_anchor,
            "proj_positive": proj_positive,
            "proj_hardneg": proj_hardneg
        }
        if return_explain:
            out["explain_anchor"] = explain_anchor
        return out


class BaselineCLIPClassifier(nn.Module):
    def __init__(self, clip_vision_dim=1024, clip_text_dim=768, dropout_rate=0.4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(clip_vision_dim + clip_text_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 2)
        )

    def forward(self, batch):
        img_pool = batch["img_pool"].to(self.classifier[0].weight.device)
        text_pool = batch["text_pool"].to(self.classifier[0].weight.device)
        fused = torch.cat((img_pool, text_pool), dim=1)
        return self.classifier(fused)

# ==============================================================================
# 5. 训练与评估函数（适配反事实加权损失）
# ==============================================================================
def train_epoch_cda(model, dataloader, optimizer, scheduler, criterions, cfg):
    model.train()
    total_loss = 0
    total_cls_loss = 0
    total_cons_loss = 0
    total_cont_loss = 0

    pbar = tqdm(dataloader, desc="Training CDA-Net")
    for batch in pbar:
        labels = batch['label'].to(cfg.device)
        # 获取策略权重（如果使用）
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

        if cfg.use_consistency_loss:
            # 传入策略权重
            loss_cons = criterions['cons'](feat_anchor, feat_positive, strategy_weights)
            total_loss_batch += cfg.lambda_consistency * loss_cons
            loss_cons_val = loss_cons.item()
            total_cons_loss += loss_cons_val

        if cfg.use_contrastive_loss:
            features = torch.cat([feat_anchor, feat_positive, feat_hard_negative], dim=0)
            b_labels = torch.cat([labels, labels, labels], dim=0)
            
            # 扩展权重以匹配拼接后的特征
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

def train_epoch_baseline(model, dataloader, optimizer, scheduler, criterion, cfg):
    model.train(); total_loss = 0
    for batch in tqdm(dataloader, desc="Training Baseline"):
        labels = batch['label'].to(cfg.device); optimizer.zero_grad()
        logits = model(batch); loss = criterion(logits, labels); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm); optimizer.step(); scheduler.step()
        total_loss += loss.item()
    print(f"Avg Baseline Training Loss: {total_loss / len(dataloader):.4f}")

def evaluate(model, dataloader, cfg, is_baseline=False):
    model.eval(); all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['label']
            if is_baseline:
                logits = model(batch)
            else:
                # 对验证集使用文本自身作为反事实（与训练逻辑一致）
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ==============================================================================
# 6. 主函数（适配新数据集和模型）
# ==============================================================================
if __name__ == "__main__":
    cfg = Config()
    set_seed(2025)
    
    os.makedirs("checkpoints", exist_ok=True)
    BEST_MODEL_PATH = f"./checkpoints/best_model_{cfg.mode}_clip_large.pth"
    
    print(f"--- Running in mode: {cfg.mode.upper()} ---")
    if cfg.mode == "cdanet":
        print("CDA-Net Arch Config:", f"MMF={cfg.use_multimodal_fusion}", f"GD={cfg.use_gated_delta}", f"AP={cfg.use_attention_pooling}")
        print("CDA-Net Loss Config:", f"Consistency={cfg.use_consistency_loss}", f"Contrastive={cfg.use_contrastive_loss}",
              f"StrategyWeighting={cfg.use_strategy_weighting}")
        model = CausalDeltaAttentionNet(
            clip_vision_dim=1024,  # CLIP-Large维度
            clip_text_dim=768,
            use_multimodal_fusion=cfg.use_multimodal_fusion,
            use_gated_delta=cfg.use_gated_delta,
            use_attention_pooling=cfg.use_attention_pooling
        ).to(cfg.device)
        # 加载CDA训练集（预提取特征）
        train_dataset = CausalPairFeatureDataset(cfg.feat_dir, split="train_cda")
        train_fn = train_epoch_cda
        
        criterions = {
            'cls': nn.CrossEntropyLoss(),
            'cons': CausalConsistencyLoss(use_strategy_weighting=cfg.use_strategy_weighting) if cfg.use_consistency_loss else None,
            'cont': SupervisedContrastiveLoss(
                temperature=cfg.contrastive_temperature,
                use_strategy_weighting=cfg.use_strategy_weighting
            ) if cfg.use_contrastive_loss else None
        }
        
        # 仅训练下游层（无CLIP参数）
        head_params = list(model.classifier.parameters()) + list(model.cda_head.parameters())
        if cfg.use_multimodal_fusion:
            head_params += list(model.image_text_cross_attention.parameters()) + list(model.fusion_norm.parameters())
        if cfg.use_gated_delta:
            head_params += list(model.delta_fusion_gate.parameters())
        if cfg.use_attention_pooling:
            head_params += list(model.attention_pooling.parameters()) + [model.pooling_query]
        optimizer = optim.AdamW(head_params, lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    
    elif cfg.mode == "baseline":
        model = BaselineCLIPClassifier(clip_vision_dim=1024, clip_text_dim=768).to(cfg.device)
        train_dataset = StandardFeatureDataset(cfg.feat_dir, split="train_baseline")
        train_fn = train_epoch_baseline
        criterions = {'cls': nn.CrossEntropyLoss()}
        optimizer = optim.AdamW(model.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError("Config.mode must be 'cdanet' or 'baseline'")

    # 验证集
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

    best_metric = 0.0  # 以ACC为指标

    print(f"Model: {model.__class__.__name__}, Feature dir: {cfg.feat_dir}")
    
    for epoch in range(1, cfg.num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{cfg.num_epochs} ---")
        if cfg.mode == "cdanet":
            train_fn(model, train_dataloader, optimizer, scheduler, criterions, cfg)
        else:
            train_fn(model, train_dataloader, optimizer, scheduler, criterions['cls'], cfg)
        
        val_acc, val_f1, val_auc = evaluate(model, dev_dataloader, cfg, is_baseline=(cfg.mode=="baseline"))
        print(f"Epoch {epoch} Results: Val Acc = {val_acc:.4f}, Val Macro-F1 = {val_f1:.4f}, Val AUC = {val_auc:.4f}")

        if val_acc > best_metric:
            best_metric = val_acc
            print(f"🎉 New best model found! ACC: {best_metric:.4f}. Saving to {BEST_MODEL_PATH}")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        
    print("\n--- Training Complete ---")
    print(f"🏆 Best validation ACC: {best_metric:.4f}")
    print(f"Best model saved at: {BEST_MODEL_PATH}")