import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalDeltaAttentionNet(nn.Module):
    def __init__(self, clip_vision_dim=1024, clip_text_dim=768, dropout_rate=0.4):
        super().__init__()
        self.use_multimodal_fusion = True
        self.use_gated_delta = True
        self.use_attention_pooling = True

        self.clip_vision_dim = clip_vision_dim
        self.clip_text_dim = clip_text_dim

        new_classifier_input_dim = clip_vision_dim + clip_text_dim + clip_text_dim

        self.image_text_cross_attention = nn.MultiheadAttention(
            embed_dim=clip_text_dim,
            kdim=clip_vision_dim,
            vdim=clip_vision_dim,
            num_heads=8,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(clip_text_dim)

        self.delta_fusion_gate = nn.Sequential(
            nn.Linear(clip_text_dim * 2, clip_text_dim),
            nn.Sigmoid()
        )

        self.cda_head = nn.MultiheadAttention(
            embed_dim=clip_text_dim,
            num_heads=8,
            batch_first=True
        )

        self.pooling_query = nn.Parameter(torch.randn(1, 1, clip_text_dim))
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=clip_text_dim,
            num_heads=8,
            batch_first=True
        )

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
        global_features = torch.cat((img_pool, text_pool), dim=1)
        text_seq_features = text_seq
        image_seq_features = None
        return global_features, text_seq_features, image_seq_features

    def get_causal_features(self, img_pool, anchor_text_pool, anchor_text_seq, cf_text_pool, cf_text_seq, return_explain=False):
        z_a, S_a, V_a = self.encode_sequence(img_pool, anchor_text_pool, anchor_text_seq)
        z_c, S_c, V_c = self.encode_sequence(img_pool, cf_text_pool, cf_text_seq)

        final_delta = z_a - z_c
        seq_delta = None
        cda_weights = None
        pool_weights = None

        S_a_fused = S_a
        S_c_fused = S_c
        if V_a is not None:
            try:
                attn_out, _ = self.image_text_cross_attention(query=S_a, key=V_a, value=V_a, need_weights=False)
                S_a_fused = self.fusion_norm(S_a + attn_out)
                attn_out_c, _ = self.image_text_cross_attention(query=S_c, key=V_a, value=V_a, need_weights=False)
                S_c_fused = self.fusion_norm(S_c + attn_out_c)
            except:
                pass

        if S_a_fused is not None and S_c_fused is not None:
            seq_delta = S_a_fused - S_c_fused

            gate_in = torch.cat([S_a_fused, S_c_fused], dim=-1)
            gate_vals = self.delta_fusion_gate(gate_in)
            seq_delta = gate_vals * seq_delta

            try:
                causal_seq, cda_weights = self.cda_head(query=S_a_fused, key=seq_delta, value=seq_delta, need_weights=True)
            except TypeError:
                causal_seq, cda_weights = self.cda_head(S_a_fused, seq_delta, seq_delta, need_weights=True)

            bsz = causal_seq.size(0)
            pool_q = self.pooling_query.expand(bsz, -1, -1)
            pooled_causal, pool_weights = self.attention_pooling(query=pool_q, key=causal_seq, value=causal_seq, need_weights=True)
            pooled_causal = pooled_causal.squeeze(1)
        else:
            pooled_causal = torch.zeros(z_a.size(0), self.clip_text_dim, device=z_a.device)

        try:
            pooled_causal_norm = self.fusion_norm(pooled_causal)
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
        img_pool = batch["img_pool"].to(self.classifier[0].weight.device)
        anchor_text_pool = batch["anchor_text_pool"].to(self.classifier[0].weight.device)
        anchor_text_seq = batch["anchor_text_seq"].to(self.classifier[0].weight.device)
        positive_text_pool = batch["positive_text_pool"].to(self.classifier[0].weight.device)
        positive_text_seq = batch["positive_text_seq"].to(self.classifier[0].weight.device)
        hardneg_text_pool = batch["hardneg_text_pool"].to(self.classifier[0].weight.device)
        hardneg_text_seq = batch["hardneg_text_seq"].to(self.classifier[0].weight.device)

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

        feat_positive = self.get_causal_features(
            img_pool, anchor_text_pool, anchor_text_seq,
            positive_text_pool, positive_text_seq, return_explain=False
        )
        feat_hard_negative = self.get_causal_features(
            img_pool, anchor_text_pool, anchor_text_seq,
            hardneg_text_pool, hardneg_text_seq, return_explain=False
        )

        logits = self.classifier(feat_anchor)
        proj_anchor = F.normalize(self.proj_head(feat_anchor), p=2, dim=1)
        proj_positive = F.normalize(self.proj_head(feat_positive), p=2, dim=1)
        proj_hardneg = F.normalize(self.proj_head(feat_hard_negative), p=2, dim=1)

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