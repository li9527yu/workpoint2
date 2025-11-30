import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput

class GatedFusion(nn.Module):
    """
    Token-wise gating: 对每个 token 的两个分支隐表示做逐 token 的门控融合
    gate = sigmoid(MLP([h1, h2]))  ->  fused = gate * h1 + (1-gate) * h2
    """
    def __init__(self, hidden_size: int, gate_hidden: int = 256):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, gate_hidden),
            nn.ReLU(),
            nn.Linear(gate_hidden, 1)
        )

    def forward(self, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
        # h1, h2: [B, L, D]
        gate_logits = self.gate(torch.cat([h1, h2], dim=-1))       # [B, L, 1]
        gate = torch.sigmoid(gate_logits)                          # [B, L, 1]
        fused = gate * h1 + (1.0 - gate) * h2
        return fused, gate


class MyFlanT5(nn.Module):
    """
    - 两路 encoder：路1=情感线索组合输入，路2=图文理解输入
    - Representation-level 融合（token-wise gating）
    - 训练阶段包含：loss_single + loss_multi + loss_fusion（与推理一致）
    - 推理阶段：使用融合后的 encoder 表示送入同一 decoder 进行生成
    """
    def __init__(
        self,
        model_path: str,
        tokenizer: T5Tokenizer,
        hidden_size: int = 768,
        gate_hidden: int = 256,
        loss_weights=(1.0, 0.5, 0.8),   # (w_single, w_multi, w_fusion)
    ):
        super().__init__()
        self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer

        # 你原本的 query 投影（来自外部模块的 input_hidden_states）
        self.language_projection = nn.Linear(hidden_size, hidden_size)
        self.imagetext_projection = nn.Linear(hidden_size, hidden_size)

        # 表示级别融合（token-wise gating）
        self.fuser = GatedFusion(hidden_size=hidden_size, gate_hidden=gate_hidden)

        # 权重
        self.w_single, self.w_multi, self.w_fusion = loss_weights

        # init
        self.init_linear_weight()

    def init_linear_weight(self):
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)
        nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.imagetext_projection.bias)

        # 初始化融合门结构
        for m in self.fuser.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- low-level helpers ----------
    def _get_input_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        通过 embedding 层将 ids -> embeddings
        """
        input_embeddings = self.language_model.get_input_embeddings()
        return input_embeddings(input_ids)

    def _concat_query_tokens(
        self,
        query_states: torch.Tensor,          # [B, Lq, D]
        query_mask: torch.Tensor,            # [B, Lq]
        token_embeds: torch.Tensor,          # [B, Lt, D]
        token_mask: torch.Tensor             # [B, Lt]
    ):
        """
        将 query 与 token embeds 按序列维拼接，同时拼接 attention_mask
        """
        inputs_embeds = torch.cat([query_states, token_embeds.to(query_states.device)], dim=1)          # [B, Lq+Lt, D]
        attention_mask = torch.cat([query_mask, token_mask.to(query_mask.device)], dim=1)               # [B, Lq+Lt]
        return inputs_embeds, attention_mask

    def _encode_once(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        """
        只跑 encoder，返回 encoder last_hidden_state
        """
        # T5ForConditionalGeneration 内的 encoder
        encoder = self.language_model.get_encoder()
        enc_outputs = encoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True)
        return enc_outputs.last_hidden_state    # [B, L, D]

    # ---------- core paths ----------
    def _encode_single_branch(
        self,
        input_ids: torch.Tensor,                 # 路1/路2 对应的模板token ids
        input_mask: torch.Tensor,
        query_states: torch.Tensor,              # 外部给的 query hidden states
        projector: nn.Linear
    ):
        """
        对单路进行：query投影 -> 拼接 -> encoder
        返回：encoder_last_hidden, 拼接后的 attention_mask
        """
        # query: [B, Lq, D]
        proj_q = projector(query_states)                             # [B, Lq, D]
        proj_q_mask = torch.ones(proj_q.size()[:-1], dtype=torch.long, device=proj_q.device)  # [B, Lq]

        token_embeds = self._get_input_embeds(input_ids)             # [B, Lt, D]
        inputs_embeds, attention_mask = self._concat_query_tokens(
            proj_q, proj_q_mask, token_embeds, input_mask
        )                                                            # [B, Lq+Lt, D], [B, Lq+Lt]

        enc_last = self._encode_once(inputs_embeds, attention_mask)  # [B, L, D]
        return enc_last, attention_mask

    def _decode_with_labels(self, enc_last: torch.Tensor, attention_mask: torch.Tensor,
                            decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        """
        用共享 decoder 计算 seq2seq loss（训练）
        """
        enc_outputs = BaseModelOutput(last_hidden_state=enc_last)
        outputs = self.language_model(
            encoder_outputs=enc_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    # ---------- forward / generate ----------
    def forward(
        self,
        input_ids,                          # 路1：情感线索组合输入 ids
        input_multi_ids,                    # 路2：图文理解输入 ids
        attention_mask,                     # 路1 mask
        input_multi_attention_mask,         # 路2 mask
        input_hidden_states,                # 外部提供的 query（如图像/跨模态编码）[B, Lq, D]
        relation=None,                      # 兼容你原来的签名（未使用）
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        return_details: bool = False
    ):
        """
        训练阶段：计算三项 loss（single/multi/fusion），并返回总损失
        """
        # --- 分支1：情感线索组合 ---
        single_enc_last, single_mask = self._encode_single_branch(
            input_ids, attention_mask, input_hidden_states, self.language_projection
        )

        # --- 分支2：图文理解 ---
        multi_enc_last, multi_mask = self._encode_single_branch(
            input_multi_ids, input_multi_attention_mask, input_hidden_states, self.imagetext_projection
        )

        # --- 各自解码（与现有做法等价，但使用共享 decoder）---
        out_single = self._decode_with_labels(single_enc_last, single_mask,
                                              decoder_input_ids, decoder_attention_mask, labels)
        out_multi  = self._decode_with_labels(multi_enc_last,  multi_mask,
                                              decoder_input_ids, decoder_attention_mask, labels)

        # --- 表示级融合（token-wise gating），保持 mask 对齐（两路长度应相同；若不同请在外部padding到同长）---
        # 如果两路长度不同，你可以在此处做对齐（截断到最短，或padding到最长并共享mask）
        if single_enc_last.size(1) != multi_enc_last.size(1):
            # 简单做法：按最短长度截断（推荐外部先对齐以免信息丢失）
            min_len = min(single_enc_last.size(1), multi_enc_last.size(1))
            single_enc_last = single_enc_last[:, :min_len, :]
            multi_enc_last  = multi_enc_last[:, :min_len, :]
            single_mask = single_mask[:, :min_len]
            multi_mask  = multi_mask[:, :min_len]

        fused_enc_last, gate = self.fuser(single_enc_last, multi_enc_last)
        # 融合后的 mask：二者的 OR（只要任一路可见即为1）
        fused_mask = torch.clamp(single_mask + multi_mask, max=1)

        # --- 融合解码（fusion loss 与推理路径一致）---
        out_fused = self._decode_with_labels(fused_enc_last, fused_mask,
                                             decoder_input_ids, decoder_attention_mask, labels)

        # --- 总损失（对齐推理方式：融合分支参与训练）---
        loss = (self.w_single * out_single.loss
                + self.w_multi * out_multi.loss
                + self.w_fusion * out_fused.loss)

        if return_details:
            return {
                "loss": loss,
                "loss_single": out_single.loss.detach(),
                "loss_multi": out_multi.loss.detach(),
                "loss_fusion": out_fused.loss.detach(),
                "gate_mean": gate.mean().detach()
            }
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        input_multi_ids,
        attention_mask,
        input_multi_attention_mask,
        input_hidden_states,
        relation=None,
        max_new_tokens: int = 16,
        do_sample: bool = False,
        num_beams: int = 1,
        **gen_kwargs
    ):
        """
        推理阶段：使用与训练相同的表示级融合路径，然后把 fused encoder_outputs 交给 decoder 生成。
        """
        # 分支1
        single_enc_last, single_mask = self._encode_single_branch(
            input_ids, attention_mask, input_hidden_states, self.language_projection
        )
        # 分支2
        multi_enc_last, multi_mask = self._encode_single_branch(
            input_multi_ids, input_multi_attention_mask, input_hidden_states, self.imagetext_projection
        )

        # 对齐长度（如有必要）
        if single_enc_last.size(1) != multi_enc_last.size(1):
            min_len = min(single_enc_last.size(1), multi_enc_last.size(1))
            single_enc_last = single_enc_last[:, :min_len, :]
            multi_enc_last  = multi_enc_last[:, :min_len, :]
            single_mask = single_mask[:, :min_len]
            multi_mask  = multi_mask[:, :min_len]

        # 融合（与训练相同）
        fused_enc_last, _ = self.fuser(single_enc_last, multi_enc_last)
        fused_mask = torch.clamp(single_mask + multi_mask, max=1)

        # 交给 decoder 生成（与 forward 的融合分支一致）
        enc_outputs = BaseModelOutput(last_hidden_state=fused_enc_last)
        gen_out = self.language_model.generate(
            encoder_outputs=enc_outputs,
            attention_mask=fused_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            **gen_kwargs
        )
        # 解码
        predicted_labels = self.tokenizer.batch_decode(
            gen_out, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return predicted_labels
