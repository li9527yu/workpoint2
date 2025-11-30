import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class MyFlanT5(nn.Module):
    def __init__(
        self,
        model_path,
        tokenizer: T5Tokenizer,
        model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl',
        label_words=('positive', 'neutral', 'negative'),
        d_rel: int = 8  # relation_s 的 embedding 维度
    ):
        super().__init__()
        self.language_model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer

        # 两路 query 投影
        self.language_projection = nn.Linear(768, 768)
        self.imagetext_projection = nn.Linear(768, 768)
        self.init_linear_weight()

        # ---- Soft Gating 相关 ----
        # 1) relation_s 的 embedding（0:无关, 1:情感相关, 2:语义相关但情感无关）
        self.relation_embedding = nn.Embedding(3, d_rel)

        # 2) 门控 MLP：输入 = [relation_embed, prob_text, prob_image]
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_rel + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 输出 alpha ∈ [0,1]，作为“文本路权重”
        )

        # 3) 三分类标签词 → token ids（用于从 vocab 中抽取类别 logits 计算置信度）
        self.label_words = tuple(label_words)
        self.class_token_ids = self._build_label_token_ids(self.label_words)  # [3]

    # ---------- 工具函数 ----------
    def init_linear_weight(self):
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)
        nn.init.normal_(self.imagetext_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.imagetext_projection.bias)

    def _build_label_token_ids(self, label_words):
        ids = []
        for w in label_words:
            tok_ids = self.tokenizer.encode(w, add_special_tokens=False)
            if len(tok_ids) == 0:
                raise ValueError(f"Label word '{w}' cannot be tokenized.")
            ids.append(tok_ids[0])  # 用首 token 代表该类
        return torch.tensor(ids, dtype=torch.long)

    def get_embeds(self, input_ids):
        emb = self.language_model.get_input_embeddings()
        return emb(input_ids)

    def _class_logits_from_vocab(self, step_logits):
        """
        从 [B, V] 的词表 logits 抽取类别 logits [B, 3]（按 self.class_token_ids）。
        """
        class_ids = self.class_token_ids.to(step_logits.device)
        return step_logits.index_select(dim=-1, index=class_ids)  # [B, 3]

    def _branch_conf_and_pred(self, step_logits):
        """
        对单一路径的某一步 vocab logits：
         1) 抽取三类 logits
         2) softmax 得到 [B,3] 概率
         3) 返回最大概率 pmax 与 预测类别索引 yhat
        """
        cls_logits = self._class_logits_from_vocab(step_logits)  # [B, 3]
        probs = F.softmax(cls_logits, dim=-1)
        pmax, yhat = probs.max(dim=-1)  # [B], [B]
        return pmax, yhat

    def _gate_weight(self, relation, p_text, p_img):
        """
        Soft gating：alpha = gate_mlp([relation_embed, p_text, p_img])
        返回 alpha（文本权重）与 beta=1-alpha（图像权重）
        """
        if not isinstance(relation, torch.Tensor):
            relation = torch.tensor(relation, dtype=torch.long, device=p_text.device)
        relation = relation.to(p_text.device)
        relation_embed = self.relation_embedding(relation)  # [B, d_rel]
        gate_input = torch.cat([relation_embed, p_text.unsqueeze(-1), p_img.unsqueeze(-1)], dim=-1)  # [B, d_rel+2]
        alpha = self.gate_mlp(gate_input)  # [B,1]
        beta = 1.0 - alpha
        return alpha, beta

    # ---------- 训练前向：融合后算 loss（训练/推理一致） ----------
    def forward(
        self,
        input_ids,
        input_multi_ids,
        attention_mask,
        input_multi_attention_mask,
        input_hidden_states,
        relation,  # [B]，取值 0/1/2
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        # 文本路径
        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(query.size()[:-1], dtype=torch.long, device=query.device)
        relation_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, relation_inputs_embeds.to(query.device)], dim=1)
        attn_mask = torch.cat([query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True
        )

        # 多模态路径
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device)
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
        multi_attn_mask = torch.cat([imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1)

        multi_outputs = self.language_model(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attn_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True
        )

        # ---- Soft gating：基于“最后一步”计算两路置信度（也可按需改为 labels 对齐的步）----
        step_text = outputs.logits[:, -1, :]      # [B, V]
        step_img  = multi_outputs.logits[:, -1, :]  # [B, V]
        p_text, _ = self._branch_conf_and_pred(step_text)  # [B]
        p_img,  _ = self._branch_conf_and_pred(step_img)   # [B]

        alpha, beta = self._gate_weight(relation, p_text, p_img)  # [B,1], [B,1]
        alpha = alpha.view(-1, 1, 1)
        beta  = beta.view(-1, 1, 1)

        # 融合 logits 并计算 loss
        fused_logits = alpha * outputs.logits + beta * multi_outputs.logits  # [B, T, V]

        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            total_loss = loss_fct(
                fused_logits.view(-1, fused_logits.size(-1)),
                labels.view(-1)
            )
        return total_loss

    # ---------- 推理：步级 soft gating 融合 ----------
    @torch.no_grad()
    def generate(
        self,
        input_ids,
        input_multi_ids,
        attention_mask,
        input_multi_attention_mask,
        input_hidden_states,
        relation,
        max_new_tokens=32,
        do_sample=False
    ):
        # 文本路径
        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(query.size()[:-1], dtype=torch.long, device=query.device)
        relation_inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, relation_inputs_embeds.to(query.device)], dim=1)
        attn_mask = torch.cat([query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 多模态路径
        imagetext_query = self.imagetext_projection(input_hidden_states)
        imagetext_query_attention_mask = torch.ones(imagetext_query.size()[:-1], dtype=torch.long, device=imagetext_query.device)
        multi_inputs_embeds = self.get_embeds(input_multi_ids)
        multi_inputs_embeds = torch.cat([imagetext_query, multi_inputs_embeds.to(imagetext_query.device)], dim=1)
        multi_attn_mask = torch.cat([imagetext_query_attention_mask, input_multi_attention_mask.to(imagetext_query_attention_mask.device)], dim=1)

        multi_outputs = self.language_model.generate(
            inputs_embeds=multi_inputs_embeds,
            attention_mask=multi_attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True
        )

        # 步级融合：每一步根据两路“当前步”logits 计算 pmax → 门控 → 融合
        fused_token_ids = []
        # relation 张量
        if not isinstance(relation, torch.Tensor):
            relation = torch.tensor(relation, dtype=torch.long, device=inputs_embeds.device)
        relation = relation.to(inputs_embeds.device)

        for step_logits_text, step_logits_img in zip(outputs.scores, multi_outputs.scores):
            # 先用类别 logits 计算两路置信度（pmax）
            p_text, _ = self._branch_conf_and_pred(step_logits_text)  # [B]
            p_img,  _ = self._branch_conf_and_pred(step_logits_img)   # [B]

            # 门控权重（文本 alpha，图像 beta）
            alpha, beta = self._gate_weight(relation, p_text, p_img)  # [B,1], [B,1]

            # 用全 vocab logits 做融合（也可仅在3类上融合：更像分类）
            fused_logits = alpha * step_logits_text + beta * step_logits_img  # [B, V]
            fused_token_ids.append(fused_logits.argmax(dim=-1))  # [B]

        fused_token_ids = torch.stack(fused_token_ids, dim=1)  # [B, T]
        predicted_labels = self.tokenizer.batch_decode(
            fused_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return predicted_labels
