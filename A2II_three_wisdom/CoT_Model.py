import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration
class MyFlanT5(nn.Module):
    def __init__(self,model_path,tokenizer,model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl'):
            super().__init__()
            self.language_model=T5ForConditionalGeneration.from_pretrained(model_path)
            self.tokenizer=tokenizer
            # 在输入部分添加 MLP 层
            # 图像特征的映射
            self.language_projection = nn.Linear(768, 768)
            self.imagetext_projection = nn.Linear(768, 768)
            # 两路模型融合后特征的映射
            self.fused_projection = nn.Linear(768, 768)
            # self.selector=nn.Linear(768, 2)
            self.init_linear_weight()
    def init_linear_weight(self):
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

    def get_embeds(self, input_ids):
        input_embeddings = self.language_model.get_input_embeddings()
        input_embeds = input_embeddings(input_ids)  # shape: [bs, seq_len, hidden_dim]
        return input_embeds

    def forward(self, input_ids, attention_mask, input_hidden_states,relation,
                decoder_input_ids=None, decoder_attention_mask=None, labels=None):

        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)
        attention_mask = torch.cat(
            [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        )
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, input_ids, attention_mask, input_hidden_states,
                 decoder_input_ids=None, decoder_attention_mask=None, labels=None):
        query = self.language_projection(input_hidden_states)
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        )
        inputs_embeds = self.get_embeds(input_ids)
        inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)
        attention_mask = torch.cat(
            [query_attention_mask, attention_mask.to(query_attention_mask.device)], dim=1
        )
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=16,
            do_sample=False
        )
        predicted_labels = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return predicted_labels

    def logit_generate_batch(self, input_ids, attention_mask, input_hidden_states, options):
        """
        批量版 logit_generate: 多问题 + 相同选项
        Args:
            input_ids: [B, Lq]  每个问题的 token ids
            attention_mask: [B, Lq]
            input_hidden_states: [B, Lh, D]
            options: dict，例如 {"positive":"positive","neutral":"neutral","negative":"negative"}
        Returns:
            results: list of {"answer": best_option, "logit_score": probs}
        """
        device = input_ids.device
        B = input_ids.size(0)
        option_texts = list(options.values())
        option_keys = list(options.keys())
        N = len(option_texts)

        # ===== 1. 构造 query + input embeddings =====
        query = self.language_projection(input_hidden_states)  # [B, Lh, D]
        query_attention_mask = torch.ones(query.size()[:-1], dtype=torch.long, device=device)
        inputs_embeds = self.get_embeds(input_ids)  # [B, Lq, D]
        inputs_embeds = torch.cat([query, inputs_embeds.to(device)], dim=1)  # [B, Lh+Lq, D]
        attn_mask = torch.cat([query_attention_mask, attention_mask.to(device)], dim=1)  # [B, Lh+Lq]

        # ===== 2. 扩展成 B*N (每个问题复制N次) =====
        expanded_inputs = inputs_embeds.unsqueeze(1).repeat(1, N, 1, 1).view(B*N, -1, inputs_embeds.size(-1))
        expanded_masks = attn_mask.unsqueeze(1).repeat(1, N, 1).view(B*N, -1)

        # ===== 3. 构造所有候选 labels =====
        labels = self.tokenizer(option_texts * B, return_tensors="pt", padding=True).to(device)  # [B*N, L_ans]

        # ===== 4. forward 一次 =====
        with torch.inference_mode():
            outputs = self.language_model(
                inputs_embeds=expanded_inputs,
                attention_mask=expanded_masks,
                labels=labels.input_ids
            )
            logits = outputs.logits   # [B*N, L_ans, V]

        # ===== 5. 计算逐选项 loss =====
        vocab_size = logits.size(-1)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels.input_ids[..., 1:].contiguous()

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        option_loss = per_token_loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)
        scores = -option_loss  # 越大越好

        # ===== 6. 还原到 [B, N] =====
        scores = scores.view(B, N)
        probs = F.softmax(scores, dim=-1).detach().cpu().numpy()

        # ===== 7. 返回结果 =====
        results = []
        for i in range(B):
            best_idx = scores[i].argmax().item()
            results.append({
                "answer": option_keys[best_idx],
                "logit_score": probs[i]
            })

        return results
