import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import  T5ForConditionalGeneration,GenerationConfig

class MyFlanT5(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.language_model=T5ForConditionalGeneration.from_pretrained(args.pretrained_model_dir)
        self.language_model.resize_token_embeddings(len(args.tokenizer))
        self.text_embeddings = self.language_model.get_input_embeddings()
        # 在输入部分添加 MLP 层
        self.language_projection = nn.Linear(768, 768)
        self.init_linear_weight()

        self.a_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, f'{args.pretrained_model_dir}/multi_task/a_generation_config.json')
        self.sra_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, f'{args.pretrained_model_dir}/multi_task/sra_generation_config.json')
        self.ra_generation_config = GenerationConfig.from_pretrained(args.pretrained_model_dir, f'{args.pretrained_model_dir}/multi_task/ra_generation_config.json')

    def init_linear_weight(self):
        # # 初始化
        nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.language_projection.bias)

    def forward(self, a_input_ids, a_attention_mask, a_decoder_output_labels, sra_input_ids, sra_attention_mask, sra_decoder_output_labels, ra_input_ids,
                ra_attention_mask, ra_decoder_output_labels, image_feature, is_eval=False):
        query=self.language_projection(image_feature) #[bs,32,768]
        query_attention_mask = torch.ones(
            query.size()[:-1], dtype=torch.long, device=query.device
        ) #[bs,32]

        # a
        a_inputs_embeds = self.text_embeddings(a_input_ids) ##[bs,128,768]
        a_inputs_embeds = torch.cat([query, a_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        a_attention_mask = torch.cat(
            [query_attention_mask, a_attention_mask.to(query_attention_mask.device)], dim=1
        )

        # sra
        sra_inputs_embeds = self.text_embeddings(sra_input_ids)
        sra_inputs_embeds = torch.cat([query, sra_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        sra_attention_mask = torch.cat(
            [query_attention_mask, sra_attention_mask.to(query_attention_mask.device)], dim=1
        )

        # ra
        ra_inputs_embeds = self.text_embeddings(ra_input_ids)
        ra_inputs_embeds = torch.cat([query, ra_inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
        ra_attention_mask = torch.cat(
            [query_attention_mask, ra_attention_mask.to(query_attention_mask.device)], dim=1
        )

        if not is_eval: 
            a_outputs = self.language_model(
                inputs_embeds=a_inputs_embeds,
                attention_mask=a_attention_mask,
                labels=a_decoder_output_labels
            )
            a_loss=a_outputs['loss']

            sra_outputs = self.language_model(
                inputs_embeds=sra_inputs_embeds,
                attention_mask=sra_attention_mask,
                labels=sra_decoder_output_labels
            )
            sra_loss=sra_outputs['loss']

            ra_outputs = self.language_model(
                inputs_embeds=ra_inputs_embeds,
                attention_mask=ra_attention_mask,
                labels=ra_decoder_output_labels
            )
            ra_loss=ra_outputs['loss']
            return a_loss, sra_loss, ra_loss
        else:
            batch_size = a_inputs_embeds.shape[0]  # 获取 inputs_embeds 的 batch_size
            a_decoder_start_token_id = 32104  # 起始符的token ID
            a_decoder_input_ids = torch.tensor(
                [[a_decoder_start_token_id]] * batch_size  # 扩展为 (batch_size, 1)
            ).to(a_inputs_embeds.device)
            r_decoder_start_token_id = 32102  # 起始符的token ID
            r_decoder_input_ids = torch.tensor(
                [[r_decoder_start_token_id]] * batch_size  # 扩展为 (batch_size, 1)
            ).to(a_inputs_embeds.device)

            a_sequence_ids = self.language_model.generate(inputs_embeds=a_inputs_embeds, attention_mask=a_attention_mask,decoder_input_ids=a_decoder_input_ids, generation_config=self.a_generation_config)
            sra_sequence_ids = self.language_model.generate(inputs_embeds=sra_inputs_embeds, attention_mask=sra_attention_mask,decoder_input_ids=r_decoder_input_ids, generation_config=self.sra_generation_config)
            ra_sequence_ids = self.language_model.generate(inputs_embeds=ra_inputs_embeds, attention_mask=ra_attention_mask,decoder_input_ids=r_decoder_input_ids, generation_config=self.ra_generation_config)
            a_sequence = self.args.tokenizer.batch_decode(a_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            sra_sequence = self.args.tokenizer.batch_decode(sra_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            ra_sequence = self.args.tokenizer.batch_decode(ra_sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            return a_sequence, sra_sequence, ra_sequence




        
