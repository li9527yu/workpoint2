import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaModel, AutoConfig,InstructBlipConfig,InstructBlipQFormerAttention,InstructBlipQFormerEmbeddings
from modeling_utils import BertSelfEncoder, BertCrossEncoder_AttnMap, BertPooler, BertLayerNorm



class MyFlanT5(nn.Module):
    def __init__(self,model_path,model_name='/public/home/ghfu/lzy/model/instructblip-flan-t5-xl'):
        super().__init__()
        # relation
        self.config=InstructBlipConfig.from_pretrained(model_name)
        config=self.config.qformer_config
        self.embeddings = InstructBlipQFormerEmbeddings(config)
        self.crossattention = InstructBlipQFormerAttention(config, is_cross_attention=True)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


        self.dropout1=nn.Dropout(0.3)
        self.gather=nn.Linear(768,1)
        self.dropout2=nn.Dropout(0.3)
        self.pred=nn.Linear(128,2)

        
        
        self.language_model=model_path
        # 在输入部分添加 MLP 层
        self.language_projection = nn.Linear(768, 768)
        self.selector=nn.Linear(768, 2)
        self.init_linear_weight()

    def init_linear_weight(self):
        # 初始化
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)) and ('roberta' not in name ): #linear/embedding
                module.weight.data.normal_(mean=0.0, std=0.02)
            elif isinstance(module, BertLayerNorm) and ('roberta' not in name ):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None and ('roberta' not in name ):
                module.bias.data.zero_()
        # nn.init.normal_(self.language_projection.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.language_projection.bias)

        # nn.init.normal_(self.feat_linear.weight, mean=0.0, std=0.02)
        # nn.init.zeros_(self.feat_linear.bias)

    # 定义获取嵌入层的函数
    def get_relation_embeds(self,input_ids):

        embeddings = self.word_embeddings(input_ids)
        embeddings = embeddings.to(self.layernorm.weight.dtype)
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
    def get_embeds(self, input_ids):
        """
        获取输入的嵌入向量
        :param input_ids: 输入的token IDs
        :return: 对应的嵌入向量
        """
        # 调用父类的 get_input_embeddings 方法来获取嵌入层
        input_embeddings = self.language_model.get_input_embeddings()
        
        # 通过嵌入层将 token IDs 转换为嵌入向量
        input_embeds = input_embeddings(input_ids)  # shape: [bs, seq_len, embedding_dim]
        
        return input_embeds

    def forward(self,rel_inputs_id,rel_inputs_mask,img_feat,rel_label, input_ids=None,input_ir_ids=None,input_hidden_state=None,input_pooler_output=None,\
        attention_mask=None, input_ir_attention_mask=None,\
        decoder_input_ids=None, decoder_attention_mask=None, labels=None):

         
        if (rel_label == -1).all():
            # 先做相关性分类 作用于指令部分
            rel_inputs_mask=rel_inputs_mask.unsqueeze(1).unsqueeze(2)
            rel_inputs_mask = rel_inputs_mask.to(dtype=next(self.parameters()).dtype)
            rel_inputs_mask = (1.0 - rel_inputs_mask) * -10000.0  
            #img_feat [8, 257, 1408]
            
            image_attention_mask = torch.ones(img_feat.size()[:-1], dtype=torch.long, device=img_feat.device)
            image_attention_mask=image_attention_mask.unsqueeze(1).unsqueeze(2)
            image_attention_mask = (1.0 - image_attention_mask) * -10000.0
            # rel_inputs_embeds = self.get_embeds(rel_inputs_id) #[8, 150, 768]
            rel_inputs_embeds = self.get_relation_embeds(rel_inputs_id)
            

            rel_outputs=self.crossattention(
                hidden_states=rel_inputs_embeds,
                attention_mask=rel_inputs_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=image_attention_mask

            )[0]  #[8, 150, 768]

            gathered_sentence_aware_image=self.gather(self.dropout1(rel_outputs)).squeeze(2)
            rel_pred=self.pred(self.dropout2(gathered_sentence_aware_image)) #  [N,2]

            probabilities = F.softmax(rel_pred, dim=-1)
            #    获取每个样本的最大概率类别
            predicted_labels = torch.argmax(probabilities, dim=-1)
            final_input_ids = input_ir_ids * (predicted_labels.unsqueeze(1) == 0).float() + \
                    input_ids * (predicted_labels.unsqueeze(1) == 1).float()
            final_attention_mask = input_ir_attention_mask * (predicted_labels.unsqueeze(1) == 0).float() + \
                    attention_mask * (predicted_labels.unsqueeze(1) == 1).float()
            # 注意类型转化为long
            final_input_ids=final_input_ids.to(torch.long)
            final_attention_mask=final_attention_mask.to(torch.long)
            # 将筛选后的输入和query进行拼接
            query=self.language_projection(input_hidden_state)
            # query=input_hidden_state
            query_attention_mask = torch.ones(
                query.size()[:-1], dtype=torch.long, device=query.device
            )
            inputs_embeds = self.get_embeds(final_input_ids)
            inputs_embeds = torch.cat([query, inputs_embeds.to(query.device)], dim=1)  # [bs, 32+128,768]
            attention_mask = torch.cat(
                [query_attention_mask, final_attention_mask.to(query_attention_mask.device)], dim=1
            )
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels
            )
            return outputs
        else:
            rel_inputs_mask=rel_inputs_mask.unsqueeze(1).unsqueeze(2)
            rel_inputs_mask = rel_inputs_mask.to(dtype=next(self.parameters()).dtype)
            rel_inputs_mask = (1.0 - rel_inputs_mask) * -10000.0  
            #img_feat [8, 257, 1408]
            
            image_attention_mask = torch.ones(img_feat.size()[:-1], dtype=torch.long, device=img_feat.device)
            image_attention_mask=image_attention_mask.unsqueeze(1).unsqueeze(2)
            image_attention_mask = (1.0 - image_attention_mask) * -10000.0
            rel_inputs_embeds = self.get_relation_embeds(rel_inputs_id) #[8, 150, 768]
            rel_outputs=self.crossattention(
                hidden_states=rel_inputs_embeds,
                attention_mask=rel_inputs_mask,
                encoder_hidden_states=img_feat,
                encoder_attention_mask=image_attention_mask

            )[0]  #[8, 150, 768]

            gathered_sentence_aware_image=self.gather(self.dropout1(rel_outputs)).squeeze(2)
            rel_pred=self.pred(self.dropout2(gathered_sentence_aware_image)) #  [N,2]
            return rel_pred
            
