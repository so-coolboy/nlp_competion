# -*- coding: utf-8 -*-
# @Time    : 2021/1/3
# @Author  : Xichuankun
# @Email   : 1364707232@qq.com
# @Software: Vs Code

import torch.nn as nn
from NEZHA.model_nezha import BertConfig
import os
import torch
from config import Config
from torch_utils.CRF_layers import CRFLayer
if Config().pretrainning_model == 'nezha':
    from NEZHA.model_nezha import BertPreTrainedModel, NEZHAModel
elif Config().pretrainning_model == 'albert':
    from transformers import AlbertModel, BertPreTrainedModel
else:
    # bert,roberta
    from transformers import RobertaModel, BertPreTrainedModel

from torch_utils import BiLSTM, IDCNN, RTransformer, TENER
config = Config()


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.params = params
        # 实体类别数
        self.num_labels = len(params.tags)
        # nezha
        if params.pretrainning_model == 'nezha':
            self.bert = NEZHAModel(config)
        elif params.pretrainning_model == 'albert':
            self.bert = AlbertModel(config)
        else:
            self.bert = RobertaModel(config)

        # 动态权重
        self.classifier = nn.Linear(config.hidden_size, 1)  # for dym's dense
        self.dense_final = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                         nn.ReLU(True))  # 动态最后的维度
        self.dym_weight = nn.Parameter(torch.ones((config.num_hidden_layers, 1, 1, 1)),
                                       requires_grad=True)
        self.pool_weight = nn.Parameter(torch.ones((2, 1, 1, 1)),
                                        requires_grad=True)
        # 结构
        self.idcnn = IDCNN(config, params, filters=params.filters, tag_size=self.num_labels,
                           kernel_size=params.kernel_size)
        self.bilstm = BiLSTM(self.num_labels, embedding_size=config.hidden_size, hidden_size=params.lstm_hidden,
                             num_layers=params.num_layers,
                             dropout=params.drop_prob, with_ln=True)
        self.tener = TENER(tag_size=self.num_labels, embed_size=config.hidden_size, dropout=params.drop_prob,
                               num_layers=params.num_layers, d_model=params.tener_hs, n_head=params.num_heads)
        self.rtransformer=RTransformer(tag_size=self.num_labels, dropout=params.drop_prob, d_model=config.hidden_size,
                                       ksize=params.k_size, h=params.rtrans_heads)

        self.base_output = nn.Linear(config.hidden_size, self.num_labels)
        # crf
        self.crf = CRFLayer(self.num_labels, params)
        if params.pretrainning_model == 'nezha':
            self.apply(self.init_bert_weights)
        else:
            self.init_weights()
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        layer_logits = []
        all_encoder_layers = outputs[1:]
        for i, layer in enumerate(all_encoder_layers):
            layer_logits.append(self.classifier(layer))
        layer_logits = torch.cat(layer_logits, 2)
        layer_dist = torch.nn.functional.softmax(layer_logits)
        with open(Config().processed_data + 'dynamic_weight_record.txt', 'a') as fw:
            fw.write(str(layer_dist))
        seq_out = torch.cat([torch.unsqueeze(x, 2) for x in all_encoder_layers], dim=2)
        pooled_output = torch.matmul(torch.unsqueeze(layer_dist, 2), seq_out)
        pooled_output = torch.squeeze(pooled_output, 2)
        ## 是否真的需要激活函数
        word_embed = self.dense_final(pooled_output)
        dym_layer = word_embed
        return dym_layer

    def get_weight_layer(self, outputs):
        """
        获取动态权重融合后的bert output(num_layer维度)
        :param outputs: origin bert output
        :return: sequence_output: 融合后的bert encoder output. (batch_size, seq_len, hidden_size[embedding_dim])
        """
        hidden_stack = torch.stack(outputs[1:], dim=0)  # (bert_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
    ):
        """
        :param input_ids: (batch_size, seq_len)
        :param attention_mask: 各元素的值为0或1，避免在padding的token上计算attention。(batch_size, seq_len)
        :param token_type_ids: token对应的句子类型id，值为0或1。为空自动生成全0。(batch_size, seq_len)
        :param position_ids: 位置编码。为空自动根据句子长度生成。(batch_size, sequence_length)
        :param head_mask: 各元素的值为0或1。为空自动生成全1，即不mask。(num_heads,) or (num_layers, num_heads)
        :param inputs_embeds: 与input_ids互斥。(batch_size, seq_len, embedding_dim)
        :param labels: (batch_size, seq_len)
    Returns:
        loss: scores对应的交叉熵损失
            returned when ``labels`` is provided)
            Classification loss.
        scores: (batch_size, sequence_length, config.num_labels)
            Classification scores (before SoftMax).
        hidden_states (Tuple): embedding层的输出和各层encoder的输出
            one for the output of the embeddings + one for the output of each layer
            returned when ``config.output_hidden_states=True`` (batch_size, sequence_length, hidden_size)
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (Tuple): 各层encoder中，各attention head的self-attention概率
            returned when ``config.output_attentions=True`` (batch_size, num_heads, sequence_length, sequence_length)
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        # pretrain model
        # Nezha
        if config.pretrainning_model == 'nezha':
            encoded_layers, pooled_output = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_all_encoded_layers=True
            )  # encoded_layers, pooled_output
            sequence_output = encoded_layers[-1]
        else:
            sequence_output, pooled_output, encoded_layers = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )  # sequence_output, pooled_output, (hidden_states), (attentions)
        if not config.use_origin_bert:
            sequence_output = self.get_weight_layer(encoded_layers)  # (batch_size, seq_len, hidden_size[embedding_dim])
            # sequence_output = self.get_dym_layer(encoded_layers) # (batch_size, seq_len, hidden_size[embedding_dim])
        # middle
        # (seq_len, batch_size, tag_size)
        if self.params.mid_struct == 'bilstm':
            feats = self.bilstm.get_lstm_features(sequence_output.transpose(1, 0), attention_mask.transpose(1, 0))
        elif self.params.mid_struct == 'idcnn':
            feats = self.idcnn(sequence_output).transpose(1, 0)
        elif self.params.mid_struct == 'tener':
            feats = self.tener(sequence_output,attention_mask).transpose(1, 0)
        elif self.params.mid_struct == 'rtransformer':
            feats = self.rtransformer(sequence_output, attention_mask).transpose(1, 0)

        elif self.params.mid_struct == 'base':
            feats = self.base_output(sequence_output).transpose(1, 0)
        else:
            raise KeyError('mid_struct must in [bilstm idcnn tener rtransformer]')
        # CRF
        if labels is not None:
            # total scores
            forward_score = self.crf(feats, attention_mask.transpose(1, 0))
            gold_score = self.crf.score_sentence(feats, labels.transpose(1, 0),
                                                 attention_mask.transpose(1, 0))
            loss = (forward_score - gold_score).mean()
            return loss
        else:
            # 维特比算法
            best_paths = self.crf.viterbi_decode(feats, attention_mask.transpose(1, 0))
            return best_paths


if __name__ == '__main__':
    """
    测试模型网络结构
    """
    params = Config()
    bert_config = BertConfig.from_json_file(os.path.join(params.model_path, 'bert_config.json'))
    model = BertForTokenClassification(bert_config, params=params)
    for n, p in model.named_parameters():
        print(n)
