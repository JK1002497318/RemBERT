# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.tensor as tensor
import paddle.nn.functional as F
from paddle.nn import TransformerEncoder, Linear, Layer, Embedding, LayerNorm, Tanh

from paddlenlp.transformers import PretrainedModel, register_base_model

__all__ = [
    'RemBertModel',
    'RemBertEmbeddings',
    'RemBertPooler',
    "RemBertPretrainedModel",
    'RemBertForSequenceClassification',
]


ACT2FN = {
    "relu": nn.functional.relu,
    "silu": nn.functional.silu,
    "swish": nn.functional.silu,
    "gelu": nn.functional.gelu,
    "tanh": nn.functional.tanh,
    "sigmoid": nn.functional.sigmoid,
}


class RemBertPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained BERT models. It provides BERT related
    `model_config_file`, `resource_files_names`, `pretrained_resource_files_map`,
    `pretrained_init_configuration`, `base_model_prefix` for downloading and
    loading pretrained models. See `PretrainedModel` for more details.
    """

    model_config_file = "config.json"
    pretrained_init_configuration = {
        # "rembert": {
        #     "_name_or_path": "artefacts/pt_model",
        #     "attention_probs_dropout_prob": 0,
        #     "bos_token_id": 312,
        #     "embedding_dropout_prob": 0,
        #     "embedding_size": 256,
        #     "eos_token_id": 313,
        #     "hidden_act": "gelu",
        #     "hidden_dropout_prob": 0,
        #     "hidden_size": 1152,
        #     "initializer_range": 0.02,
        #     "input_embedding_size": 256,
        #     "intermediate_size": 4608,
        #     "layer_norm_eps": 1e-12,
        #     "max_position_embeddings": 512,
        #     "model_type": "rembert",
        #     "num_attention_heads": 18,
        #     "num_hidden_layers": 32,
        #     "output_embedding_size": 1664,
        #     "pad_token_id": 0,
        #     "tie_word_embeddings": False,
        #     "transformers_version": "4.4.0.dev0",
        #     "type_vocab_size": 2,
        #     "use_cache": True,
        #     "vocab_size": 250300
        # }
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
            "bert-base-uncased":
            "https://paddlenlp.bj.bcebos.com/models/transformers/bert-base-uncased.pdparams",
        }
    }
    base_model_prefix = "rembert"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.bert.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class RemBertModel(RemBertPretrainedModel):
    """
    The bare BERT Model transformer outputting raw hidden-states without any specific head on top.

    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Check the superclass documentation for the generic methods and the library implements for all its model.

    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.

    Args:
        vocab_size (`int`):
            Vocabulary size of the XLNet model. Defines the number of different tokens that can
            be represented by the `inputs_ids` passed when calling XLNetModel.
        hidden_size (`int`, optional):
            Dimensionality of the encoder layers and the pooler layer. Defaults to ``768``.
        num_hidden_layers (`int`, optional):
            Number of hidden layers in the Transformer encoder. Defaults to ``12``.
        num_attention_heads (`int`, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to ``12``.
        intermediate_size (`int`, optional):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            Defaults to ``3072``.
        hidden_act (`str`, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to ``0.1``.
        attention_probs_dropout_prob (`float`, optional):
            The dropout probability for all fully connected layers in the pooler.
            Defaults to ``0.1``.
        initializer_range (`float`, optional):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            Defaults to ``0.02``.
    """

    def __init__(self,
                 _name_or_path= "artefacts/pt_model",
                 attention_probs_dropout_prob=0,
                 bos_token_id=312,
                 embedding_dropout_prob=0,
                 embedding_size=256,
                 eos_token_id=313,
                 hidden_act="gelu",
                 hidden_dropout_prob=0,
                 hidden_size=1152,
                 initializer_range=0.02,
                 input_embedding_size=256,
                 intermediate_size=4608,
                 layer_norm_eps=1e-12,
                 max_position_embeddings=512,
                 model_type="rembert",
                 num_attention_heads=18,
                 num_hidden_layers=32,
                 output_embedding_size=1664,
                 pad_token_id=0,
                 tie_word_embeddings=False,
                 transformers_version="4.4.0.dev0",
                 type_vocab_size=2,
                 use_cache=True,
                 vocab_size=250300,
                 pool_act="tanh"):
        super(RemBertModel, self).__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = RemBertEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size,
            num_attention_heads,
            intermediate_size,
            dropout=hidden_dropout_prob,
            activation=hidden_act,
            attn_dropout=attention_probs_dropout_prob,
            act_dropout=0)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        self.pooler = RemBertPooler(hidden_size, pool_act)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                output_hidden_states=False):
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        if output_hidden_states:
            output = embedding_output
            encoder_outputs = []
            for mod in self.encoder.layers:
                output = mod(output, src_mask=attention_mask)
                encoder_outputs.append(output)
            if self.encoder.norm is not None:
                encoder_outputs[-1] = self.encoder.norm(encoder_outputs[-1])
            pooled_output = self.pooler(encoder_outputs[-1])
        else:
            sequence_output = self.encoder(embedding_output, attention_mask)
            pooled_output = self.pooler(sequence_output)
        if output_hidden_states:
            return encoder_outputs, pooled_output
        else:
            return sequence_output, pooled_output


class RemBertEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """
    def __init__(self,
                 vocab_size=250300,
                 hidden_size=768,
                 pad_token_id=0,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 input_embedding_size=256,
                 layer_norm_eps=1e-12,
                 hidden_dropout_prob=0):
        super(RemBertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, input_embedding_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                input_embedding_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.layer_norm = nn.LayerNorm(input_embedding_size, epsilon=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)

            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class RemBertPooler(Layer):
    """
    """

    def __init__(self, hidden_size, pool_act="tanh"):
        super(RemBertPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.pool_act = pool_act

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        if self.pool_act == "tanh":
            pooled_output = self.activation(pooled_output)
        return pooled_output


class RemBertForSequenceClassification(RemBertPretrainedModel):
    """
    Model for sentence (pair) classification task with BERT.
    Args:
        rembert (RemBertModel): An instance of BertModel.
        num_classes (int, optional): The number of classes. Default 2
        dropout (float, optional): The dropout probability for output of RemBERT.
            If None, use the same value as `hidden_dropout_prob` of `RemBertModel`
            instance `rembert`. Default None
    """

    def __init__(self, rembert, num_classes=3, dropout=None):
        super(RemBertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.rembert = rembert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.rembert.config["embedding_dropout_prob"])
        self.classifier = nn.Linear(self.rembert.config["hidden_size"],
                                    num_classes)
        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                labels=None):
        _, pooled_output = self.rembert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            attention_mask=attention_mask)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = paddle.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = paddle.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return loss, logits
