# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
# limitations unde r the License.
"""Tokenization classes for RemBERT."""


import copy
import io
import json
import os
import six
import unicodedata
import sentencepiece as spm

from paddlenlp.transformers import PretrainedTokenizer
from paddlenlp.data.vocab import Vocab
from paddlenlp.transformers.tokenizer_utils import convert_to_unicode, whitespace_tokenize, _is_whitespace, _is_control, _is_punctuation

__all__ = ['RemBertTokenizer']


class RemBertTokenizer(PretrainedTokenizer):
    """
    Constructs a RemBERT tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    Args:
        vocab_file (str): file path of the vocabulary
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the RemBERT pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
        unk_token (str): The special token for unkown words. Default: "[UNK]".
        sep_token (str): The special token for separator token . Default: "[SEP]".
        pad_token (str): The special token for padding. Default: "[PAD]".
        cls_token (str): The special token for cls. Default: "[CLS]".
        mask_token (str): The special token for mask. Default: "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddle.hapi.text import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # the following line get: ['he', 'was', 'a', 'puppet', '##eer']
            tokens = tokenizer('He was a puppeteer')
            # the following line get: 'he was a puppeteer'
            tokenizer.convert_tokens_to_string(tokens)
    """
    resource_files_names = {"vocab_file": "sentencepiece.model"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {}
    }
    pretrained_init_configuration = {}
    padding_side = 'right'

    def __init__(
        self,
        vocab_file,
        do_lower_case=False,
        remove_space=True,
        keep_accents=True,
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        **kwargs
    ):
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(vocab_file)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.vocab = self.get_vocab()

    @property
    def vocab_size(self):
        """
        return the size of vocabulary.
        Returns:
            int: the size of vocabulary.
        """
        return len(self.sp_model)

    def get_vocab(self):
        token_to_idx = {self.sp_model.IdToPiece(i): i for i in range(self.vocab_size)}
        vocab = Vocab.from_dict(
            token_to_idx,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            )
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text):
        """Tokenize a string."""
        pieces = self.sp_model.EncodeAsPieces(text)
        return pieces

    def tokenize(self, text):
        """
        End-to-end tokenization for RemBERT models.
        Args:
            text (str): The text to be tokenized.
        
        Returns:
            list: A list of string representing converted tokens.
        """
        return self._tokenize(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        """
        Converts a sequence of tokens (list of string) in a single string. Since
        the usage of WordPiece introducing `##` to concat subwords, also remove
        `##` when converting.
        Args:
            tokens (list): A list of string representing tokens to be converted.
        Returns:
            str: Converted string from tokens.
        """
        out_string = self.sp_model.decode_pieces(tokens)
        return out_string

    def num_special_tokens_to_add(self, pair=False):
        """
        Returns the number of added tokens when encoding a sequence with special tokens.

        Note:
            This encodes inputs and checks the number of added tokens, and is therefore not efficient. Do not put this
            inside your training loop.

        Args:
            pair: Returns the number of added tokens in the case of a sequence pair if set to True, returns the
                number of added tokens in the case of a single sequence if set to False.

        Returns:
            Number of tokens added to sequences
        """
        token_ids_0 = []
        token_ids_1 = []
        return len(
            self.build_inputs_with_special_tokens(token_ids_0, token_ids_1
                                                  if pair else None))

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. 
        
        A RemBERT sequence has the following format:
        ::
            - single sequence: ``[CLS] X [SEP]``
            - pair of sequences: ``[CLS] A [SEP] B [SEP]``

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of input_id with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        _cls = [self.cls_token_id]
        _sep = [self.sep_token_id]
        return _cls + token_ids_0 + _sep + token_ids_1 + _sep

    def build_offset_mapping_with_special_tokens(self,
                                                 offset_mapping_0,
                                                 offset_mapping_1=None):
        """
        Build offset map from a pair of offset map by concatenating and adding offsets of special tokens. 
        
        A RemBERT offset_mapping has the following format:
        ::
            - single sequence: ``(0,0) X (0,0)``
            - pair of sequences: `(0,0) A (0,0) B (0,0)``
        
        Args:
            offset_mapping_ids_0 (:obj:`List[tuple]`):
                List of char offsets to which the special tokens will be added.
            offset_mapping_ids_1 (:obj:`List[tuple]`, `optional`):
                Optional second list of char offsets for offset mapping pairs.

        Returns:
            :obj:`List[tuple]`: List of char offsets with the appropriate offsets of special tokens.
        """
        if offset_mapping_1 is None:
            return [(0, 0)] + offset_mapping_0 + [(0, 0)]

        return [(0, 0)] + offset_mapping_0 + [(0, 0)
                                              ] + offset_mapping_1 + [(0, 0)]

    def create_token_type_ids_from_sequences(self,
                                             token_ids_0,
                                             token_ids_1=None):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. 

        A RemBERT sequence pair mask has the following format:
        ::

            0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
            | first sequence    | second sequence |

        If :obj:`token_ids_1` is :obj:`None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.

        Returns:
            :obj:`List[int]`: List of token_type_id according to the given sequence(s).
        """
        _sep = [self.sep_token_id]
        _cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(_cls + token_ids_0 + _sep) * [0]
        return len(_cls + token_ids_0 + _sep) * [0] + len(token_ids_1 + _sep) * [1]

    def get_special_tokens_mask(self,
                                token_ids_0,
                                token_ids_1=None,
                                already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``encode`` methods.

        Args:
            token_ids_0 (List[int]): List of ids of the first sequence.
            token_ids_1 (List[int], optinal): List of ids of the second sequence.
            already_has_special_tokens (bool, optional): Whether or not the token list is already 
                formatted with special tokens for the model. Defaults to None.

        Returns:
            results (List[int]): The list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + (
                [0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]


if __name__ == "__main__":
    tokenizer = RemBertTokenizer.from_pretrained("../rembert")
    sentence = "Welcome to use paddlepaddle and paddleNLP!"
    print(tokenizer(sentence))