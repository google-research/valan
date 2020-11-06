# coding=utf-8
# Copyright 2019 Google LLC
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


"""Utility functions required by files."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import re
import string

from absl import flags

import numpy as np

from valan.r2r import constants

FLAGS = flags.FLAGS


class Tokenizer(object):
  """A class to tokenize and encode sentences."""
  # Split on any non-alphanumeric character.
  # https://github.com/ronghanghu/speaker_follower
  SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

  @staticmethod
  def split_sentence(sentence):
    """Break sentence into a list of words and punctuation."""
    toks = []
    for s in Tokenizer.SENTENCE_SPLIT_REGEX.split(sentence.strip()):
      if s.strip():
        word = s.strip().lower()
        if (all(c in string.punctuation for c in word) and
            not all(c in '.' for c in word)):
          toks += list(word)
        else:
          toks.append(word)
    return toks


def get_token_ids(sentence, fixed_length, vocab):
  """Splits the sentence into tokens and returns their ids.

  Args:
    sentence: The input natural language instruction (sequence of characters).
    fixed_length: The maximum instruction length in R2R.
    vocab: R2R vocabulary dictionary.

  Returns:
    token_ids: A numpy array of length fixed_length with token indices.
    num_tokens: The number of tokens in the instruction.
  """
  tokens = Tokenizer.split_sentence(sentence)
  num_tokens = len(tokens)
  token_ids = []
  oov_token_id = vocab[constants.OOV_TOKEN]
  pad_token_id = vocab[constants.PAD_TOKEN]
  for i in range(fixed_length):
    if i < num_tokens:
      token_ids.append(vocab.get(tokens[i], oov_token_id))
    else:
      token_ids.append(pad_token_id)
  return np.array(token_ids), num_tokens
