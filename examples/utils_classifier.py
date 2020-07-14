# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" BERT multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension  """

from __future__ import absolute_import, division, print_function


import logging
import os
import copy
import sys
from io import open
import json
import csv
import glob
import tqdm
import torch
import re
import kaldiio

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, guid, text, label=None, feats=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            text: list of str. The untokenized text of the utterance.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.label = label
        self.text = text
        self.feats = feats


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""
    def __init__(self, data_dir, speech_data_dir=None):
        super(DataProcessor, self).__init__()
        self.space_and_punct = re.compile(" (['.,?!]|n't|-)")
        self.data_dir = data_dir
        self.speech_data_dir = speech_data_dir
        self.examples = {}

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        labels = []

        if hasattr(self, 'labels'):
            labels = self.labels
        else:
            labels = sorted([str(x) for x in list(set([
                            e.label for e in
                                self.get_train_examples() +
                                self.get_dev_examples() +
                                self.get_test_examples()
                        ]))])
        return labels

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_csv(cls, input_file, quotechar='"'):
        """Reads a comma separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    def _clean_text(self, text):
        """Cleans text."""
        text = ' '.join(text.split())
        text = self.space_and_punct.sub('\\1', text)
        text = text.replace('- ', '-')

        if text[-1] != '.' and text[-1] != '?':
            text += '.'

        return text

    @classmethod
    def _read_scp(self, input_file):
        loader = kaldiio.load_scp(input_file)
        feats = {}

        for k in loader:
            feats[k] = loader[k]

        return feats

class DaProcessor(DataProcessor):
    """Processor for the SwitchBoard data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "ordered_sentences.train")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "ordered_sentences.valid")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "ordered_sentences.test")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type in self.examples:
            examples = self.examples[set_type]
        else:
            feats = None

            if self.speech_data_dir:
                feats = self._read_scp(os.path.join(self.speech_data_dir, set_type + '.scp'))

            for (i, line) in enumerate(lines):
                if line[0][0] == "#":
                    continue
                guid = "%s" % (line[1])
                label = line[0]
                text = self._clean_text(line[3])

                if feats:
                    if guid in feats:
                        examples.append(
                            InputExample(guid=guid, text=text, label=label, feats=feats[guid]))
                else:
                    examples.append(
                        InputExample(guid=guid, text=text, label=label))

            self.examples[set_type] = examples

        return examples

class SwbdProcessor(DaProcessor):
    """Processor for the SwitchBoard data set."""
    def get_labels(self):
        """Gets the list of labels for this data set."""
        labels = ['0', '1', '10', '11', '12', '13', '14', '15', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '5', '6', '7', '8', '9']
        return labels


class MrdaProcessor(DaProcessor):
    """Processor for the MRDA data set."""
    def get_labels(self):
        labels = ['0', '1', '2', '3', '4', '5']
        return labels


class FluentAIProcessor(DataProcessor):
    """Processor for the SwitchBoard data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "train_data.csv")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "valid_data.csv")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_csv(os.path.join(self.data_dir, "test_data.csv")), "test")

    def get_labels(self):
        labels = ['activate|lamp|none', 'activate|lights|bedroom', 'activate|lights|kitchen', 'activate|lights|none', 'activate|lights|washroom', 'activate|music|none', 'bring|juice|none', 'bring|newspaper|none', 'bring|shoes|none', 'bring|socks|none', 'change language|Chinese|none', 'change language|English|none', 'change language|German|none', 'change language|Korean|none', 'change language|none|none', 'deactivate|lamp|none', 'deactivate|lights|bedroom', 'deactivate|lights|kitchen', 'deactivate|lights|none', 'deactivate|lights|washroom', 'deactivate|music|none', 'decrease|heat|bedroom', 'decrease|heat|kitchen', 'decrease|heat|none', 'decrease|heat|washroom', 'decrease|volume|none', 'increase|heat|bedroom', 'increase|heat|kitchen', 'increase|heat|none', 'increase|heat|washroom', 'increase|volume|none']
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        if set_type in self.examples:
            examples = copy.deepcopy(self.examples[set_type])
        else:
            feats = None

            if self.speech_data_dir:
                feats = self._read_scp(os.path.join(self.speech_data_dir, set_type + '.scp'))

            for (i, line) in enumerate(lines):
                if line[0] == '':
                    continue
                guid = "%s-%s" % (set_type, line[0])
                label = '|'.join(line[4:7])
                text = self._clean_text(line[3])

                if feats:
                    examples.append(
                        InputExample(guid=guid, text=text, label=label, feats=feats[guid]))
                else:
                    examples.append(
                        InputExample(guid=guid, text=text, label=label))

            self.examples[set_type] = examples

        return examples


def convert_examples_to_features(examples, label_map, max_seq_length,
                                 tokenizer, transformer,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 sep_token_extra=False,
                                 pad_token_segment_id=0,
                                 pad_on_left=False,
                                 pad_token=0,
                                 mask_padding_with_zero=True,
                                 device='cpu'):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        label = label_map[example.label]

        tokens_a = tokenizer.tokenize(example.text)
        tokens_b = None
        special_tokens_count = 4 if sep_token_extra else 3

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]

        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: {}".format(example.guid))
            logger.info("tokens: {}".format(' '.join(tokens)))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(' '.join(map(str, input_mask))))
            logger.info("segment_ids: {}".format(' '.join(map(str, segment_ids))))
            logger.info("label: {}".format(label))

        embeddings = torch.mean(transformer(
                input_ids=torch.tensor([input_ids], dtype=torch.long).to(device),
                attention_mask=torch.tensor([input_mask], dtype=torch.long).to(device),
                token_type_ids=torch.tensor([segment_ids], dtype=torch.long).to(device)
                )[0], 1).detach()

        features.append(
            {
                'embeddings': embeddings,
                'label': label
            }
        )

    return features


processors = {
    "swbd": SwbdProcessor,
    "mrda": MrdaProcessor,
    "fluentai": FluentAIProcessor
}

data_dirs = {
    "swbd": 'corpora/da/SWBD',
    "mrda": 'corpora/da/MRDA',
    "fluentai": 'corpora/fluent_speech_commands_dataset/data'
}

speech_data_dirs = {
    "swbd": 'espnet/egs/multi_en/slu1/feats/swbd',
    "mrda": 'espnet/egs/multi_en/slu1/feats/mrda',
    "fluentai": 'espnet/egs/multi_en/slu1/feats/fluentai'
}


