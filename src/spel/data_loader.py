"""
The enwiki/conll Dataset reader/provider using torchtext.
The datasets were crated using the scripts from:
    https://github.com/samuelbroscheit/entity_knowledge_in_bert/tree/master/bert_entity/preprocessing
The get_dataset.collate_batch function is influenced by:
    https://raw.githubusercontent.com/samuelbroscheit/entity_knowledge_in_bert/master/bert_entity/data_loader_wiki.py

Please note that the pre-processed fine-tuning data will be automatically downloaded upon instantiation of the data
 readers and the result will be saved under /home/<user_name>/.cache/torch/text/datasets/ (in linux systems)

The expected sizes of the auto-downloaded datasets:
    - Step 1 (general knowledge fine-tuning):
            enwiki-2023-spel-roberta-tokenized-aug-27-2023.tar.gz: 19.1 GBs
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            * You can delete the file above once fine-tuning step 1 is done, and you are moving on to step 2.         *
            * in the cleaning up process, make sure you remove the cached validation set files under .checkpoints     *
            * directory as well                                                                                       *
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    - Step 2 (general knowledge fine-tuning):
            enwiki-2023-spel-roberta-tokenized-aug-27-2023-retokenized.tar.gz: 17.5 GBs
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
            * You can delete the file above once fine-tuning step 2 is done, and you are moving on to step 3.         *
            * in the cleaning up process, make sure you remove the cached validation set files under .checkpoints     *
            * directory as well                                                                                       *
            * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    - Step 3 (domain specific fine-tuning):
            aida-conll-spel-roberta-tokenized-aug-23-2023.tar.gz: 5.1 MBs

No extra preprocessing step will be required, as soon as you start the fine-tuning script for each step,
 the proper fine-tuning dataset will be downloaded and will be served **without** the need for unzipping.
"""
import os
import json
import numpy
from functools import partial
from collections import OrderedDict
from tqdm import tqdm
from typing import Union, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, BatchEncoding

from src.spel.configuration import (get_aida_plus_wikipedia_plus_out_of_domain_vocab, get_aida_train_canonical_redirects,
                                get_aida_vocab, get_ood_vocab, get_checkpoints_dir, get_base_model_name, device)

BERT_MODEL_NAME = get_base_model_name()
MAX_SPAN_ANNOTATION_SIZE = 4


class StaticAccess:
    def __init__(self):
        self.mentions_vocab, self.mentions_itos = None, None
        self.set_vocab_and_itos_to_all()
        self.aida_canonical_redirects = get_aida_train_canonical_redirects()
        self._all_vocab_mask_for_aida = None
        self._all_vocab_mask_for_ood = None

    def set_vocab_and_itos_to_all(self):
        self.mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
        self.mentions_itos = [w[0] for w in sorted(self.mentions_vocab.items(), key=lambda x: x[1])]

    @staticmethod
    def get_aida_vocab_and_itos():
        aida_mentions_vocab = get_aida_vocab()
        aida_mentions_itos = [w[0] for w in sorted(aida_mentions_vocab.items(), key=lambda x: x[1])]
        return aida_mentions_vocab, aida_mentions_itos

    def shrink_vocab_to_aida(self):
        self.mentions_vocab, self.mentions_itos = self.get_aida_vocab_and_itos()

    def get_all_vocab_mask_for_aida(self):
        if self._all_vocab_mask_for_aida is None:
            mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
            mask = torch.ones(len(mentions_vocab)).to(device)
            mask = mask * -10000
            mask[torch.Tensor([mentions_vocab[x] for x in get_aida_vocab()]).long()] = 0
            self._all_vocab_mask_for_aida = mask
        return self._all_vocab_mask_for_aida

    def get_all_vocab_mask_for_ood(self):
        if self._all_vocab_mask_for_ood is None:
            mentions_vocab = get_aida_plus_wikipedia_plus_out_of_domain_vocab()
            mask = torch.ones(len(mentions_vocab)).to(device)
            mask = mask * -10000
            mask[torch.Tensor([mentions_vocab[x] for x in get_ood_vocab()]).long()] = 0
            self._all_vocab_mask_for_ood = mask
        return self._all_vocab_mask_for_ood


dl_sa = StaticAccess()


tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME, cache_dir=get_checkpoints_dir() / "hf")

WIKI_EXTRACTED_FILES = {"train": "train.json", "valid": "valid.json", "test": " test.json"}


def wiki_filter_fn(split, fname_and_stream):
    return WIKI_EXTRACTED_FILES[split] in fname_and_stream[0]


def wiki_data_record_convert(line):
    element = json.loads(line)
    r = {'tokens': element['tokens'], 'mentions': [], 'mention_entity_probs': [], 'mention_probs': []}
    for token, mentions, mention_entity_probs, mention_probs in zip(element['tokens'], element['mentions'],
                                                                    element['mention_entity_probs'],
                                                                    element['mention_probs']):
        if len(mention_probs) < len(mentions):
            mention_probs.extend([1.0 for _ in range(len(mentions) - len(mention_probs))])
        sorted_mentions = sorted(list(zip(mentions, mention_entity_probs, mention_probs)),
                                 key=lambda x: x[1], reverse=True)
        mentions_ = [dl_sa.aida_canonical_redirects[x[0]] if x[0] in dl_sa.aida_canonical_redirects else x[0]
                     for x in sorted_mentions if x[0]]  # ignore mentions that are None
        mention_entity_probs_ = [x[1] for x in sorted_mentions if x[0]]  # ignore prob. for None mentions
        mention_probs_ = [x[2] for x in sorted_mentions if x[0]]  # ignore m_probs for None mentions
        r['mentions'].append(mentions_[:MAX_SPAN_ANNOTATION_SIZE])
        r['mention_probs'].append(mention_probs_[:MAX_SPAN_ANNOTATION_SIZE])
        r['mention_entity_probs'].append(mention_entity_probs_[:MAX_SPAN_ANNOTATION_SIZE])
        if len(mentions_) > MAX_SPAN_ANNOTATION_SIZE:
            r['mention_entity_probs'][-1] = [x / sum(r['mention_entity_probs'][-1])
                                             for x in r['mention_entity_probs'][-1]]
    return r



def aida_select_split(s, file_name_data):
    return file_name_data[1][s]


def aida_data_record_convert(r):
    for x in r:  # making sure each token comes with exactly one annotation
        assert len(x) == 7 or len(x) == 8  # whether it contains the candidates or not
        return {"tokens": [x[0] for x in r], "mentions": [[x[4] if x[4] else "|||O|||"] for x in r],
                "mention_entity_probs": [[1.0] for _ in r], "mention_probs": [[1.0] for _ in r],
                "candidates": [x[7] if x[7] else [] for x in r] if len(x) == 8 else [[] for x in r]}


class DistributableDataset(Dataset):
    """
    Based on the documentations in torch.utils.data.DataLoader, `IterableDataset` does not support custom `sampler`
    Therefore we cannot use the DistributedSampler with the DataLoader to split the data samples.
    This class is a workaround to make the IterableDataset work with the DistributedSampler.
    """
    def __init__(self, dataset, size, world_size, rank):
        self.size = size
        self.data = iter(dataset)
        self.world_size = world_size
        self.rank = rank
        self.initial_fetch = True

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # Since we don't have a means of accessing the data by indices, we try skipping the indices that we believe
        #  belong to other processes
        skip_size = self.rank if self.initial_fetch else self.world_size - 1
        self.initial_fetch = False
        for _ in range(skip_size):
            next(self.data)
        return next(self.data)


def convert_is_in_mention_to_bioes(is_in_mention):
    # B = 0, I = 1, O = 2, E = 3, S = 4
    bioes = []
    for iim, current in enumerate(is_in_mention):
        before = is_in_mention[iim - 1] if iim > 0 else 0
        after = is_in_mention[iim + 1] if iim < len(is_in_mention) - 1 else 0
        bioes.append(
            2 if not current else (4 if not before and not after else (0 if not before else (3 if not after else 1))))
    return bioes


def create_output_with_negative_examples(batch_entity_ids, batch_entity_probs, batch_size, maxlen, label_vocab_size,
                                         label_size, labels_with_high_model_score=None):
    all_entity_ids = OrderedDict()
    for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
    ):
        for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)
        ):
            for eid in token_entity_ids:
                if eid not in all_entity_ids:
                    all_entity_ids[eid] = len(all_entity_ids)
    # #####################################################
    shared_label_ids = list(all_entity_ids.keys())

    if len(shared_label_ids) < label_size and labels_with_high_model_score is not None:
        negative_examples = set(labels_with_high_model_score)
        negative_examples.difference_update(shared_label_ids)
        shared_label_ids += list(negative_examples)

    if len(shared_label_ids) < label_size:
        negative_samples = set(numpy.random.choice(label_vocab_size, label_size, replace=False))
        negative_samples.difference_update(shared_label_ids)
        shared_label_ids += list(negative_samples)

    shared_label_ids = shared_label_ids[: label_size]

    all_batch_entity_ids, batch_shared_label_ids = all_entity_ids, shared_label_ids
    if label_size > 0:
        label_probs = torch.zeros(batch_size, maxlen, len(batch_shared_label_ids))
    else:
        label_probs = torch.zeros(batch_size, maxlen, label_vocab_size)
    # loop through the batch x tokens x (label_ids, label_probs)
    for batch_offset, (batch_item_token_item_entity_ids, batch_item_token_entity_probs) in enumerate(
            zip(batch_entity_ids, batch_entity_probs)
    ):
        # loop through tokens x (label_ids, label_probs)
        for tok_id, (token_entity_ids, token_entity_probs) in enumerate(
                zip(batch_item_token_item_entity_ids, batch_item_token_entity_probs)):
            if label_size is None:
                label_probs[batch_offset][tok_id][torch.LongTensor(token_entity_ids)] = torch.Tensor(
                    batch_item_token_item_entity_ids)
            else:
                label_probs[batch_offset][tok_id][
                    torch.LongTensor(list(map(all_batch_entity_ids.__getitem__, token_entity_ids)))
                ] = torch.Tensor(token_entity_probs)

    label_ids = torch.LongTensor(batch_shared_label_ids)
    return BatchEncoding({
        "ids": label_ids,  # of size label_size
        "probs": label_probs,  # of size input_batch_size x input_max_len x label_size
        "dictionary": {v: k for k, v in all_batch_entity_ids.items()}  # contains all original ids for mentions in batch
    })