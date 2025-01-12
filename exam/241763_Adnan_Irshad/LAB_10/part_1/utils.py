import json
import os
from collections import Counter

import torch
from torch.utils import data

device = 'cuda:0'  # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side
PAD_TOKEN = 0


class Lang:
    """Class to convert words, intents and slots to ids and vice-versa
    @param words: list of words
    @param intents: list of intents
    @param slots: list of slots
    @param cutoff: minimum number of occurrences of a word to be included in the vocabulary
    """

    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    @staticmethod
    def w2id(elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    @staticmethod
    def lab2id(elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)

        return vocab


class IntentsAndSlots(data.Dataset):
    """Customize the Dataset class to load the data and convert it to ids using the Lang class
    @param dataset: list of dictionaries with the keys 'utterance', 'slots' and 'intent'
    @param lang: Lang object
    @param unk: unknown token to be used when a word is not in the vocabulary
    """
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    # Auxiliary methods
    def mapping_lab(self, data_, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data_]

    def mapping_seq(self, data_, mapper):  # Map sequences to number
        res = []
        for seq in data_:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


def load_data(path):
    """
    Load the data from a json file
        input: path/to/data
        output: json
    """
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# Collate_fn function which is passed to the DataLoader class to create batches
def collate_fn(data_):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)

        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix

        # print(padded_seqs)
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph

        return padded_seqs, lengths

    # Sort data by seq lengths
    data_.sort(key=lambda x: len(x['utterance']), reverse=True)

    new_item = {}
    for key in data_[0].keys():
        new_item[key] = [d[key] for d in data_]

    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    src_utt = src_utt.to(device)  # We load the Tensor on our selected device
    y_slots = y_slots.to(device)
    intent = intent.to(device)
    y_lengths = torch.LongTensor(y_lengths).to(device)

    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths

    return new_item
