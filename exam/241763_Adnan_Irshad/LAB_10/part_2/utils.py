import copy
import json
import torch
from torch.utils.data import TensorDataset


class InputSample:
    def __init__(self, id_, words_, intent_, slot_labels_):
        self.id = id_
        self.words = words_
        self.intent = intent_
        self.slot_labels = slot_labels_

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures:
    def __init__(self, input_id, attention_mask, token_type_id, intent_label_id, slot_labels_ids):
        self.input_id = input_id
        self.attention_mask = attention_mask
        self.token_type_id = token_type_id
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor:
    def __init__(self, train_raw_, dev_raw_, test_raw_, intent_labels_, slot_labels_):
        self.train_raw = train_raw_
        self.dev_raw = dev_raw_
        self.test_raw = test_raw_

        self.intent_labels = intent_labels_
        self.slot_labels = slot_labels_

    def create_birt_format(self, mode):
        data_ = []

        if mode == "train":
            raw = self.train_raw
        elif mode == "dev":
            raw = self.dev_raw
        elif mode == "test":
            raw = self.test_raw
        else:
            raise ValueError("The mode should be in train, dev or test")

        for i_, input_dict in enumerate(raw):
            input_id = "%s-%d" % (mode, i_)

            words_ = input_dict['utterance'].split()
            intent_ = input_dict['intent']
            intent_label_ = self.intent_labels.index(
                intent_) if intent_ in self.intent_labels else self.intent_labels.index('UNK')
            slots_ = input_dict['slots'].split()
            slot_labels = [self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index('UNK') for
                           s in slots_]

            assert len(words_) == len(slot_labels)  # Sanity check

            data_.append(InputSample(id_=input_id, words_=words_, intent_=intent_label_, slot_labels_=slot_labels))

        return data_


# Function to load the data from the json file
def load_data(path):
    """
        input: path/to/data
        output: json
    """
    with open(path) as f:
        dataset = json.loads(f.read())

    return dataset


def input2features(input_sample, max_seq_len, tokenizer, pad_token_label_id=100, cls_token_segment_id=0,
                   pad_token_segment_id=0, sequence_a_segment_id=0, mask_padding_with_zero=True):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []

    for i, sample in enumerate(input_sample):
        tokens = []
        slot_labels_ids = []

        for word, slot_label in zip(sample.words, sample.slot_labels):
            word_tokens = tokenizer.tokenize(word)

            # handle bad encoded word
            if len(word_tokens) == 0:
                word_tokens = [unk_token]

            tokens.extend(word_tokens)

            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        # special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[: (max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[: (max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # 0 padding up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids += ([pad_token_id] * padding_length)
        attention_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        slot_labels_ids += ([pad_token_label_id] * padding_length)
        token_type_ids += ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with input length {} vs {}".format(len(attention_mask),
                                                                                             max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                             max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with input length {} vs {}".format(len(slot_labels_ids),
                                                                                              max_seq_len)

        intent_label_id = int(sample.intent)

        features.append(
            InputFeatures(
                input_id=input_ids,
                attention_mask=attention_mask,
                token_type_id=token_type_ids,
                slot_labels_ids=slot_labels_ids,
                intent_label_id=intent_label_id
            )
        )

    return features


def convert_to_tensor(features):
    all_input_ids = torch.tensor([f.input_id for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_labels_ids,
                            all_intent_label_ids)

    return dataset
