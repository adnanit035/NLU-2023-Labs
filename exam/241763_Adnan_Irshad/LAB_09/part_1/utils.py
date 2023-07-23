import torch
import torch.utils.data as data

device = 'cuda:0'


def read_file(path, eos_token="<eos>"):
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line + eos_token)
    return output


def get_vocab(corpus, special_tokens=None):
    if special_tokens is None:
        special_tokens = []
    output = {}
    i = 0
    for st in special_tokens:
        output[st] = i
        i += 1
    for sentence in corpus:
        for w in sentence.split():
            if w not in output:
                output[w] = i
                i += 1
    return output


class PennTreeBank(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, corpus, lang):
        self.source = []
        self.target = []

        for sentence in corpus:
            self.source.append(sentence.split()[0:-1])  # We get from the first token till the second-last token
            self.target.append(sentence.split()[1:])  # We get from the second token till the last token
            # See example in section 6.2

        self.source_ids = self.mapping_seq(self.source, lang)
        self.target_ids = self.mapping_seq(self.target, lang)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        src = torch.LongTensor(self.source_ids[idx])
        trg = torch.LongTensor(self.target_ids[idx])
        sample = {'source': src, 'target': trg}
        return sample

    # Auxiliary methods
    @staticmethod
    def mapping_seq(data_, lang):  # Map sequences to number
        res = []
        for seq in data_:
            tmp_seq = []
            for x in seq:
                if x in lang.word2id:
                    tmp_seq.append(lang.word2id[x])
                else:
                    print('OOV found!')
                    print('You have to deal with that')
                    break
            res.append(tmp_seq)
        return res


class Lang:
    """Simple vocabulary wrapper."""

    def __init__(self, corpus, special_tokens=None):
        """
        :param corpus:
        :param special_tokens:
        """
        if special_tokens is None:
            special_tokens = []
        self.word2id = self.get_vocab(corpus, special_tokens)
        self.id2word = {v: k for k, v in self.word2id.items()}

    @staticmethod
    def get_vocab(corpus, special_tokens=None):
        """
        description: function to get the vocabulary of the corpus
        :param corpus:
        :param special_tokens:
        :return: dict of word to id
        """
        if special_tokens is None:
            special_tokens = []
        output = {}
        i = 0
        for st in special_tokens:
            output[st] = i
            i += 1
        for sentence in corpus:
            for w in sentence.split():
                if w not in output:
                    output[w] = i
                    i += 1
        return output


def collate_fn(data_, pad_token):
    def merge(sequences):
        """
        merge from batch * sent_len to batch * max_len
        """
        lengths_ = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths_) == 0 else max(lengths_)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(pad_token)
        for i, seq in enumerate(sequences):
            end = lengths_[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths_

    # Sort data by seq lengths in descending order
    data_.sort(key=lambda x: len(x["source"]), reverse=True)
    new_item = {}
    for key in data_[0].keys():
        new_item[key] = [d[key] for d in data_]

    source, _ = merge(new_item["source"])
    target, lengths = merge(new_item["target"])

    new_item["source"] = source.to(device)
    new_item["target"] = target.to(device)
    new_item["number_tokens"] = sum(lengths)
    return new_item