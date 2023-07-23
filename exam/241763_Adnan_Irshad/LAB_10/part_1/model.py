import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0):
        super(ModelIAS, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # LSTM layer with bi-directionality
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer, bidirectional=True)
        # Output Linear layers for slot filling and intent classification
        # hid_size * 2 because of bi-directionality (concatenation of the forward and backward hidden states)
        self.slot_out = nn.Linear(hid_size * 2, out_slot)  # bidirectional = True -> hid_size * 2
        self.intent_out = nn.Linear(hid_size * 2, out_int)  # bidirectional = True -> hid_size * 2

        # Dropout layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = utt_emb.permute(1, 0, 2)  # we need seq len first -> seq_len X batch_size X emb_size

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)

        # Apply dropout to the output of the LSTM layer
        # Since LSTM layer is bidirectional, we need to apply dropout to both directions (forward and backward)
        # For this reason we need to split the tensor in two parts and apply dropout to each part
        #  OR we need to apply dropout to the concatenation of the two parts: forward and backward hidden states
        # Concatenate the final forward (last_hidden[-2,:,:]) and backward (last_hidden[-1,:,:]) hidden layers
        # and apply dropout
        last_hidden = self.dropout(torch.cat((last_hidden[-2, :, :], last_hidden[-1, :, :]), dim=1))
        utt_encoded = self.dropout(utt_encoded)

        # Compute slot logits
        slots = self.slot_out(utt_encoded)
        # Compute intent logits
        intent = self.intent_out(last_hidden)
        # Slot size: seq_len, batch size, classes
        slots = slots.permute(1, 2, 0)  # We need this for computing the loss
        # Slot size: batch_size, classes, seq_len
        return slots, intent
