from seqeval.metrics import f1_score
from transformers import BertModel, BertPreTrainedModel
from torch import nn
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import numpy as np


class BertForIntentClassificationAndSlotFilling(BertPreTrainedModel):
    def __init__(self, config, intent_label_lst, slot_label_lst, dropout_rate=0.1, ignore_index=0, slot_loss_coef=1.0):
        super(BertForIntentClassificationAndSlotFilling, self).__init__(config)
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.ignore_index = ignore_index
        self.slot_loss_coef = slot_loss_coef

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_rate)
        self.intent_classifier = nn.Linear(config.hidden_size, self.num_intent_labels)
        self.slot_classifier = nn.Linear(config.hidden_size, self.num_slot_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_label_ids, pad_token_label_id):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        if intent_label_ids is not None and slot_label_ids is not None:
            # Intent Classification Loss
            intent_loss_fct = nn.CrossEntropyLoss()
            intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

            # Slot Filling Loss
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token_label_id)
            if attention_mask is not None:
                slot_active_loss = attention_mask.view(-1) == 1
                slot_active_logits = slot_logits.view(-1, self.num_slot_labels)[slot_active_loss]
                slot_active_labels = slot_label_ids.view(-1)[slot_active_loss]
                slot_loss = slot_loss_fct(slot_active_logits, slot_active_labels)
            else:
                slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_label_ids.view(-1))

            total_loss += slot_loss * self.slot_loss_coef

        outputs = ((intent_logits, slot_logits),) + outputs[2:]
        outputs = (total_loss,) + outputs

        return outputs
