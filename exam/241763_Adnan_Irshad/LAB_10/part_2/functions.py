import numpy as np
import torch
from seqeval.metrics import f1_score
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup


def train(model, train_dataset, optimizer, n_epochs, gradient_accumulation_steps, warmup_steps, max_grad_norm, device,
          train_batch_size, pad_token_label_id):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_dataloader) // gradient_accumulation_steps * n_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    global_step = 0
    tr_loss = 0.0
    model.zero_grad()

    train_iterator = trange(int(n_epochs), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(b.to(device) for b in batch)

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'intent_label_ids': batch[4],
                      'slot_label_ids': batch[3],
                      'pad_token_label_id': pad_token_label_id}

            outputs = model(**inputs)
            loss = outputs[0]

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

    return global_step, tr_loss / global_step


# evaluation method
def evaluate_test(model, dataset, slot_label_lst, batch_size, device, pad_token_label_id):
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=batch_size)

    loss = 0.0
    nb_steps = 0
    intent_preds = None
    slot_preds = None
    out_intent_label_ids = None
    out_slot_labels_ids = None

    model.eval()

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(b.to(device) for b in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2],
                      'intent_label_ids': batch[4],
                      'slot_label_ids': batch[3],
                      'pad_token_label_id': pad_token_label_id}

            outputs = model(**inputs)
            temp_loss, (intent_logits, slot_logits) = outputs[:2]

            loss += temp_loss.mean().item()

        nb_steps += 1

        # intent prediction
        if intent_preds is None:
            intent_preds = intent_logits.detach().cpu().numpy()
            out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
        else:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
            out_intent_label_ids = np.append(out_intent_label_ids,
                                             inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

        # slot prediction/
        if slot_preds is None:
            slot_preds = slot_logits.detach().cpu().numpy()
            out_slot_labels_ids = inputs["slot_label_ids"].detach().cpu().numpy()
        else:
            slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
            out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_label_ids"].detach().cpu().numpy(),
                                            axis=0)

    eval_loss = loss / nb_steps
    results = {
        "loss": eval_loss
    }

    # Intent result
    intent_preds = np.argmax(intent_preds, axis=1)
    out_intent_label_ids = out_intent_label_ids.reshape(-1)
    intent_result = (intent_preds == out_intent_label_ids).mean()

    # Slot result
    slot_preds = np.argmax(slot_preds, axis=2)

    # Remove ignored index (special tokens)
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]
    slot_label_list = [[] for _ in range(slot_preds.shape[0])]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if out_slot_labels_ids[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_lst[slot_preds[i][j]])
                slot_label_list[i].append(slot_label_lst[out_slot_labels_ids[i][j]])

    slot_result = f1_score(slot_label_list, slot_preds_list)

    results["intent_result"] = intent_result
    results["slot_result"] = slot_result

    return results
