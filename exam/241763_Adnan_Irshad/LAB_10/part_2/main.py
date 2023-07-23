import os
from collections import Counter
from pprint import pprint
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import BertConfig
from transformers import BertTokenizer
from torch.optim import Adam

from functions import *
from model import *
from utils import *

if __name__ == "__main__":
    # 0. Set the dataset path
    dataset_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(os.path.dirname(dataset_dir), "dataset")

    # To save the best model
    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(model_dir, "bin")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # 1. Load the data
    tmp_train_raw = load_data(os.path.join(dataset_dir, 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join(dataset_dir, 'ATIS', 'test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    # 2. Create a train, test and dev set
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10) / (len(tmp_train_raw)), 2)
    intents = [x['intent'] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occur once only, we put them in training
            X.append(tmp_train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(tmp_train_raw[id_y])

    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(
        X, Y, test_size=portion, random_state=42, shuffle=True, stratify=Y)
    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]
    # Intent distribution
    print('Train:')
    pprint({k: round(v / len(y_train), 3) * 100 for k, v in sorted(Counter(y_train).items())})
    print('Dev:'),
    pprint({k: round(v / len(y_dev), 3) * 100 for k, v in sorted(Counter(y_dev).items())})
    print('Test:')
    pprint({k: round(v / len(y_test), 3) * 100 for k, v in sorted(Counter(y_test).items())})
    print('=' * 89)
    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    # 3. Prepare data for BERT model
    words = sum([x['utterance'].split() for x in train_raw], [])  # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not want unk labels, # however this depends on the research purpose
    slots_ = set(sum([line['slots'].split() for line in corpus], []))
    intents_ = set([line['intent'] for line in corpus])

    slots = list(slots_)
    intents = list(intents_)
    intents.append("UNK")
    slots.append("UNK")

    # 4. Create a data processor that will create the BERT input data
    data_processor = DataProcessor(X_train, X_dev, test_raw, intents, slots)
    train_data = data_processor.create_birt_format("train")
    dev_data = data_processor.create_birt_format("dev")
    test_data = data_processor.create_birt_format("test")

    pad_token_label_id = 0  # ignore index 0: "O" token
    max_seq_len = 50 # maximum sequence length for BERT as suggested by the paper

    # Global variables
    device = 'cuda:0'  # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Used to report errors on CUDA side

    # BERT tokenizer for tokenizing the text
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 5. Convert data to features
    train_features = input2features(train_data, max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)
    test_features = input2features(test_data, max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)
    dev_features = input2features(dev_data, max_seq_len, tokenizer, pad_token_label_id=pad_token_label_id)

    # 6. Convert features to tensors
    train_dataset = convert_to_tensor(train_features)
    test_dataset = convert_to_tensor(test_features)
    dev_dataset = convert_to_tensor(dev_features)

    # Hyperparameters for training
    train_batch_size = 128  # batch size for training as suggested by the paper
    eval_batch_size = 64  # batch size for evaluation
    n_epochs = 40.0  # Number of epochs for training
    gradient_accumulation_steps = 3  # Accumulate gradients on several steps.
    lr = 0.0001  # 5e-5 equivalent to 0.00005
    adam_epsilon = 1e-8  # Epsilon for Adam optimizer to avoid division by zero error as per official implementation
    warmup_steps = 0  # Linear warmup over warmup_steps. It is used to avoid over-fitting
    max_grad_norm = 1.0  # Maximum norm for the gradients. it is used for exploding gradients problem

    model_name = "Lab-10-Ex-2-Intent-Classification-and-Slot-Filling.pt"

    no_decay = ['bias', 'LayerNorm.weight']  # no decay for bias and LayerNorm.weight

    train_loss_set = []
    intent_res_set = []
    slot_res_set = []
    best_model = None

    print("Training Model: ", model_name)
    print("Training Parameters: ")
    print("Train Batch Size: {}, Eval Batch Size: {}, no. of epochs: {}".format(
        train_batch_size, eval_batch_size, n_epochs))
    print("Learning Rate: {}, Adam Epsilon: {}, Warmup Steps: {}, Max Grad Norm: {}".format(
        lr, adam_epsilon, warmup_steps, max_grad_norm))
    print("Optimizer: AdamW, Loss Function: Cross Entropy Loss")

    runs = 3
    for run in tqdm(range(0, runs)):
        print("\nRun: ", run + 1)
        # 7. Initiate Model
        # Load BERT pretrained model
        config = BertConfig.from_pretrained(
            'bert-base-uncased', finetuning_task='intent-classification-and-slot-filling')
        model = BertForIntentClassificationAndSlotFilling.from_pretrained(
            'bert-base-uncased', config=config, intent_label_lst=intents, slot_label_lst=slots)
        model.to(device)

        # Prepare optimizer and schedule (linear warmup and decay) for training
        # The BERT paper recommends: AdamW with weight decay fix
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if
                        not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if
                        any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        # AdamW optimizer with weight decay fix
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=adam_epsilon)

        # 9. Train and evaluate
        _, tr_loss = train(model, train_dataset, optimizer, n_epochs, gradient_accumulation_steps, warmup_steps,
                           max_grad_norm, device, train_batch_size, pad_token_label_id)

        print("Train loss: {}".format(tr_loss))

        # 10. Evaluate on dev dataset
        results = evaluate_test(model, test_dataset, slots, eval_batch_size, device, pad_token_label_id)
        print("Intent Accuracy: {}".format(results["intent_result"]))
        print("Slot F1-Score: {}".format(results["slot_result"]))

        train_loss_set.append(tr_loss)
        intent_res_set.append(results["intent_result"])
        slot_res_set.append(results["slot_result"])

        # set to best model if it is the first run or if the current model is better than the best model
        if slot_res_set.__len__() == 1 and intent_res_set.__len__() == 1:
            best_model = model
        elif slot_res_set[-1] > results["slot_result"] or intent_res_set[-1] > results["intent_result"]:
            best_model = model

    print("\n\nTraining complete!")
    print("\nAverage Train loss: {}".format(sum(train_loss_set) / len(train_loss_set)))
    print("Average Intent Accuracy: {}".format(sum(intent_res_set) / len(intent_res_set)))
    print("Average Slot F1-Score: {}".format(sum(slot_res_set) / len(slot_res_set)))

    # 10. Save the best model
    torch.save(best_model.state_dict(), os.path.join(model_dir, model_name))
    print("\nBest model saved to ", os.path.join(model_dir, model_name))
