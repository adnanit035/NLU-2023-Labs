# Importing libraries
from pprint import pprint

# import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    # 1. Load data, split into train and test set
    tmp_train_raw = load_data(os.path.join(dataset_dir, 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join(dataset_dir, 'ATIS', 'test.json'))
    print('Train samples:', len(tmp_train_raw))
    print('Test samples:', len(test_raw))

    # 2. Create a dev set
    # First we get the 10% of dataset,
    # then compute the percentage of these examples on the training set which is around 11%
    portion = round(((len(tmp_train_raw) + len(test_raw)) * 0.10) / (len(tmp_train_raw)), 2)
    intents = [x['intent'] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    # 3. Split the training set into train and dev set with stratification (same distribution of intents)
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

    # 4. Convert words to numbers (word2id)
    words = sum([x['utterance'].split() for x in train_raw], [])  # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not want unk labels, # however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    # 5. Create the lang object that will be used to convert words to numbers
    lang = Lang(words, intents, slots, cutoff=0)

    # 6. Create the datasets and dataloaders for train, dev and test set
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiation
    train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=64, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)

    # 7. Training set up
    hid_size = 200
    emb_size = 300

    lr = 0.0001  # learning rate
    clip = 5  # Clip the gradient

    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    runs = 5
    slot_f1s, intent_acc = [], []
    best_model = None

    model_name = "Lab-10-Ex-1-IAS_Adam.pt"
    print("Training model: {}".format(model_name))
    print("Hyperparameters: hid_size={}, emb_size={}, lr={}, clip={}, Optimizer={}".format(
        hid_size, emb_size, lr, clip, "Adam"))

    for x in tqdm(range(0, runs)):
        print("Run: {}".format(x + 1))
        model = ModelIAS(hid_size, out_slot, out_int, emb_size, vocab_len, pad_index=PAD_TOKEN).to(device)
        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
        criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token in intents

        # 8. Train a neural network model for intent detection and slot filling (with early stopping)
        n_epochs = 200
        patience = 3  # Early stopping patience
        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0

        for i in tqdm(range(1, n_epochs)):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            if i % 5 == 0:
                sampled_epochs.append(i)
                losses_train.append(np.asarray(loss).mean())
                results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model,
                                                              lang)
                losses_dev.append(np.asarray(loss_dev).mean())
                f1 = results_dev['total']['f']

                if f1 > best_f1:
                    best_f1 = f1
                    patience = 3
                    best_model = model
                else:
                    patience -= 1
                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        # 9. Evaluate the model on the test set
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])

        slot_f1s.append(results_test['total']['f'])
        intent_acc.append(intent_test['accuracy'])
        print()
    
    print('Average Slot F1: {} +/- {}'.format(np.mean(slot_f1s), np.std(slot_f1s)))
    print('Average Intent Accuracy: {} +/- {}'.format(np.mean(intent_acc), np.std(intent_acc)))

    # 10. Plot of the train and valid losses during training (one plot for the train and one for the dev)
    # plt.figure(num=3, figsize=(8, 5)).patch.set_facecolor('white')
    # plt.title('Train and Dev Losses')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.plot(sampled_epochs, losses_train, label='Train loss')
    # plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    # plt.legend()
    # plt.show()

    # 11. Save the model
    torch.save(best_model.state_dict(), os.path.join(model_dir, model_name))
    print('Model saved')
