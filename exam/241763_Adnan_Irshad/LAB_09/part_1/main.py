from functools import partial

from torch.utils.data import DataLoader

from functions import *
from utils import *
from model import *

from tqdm import tqdm
import copy
import math
import os
from torch import optim


if __name__ == "__main__":
    # 0. Set the dataset path
    dataset_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(os.path.dirname(dataset_dir), "dataset")

    # to save the best model
    model_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(model_dir, "bin")

    # 1. Read the data
    train_raw = read_file(os.path.join(dataset_dir, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(dataset_dir, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(dataset_dir, "ptb.test.txt"))

    # 2. Create the vocabulary for the training set
    vocab = get_vocab(train_raw, ["<pad>", "<eos>"])

    # 3. Create the language object for the training set
    lang = Lang(train_raw, ["<pad>", "<eos>"])

    # 4. Create the dataset objects
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    # 5. Create the data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=256, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(
        dev_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(
        test_dataset, batch_size=1024, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))

    # 6. Create the model
    hid_size = 200
    emb_size = 300

    ################################################################################################################
    # 1. Replace RNN with LSTM (output the PPL)
    ################################################################################################################
    model_name = "Lab-9-Part-1_LSTM_SGD.pt"
    lr = 1.0  # By experiments, I found this learning rate to be the best for the LSTM with dropout and the SGD
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))

    print('\nTraining Model: {}'.format(model_name))
    print('\nHyperparameters:')
    print('Learning Rate: ', lr)
    print('Clip: ', clip)
    print('Hidden Size: ', hid_size)
    print('Embedding Size: ', emb_size)
    print('Number of Epochs: ', n_epochs)
    print('Patience: ', patience)
    print(f'Train Batch: {256}, Test Batch: {1024}, Val Batch: {1024}')
    print('Optimizer: ', 'SGD')
    print('Loss Function: ', 'CrossEntropyLoss')
    print()

    device = 'cuda:0'
    vocab_len = len(lang.word2id)

    experiments_ppl = []
    no_of_runs_for_each_experiment = 3
    best_lstm_model = None
    for i in range(no_of_runs_for_each_experiment):
        best_model = None
        print("Run: ", i+1)
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 5
                else:
                    patience -= 1

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print("Test PPL: %f" % final_ppl)
        experiments_ppl.append(final_ppl)
        if i == 0:
            best_lstm_model = best_model
        else:
            if final_ppl < experiments_ppl[i-1]:
                best_lstm_model = best_model

    print("\nBest LSTM Model PPL: ", min(experiments_ppl))
    print("Average LSTM Model PPL: ", sum(experiments_ppl)/len(experiments_ppl))

    # 7. Save the model
    torch.save(best_lstm_model.state_dict(), os.path.join(model_dir, model_name))
    print()
    print("-" * 100)
    ################################################################################################################
    # 2. Add dropout to the LSTM (output the PPL) with SGD optimizer
    #   - one on embedding layer
    #   - one on the output
    ################################################################################################################
    model_name = "Lab-9-Part-1_LSTM_Dropout_SGD.pt"
    lr = 1.0  # By experiments, I found this learning rate to be the best for the LSTM with dropout and the SGD
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))

    print('\nTraining Model: {}'.format(model_name))
    print('\nHyperparameters:')
    print('Learning Rate: ', lr)
    print('Clip: ', clip)
    print('Hidden Size: ', hid_size)
    print('Embedding Size: ', emb_size)
    print('Number of Epochs: ', n_epochs)
    print('Patience: ', patience)
    print(f'Train Batch: {256}, Test Batch: {1024}, Val Batch: {1024}')
    print('Optimizer: ', 'SGD')
    print('Loss Function: ', 'CrossEntropyLoss')
    print('Model: ', model_name)
    print()

    device = 'cuda:0'
    vocab_len = len(lang.word2id)

    experiments_ppl = []
    no_of_runs_for_each_experiment = 3
    best_lstm_dropout_model = None

    for i in range(no_of_runs_for_each_experiment):
        best_model = None
        print("Run: ", i+1)
        model = LM_LSTM_Dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 5
                else:
                    patience -= 1

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print("Test PPL: %f" % final_ppl)
        experiments_ppl.append(final_ppl)
        if i == 0:
            best_lstm_model = best_model
        else:
            if final_ppl < experiments_ppl[i - 1]:
                best_lstm_model = best_model

    print("\nBest LSTM Model PPL: ", min(experiments_ppl))
    print("Average LSTM Model PPL: ", sum(experiments_ppl) / len(experiments_ppl))

    # 7. Save the model
    torch.save(best_lstm_model.state_dict(), os.path.join(model_dir, model_name))
    print()
    print("-" * 100)
    ################################################################################################################
    # 3. Replace the SGD optimizer with AdamW & without dropout layers (output the PPL)
    ################################################################################################################
    model_name = "Lab-9-Part-1_LSTM_AdamW.pt"
    lr = .0001  # By experiments, I found this learning rate to be the best for the LSTM with dropout and the Adam
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))

    print('\nTraining Model: {}'.format(model_name))
    print('\nHyperparameters:')
    print('Learning Rate: ', lr)
    print('Clip: ', clip)
    print('Hidden Size: ', hid_size)
    print('Embedding Size: ', emb_size)
    print('Number of Epochs: ', n_epochs)
    print('Patience: ', patience)
    print(f'Train Batch: {256}, Test Batch: {1024}, Val Batch: {1024}')
    print('Optimizer: ', 'AdamW')
    print('Loss Function: ', 'CrossEntropyLoss')
    print('Model: ', model_name)
    print()

    device = 'cuda:0'
    vocab_len = len(lang.word2id)

    experiments_ppl = []
    no_of_runs_for_each_experiment = 3
    best_lstm_dropout_model = None

    for i in range(no_of_runs_for_each_experiment):
        best_model = None
        print("Run: ", i + 1)
        model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 5
                else:
                    patience -= 1

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print("Test PPL: %f" % final_ppl)
        experiments_ppl.append(final_ppl)
        if i == 0:
            best_lstm_model = best_model
        else:
            if final_ppl < experiments_ppl[i - 1]:
                best_lstm_model = best_model

    print("\nBest LSTM Model PPL: ", min(experiments_ppl))
    print("Average LSTM Model PPL: ", sum(experiments_ppl) / len(experiments_ppl))

    # 7. Save the model
    torch.save(best_lstm_model.state_dict(), os.path.join(model_dir, model_name))
    print()
    print("-" * 100)
    ################################################################################################################
    # 4. Replace the SGD optimizer with AdamW & with dropout layers (output the PPL)
    ################################################################################################################
    model_name = "Lab-9-Part-1_LSTM_Dropout_AdamW.pt"
    lr = .0001  # By experiments, I found this learning rate to be the best for the LSTM with dropout and the Adam
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))

    print('\nTraining Model-4 with These Parameters: ')
    print('Learning Rate: ', lr)
    print('Clip: ', clip)
    print('Hidden Size: ', hid_size)
    print('Embedding Size: ', emb_size)
    print('Number of Epochs: ', n_epochs)
    print('Patience: ', patience)
    print(f'Train Batch: {256}, Test Batch: {1024}, Val Batch: {1024}')
    print('Optimizer: ', 'AdamW')
    print('Loss Function: ', 'CrossEntropyLoss')
    print('Model: ', model_name)
    print()

    device = 'cuda:0'
    vocab_len = len(lang.word2id)

    experiments_ppl = []
    no_of_runs_for_each_experiment = 3
    best_lstm_dropout_model = None

    for i in range(no_of_runs_for_each_experiment):
        best_model = None
        print("Run: ", i + 1)
        model = LM_LSTM_Dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, clip)

            if epoch % 1 == 0:
                sampled_epochs.append(epoch)
                losses_train.append(np.asarray(loss).mean())
                ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
                losses_dev.append(np.asarray(loss_dev).mean())
                pbar.set_description("PPL: %f" % ppl_dev)
                if ppl_dev < best_ppl:  # the lower, the better
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model).to('cpu')
                    patience = 5
                else:
                    patience -= 1

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print("Test PPL: %f" % final_ppl)
        experiments_ppl.append(final_ppl)
        if i == 0:
            best_lstm_model = best_model
        else:
            if final_ppl < experiments_ppl[i - 1]:
                best_lstm_model = best_model

    print("\nBest LSTM Model PPL: ", min(experiments_ppl))
    print("Average LSTM Model PPL: ", sum(experiments_ppl) / len(experiments_ppl))

    # 7. Save the model
    torch.save(best_lstm_model.state_dict(), os.path.join(model_dir, model_name))
    print()
    print("-" * 100)
