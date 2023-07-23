import copy
import os
from functools import partial

from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # learning rate scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import *
from utils import *

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
    # 4. Combined LSTM: Variational Dropout + NT-ASGD + Weight Tying
    ################################################################################################################
    model_name = "Lab-9-Part-2_LSTM_AllToGather.pt"
    lr = 1.0  # By experiments, I found this learning rate to be the best for the LSTM with dropout and the SGD
    clip = 5  # Clip the gradient
    n_epochs = 100
    patience = 5
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    pbar = tqdm(range(1, n_epochs))

    print('\nTraining Model with Combined Variational Dropout + NT-ASGD + Weight Tying: {}'.format(model_name))
    print('\nHyperparameters:')
    print('Learning Rate: ', lr)
    print('Clip: ', clip)
    print('Hidden Size: ', hid_size)
    print('Embedding Size: ', emb_size)
    print('Number of Epochs: ', n_epochs)
    print('Patience: ', patience)
    print(f'Train Batch: {256}, Test Batch: {1024}, Val Batch: {1024}')
    print('Optimizer: ', 'ASGD')
    print('Loss Function: ', 'CrossEntropyLoss')
    print()

    device = 'cuda:0'
    vocab_len = len(lang.word2id)

    experiments_ppl = []
    no_of_runs_for_each_experiment = 3
    best_model = None
    best_lstm_model = None
    for i in range(no_of_runs_for_each_experiment):
        print("Run: ", i + 1)
        model = LM_LSTM_Dropout(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(device)
        model.apply(init_weights)

        # Changing optimizer to ASGD, which is the basis for NT-ASGD (Non-monotonically Triggered AvSGD)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        # make copy of model
        average_model = copy.deepcopy(model)
        check_interval = 5  # Set check interval for NT-AvSGD
        non_monotonic_trigger = 2  # Set trigger for NT-AvSGD
        last_losses = []  # Store losses of last check_interval epochs

        criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
        criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

        # LR scheduler that reduces LR when a metric has stopped improving (patience=3 means reducing LR after 3 epochs)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience, factor=0.1, verbose=True)

        # If the PPL is too high try to change the learning rate
        for epoch in pbar:
            loss = train_loop(train_loader, optimizer, criterion_train, model, average_model, clip)

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

                # NT-AvSGD: logic for non-monotonic triggering
                last_losses.append(loss_dev)
                if len(last_losses) > check_interval:
                    last_losses.pop(0)
                    # Check if the last check_interval losses are not monotonically decreasing (trigger)
                    # and switch to average model
                    if sum(x > y for x, y in zip(last_losses[1:], last_losses[:-1])) >= non_monotonic_trigger:
                        model.load_state_dict(average_model.state_dict())  # Switch to average model

                if patience <= 0:  # Early stopping with patience
                    break  # Not nice but it keeps the code clean

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)
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
    print('\nAnalogy Task:')
    # Our model is trained on WSJ news queen and king should be OOV or very rare tokens
    # Try with different words
    analogy_our_model('man', 'woman', 'u.s.', best_lstm_model, lang)

    # Our model is trained on WSJ news queen and king should be OOV or very rare tokens
    analogy_our_model('a', 'woman', 'queen', best_lstm_model, lang)
    print()
    print("-" * 100)
