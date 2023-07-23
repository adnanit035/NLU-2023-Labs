import numpy as np
from numpy.linalg import norm
import torch
import torch.nn as nn
import math


def cosine_similarity(v, w):
    return np.dot(v, w) / (norm(v) * norm(w))


# Training and evaluation loops
def train_loop(data, optimizer, criterion, model, average_model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid explosion gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

        # Update the average model
        # for param, avg_param in zip(model.parameters(), average_model.parameters()):
        #     avg_param.data.mul_(0.999).add_(0.001, param.data)

        #  Update the average model.
        for param, avg_param in zip(model.parameters(), average_model.parameters()):
            avg_param.data.mul_(0.9).add_(param.data, alpha=0.1)

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad():  # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return


def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


def analogy_our_model(w1, w2, w3, model, lang):
    model.eval().to('cpu')

    # Suggest: make use of torch.LongTensor and check if the word is in the vocab
    # Get word ids
    temp_w1 = lang.word2id[w1]
    temp_w2 = lang.word2id[w2]
    temp_w3 = lang.word2id[w3]

    # Get word vectors
    v1 = model.get_word_embedding(torch.LongTensor([temp_w1]))
    v2 = model.get_word_embedding(torch.LongTensor([temp_w2]))
    v3 = model.get_word_embedding(torch.LongTensor([temp_w3]))

    # relation vector
    rv = v3 + v2 - v1

    # Get the most similar word
    ms = model.get_most_similar(rv, top_k=10)

    # getting words & scores
    for i, key in enumerate(ms[0]):
        print(lang.id2word[key], ms[1][i])
