from functions import *


class VariationalDropout(nn.Module):
    def __init__(self, p=0.5, batch_first=False):
        super().__init__()
        self.p = p
        self.batch_first = batch_first

    def forward(self, x):
        if not self.training or not self.p:
            return x
        m = x.data.new(x.size(0), 1, x.size(2) if self.batch_first else 1).bernoulli_(1 - self.p)
        mask = m.div_(1 - self.p)
        mask = mask.expand_as(x)
        return mask * x


class LM_LSTM_Dropout(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1, emb_dropout=0.1, n_layers=1,
                 tied_weights=False, variational_dropout=True):
        super(LM_LSTM_Dropout, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        if variational_dropout:  # added variational dropout to embedding
            self.emb_dropout = VariationalDropout(emb_dropout)
        else:
            self.emb_dropout = nn.Dropout(emb_dropout)

        # Pytorch LSTM layer
        self.rnn = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False)  # Replaced RNN with LSTM

        if tied_weights:
            # Linear layer to project the hidden layer to our output space
            self.output = nn.Linear(hidden_size, output_size, bias=False)  # no bias if weights are tied
            self.output.weight = self.embedding.weight  # tie weights
        else:
            # Linear layer to project the hidden layer to our output space
            self.output = nn.Linear(hidden_size, output_size)

        if variational_dropout:
            # added variational dropout to output
            self.out_dropout = VariationalDropout(out_dropout)
        else:
            self.out_dropout = nn.Dropout(out_dropout)

        # pad token index for loss function
        self.pad_token = pad_index

    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        emb = self.emb_dropout(emb)  # Applied Dropout after embedding
        rnn_out, _ = self.rnn(emb)
        rnn_out = self.out_dropout(rnn_out)  # Applied Dropout before output
        output = self.output(rnn_out).permute(0, 2, 1)
        return output

    def get_word_embedding(self, token):
        return self.embedding(token).squeeze(0).detach().cpu().numpy()

    def get_most_similar(self, vector, top_k=10):
        embeds = self.embedding.weight.detach().cpu().numpy()
        # Our function that we used before
        scores = []
        for i, x in enumerate(embeds):
            if i != self.pad_token:
                scores.append(cosine_similarity(x, vector))
        # Take ids of the most similar tokens
        scores = np.asarray(scores)
        indexes = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[indexes]
        return indexes, top_scores
