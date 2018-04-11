#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#
use_cuda = torch.cuda.is_available()
MAX_LENGTH=10

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_dims=100, vectors=None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if vectors is None:
            self.embedding = nn.Embedding(input_size, embedding_dims)
        else:
            self.embedding = nn.Embedding(vectors.shape[0], vectors.shape[1]) # .from_pretrained(torch.FloatTensor(vectors), freeze=False)
            self.embedding.data = torch.FloatTensor(vectors)

        self.gru = nn.GRU(embedding_dims, hidden_size, bidirectional=True)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#


######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, embedding=None):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = embedding
        self.embedding_size = self.embedding.embedding_dim

        self.gru = nn.GRU(self.embedding_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.has_attention=False

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
