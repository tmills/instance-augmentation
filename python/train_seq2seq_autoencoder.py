#!/usr/bin/env python

import argparse
from io import open
import unicodedata
import string
import re
import random
import sys
import pickle
from zipfile import ZipFile

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from .data import Lang, getRandomSentences, variablesFromSents, readVectors

## "Global" variables
use_cuda = torch.cuda.is_available()
MAX_LENGTH = 20
hidden_size = 256
#SOS_token = 0
#EOS_token = 1


def main(args):
    parser = argparse.ArgumentParser(description='Seq2seq trainer')
    parser.add_argument('--data', type=str, required=True,
                    help='Sentlines file to train on')
    parser.add_argument('--attention', action="store_true", help='Use attention model')
    parser.add_argument('--model', type=str, default=None,
                    help='Partially-trained model to resume learning')
    parser.add_argument('--vectors', type=str, default=None,
                    help='Pre-trained word embeddings file to use')

    args = parser.parse_args(args)

    lang = None
    if not args.vectors is None:
        vectors, lang = readVectors(args.vectors)

    lang, training_sents, max_length = getRandomSentences(args.data, MAX_LENGTH, lang=lang)
    
    if not args.model is None:
        print("Resuming training with partially-trained model.")
        ## TODO: This lang is the one we want since its word->index mapping is needed, however if the dataset
        ## given above is different from the one originally used to create the language then there will be
        ## issues (with OOV), so something is needed to reconcile if I ever try that method.
        encoder1, decoder1, lang = loadModels(args.model)
        attention = ('attn' in dir(decoder1))
    elif args.attention:
        from .basic_model import EncoderRNN
        from .attention_model import AttnDecoderRNN as DecoderRNN
        attention = True
        print("Training attention-based decoder.")
        encoder1 = EncoderRNN(lang.n_words, hidden_size, embedding_dims=100, vectors=vectors)
        decoder1 = DecoderRNN(hidden_size, lang.n_words, embedding=encoder1.embedding, max_length=MAX_LENGTH)
    else:
        attention = False
        from .basic_model import EncoderRNN, DecoderRNN
        print("Training basic RNN decoder.")
        encoder1 = EncoderRNN(lang.n_words, hidden_size, embedding_dims=100, vectors=vectors)
        decoder1 = DecoderRNN(hidden_size, lang.n_words, embedding=encoder1.embedding)

    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    trainIters(training_sents, encoder1, decoder1, 1000000, lang, print_every=5000, attention=attention, max_length=MAX_LENGTH, learning_rate=0.01)


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#

teacher_forcing_ratio = 0.5


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH, attention=False):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    ## Used only for attention decoder:
    if attention:
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        if attention:
            encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[Lang.SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            # If using attention, add encoder_outputs as the 3rd argument and decoder attention as the third output
            # import pdb; pdb.set_trace()
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)

            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            # If using attention, add encoder_outputs as the 3rd argument and decoder attention as the third output
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == Lang.EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(sents, encoder, decoder, n_iters, lang, print_every=1000, plot_every=100, learning_rate=0.01, attention=False, max_length=MAX_LENGTH):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [variablesFromSents(lang, random.choice(sents))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, attention=attention, max_length=max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            ## Checkpoint the models:
            saveModels(encoder, decoder, lang)

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def saveModels(encoder, decoder, lang):
    with open('encoder.pth', 'wb') as f:
        torch.save(encoder, f)
    with open('decoder.pth', 'wb') as f:
        torch.save(decoder, f)
    with open('lang.pkl', 'wb') as f:
        pickle.dump(lang, f)
    with ZipFile('model.ptz', 'w') as myzip:
        myzip.write('encoder.pth')
        myzip.write('decoder.pth')
        myzip.write('lang.pkl')

def loadModels(path):
    with ZipFile(path, 'r') as myzip:
        myzip.extract('encoder.pth')
        myzip.extract('decoder.pth')
        myzip.extract('lang.pkl')
    
    encoder = torch.load('encoder.pth')
    encoder.gru.flatten_parameters()
    decoder = torch.load('decoder.pth')
    decoder.gru.flatten_parameters()
    with open('lang.pkl', 'rb') as f:
        lang = pickle.load(f)
    
    return encoder, decoder, lang


if __name__ == '__main__':
    main(sys.argv[1:])
