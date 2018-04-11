#!/usr/bin/env python

from io import open
import unicodedata
import string
import re
import random
import sys

import numpy as np
import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()

################################################
# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.
#

#SOS_token = 0
#EOS_token = 1

class Lang:
    SOS_token = 0
    EOS_token = 1
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Lang.SOS_token: "SOS", Lang.EOS_token: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# class PretrainLang(Lang):
#     def __init__(self, name):
#         self.name = name
#         self.word2index = {}
#         self.word2count = {}
#         self.index2word = {}
#         self.n_words = 0

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(lang.EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromSents(lang, sent):
    input_variable = variableFromSentence(lang, sent)
    target_variable = variableFromSentence(lang, sent)
    return (input_variable, target_variable)

def getRandomSentences(fn, max_length=-1, lang=None):
    f = open(fn, 'r')
    if lang is None:
        lang = Lang(fn.split('.')[0])

    actual_max_length = 0
    num_total_sents = 0

    all_sents = []
    for line in f:
        num_total_sents += 1
        line = line.rstrip()
        tokens = line.split()
        if max_length == -1 or len(tokens) < max_length:
            all_sents.append(line)
            lang.addSentence(line)
            actual_max_length = max(actual_max_length, len(tokens))
    
    print("Collected %d out of %d sentences for training with the proper length" % (len(all_sents), num_total_sents))

    f.close()
    return lang, all_sents, actual_max_length

def readVectors(fn):
    f = open(fn, encoding='utf-8')
    header = f.readline().strip()
    vlen, dims = [int(x) for x in header.split()]
    # Add EOS and SOS vectors
    vecs = np.zeros((vlen+2, dims))
    word_ind = 2
    lang = Lang('Lang_%s' % (f))

    for line in f.readlines():
        vals = line.rstrip().split(' ')
        word = vals[0]
        lang.addWord(word)
        vecs[word_ind,:] += [float(x) for x in vals[1:]]
        word_ind += 1

    return vecs, lang
