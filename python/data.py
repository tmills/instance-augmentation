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
#UNK_token = 2

class Lang:
    SOS_token = 0
    EOS_token = 1
    UNK_token = 2
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {Lang.SOS_token: "SOS", Lang.EOS_token: "EOS", Lang.UNK_token: "UNK"}
        self.n_words = 3  # Count SOS, EOS, UNK

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
    words = sentence.split(' ')
    # First replace unknown words with unk token if we have it:
    if 'unk' in lang.word2index:
        unk_words = [word if word in lang.word2index else 'unk' for word in words]
    else:
        unk_words = words

    # Then replace strings with int index.
    return [lang.word2index[word] if word in lang.word2index else Lang.UNK_token for word in sentence.split(' ')]

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
        pretrained = False
        lang = Lang(fn.split('.')[0])
    else:
        # print("Size of dictionary before getting data is: %d" % (lang.n_words))
        pretrained = True

    actual_max_length = 0
    num_total_sents = 0

    all_sents = []
    for line in f:
        num_total_sents += 1
        line = line.rstrip()
        tokens = line.split()
        if max_length == -1 or len(tokens) < max_length:
            all_sents.append(line)
            if not pretrained:
                lang.addSentence(line)
            actual_max_length = max(actual_max_length, len(tokens))
    
    # print("Size of dictionary after getting data is: %d" % (lang.n_words))
    print("Collected %d out of %d sentences for training with the proper length" % (len(all_sents), num_total_sents))

    f.close()
    return lang, all_sents, actual_max_length

def readVectors(fn):
    f = open(fn, encoding='utf-8')
    header = f.readline().strip()
    vlen, dims = [int(x) for x in header.split()]
    # Add EOS, SOS, and UNK vectors
    vecs = np.zeros((vlen+3, dims))
    word_ind = 3
    lang = Lang('Lang_%s' % (f))

    for line in f.readlines():
        vals = line.rstrip().split(' ')
        word = vals[0]
        lang.addWord(word)
        vecs[word_ind,:] += [float(x) for x in vals[1:]]
        word_ind += 1

    return vecs, lang
