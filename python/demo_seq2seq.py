#!/usr/bin/env python

import sys
from os.path import join

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from .data import variableFromSentence, Lang
from .basic_model import EncoderRNN, DecoderRNN
# from .attention_model import EncoderRNN, AttnDecoderRNN
from .train_seq2seq_autoencoder import loadModels

MAX_LENGTH = 20
use_cuda = torch.cuda.is_available()

def main(args):
    if len(args) < 1:
        sys.stderr.write("One required argument: <model path>\n")
        sys.exit(-1)

    encoder, decoder, lang = loadModels(args[0])

    for line in sys.stdin.readlines():
        sent = line.rstrip()
        if sent == '':
            break
        if len(sent.split()) >= MAX_LENGTH:
            sys.stderr.write("Skipping sentence with too many tokens\n")
            continue

        output_words = evaluate(encoder, decoder, lang, sent, MAX_LENGTH)
        print(' '.join(output_words))

    

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, lang, sentence, max_length=MAX_LENGTH, attention=False):
    input_variable = variableFromSentence(lang, sentence)
    input_length = input_variable.size()[0]
    #print("Input variable has length %d and is: %s" % (input_length, str(input_variable)))

    if hasattr(decoder, 'attn'):
        attention=True
        # print("Found an attention model in the decoder.")

    encoder_hidden = encoder.initHidden()

    if attention:
        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
        #print("Created encoder outputs for attention with shape %s" % (str(encoder_outputs.shape)))

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        if attention:
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[Lang.SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        if attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
        else:            
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == Lang.EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words #, decoder_attentions[:di + 1]

if __name__ == '__main__':
    main(sys.argv[1:])
