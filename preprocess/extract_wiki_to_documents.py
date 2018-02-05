#!/usr/bin/env python

import sys
import re

import nltk
from nltk.tokenize import wordpunct_tokenize

def main(args):
    if len(args) < 1:
        sys.stderr.write('One required argument: <wiki xml>\n')
        sys.exit(-1)

    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    num_docs = 0
    doc = ''
    f = open(args[0], 'r')

    for line in f:
        if line[0:6] == '</doc>':
            ## At the end of a document start processing it into
            ## paragraphs, sentences, tokens
            num_docs += 1
            paragraphs = re.split("\n\n+", doc)
            for paragraph in paragraphs:
                sentences = sent_detector.tokenize(paragraph.strip())
                for sentence in sentences:
                    words = wordpunct_tokenize(sentence)
                    for word in words:
                        sys.stdout.write(word.lower())
                        sys.stdout.write(' ')
                    sys.stdout.write('</s> ')
                sys.stdout.write('</p> ')
            sys.stdout.write('\n')
        elif line[0:4] == "<doc":
            doc = ''
            line_num = 0
        else:
            if line_num > 1:
                ## First 2 lines are repeat of the title and a blank line.
                doc += line
            line_num += 1

        
    sys.stderr.write("Wrote %d documents\n" % (num_docs))

if __name__ == '__main__':
    main(sys.argv[1:])
