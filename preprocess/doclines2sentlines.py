#!/usr/bin/env python

import glob
import sys

def main(args):
    if len(args) < 1:
        sys.stderr.write('One required argument: <doclines directory> [optional max length]\n')
        sys.exit(-1)

    max_length = -1 if len(args) < 2 else int(args[1])

    for fn in glob.glob("%s/*wiki*.txt" % (args[0])):
        f = open(fn, 'r')
        for line in f:
            line = line.replace("</p> ", '')
            sents = line.split("</s> ")
            ## Last sentence is empty because of document-ending <s> and <p> tags
            for sent in sents[:-1]:
                if max_length < 0 or len(sent.split()) < max_length:
                    print(sent.rstrip())
        f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
