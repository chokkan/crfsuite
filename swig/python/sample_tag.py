#!/usr/bin/env python

import crfsuite
import sys

def instances(fi):
    xseq = crfsuite.ItemSequence()

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An end of sequence.
            yield xseq
            xseq = crfsuite.ItemSequence()
            continue

        # Obtain a label and attributes.
        fields = line.split('\t')
        item = crfsuite.Item()
        for field in fields[1:]:
            p = field.rfind(':')
            if p == -1:
                # The weight of this item is not specified (weight = 1)
                item.append(crfsuite.Attribute(field))
            else:
                # The weight of this item is specified by ';'
                item.append(crfsuite.Attribute(field[:p], float(field[p+1:])))

        xseq.append(item)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    # Create a tagger.
    tagger = crfsuite.Tagger()

    # Load a model for the tagger.
    tagger.open(sys.argv[1])

    # Loop for a sequence (refer to instances(fi)).
    for xseq in instances(fi):
        # Set the sequence
        tagger.set(xseq)

        # Find the best label sequence (using the Viterbi algorithm).
        yseq = tagger.viterbi()

        # Show the probability of the best label sequence.
        fo.write('%f\n' % tagger.probability(yseq))

        # Output labels with marginal probabilities.
        for t in range(len(yseq)):
            fo.write('%s:%f\n' % (yseq[t], tagger.marginal(yseq[t], t)))
        fo.write('\n')
