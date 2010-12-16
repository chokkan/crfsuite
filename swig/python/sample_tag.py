#!/usr/bin/env python

import crfsuite
import sys

def instances(fi):
    xseq = crfsuite.ItemSequence()
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield xseq
            xseq = crfsuite.ItemSequence()
            continue

        fields = line.split('\t')
        item = crfsuite.Item()
        for field in fields[1:]:
            p = field.rfind(':')
            if p == -1:
                item.append(crfsuite.Attribute(field))
            else:
                item.append(crfsuite.Attribute(field[:p], float(field[p+1:])))
        xseq.append(item)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    tagger = crfsuite.Tagger()
    tagger.open(sys.argv[1])

    for xseq in instances(fi):
        tagger.set(xseq)
        yseq = tagger.viterbi()
        print tagger.probability(yseq)
        for y in yseq:
            fo.write('%s\n' % y)
        fo.write('\n')

