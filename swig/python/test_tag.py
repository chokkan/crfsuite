#!/usr/bin/env python

import crfsuite
import sys

def instances(fi):
    xseq = crfsuite.ItemSequence()
    for line in fi:
        line = line.strip('\n')
        if not line:
            if 0 < len(xseq):
                yield xseq
            xseq = crfsuite.ItemSequence()

        fields = line.split('\t')
        item = crfsuite.Item()
        for field in fields[1:]:
            p = field.rfind(':')
            if p == -1:
                item.append(crfsuite.Attribute(field, 1.))
            else:
                item.append(crfsuite.Attribute(field[:p], float(field[p+1:])))
        xseq.append(item)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    tagger = crfsuite.Tagger()
    tagger.open(sys.argv[1])

    for xseq in instances(fi):
        print tagger.tag(xseq)


