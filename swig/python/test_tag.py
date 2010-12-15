#!/usr/bin/env python

import crfsuite
import sys

def instances(fi):
    inst = crfsuite.instance()
    for line in fi:
        line = line.strip('\n')
        if not line:
            if 0 < len(inst.xseq):
                yield inst
            inst = crfsuite.instance()

        fields = line.split('\t')
        inst.yseq.append(fields[0])
        
        item = crfsuite.Item()
        for field in fields[1:]:
            p = field.rfind(':')
            if p == -1:
                item.append(crfsuite.feature(field, 1.))
            else:
                item.append(crfsuite.feature(field[:p], float(field[p+1:])))
        inst.xseq.append(item)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    tagger = crfsuite.tagger()
    tagger.open(sys.argv[1])

    for inst in instances(fi):
        print tagger.tag(inst)


