#!/usr/bin/env python

import crfsuite
import sys

class Trainer(crfsuite.Trainer):
    def message(self, s):
        sys.stdout.write(s)

def instances(fi):
    xseq = crfsuite.ItemSequence()
    yseq = crfsuite.LabelSequence()
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield xseq, yseq
            xseq = crfsuite.ItemSequence()
            yseq = crfsuite.LabelSequence()

        fields = line.split('\t')
        item = crfsuite.Item()
        for field in fields[1:]:
            p = field.rfind(':')
            if p == -1:
                item.append(crfsuite.Attribute(field))
            else:
                item.append(crfsuite.Attribute(field[:p], float(field[p+1:])))
        xseq.append(item)
        yseq.append(fields[0])

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    trainer = Trainer()
    for xseq, yseq in instances(fi):
        trainer.append(xseq, yseq, 0)

    trainer.set('c2', '10')
    trainer.train('crf1d', 'l2sgd', sys.argv[1], -1)

