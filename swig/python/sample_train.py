#!/usr/bin/env python

import crfsuite
import sys

# Extend crfsuite.Trainer to receive progress messages from the training
# algorithm (this is achieved by cross language polymorphism!)
class Trainer(crfsuite.Trainer):
    def message(self, s):
        # Overwrite the member function 'message' to receive messages.
        sys.stdout.write(s)

def instances(fi):
    xseq = crfsuite.ItemSequence()
    yseq = crfsuite.StringList()

    for line in fi:
        line = line.strip('\n')
        if not line:
            # An end of sequence.
            yield xseq, tuple(yseq)
            xseq = crfsuite.ItemSequence()
            yseq = crfsuite.StringList()
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
        yseq.append(fields[0])

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    # How to obtain CRFsuite version
    fo.write('CRFsuite version %s\n' % crfsuite.version())

    # Create an instance of a trainer.
    trainer = Trainer()

    # Read training instances from STDIN (refer to instances(fi))
    #  xseq: crfsuite.ItemSequence()
    #  yseq: crfsuite.StringList()
    #  0: group number
    for xseq, yseq in instances(fi):
        trainer.append(xseq, yseq, 0)

    # Select a training algorithm and a graphical model.
    trainer.select('l2sgd', 'crf1d')

    # Show the list of parameters for the trainer.
    fo.write('===== List of training parameters =====\n')
    for name in trainer.params():
        fo.write('%s=%s: %s\n' % (name, trainer.get(name), trainer.help(name)))
    fo.write('\n')

    # How to set/get a training parameter.
    trainer.set('feature.possible_states', '1')
    fo.write('c2 = %s\n' % trainer.get('c2'))
    fo.write('\n')

    # Start training:
    #  the first argument: a filename of a model
    #  the second argument: a holdout group (-1: no holdout evaluation)
    fo.write('===== Training =====\n')
    trainer.train(
        sys.argv[1] if len(sys.argv) > 1 else '',
        -1
        )
