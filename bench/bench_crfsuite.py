#!/usr/bin/env python

import sys
import os
import string
from bench import *

CRFSUITE='/home/okazaki/projects/crfsuite/frontend/crfsuite'
OUTDIR='crfsuite/'

training_patterns = (
    ('num_features', r'^Number of features: (\d+)', 1, int, last),
    ('time', r'^Total seconds required for training: ([\d.]+)', 1, float, last),
    ('iterations', r'^\*\*\*\*\* (Iteration|Epoch) #(\d+)', 2, int, last),
    ('update', r'^Seconds required for this iteration: ([\d.]+)', 1, float, min),
    ('loss', r'^Loss: ([-\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Item accuracy: \d+ / \d+ \(([\d.]+)\)', 1, float, last),
)

params = {
    'lbfgs-sparse': '-a lbfgs -p feature.possible_states=0 -p feature.possible_transitions=0',
    'lbfgs-dense': '-a lbfgs -p feature.possible_states=1 -p feature.possible_transitions=1',
    'l2sgd-sparse': '-a l2sgd -p feature.possible_states=0 -p feature.possible_transitions=0',
    'l2sgd-dense': '-a l2sgd -p feature.possible_states=1 -p feature.possible_transitions=1',
    'ap-sparse': '-a ap -p feature.possible_states=0 -p feature.possible_transitions=0 -p max_iterations=50',
    'ap-dense': '-a ap -p feature.possible_states=1 -p feature.possible_transitions=1 -p max_iterations=50',
}

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'crfsuite-' + name + '.txt'
        tglog = OUTDIR + name + '.tg.log'

        s = string.Template(
            '$crfsuite learn $param -m $model train.crfsuite > $trlog'
            )
        cmd = s.substitute(
            crfsuite=CRFSUITE,
            param=param,
            model=model,
            trlog=trlog
            )

        fe.write(cmd)
        fe.write('\n')
        #os.system(cmd)

        fo = open(trtxt, 'w')
        fo.write('$ %s\n' % cmd)
        fo.write(open(trlog, 'r').read())

        s = string.Template(
            '$crfsuite tag -m $model -qt test.crfsuite > $tglog'
            )
        cmd = s.substitute(
            crfsuite=CRFSUITE,
            model=model,
            tglog=tglog
            )

        fe.write(cmd)
        fe.write('\n')
        #os.system(cmd)

        D = analyze_log(open(trlog), training_patterns)
        D.update(analyze_log(open(tglog), tagging_patterns))
        D['logfile'] = trtxt
        R[name] = D

    print repr(R)
