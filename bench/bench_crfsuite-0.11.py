#!/usr/bin/env python

import sys
import os
import string
from bench import *

CRFSUITE='/home/okazaki/install/crfsuite-0.11/frontend/crfsuite'
OUTDIR='crfsuite-0.11/'

training_patterns = (
    ('num_features', r'^Number of features: (\d+)', 1, int, last),
    ('time', r'^Total seconds required for L-BFGS: ([\d.]+)', 1, float, last),
    ('iterations', r'^\*\*\*\*\* (Iteration|Epoch) #(\d+)', 2, int, last),
    ('update', r'^Seconds required for this iteration: ([\d.]+)', 1, float, min),
    ('loss', r'^Log-likelihood: -([\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Item accuracy: \d+ / \d+ \(([\d.]+)\)', 1, float, last),
)

params = {
    'lbfgs-sparse': '-p regularization.sigma=0.70710678118654746 -p feature.possible_states=0 -p feature.possible_transitions=0',
    'lbfgs-dense': '-p regularization.sigma=0.70710678118654746 -p feature.possible_states=1 -p feature.possible_transitions=1',
}

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'crfsuite0.11-' + name + '.txt'
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
