#!/usr/bin/env python

import sys
import os
import string
from bench import *

CRFPP_LEARN='/home/okazaki/local/bin/crf_learn'
CRFPP_TEST='/home/okazaki/local/bin/crf_test'
OUTDIR='crfpp/'

training_patterns = (
    ('num_features', r'^Number of features:[ ]*(\d+)', 1, int, last),
    ('time', r'^Done!([\d.]+)', 1, float, last),
    ('iterations', r'^iter=(\d+)', 1, int, last),
    ('update', r'time=([\d.]+)', 1, float, min),
    ('loss', r'obj=([\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Item accuracy: ([\d.]+)', 1, float, last),
)

params = {
    'lbfgs': '-a CRF-L2',
    'mira': '-a MIRA',
}

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'crfpp-' + name + '.txt'
        tglog = OUTDIR + name + '.tg.log'

        s = string.Template(
            '$crfpp_learn $param template.crfpp train.txt $model > $trlog'
            )
        cmd = s.substitute(
            crfpp_learn=CRFPP_LEARN,
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
            '$crfpp_test -m $model test.txt | ./accuracy.py > $tglog'
            )
        cmd = s.substitute(
            crfpp_test=CRFPP_TEST,
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
