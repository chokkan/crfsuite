#!/usr/bin/env python

import sys
import os
import string
from bench import *

CRFSGD='/home/okazaki/install/sgd-1.3/crf/crfsgd'
OUTDIR='crfsgd/'

training_patterns = (
    ('num_features', r'features: (\d+)', 1, int, last),
    ('time', r'^Done!  ([\d.]+)', 1, float, last),
    ('iterations', r'^\[Epoch (\d+)\]', 1, int, last),
    ('update', r'^\[Epoch \d+\][^a-z]+wnorm:[^a-z]+total time: ([\d.]+) seconds$', 1, float, diffmin),
    ('loss', r'loss: ([\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Item accuracy: ([\d.]+)', 1, float, last),
)

params = {
    'default': "-f 1 -r 100 -e ''",
}

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'crfsgd-' + name + '.txt'
        tglog = OUTDIR + name + '.tg.log'

        s = string.Template(
            '$crfsgd $param $model template.crfpp train.txt > $trlog'
            )
        cmd = s.substitute(
            crfsgd=CRFSGD,
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
            '$crfsgd -t $model test.txt | ./accuracy.py > $tglog'
            )
        cmd = s.substitute(
            crfsgd=CRFSGD,
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

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)
