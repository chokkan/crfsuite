#!/usr/bin/env python

import sys
import os
import string
from bench import *

WAPITI='/home/okazaki/install/wapiti-1.1.3/wapiti'
OUTDIR='wapiti/'

training_patterns = (
    ('num_features', r'nb features: (\d+)', 1, int, last),
    ('time', r'^([\d.]+)user ([\d.]+)system', (1, 2), float, sum),
    ('iterations', r'\[\s*(\d+)\]', 1, int, last),
    ('update', r'time=([\d.]+)', 1, float, min),
    ('loss', r'obj=([\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Item accuracy: ([\d.]+)', 1, float, last),
)

params = {
    'lbfgs': '-a l-bfgs --rho2 0.70710678118654746 --maxiter 1000 --stopeps 0.00001 --stopwin 10',
    'rprop': '-a rprop --rho3 0.70710678118654746 --maxiter 1000',
}

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'wapiti-' + name + '.txt'
        tglog = OUTDIR + name + '.tg.log'

        s = string.Template(
            'time $wapiti train $param -p template.wapiti train.txt $model > $trlog 2>&1'
            )
        cmd = s.substitute(
            wapiti=WAPITI,
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
            '$wapiti label -m $model test.txt | ./accuracy.py > $tglog'
            )
        cmd = s.substitute(
            wapiti=WAPITI,
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
