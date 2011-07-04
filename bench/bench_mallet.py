#!/usr/bin/env python

import sys
import os
import string
from bench import *

MALLET='java -cp "/home/okazaki/install/mallet-2.0.6/class:/home/okazaki/install/mallet-2.0.6/lib/mallet-deps.jar" cc.mallet.fst.SimpleTagger'
OUTDIR='mallet/'

training_patterns = (
    ('num_features', r'^Number of weights = (\d+)', 1, int, last),
    ('time', r'^([\d.]+)user ([\d.]+)system', (1, 2), float, sum),
    ('iterations', r'^CRF finished one iteration of maximizer, i=(\d+)', 1, int, len),
#    ('update', r'^Seconds required for this iteration: ([\d.]+)', 1, float, min),
    ('loss', r'^getValue\(\) \(loglikelihood, optimizable by label likelihood\) = -([\d.]+)', 1, float, last),
)

tagging_patterns = (
    ('accuracy', r'^Testing accuracy=([\d.]+)', 1, float, last),
)

params = {
    'default': '--gaussian-variance 0.70710678118654746',
}

if __name__ == '_main__':
    print analyze_log(sys.stdin, training_patterns)

if __name__ == '__main__':
    fe = sys.stderr

    R = {}
    for name, param in params.iteritems():
        model = OUTDIR + name + '.model'
        trlog = OUTDIR + name + '.tr.log'
        trtxt = LOGDIR + 'mallet-' + name + '.txt'
        tglog = OUTDIR + name + '.tg.log'

        s = string.Template(
            'time $mallet --train true $param --model-file $model train.mallet > $trlog 2>&1'
            )
        cmd = s.substitute(
            mallet=MALLET,
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
            '$mallet --model-file $model --test lab test.mallet > $tglog 2>&1'
            )
        cmd = s.substitute(
            mallet=MALLET,
            model=model,
            tglog=tglog
            )

        fe.write(cmd)
        fe.write('\n')
        #os.system(cmd)

        D = analyze_log(open(trlog), training_patterns)
        D['update'] = 0.
        D.update(analyze_log(open(tglog), tagging_patterns))
        D['logfile'] = trtxt
        R[name] = D

    print repr(R)
