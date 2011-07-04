#!/usr/bin/env python

import sys
import os

scripts = (
    ('CRFsuite 0.12', './bench_crfsuite.py'),
    ('CRFsuite 0.11', './bench_crfsuite-0.11.py'),
    ('Wapiti v1.1.3', './bench_wapiti.py'),
    ('sgd 1.3', './bench_crfsgd.py'),
    ('CRF++ 0.54', './bench_crfpp.py'),
    ('MALLET 2.0.6', './bench_mallet.py'),
)

fields = (
    ('# Features', 'num_features'),
    ('Time', 'time'),
    ('# Iters', 'iterations'),
    ('Update', 'update'),
    ('Loss', 'loss'),
    ('Log', 'log'),
)

def number(x):
    y = ''
    p = x.find('.')
    if p == -1:
        p = len(x)
    for i in range(p):
        if i % 3 == 0 and i != 0:
            y = ' ' + y
        y = x[p-i-1] + y
    return y + x[p:]

def read():
    R = {}
    for name, script in scripts:
        fi = os.popen(script, 'r')
        R[name] = eval(fi.read())
    return R

def output_update(fo, R):
    for name, script in scripts:
        for param, result in R[name].iteritems():
            fo.write('%s\t%s\t%f\n' % (name, param, result.get('update', 0.)))

def output_table(fo, R):
    for name, script in scripts:
        for param, result in R[name].iteritems():
            fo.write('<row>\n')
            fo.write('<entry>%s</entry>\n' % name)
            fo.write('<entry>%s</entry>\n' % param)
            fo.write('<entry></entry>\n')
            fo.write('<entry>%s</entry>\n' % number('%d' % result['num_features']))
            fo.write('<entry>%s</entry>\n' % number('%.1f' % result['time']))
            fo.write('<entry>%s</entry>\n' % number('%d' % result['iterations']))
            fo.write('<entry>%s</entry>\n' % number('%.1f' % result['update']))
            fo.write('<entry>%s</entry>\n' % number('%.1f' % result['loss']))
            fo.write('<entry>%.3f</entry>\n' % (100. * result['accuracy']))
            fo.write('<entry><ulink url="%s">Log</ulink></entry>\n' % result['logfile'])
            fo.write('</row>\n')
            fo.write('\n')


if __name__ == '__main__':
    R = read()
    output_table(sys.stdout, R)
