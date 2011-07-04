#!/usr/bin/env python

import sys
import re

re_iteration = re.compile(r'^\*\*\*\*\* (Iteration|Epoch) #(\d+) \*\*\*\*\*')
patterns = {
    'loss': re.compile(r'^Loss: ([\d.]+)'),
    'accuracy': re.compile(r'^Item accuracy: \d+ / \d+ \(([\d.]+)\)'),
    'norm': re.compile(r'^Feature [L2-]+norm: ([\d.]+)'),
}

def read(fi):
    D = []
    for line in fi:
        line = line.strip('\n')
        m = re_iteration.match(line)
        if m is not None:
            if len(D)+1 != int(m.group(2)):
                sys.stderr.write('ERROR: sync\n')
                sys.exit(1)
            D.append({})
            continue

        if D:
            for name, pattern in patterns.iteritems():
                m = pattern.match(line)
                if m is not None:
                    D[-1][name] = float(m.group(1))
    
    return D

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    i = 1
    D = read(fi)
    for item in D:
        fo.write('%d' % i)
        i += 1
        for name in patterns.iterkeys():
            fo.write(' %f' % item[name])
        fo.write('\n')
