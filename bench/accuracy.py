#!/usr/bin/env python

import sys

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout
    n = 0
    m = 0

    for line in fi:
        line = line.strip()
        if line:
            fields = line.split()
            if len(fields) >= 2:
                if fields[-1] == fields[-2]:
                    m += 1
                n += 1

    print 'Item accuracy: %f' % (m / float(n))
