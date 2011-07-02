#!/usr/bin/env python

import sys

fi = sys.stdin
fo = sys.stdout

for line in fi:
    line = line.strip('\n')
    if not line:
        fo.write('\n')

    fields = line.split('\t')
    fo.write('%s %s\n' % (' '.join(fields[1:]), fields[0]))

    
