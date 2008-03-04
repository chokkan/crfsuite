#!/bin/env python

import sys

fi = sys.stdin
fo = sys.stdout

for line in fi:
    line = line.strip('\n')
    if not line:
        fo.write('\n')
    else:
        fields = line.split('\t')
        if len(fields) > 1:
            fo.write('\t'.join(fields[1:]))
            fo.write('\t')
        fo.write(fields[0])
        fo.write('\n')
