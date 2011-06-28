#!/usr/bin/env python

import re
import sys

class FeatureExtractor:
    def __init__(self):
        self.macro = re.compile(r'%x\[(?P<row>[\d-]+),(?P<col>[\d]+)\]')
        self.inst = []
        self.t = 0
        self.templates = []

    def read(self, fi):
        self.templates = []
        for line in fi:
            line = line.strip()
            if line.startswith('#'):
                continue
            if line.startswith('U'):
                self.templates.append(line.replace(':', '='))
            elif line == 'B':
                continue
            elif line.startswith('B'):
                sys.stderr(
                    'ERROR: bigram templates not supported: %s\n' % line)
                sys.exit(1)

    def replace(self, m):
        row = self.t + int(m.group('row'))
        col = int(m.group('col'))
        if row in range(0, len(self.inst)):
            return self.inst[row]['x'][col]
        else:
            return ''

    def apply(self, inst, t):
	self.inst = inst
	self.t = t
        for template in self.templates:
            f = re.sub(self.macro, self.replace, template)
            self.inst[t]['F'].append(f)

def readiter(fi, sep=None):
    X = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield X
            X = []
        else:
            fields = line.split(sep)
            item = {
                'x': fields[0:-1],
                'y': fields[-1],
                'F': []
                }
            X.append(item)

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout

    F = FeatureExtractor()
    F.read(open(sys.argv[1]))

    for inst in readiter(fi):
        for t in range(len(inst)):
            F.apply(inst, t)
            fo.write('%s' % inst[t]['y'])
            for attr in inst[t]['F']:
                fo.write('\t%s' % attr.replace(':', '__COLON__'))
            fo.write('\n')
        fo.write('\n')
