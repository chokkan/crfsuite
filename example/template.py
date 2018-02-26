#!/usr/bin/env python
"""[summary]."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import sys
from builtins import map, object, range
from multiprocessing import Pool

from future import standard_library

standard_library.install_aliases()

CRFPP_PATTERN = re.compile(r'%x\[(?P<row>-?\d+),(?P<col>\d+)\]')
CRFPP_COORDINATE_DELIMITER = '/'
TEMPLATE_QUOTE = ':'


class FeatureExtractor(object):
    """FeatureExtractor."""

    def __init__(self, template_file):
        """[summary].

        Arguments:
            template_file {[type]} -- [description]

        Raises:
            ValueError -- [description]

        """
        self.inst = []
        self.inst_len = 0
        self.t = 0
        self.templates = []
        for line in template_file:
            line = line.strip()
            if not line or line == 'B':
                continue
            line_head = line[0]
            if line_head == '#':
                continue
            if line_head != 'U':
                raise ValueError('ERROR: unsupported: %s\n' % line)

            elements = line.split(TEMPLATE_QUOTE)
            name = elements[0] + '='
            pattern = elements[1]
            scale = ''
            if len(elements) == 3:
                scale = elements[2]
                try:
                    float(scale)
                except:
                    raise ValueError(
                        'ERROR: invalid scaling value: %s\n' % scale)
                scale = TEMPLATE_QUOTE + scale
            coordinates = [list(map(
                int, CRFPP_PATTERN.match(coordinate_pattern).groups()))
                for coordinate_pattern in pattern.split(
                    CRFPP_COORDINATE_DELIMITER)]
            self.templates.append({
                'name': name, 'coordinates': coordinates, 'scale': scale})

    def apply(self, inst, inst_len, t):
        """[summary].

        Arguments:
            inst {[type]} -- [description]
            inst_len {[type]} -- [description]
            t {[type]} -- [description]
        """
        self.inst = inst
        self.inst_len = inst_len
        self.t = t
        self.inst[t]['F'] = [
            template['name'] + CRFPP_COORDINATE_DELIMITER.join([
                self.inst[row + self.t]['x'][col]
                if 0 <= row + self.t < self.inst_len else ''
                for row, col in template['coordinates']
            ]) + template['scale']
            for template in self.templates
        ]


def readiter(input_feature_tsv, sep):
    """[summary].

    Arguments:
        input_feature_tsv {[type]} -- [description]
        sep {[type]} -- [description]

    Yields:
        [type] -- [description]

    """
    X = []
    for line in input_feature_tsv:
        line = line.strip('\n')
        if not line:
            yield X
            X = []
        else:
            fields = line.split(sep)
            item = {
                'x': [x.replace('\\', '\\\\').replace(':', r'\:')
                      for x in fields[0:-1]],
                'y': fields[-1],
                'F': []
                }
            X.append(item)


if __name__ == '__main__':
    import argparse
    description = '''
This utility reads a data set from INPUT_FILE_PATH or STDIN, applies
feature templates compatible with CRF++, and outputs attributes to
OUTPUT_FILE_PATH or STDOUT, repectively. Each line of a data set must
consist of field values separated by SEPARATOR characters (customizable
with -s option).'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('template_file',
                        metavar='TEMPLATE_FILE_PATH',
                        type=argparse.FileType('r'))
    parser.add_argument('input_feature_tsv',
                        nargs='?',
                        metavar='INPUT_FILE_PATH',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('output_crfsuite_feature_tsv',
                        nargs='?',
                        metavar='OUTPUT_FILE_PATH',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    parser.add_argument('-s', '--sep',
                        default='\t',
                        help='specify the separator of columns of input data'
                             ' [default: "\\t"]')
    args = parser.parse_args()

    F = FeatureExtractor(args.template_file)

    def _apply_F(inst):
        inst_len = len(inst)
        rows = []

        for t in range(inst_len):
            F.apply(inst, inst_len, t)
            columns = [inst[t]['y']] + inst[t]['F']
            rows.append('\t'.join(columns))
        rows.append('')
        return rows

    with Pool(processes=4) as pool:
        sentence_iter = pool.imap(_apply_F,
                                  readiter(args.input_feature_tsv, args.sep),
                                  50)
        for rows in sentence_iter:
            args.output_crfsuite_feature_tsv.write('\n'.join(rows) + '\n')
