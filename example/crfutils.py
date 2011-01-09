"""
A miscellaneous utility for sequential labeling.
Copyright 2010,2011 Naoaki Okazaki.
"""

import optparse
import sys

def readiter(fi, names, sep=' '):
    """
    Return an iterator for item sequences read from a file object.
    This function reads a sequence from a file object L{fi}, and
    yields the sequence as a list of mapping objects. Each line
    (item) from the file object is split by the separator character
    L{sep}. Separated values of the item are named by L{names},
    and stored in a mapping object. Every item has a field 'F' that
    is reserved for storing features.

    @type   fi:     file
    @param  fi:     The file object.
    @type   names:  tuple
    @param  names:  The list of field names.
    @type   sep:    str
    @param  sep:    The separator character.
    @rtype          list of mapping objects
    @return         An iterator for sequences.
    """
    X = []
    for line in fi:
        line = line.strip('\n')
        if not line:
            yield X
            X = []
        else:
            fields = line.split(sep)
            if len(fields) < len(names):
                raise ValueError(
                    'Too few fields (%d) for %r\n%s' % (len(fields), names, line))
            item = {'F': []}    # 'F' is reserved for features.
            for i in range(len(names)):
                item[names[i]] = fields[i]
            X.append(item)

def escape(src):
    """
    Escape colon characters from feature names.

    @type   src:    str
    @param  src:    A feature name
    @rtype          str
    @return         The feature name escaped.
    """
    return src.replace(':', '__COLON__')

def output_features(fo, X, field=''):
    """
    Output features (and reference labels) of a sequence in CRFSuite
    format. For each item in the sequence, this function writes a
    reference label (if L{field} is a non-empty string) and features.

    @type   fo:     file
    @param  fo:     The file object.
    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @type   field:  str
    @param  field:  The field name of reference labels.
    """
    for t in range(len(X)):
        if field:
            fo.write('%s' % X[t][field])
        for a in X[t]['F']:
            if isinstance(a, str):
                fo.write('\t%s' % escape(a))
            else:
                fo.write('\t%s:%f' % (escape(a[0]), a[1]))
        fo.write('\n')
    fo.write('\n')

def to_crfsuite(X):
    """
    Convert an item sequence into an object compatible with crfsuite
    Python module.

    @type   X:      list of mapping objects
    @param  X:      The sequence.
    @rtype          crfsuite.ItemSequence
    @return        The same sequence in crfsuite.ItemSequence type.
    """
    import crfsuite
    xseq = crfsuite.ItemSequence()
    for x in X:
        item = crfsuite.Item()
        for f in x['F']:
            if isinstance(f, str):
                item.append(crfsuite.Attribute(escape(f)))
            else:
                item.append(crfsuite.Attribute(escape(f[0]), f[1]))
        xseq.append(item)
    return xseq

def main(feature_extractor, fields='w pos y', sep=' '):
    fi = sys.stdin
    fo = sys.stdout

    # Parse the command-line arguments.
    parser = optparse.OptionParser(usage="""usage: %prog [options]
This utility reads a data set from STDIN, and outputs features (without -t
option) or tagging results (with -t option) to STDOUT."""
        )
    parser.add_option(
        '-t', dest='model',
        help='tag the input using the model (requires "crfsuite" module)'
        )
    parser.add_option(
        '-f', dest='fields', default=fields,
        help='specify the column names of input data [default: "%default"]'
        )
    parser.add_option(
        '-s', dest='separator', default=sep,
        help='specify the separator of columns of input data [default: "%default"]'
        )
    (options, args) = parser.parse_args()

    # The fields of input: ('w', 'pos', 'y) by default.
    F = options.fields.split(' ')

    if not options.model:
        # The generator function readiter() reads a sequence from a 
        for X in readiter(fi, F, options.separator):
            feature_extractor(X)
            output_features(fo, X, 'y')

    else:
        # Create a tagger with an existing model.
        import crfsuite
        tagger = crfsuite.Tagger()
        tagger.open(options.model)

        # For each sequence from STDIN.
        for X in readiter(fi, F, options.separator):
            # Obtain features.
            feature_extractor(X)
            xseq = to_crfsuite(X)
            yseq = tagger.tag(xseq)
            for t in range(len(X)):
                v = X[t]
                fo.write('\t'.join([v[f] for f in F]))
                fo.write('\t%s\n' % yseq[t])
            fo.write('\n')
