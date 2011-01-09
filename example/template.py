#!/usr/bin/env python

"""
A utility for feature templates.
Copyright 2010,2011 Naoaki Okazaki.
"""

def apply(X, templates):
    """
    Generate features for an item sequence by applying feature templates.
    A feature template consists of a tuple of (name, offset) pairs,
    where name and offset specify a field name and offset from which
    the template extracts a feature value. Generated features are stored
    in the 'F' field of each item in the sequence.

    @type   X:      list of mapping objects
    @param  X:      The item sequence.
    @type   template:   tuple of (str, int)
    @param  template:   The feature template.
    """
    for template in templates:
        name = '|'.join(['%s[%d]' % (f, o) for f, o in template])
        for t in range(len(X)):
            values = []
            for field, offset in template:
                p = t + offset
                if p not in range(len(X)):
                    values = []
                    break
                values.append(X[p][field])
            if values:
                X[t]['F'].append('%s=%s' % (name, '|'.join(values)))

if __name__ == '__main__':
    # A sample of an item sequence.
    X = [
        {'w': 'Brown',    'pos': 'NNP', 'F': []},
        {'w': 'promises', 'pos': 'VBZ', 'F': []},
        {'w': 'change',   'pos': 'NN',  'F': []},
        ]

    # A sample of feature templates.
    templates = (
        (('w', -1), ),              # Previous word.
        (('w',  0), ),              # Current word.
        (('w',  1), ),              # Subsequent word.
        (('w', -1), ('w',  0)),     # Previous and current words.
        (('w',  0), ('w',  1)),     # Current and subsequent words.
        (('pos', -1), ),            # Previous POS.
        (('pos',  0), ),            # Current POS.
        (('pos',  1), ),            # Subsequent POS.
        (('pos', -1), ('pos',  0)), # Previous and current POSs.
        (('pos',  0), ('pos',  1)), # Current and subsequent POSs.
        )

    # Apply feature templates to generate features.
    apply(X, templates)

    # Print features in TAB-separated-value format.
    for x in X:
        print('\t'.join(x['F']))

