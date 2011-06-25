#!/usr/bin/env python

"""
An example for chunking.
Copyright 2010,2011 Naoaki Okazaki.
"""

import crfutils
import template
import sys

# Feature template. This template is identical to the one bundled in CRF++
# distribution, but written in a Python object.
templates = (
    (('w', -2), ),
    (('w', -1), ),
    (('w',  0), ),
    (('w',  1), ),
    (('w',  2), ),
    (('w', -1), ('w',  0)),
    (('w',  0), ('w',  1)),
    (('pos', -2), ),
    (('pos', -1), ),
    (('pos',  0), ),
    (('pos',  1), ),
    (('pos',  2), ),
    (('pos', -2), ('pos', -1)),
    (('pos', -1), ('pos',  0)),
    (('pos',  0), ('pos',  1)),
    (('pos',  1), ('pos',  2)),
    (('pos', -2), ('pos', -1), ('pos',  0)),
    (('pos', -1), ('pos',  0), ('pos',  1)),
    (('pos',  0), ('pos',  1), ('pos',  2)),
    )

def feature_extractor(X):
    # Apply feature templates to obtain features (in fact, attributes)
    template.apply(X, templates)
    if X:
	# Append BOS and EOS features manually
        X[0]['F'].append('__BOS__')     # BOS feature
        X[-1]['F'].append('__EOS__')    # EOS feature

if __name__ == '__main__':
    fi = sys.stdin
    fo = sys.stdout
    
    # Each line of the input text is assumed to have a token surface,its
    # part-of-speech, and its chunk label separated by SPACE (' ')
    # characters in this order. An empty line presents an end of a sentence.
    fields = ('w', 'pos', 'y')
    
    if len(sys.argv) < 3:
        # The command line does not have "-m MODEL" specified.
        # The below demonstrates how to convert an input text into the
        # label-attribute format compatible with the CRFsuite utility

        # For each sequence from STDIN.
        for X in crfutils.readiter(fi, fields, ' '):
            # print X   # uncomment this to see how a sequence is stored in X
            feature_extractor(X)

            # Output the labels and attributes.
            for x in X:
            	fo.write('%s' % x['y'])
	        for a in x['F']:
                    fo.write('\t%s' % a.replace(':', '__COLON__'))
                fo.write('\n')
            fo.write('\n')
        
    elif sys.argv[1] == '-m':
        # The command line has "-m MODEL" specified.
        # The below is an example of using the python module for tagging
        
        import crfsuite
        tagger = crfsuite.Tagger()
        tagger.open(sys.argv[2])

        # For each sequence from STDIN.
        for X in crfutils.readiter(fi, fields, ' '):
            # Obtain features.
            feature_extractor(X)
            xseq = crfutils.to_crfsuite(X)
            
            # Tag the current sentence.
            yseq = tagger.tag(xseq)
            for t in range(len(X)):
                v = X[t]
                # Thru the input text.
                fo.write(' '.join([v[f] for f in F]))
                # Append the predicted label
                fo.write(' %s\n' % yseq[t])
            fo.write('\n')

