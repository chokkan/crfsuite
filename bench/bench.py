import re
import collections

LOGDIR='log/'

def seconds(s):
    p = s.find(':')
    q = s.find(':', p+1)
    return int(s[:p]) * 3600 + int(s[p+1:q]) * 60 + int(s[q+1:])

def last(X):
    if len(X) >= 1:
        return X[-1]
    else:
        return None

def diffmin(X):
    D = []
    prev = None
    for x in X:
        if prev is not None:
            D.append(x - prev)
        prev = x
    return min(D)

def analyze_log(fi, patterns):
    P = {}
    for name, pattern, index, cast, func in patterns:
        P[name] = (re.compile(pattern), index, cast, func)

    D = collections.defaultdict(list)
    for line in fi:
        line = line.strip('\n')
        for name, (regex, index, cast, func) in P.iteritems():
            m = regex.search(line)
            if m is not None:
                if isinstance(index, tuple):
                    for i in index:
                        D[name].append(cast(m.group(i)))
                elif isinstance(index, int):
                    D[name].append(cast(m.group(index)))


    R = {}
    for name, (regex, index, cast, func) in P.iteritems():
        R[name] = func(D[name])
    return R
