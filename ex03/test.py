print "final error:", 0.9852
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
def dropout(X, p_use=1.):
    if p_use < 1.:
        rs = RandomStreams()
        out = rs.multinomial(pvals=[[p_use, 1.-p_use]])
        print out.flatten()

        print dir(out.T)

    else:
        return X

X = T.fmatrix()

print X
dropout(X, 0.5)

print help(RandomStreams())
# test = T.fvector()
# print test
