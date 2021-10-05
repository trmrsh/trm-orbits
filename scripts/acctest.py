#!/usr/bin/env python

import sys
import numpy as np
import emcee
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s times log ntrial acc1 acc2 nacc

tests the effect of changing the accuracy parameter. It will report the RMS and 
worst deviation relative to model based upon the smallest accuracy parameter specified.

Arguments:

times  -- file of eclipse times with cycle number, time and error as
          columns. Optionally at the top of the file there can be a line of the
          form '# offset = 1234' (NB the spaces must be there) which is an
          integer that will be subtracted from the cycle numbers in the file to
          reduce the correlation between the T0 and period terms of the
          ephemeris.

log    -- name of log file with MCMC results

ntrial -- number of trials to consider (starting from the last in the log file). For each one
          model data points will be stored so requiring memory for ndata*ntrial floats.

acc1   -- first accuracy parameter (the smallest you want to consider)

acc2   -- last accuracy parameter

nacc   -- number of them (they will logarithmically spaced)
"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('times',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('log',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('ntrial',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('acc1',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('acc2',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nacc',    inp.Input.LOCAL, inp.Input.PROMPT)

# load the data
times   = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))
data    = np.loadtxt(times)

# check for a cycle number offset and softening factor
fp = open(times)
offset = 0
soften = 0.
for line in fp:
    if line.startswith('# offset = '):
        offset = int(line[11:])
    if line.startswith('# soften = '):
        soften = float(line[11:])
fp.close()
print 'A cycle number offset =',offset,'will be applied.'
cycle   = data[:,0] - offset
time    = data[:,1]
etime   = np.sqrt(soften**2 + data[:,2]**2)

log     = inpt.get_value('log',  'log file for storage of mcmc output', Fname('nnser', '.log', Fname.OLD))

chain = mcmc.Fchain(log)
pars  = chain.model.keys()
print 'read chain'
if chain.vals is None or len(chain.vals) == 0 or len(chain.vals.shape) != 2 or chain.vals.shape[1] < 1:
    print 'File =',log,'contains no results to start from!'
    exit(1)

# Use model read in from header to define the starting model,
# taking the last line of the file to update the variable parameters.
# last three parameters (chi**2 etc) excluded in all cases.
vvar    = chain.names[:-3]
model   = chain.model

ntrial  = inpt.get_value('ntrial', 'number of trials to use', 100, 1)
acc1    = inpt.get_value('acc1', 'first accuracy parameter', 1.e-10, 0.0)
acc2    = inpt.get_value('acc2', 'last accuracy parameter', max(acc1, 1.e-6), acc1)
nacc    = inpt.get_value('nacc', 'number of accuracy parameters', 10, 2)

# grab space
bmodels = np.empty((ntrial,len(cycle)))
cmodels = np.empty((ntrial,len(cycle)))

# compute initial models
print 'Computing models for acc =',acc1
for n in xrange(ntrial):
    p = orbits.fixpar(chain.model, zip(vvar, chain.vals[-n-1,:-3]))
    bmodels[n,:] = orbits.etimes(p,cycle,acc=acc1)


# now compute others
print 'Computing models for other accuracies which will be rated against the first set'
for na in xrange(nacc-1):
    acc = np.exp(np.log(acc1)+(np.log(acc2)-np.log(acc1))*(na+1)/(nacc-1))
    
    for n in xrange(ntrial):
        p = orbits.fixpar(chain.model, zip(vvar, chain.vals[-n-1,:-3]))
        cmodels[n,:] = orbits.etimes(p,cycle,acc=acc)

    print 'acc =',acc,'RMS(diff) = ',(cmodels-bmodels).std(),'worst =',np.abs(cmodels-bmodels).max()
