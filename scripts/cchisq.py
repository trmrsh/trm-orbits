#!/usr/bin/env python

import sys
import numpy as np
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s times log nstart mdiff

check chi**2 of a set of models from a log, compares with values recorded.

Arguments:

times  -- file of eclipse times with cycle number, time and error as
          columns. Optionally at the top of the file there can be a line of the
          form '# offset = 1234' (NB the spaces must be there) which is an
          integer that will be subtracted from the cycle numbers in the file to
          reduce the correlation between the T0 and period terms of the
          ephemeris.

log    -- name of log file to store MCMC results in. 

nstart -- first model to start from

mdiff  -- minimum difference to report.
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
inpt.register('nstart',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('mdiff',   inp.Input.LOCAL, inp.Input.PROMPT)

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

log     = inpt.get_value('log',  'log file for storage of mcmc output', \
                             Fname('nnser', '.log', Fname.NEW))

nstart  = inpt.get_value('nstart',  'starting model', 1, 1)
mdiff   = inpt.get_value('mdiff',  'minimum difference to report', 10., 0.)

chain  = mcmc.Chain(log)
pars   = chain.model.keys()
nchisq = chain.names.index('chisq') 
vvar   = chain.vpars()

iworst = 1
worst  = 0.
for i,row in enumerate(chain.vals):
    if i+1 >= nstart:
        pinit = orbits.fixpar(chain.model, zip(vvar, row[:chain.nvpars()]))
        chisq = orbits.tchisq(pinit, cycle, time, etime)
        if abs(chisq-row[nchisq]) > mdiff:
            print 'Model',i+1,'chisq =',chisq,'cf',row[nchisq]
        if abs(chisq-row[nchisq]) > worst:
            worst  = abs(chisq-row[nchisq])
            iworst = i

print 'The worst model was number',i+1,'which showed a discrepancy of',worst


