#!/usr/bin/env python

import sys
import numpy as np
from ppgplot import * 
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
import trm.subs.plot as plot
from   trm.subs import Fname

usage = """
usage: %s times log ntrial xindex yindex frac [device]

computes the difference between orbit integration versus Keplerian orbits. It reads in
a log file of orbits and then for each one computes an RMS difference between integrating
each one versus assuming they are Keplerian. It then plots a colour-coded scatter diagram dividing
the results into two classes, split on a user-defined fraction. This is really just to test what
it is that causes the difference between the two approaches over short intervals with the expectation 
of seeing mean-motion resonances.


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

xindex -- index of X parameter

yindex -- index of Y parameter

frac   -- fraction (0-1) to define split between two classes.

device -- plot device
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
inpt.register('xindex',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('yindex',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('frac',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('device',  inp.Input.LOCAL,  inp.Input.HIDE)

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
xindex  = inpt.get_value('xindex', 'index of X axis variable', 1, 1)
yindex  = inpt.get_value('yindex', 'index of Y axis variable', 2, 1)
frac    = inpt.get_value('frac', 'fraction at which to divide into two groups', 0.5, 0., 1.)
device  = inpt.get_value('device',  'plot device', '/xs')

xindex -= 1
yindex -= 1
                       
# these are for future-proofing  
xoffset, yoffset = 0, 0

x = chain.vals[-ntrial:,xindex] - xoffset
y = chain.vals[-ntrial:,yindex] - yoffset

def limits(x):
    x1 = x.min()
    x2 = x.max()
    xr = x2 - x1
    x1 -= xr/10.
    x2 += xr/10.
    return (x1,x2)


(x1,x2) = limits(x)             
(y1,y2) = limits(y)

# compute orbits
rms = np.empty_like(x)
ok  = x == x
for n in xrange(ntrial):
    p = orbits.fixpar(chain.model, zip(vvar, chain.vals[-n-1,:-3]))

    try:
        # orbit integration
        p['integ'] = 1
        tint = orbits.etimes(p,cycle)
        
        # keplerian
        p['integ'] = 0
        tkep = orbits.etimes(p,cycle)

        rms[n] = (tint-tkep).std()

    except:
        print 'integration failed'
        ok[n] = False

srms = np.array(rms[ok])
srms.sort()
level = srms[frac*len(srms)]

high = (rms > level) & ok

print 'Critical level =',level,', range =',rms[ok].min(),'to',rms[ok].max(),'median =',np.median(rms[ok])

pgopen(device)
#pgpap(width, aspect)
plot.defcol()
pgsch(1.5)
pgscf(2)
pgslw(4)
pgsci(4)
pgenv(x1, x2, y1, y2, 0, 0)
pgsci(2)

if xoffset < 0.:
    xlabel = chain.names[xindex] + ' + ' + str(-xoffset)
elif xoffset > 0.:
    xlabel = chain.names[xindex] + ' - ' + str(xoffset)
else:
    xlabel = chain.names[xindex]

if yoffset < 0.:
    ylabel = chain.names[yindex] + ' + ' + str(-yoffset)
elif yoffset > 0.:
    ylabel = chain.names[yindex] + ' - ' + str(yoffset)
else:
    ylabel = chain.names[yindex]

pglab(xlabel, ylabel, ' ')
pgsci(1)
pgpt(x[~high],y[~high],1)
pgsci(2)
pgscr(2,1,0,0)
pgsch(0.5)
pgpt(x[high],y[high],17)
pgsci(4)
pgscr(4,0,0,1)
pgsch(0.5)
pgpt(x[~ok],y[~ok],17)
pgclos()
