#!/usr/bin/env python

usage = """
usage: %s times log nskip nthin

Checks deviation between different parameterisations of 
Kepler orbits versus full-tilt N-body ints. Reports mean
standard deviation of each coordinate type.

times -- file of eclipse times log -- mcmc data based upon file preferably
nskip -- number of point groups to skip from start of log. If set negative,
         just the final ones will be loaded.  
nthin -- the number of points to skip for each one read in.
""" 
import sys
import numpy as np
from trm import subs, orbits, mcmc
import trm.subs.input as inp
from   trm.subs import Fname

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

inpt.register('times',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('log',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nskip',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nthin',   inp.Input.LOCAL, inp.Input.PROMPT)

# load the data
times   = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))

# load the data
cycle,time,etime,offset,soften = orbits.load_times(times)

log     = inpt.get_value('log',  'log of mcmc output', \
                             Fname('nnser', '.log', Fname.OLD))

nskip   = inpt.get_value('nskip', 'number of point groups to skip at start (-ve just to read at end)', 0)
nskip   = inpt.get_value('nthin', 'number of point to skip per point read', 0, 0)

# load models 
chain = mcmc.Chain(log, 0, nskip)

model = chain.model
vvar  = chain.vpars()

coords = ('Astro','Jacobi','Marsh')

rms = ([],[],[])
avals = chain.vals[:,:len(vvar)]
for i,vals in enumerate(avals):

    # generate parameter dictionary equivalent to line just read in
    # and thus compute the cooresponding times
    pd   = orbits.fixpar(model, zip(vvar, vals[:len(vvar)]))
    ts   = orbits.ephmod(cycle,pd['t0'],pd['period'])

    # for each coord list the RMS difference between the Keplerian
    # and N-nody computations
    rline = []
    for coord,rm in zip(coords,rms):
        pd['coord'] = coord

        orbs = orbits.ptolorb(pd)
        tdk  = orbits.tdelay(ts, orbs)

        # compute delays on N-body basis
        lrvm = orbits.ptolrvm(pd)
        try:
            ttime,ntest,eratio,npoint,ierr,tnext,nstore,arr = orbits.integrate(lrvm, ts-pd['tstart'],stoerm=True)
            tdn  = -subs.AU*arr[:,3]/subs.C
            diff = tdk-tdn
            rm.append(diff.std())
        except RuntimeError, err:
            print err
            print pd
            print 'Exception raised on line',i+1

    if (i+1) % 1000 == 0:
        print 'Reached line',i+1,'out of',len(avals)

for coord, rm in zip(coords,rms):
    rm = np.array(rm)
    print 'Coordinates %-6s have min,median,mean,max RMS = %5.2f, %5.2f, %5.2f, %5.2f seconds' % (coord,rm.min(),np.median(rm),rm.mean(),rm.max())

