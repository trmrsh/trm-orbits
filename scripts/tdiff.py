#!/usr/bin/env python

import sys, signal, os
import math as m
import numpy as np
from multiprocessing import Pool
from trm import subs, mcmc, orbits
import trm.subs.input as inp
from trm.subs import Fname

usage = """
usage: %s times inlog outlog

Adds time differences to an orbit MCMC run to indicate the difference between Newtonian and 
Keplerian models for each set of elements. 6 times are added jfdelt, mfdelt, afdelt, jddelt,
mddelt, addelt referring to the maximum time difference for Jacobian, Modified Jacobian and 
Astrocentric coordinates for finely spaced times (f) and the actual data (d) over the range
of time spanned by the data.


Arguments:
 
 times   -- eclipse times.

 inlog   -- input log file of orbits. This will be used to generate a new log file with the
            same orbits but with additional stability info tacked on.

 nfine   -- number of points to use for the finely-spaced array.
 
 outlog  -- output log file of orbits. Ratio 'hill' will be added. If > 1 this should stop
            orbit crossing. However, this does not mean that mean-motion resonances can't
            stabilize cases for < 1. See also 'stability.py'
"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('times',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nfine',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('inlog',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('outlog',  inp.Input.LOCAL, inp.Input.PROMPT)

times   = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))
nfine   = inpt.get_value('nfine', 'number of finely-spaced points', 100, 2)
inlog   = inpt.get_value('inlog', 'input log file containing orbits', Fname('nnser', '.log', Fname.OLD))
outlog  = inpt.get_value('outlog', 'output log file containing orbits and time differences', 
                         Fname('nnser_stab', '.log', Fname.NOCLOBBER))

# load the data
cycle,time,etime,offset,soften = orbits.load_times(times)

# load the input log file
chain = mcmc.Chain(inlog)
model = chain.model
vvar  = chain.vpars()

extras = chain.names[len(vvar):] + ['jfdelt', 'jddelt', 'mfdelt', 'mddelt', 'afdelt', 'addelt']

# create output from scratch
ochain = mcmc.Fchain(chain.model, chain.nstore, chain.method, chain.jump, \
                         chain.nwalker, chain.stretch, extras, \
                         chain.chmin, outlog)

cfine  = np.linspace(cycle.min(), cycle.max(), nfine)
nvar   = len(vvar)
nentry = 0
nchain = len(chain) 
for vals in chain.vals:
    td = []
    pd = orbits.fixpar(model, zip(vvar, vals[:nvar]))

    # Jacobi
    pd['coord'] = 'Jacobi'
    try:
        pd['integ'] = False
        kep = orbits.etimes(pd, cfine) 
        pd['integ'] = True
        new = orbits.etimes(pd, cfine) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    try:
        pd['integ'] = False
        new = orbits.etimes(pd, cycle) 
        pd['integ'] = True
        kep = orbits.etimes(pd, cycle) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    # Modified Jacobi
    pd['coord'] = 'Marsh'
    try:
        pd['integ'] = False
        kep = orbits.etimes(pd, cfine) 
        pd['integ'] = True
        new = orbits.etimes(pd, cfine) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    try:
        pd['integ'] = False
        new = orbits.etimes(pd, cycle) 
        pd['integ'] = True
        kep = orbits.etimes(pd, cycle) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    # Astrocentric
    pd['coord'] = 'Astro'
    try:
        pd['integ'] = False
        kep = orbits.etimes(pd, cfine) 
        pd['integ'] = True
        new = orbits.etimes(pd, cfine) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    try:
        pd['integ'] = False
        new = orbits.etimes(pd, cycle) 
        pd['integ'] = True
        kep = orbits.etimes(pd, cycle) 
        td.append(subs.DAY*np.abs(new-kep).max())
    except:
        td.append(-1.)

    ochain.add_line(list(vals) + td)
    nentry += 1
    if nentry % 1000 == 0:
        print 'Reached entry',nentry,'out of',nchain

