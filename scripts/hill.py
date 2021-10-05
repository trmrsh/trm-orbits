#!/usr/bin/env python

import sys, signal, os
import math as m
import numpy as np
from multiprocessing import Pool
from trm import subs, mcmc, orbits
import trm.subs.input as inp
from trm.subs import Fname

usage = """
usage: %s inlog 

Adds a ratio indicative of Hill stability to an orbit MCMC run. Hill stability worked out
from planetary case in Marchal & Bozis (http://adsabs.harvard.edu/abs/1982CeMec..26..311M)

Arguments:
 
 inlog   -- input log file of orbits. This will be used to generate a new log file with the
            same orbits but with additional stability info tacked on.

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
inpt.register('inlog',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('outlog',  inp.Input.LOCAL, inp.Input.PROMPT)

inlog   = inpt.get_value('inlog', 'input log file containing orbits', Fname('nnser', '.log', Fname.OLD))
outlog  = inpt.get_value('outlog', 'output log file containing orbits and Hill stability ratio', 
                         Fname('nnser_stab', '.log', Fname.NOCLOBBER))

# load the input log file
chain = mcmc.Chain(inlog)
model = chain.model
vvar  = chain.vpars()

extras = chain.names[len(vvar):] + ['hill',]

# create output from scratch
ochain = mcmc.Fchain(chain.model, chain.nstore, chain.method, chain.jump, \
                         chain.nwalker, chain.stretch, extras, \
                         chain.chmin, outlog)
def hstab(pd):
    """
    Computes beta and beta_crit parameters to do with Hill stability
    """
    lrvm = orbits.ptolrvm(pd)

    # compute angular momentum and energy. Note that
    # various factors of MSUN are removed from the physical
    # quantities since they cancel
    L = subs.Vec3()
    E = 0
    for i, rvmi in enumerate(lrvm):
        # angular momentum
        L += (subs.AU**2/subs.DAY)*rvmi.m*subs.cross(rvmi.r,rvmi.v)

        # kinetic energy
        E += (subs.AU/subs.DAY)**2*rvmi.m*rvmi.v.sqnorm()/2.
        
        # potential energy
        for rvmj in lrvm[i+1:]:
            E -= (subs.GMSUN/subs.AU)*rvmi.m*rvmj.m/(rvmj.r-rvmi.r).norm()    

    # get masses, order so that make m1 > m2
    mstar = lrvm[0].m
    m1    = lrvm[1].m
    m2    = lrvm[2].m
    if m1 < m2:
        ms = m1
        m1 = m2
        m2 = ms

    # Compute beta and betac. Ratio should be > 1 to be "Hill-stable"
    beta  = -2.*(mstar+m1+m2)*L.sqnorm()*E/subs.GMSUN**2/(m1*m2+mstar*m1+mstar*m2)**3

    betac = 1+3.**(4./3.)*m1*m2/mstar**(2./3.)/(m1+m2)**(4./3.)-\
        m1*m2*(11.*m1+7.*m2)/(3.*mstar*(m1+m2)**2)
    return (beta,betac)

it = chain.names.index('ttime')

ns, nu = 0, 0
for vals in chain.vals:
    pd = orbits.fixpar(model, zip(vvar, vals[:len(vvar)]))
    norbit, nephem = orbits.norbeph(pd)
    if norbit != 2:
        print 'hill.py only works for 2 orbits.'
        exit(1)
    beta, betac = hstab(pd)
    if vals[it] > 1.e6:
        ns += 1
    else:
        nu += 1
    ochain.add_line(list(vals) + [beta/betac,])
        
