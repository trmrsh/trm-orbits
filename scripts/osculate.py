#!/usr/bin/env python

import sys
import math as m
import numpy as np
import matplotlib.pyplot as plt
from trm import subs, orbits
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s model tmax

Integrates a model to plot osculating parameters (a, e, omega, period).

Arguments:

model   -- orbit model. Elements of planets are plotted, not the central object.

tmax    -- time to integrate for (can be -ve)

nmax    -- maximum number of steps. Can roughly expect 1 step per radian of fastest planet.

nkeep   -- how often to save data (determines size of plotting arrays)

acc     -- accuracy of integration. 1.e-12

efactor -- maximum KE/PE ratio. Integration is stopped if any of the objects exceed this.

rescape -- maximum distance from barycentre in AU. Integration is stopped if any object exceeds
           this.
stoerm  -- use Stoermer's rule for the integration or not. Stoermer's rule, which applies
           for 2nd-order conservative equations, is twice as fast as the alternative which 
           is maintained for comparison.

bary    -- use barycentric or astrocentric coords for integration. Should be no difference in results, but
           this is maintained for testing.
"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('model',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('tmax',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nmax',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nkeep',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('acc',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('efactor', inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('rescape', inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('stoerm',  inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('bary',    inp.Input.GLOBAL, inp.Input.PROMPT)

mfile = inpt.get_value('model',  'timing model', Fname('nnser', '.mod'))
head, vvar, pars, model = orbits.read_model(mfile)     
p     = orbits.fixpar(model)
lrvm  = orbits.ptolrvm(p)

tmax    = inpt.get_value('tmax',  'maximum time to integrate (years)', 10000.)
nmax    = inpt.get_value('nmax',  'maximum number of steps', 1000000, 1)
nkeep   = inpt.get_value('nkeep', 'how often to save data', 1000, 1)
acc     = inpt.get_value('acc', 'accuracy of integrations', 1.e-12, 1.e-20)
efactor = inpt.get_value('efactor', 'energy ratio factor to stop integrations', 1.2,0.5)
rescape = inpt.get_value('rescape', 'escape radius (AU)', 20., 0.)
stoerm  = inpt.get_value('stoerm', "use Stoermer's rule for integration?", True)
bary    = inpt.get_value('bary', "use barycentric (else astrocentric) coords for integration?", True)

reverse = tmax < 0.

# Carry out the integration
ttime, ntest, eratio, npoint, ierr, tnext, nstore, arr = \
    orbits.integrate(lrvm, nmax, abs(tmax), nkeep=nkeep, reverse=reverse, efactor=efactor, rescape=rescape, acc=acc, stoerm=stoerm, bary=bary)

norb, neph = orbits.norbeph(p)

if bary:
    IOFF = 7
else:
    IOFF = 1

    # need to correct the orbits by the C-of-M
    wrv = np.zeros_like(arr[:,:6])
    tmass = p['mass']
    for nob in range(norb):
        ap, mp  = orbits.amthird(p['a'+str(nob+1)], p['mass'], p['period'+str(nob+1)])
        wrv    += mp*arr[:,IOFF+6*nob:IOFF+6*(nob+1)]
        tmass  += mp
    wrv /= tmass
    for nob in range(norb):
        arr[:,IOFF+6*nob:IOFF+6*(nob+1)] -= wrv

print '\n',orbits.trans_code(ntest),'It lasted',ttime,'years.'

ea      = np.empty((norb,nstore))
aa      = np.empty((norb,nstore))
omegaa  = np.empty((norb,nstore))
perioda = np.empty((norb,nstore))

for nob in range(norb):
    off = IOFF+6*nob
    ap, m  = orbits.amthird(p['a'+str(nob+1)], p['mass'], p['period'+str(nob+1)])
    print 'Planet',nob+1,'has a mass of',m,'solar masses and a semi-major axis of',ap
    for i, row in enumerate(arr[:nstore]):
        r = subs.Vec3(row[off],row[off+1],row[off+2])
        v = subs.Vec3(row[off+3],row[off+4],row[off+5])
        a,iangle,e,omega,Omega_a,period,tanom = orbits.pspace2elem(r,v,m,p['mass'])
        ea[nob,i] = e
        aa[nob,i] = a
        omegaa[nob,i] = omega
        perioda[nob,i] = period

plt.subplot(411)
for nob in range(norb):
    plt.plot(arr[:nstore,0], aa[nob,:])
plt.ylabel('a')

plt.subplot(412)
for nob in range(norb):
    plt.plot(arr[:nstore,0], ea[nob,:])
plt.ylabel('e')

plt.subplot(413)
for nob in range(norb):
    plt.plot(arr[:nstore,0], omegaa[nob,:])
plt.ylabel('omega')

plt.subplot(414)
for nob in range(norb):
    plt.plot(arr[:nstore,0], perioda[nob,:])
plt.ylabel('period')

plt.show()
