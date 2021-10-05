#!/usr/bin/env python

import sys
import numpy as np
import trm.orbits as orbits
import matplotlib.pyplot as plt
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s times model [stoerm bary]

plots a model over some times.

Arguments:

times -- file of eclipse times with cycle number, time and error as
         columns. Optionally at the top of the file there can be a line of the
         form '# offset = 1234' (NB the spaces must be there) which is an
         integer that will be subtracted from the cycle numbers in the file to
         reduce the correlation between the T0 and period terms of the
         ephemeris.

model -- an initial model file with names of parameters, equals sign, value
         and v or f for variable or fixed. Only some parameter names are
         possible. They are t0, period, quad, a#, mass#, e#, epoch#, lperi#,
         iangle#, Omega#, eperi#, eomega# where # = integer and all with a given # must
         be present. Example (take care to separate with spaces):

         t0 = 55000.3 v
         period = 0.13 v
         a1 = 12.3 f
         mass = 1.5 v
         integ = 1 f
         .
         .
         etc

qrem   -- remove quadratic term of binary ephemeris (or not)

stoerm -- use Stoermer's rule for the integration or not. Stoermer's rule, which applies
          for 2nd-order conservative equations, is twice as fast as the alternative which 
          is maintained for comparison.

"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('times',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('model',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('qrem',   inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('stoerm', inp.Input.LOCAL, inp.Input.HIDE)

# load the data
times = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))
cycle,time,etime,offset,soften = orbits.load_times(times)

mfile = inpt.get_value('model',  'initial timing model', Fname('nnser', '.mod'))
head, vvar, pars, model = orbits.read_model(mfile)     

qrem  = inpt.get_value('qrem', "remove quadratic term of binary ephemeris", True)

inpt.set_default('stoerm', True)
stoerm  = inpt.get_value('stoerm', "use Stoermer's rule for integration?", True)

try:
    orbits.check_tpar(model)
except Exception, err:
    print 'check_tpar problem:',err
    exit(1)

pdict  = orbits.fixpar(model)

t0     = pdict['t0']
period = pdict['period']
if qrem and 'quad' in pdict:
    quad = pdict['quad']
else:
    quad = 0.

# plot data plus fits relative to linear part of ephemeris
ax = plt.subplot(2,1,1)
plt.errorbar(cycle+offset, subs.DAY*(time - orbits.ephmod(cycle,t0,period,quad)), 
             subs.DAY*etime, fmt='og', capsize=0)
fcycle = np.linspace(cycle.min(),cycle.max(),400)
pred   = orbits.etimes(pdict, fcycle, stoerm=stoerm) - orbits.ephmod(fcycle,t0,period,quad)

plt.plot(fcycle+offset, subs.DAY*pred, 'r--')
plt.xlim(cycle.min()+offset-100., cycle.max()+offset+100.)

# plot residuals
plt.subplot(2,1,2,sharex=ax)
pred = orbits.etimes(pdict, cycle, stoerm=stoerm)
plt.errorbar(cycle+offset, subs.DAY*(time - pred), subs.DAY*etime, fmt='og', capsize=0)
plt.xlim(cycle.min()+offset-100., cycle.max()+offset+100.)

chisq = (((time-pred)/etime)**2).sum()
sumw  = (1/etime**2).sum()
nvar  = len(vvar)
ndata = len(etime)
print 'Chisq =',chisq,', residuals =',np.sqrt(ndata/(ndata-nvar)*chisq/sumw)

plt.show()
