#!/usr/bin/env python

import sys
import numpy as np
import trm.orbits as orbits
from trm import mcmc, subs
import trm.subs.input as inp
from trm.subs import Fname

usage = """
usage: %s model otimes ntimes errors [stoerm bary]

generates timing data given a model

Arguments:

model -- an initial model file with names of parameters, equals sign, value
         and v or f for variable or fixed. Only some parameter names are
         possible. They are t0, period, quad, a#, period#, e#, epoch#, lperi#,
         iangle#, Omega#, type# where # = integer and all with a given # must
         be present. Example (take care to separate with spaces):

         t0 = 55000.3 v
         period = 0.13 v
         a1 = 12.3 f
         period1 = 5600. v
         mass = 1.5 v
         integ = 1 f
         .
         .
         etc

otimes -- template file of eclipse times with cycle number, time and error as
         columns. Optionally at the top of the file there can be a line of the
         form '# offset = 1234' (NB the spaces must be there) which is an
         integer that will be subtracted from the cycle numbers in the file to
         reduce the correlation between the T0 and period terms of the
         ephemeris. Th

ntimes -- new file to generate

errors -- True to add uncertainties

stoerm -- use Stoermer's rule for the integration or not. Stoermer's rule, which applies
          for 2nd-order conservative equations, is twice as fast as the alternative which 
          is maintained for comparison.

bary   -- integrate using barycentric coordinates or not. The alternatives are astrocentric.
          They should give the same answer (its an error if they don't) and the option is just
          for test purposes.
"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('model',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('otimes',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('ntimes',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('errors',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('stoerm',  inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('bary',    inp.Input.LOCAL, inp.Input.HIDE)

mfile = inpt.get_value('model',  'initial timing model', Fname('nnser', '.mod'))
head, vvar, pars, model = orbits.read_model(mfile)     

try:
    orbits.check_tpar(model)
except Exception, err:
    print 'check_tpar problem:',err
    exit(1)

# load the data
otimes = inpt.get_value('otimes', 'old file of eclipses times', Fname('nnser', '.dat'))
times = orbits.Times(otimes)

# output
ntimes = inpt.get_value('ntimes', 'new file of eclipses times', Fname('nnser', '.dat', Fname.NOCLOBBER))

errors = inpt.get_value('errors', 'generate and add uncertainties', True)

inpt.set_default('stoerm', True)
stoerm  = inpt.get_value('stoerm', "use Stoermer's rule for integration?", True)

inpt.set_default('bary', True)
bary  = inpt.get_value('bary', "barycentric (else astrocentric) coordinates?", True)
    
pdict  = orbits.fixpar(model)

# generate data, add errors if wanted, set and write to disk
cycle,tim,terr = times.get()
pred = orbits.etimes(pdict, cycle, stoerm=stoerm, bary=bary)

if errors:
    pred = np.random.normal(pred,terr)

times.set_times(pred)

times.write(ntimes)

print 'Written orbit to',ntimes
