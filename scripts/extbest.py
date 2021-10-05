#!/usr/bin/env python

import sys, re
import numpy as np
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s log best method

Extracts best model from a timing orbit log file. This can then be used for re-starting a run
for example.

log     -- name of log file with MCMC results. 
best    -- name of output model.
method  -- 'c' for minimum chi**2, 'p' for maximum posterior probability.
"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('log',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('best',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('method',  inp.Input.LOCAL, inp.Input.PROMPT)

log    = inpt.get_value('log',  'log file containing MCMC output', Fname('nnser', '.log', Fname.OLD))
best   = inpt.get_value('best',  'file for output best retrieved model', Fname('nnser', '.mod', Fname.NEW))
method = inpt.get_value('method',  "method to use ('c' for min chi**2, 'p' for max posterior prob)", 'p', \
                            lvals=['c','p'])

# read chain
chain = mcmc.Fchain(log)

# get variable parameter names
vpars   = chain.vpars()

# extract best model values
if method == 'c':
    bestrow = chain.vals[chain.chisq().argmin(),:]
else:
    bestrow = chain.vals[chain.get('lpost').argmax(),:]

# Create best model dictionary bu updating model with best values
mbest = orbits.fixpar(chain.model, zip(vpars, bestrow[:chain.nvpars()]))

# Write out best model
fmo = open(best, 'w')

# Some header information
fmo.write('# Best model extracted from ' + log + '\n')
if method == 'c':
    fmo.write('# Method used = minimum chi**2\n')
else:
    fmo.write('# Method used = maximum posterior probability\n')

cbest = bestrow[chain.names.index('chisq')]
pbest = bestrow[chain.names.index('lpost')]

fmo.write('#\n# chi**2 = ' + str(cbest) + ', ln(posterior prob) = ' +  str(pbest) + '\n#\n')

# Now write out the model
for pname,value in mbest.iteritems():
    if pname == 'coord':
        fmo.write('%s = %s f\n' % (pname,value))
    elif pname.startswith('eperi') or pname.startswith('eomega') or pname == 'integ':
        fmo.write('%s = %d f\n' % (pname,value))
    else:
        if pname in vpars:
            fmo.write('%s = %.14e v\n' % (pname,value))
        else:
            fmo.write('%s = %.14e f\n' % (pname,value))

fmo.close()

print 'Best model which has chi**2 =',cbest,', ln(post prob) =',pbest,'written to',best

if method == 'c':
    print 'Model number',chain.chisq().argmin()+1,'saved.'
else:
    print 'Model number',chain.get('lpost').argmax()+1,'saved.'

