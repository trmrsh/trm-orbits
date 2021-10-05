#!/usr/bin/env python

import sys
import numpy as np
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s rvs prior log append (cmin model covar sfac nstore) plot ntrial best

fits radial velocities

Arguments:

rvs    -- file of radial velocities with time, velocity and uncertainty as columns. 

sigma  -- error to add in quadrature to input errors (sometimes useful if one suspects something like
          slit filling errors are the limiting factor)

smfac  -- multiplicative factor to apply to errors, applied after sigma.

prior  -- python file which implements a user-defined prior. This file, if specified, must contain a 
          function called 'prior' which returns ln(prior probability) [this is in addition to an internal 
          Jeffries prior on the orbital separations i.e. 1/a]. The function is fed the parameter names  
          and values via a dictionary, its only argument. The essential use of this is to control 
          unconstrained problems where to get any sort of sensible answer you have to impose arbitrary 
          constraints on parameters. You should normally try not to use it, but sometimes you won't get 
          a result without it. 

log     -- name of log file to store MCMC results in. 

append  -- True to append an old file as opposed to creating one from scratch

If not append:

  cmin    -- minimum chi**2 achievable. Needed to define the jump probability by effectively 
             scaling the uncertainties so that the reduced chi**2 = 1. You need to experiment 
             before knowing what this should be typically.

  model   -- an initial model file with names of parameters, equals sign, value and v or f for 
             variable or fixed. Only some parameter names are possible. They are t0, period, 
             quad, a#, period#, e#, epoch#, lperi#, iangle#, Omega#, type# where # = integer 
             and all with a given # must be present. Example (take care to separate with spaces):

             t0 = 55000.3 v
             period = 0.13 v
             a = 12.3 f
             gamma = 0. v
             Omega = 0. f
             iangle = 90. f
             ex = 0.1 v
             ey = 0.0 v
             type = 1 f

  covar   -- Covariance file. Use genlmq.py to start, covar.py once you have a log file to work from. This is used
             to define the MCMC jumps.

  sfac    -- Scale factor to scale jump sizes from raw values. Needs experimentation; should be aiming for ~25%
             acceptance.

  nstore  -- how often to store models. Not worth storing more often than correlation length.

plot    -- make a plot of the best fit so far (or not) [this could just be your initial model]

ntrial  -- number of trials

best    -- name of file for best model

"""

# print help
if len(sys.argv) == 2 and sys.argv[1] == '-h':
    print usage % (sys.argv[0][sys.argv[0].rfind('/')+1:],)
    exit(0)

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

# register parameters
inpt.register('rvs',     inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('sigma',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('smfac',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('prior',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('log',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('append',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('cmin',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('model',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('covar',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('sfac',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nstore',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('plot',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('ntrial',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('best',    inp.Input.LOCAL, inp.Input.PROMPT)

rvs     = inpt.get_value('rvs', 'file of radial velocities', Fname('pg1018', '.dat'))
sigma   = inpt.get_value('sigma', 'uncertainty to add in quadrature', 0., 0.)
smfac   = inpt.get_value('smfac', 'multiplicative error correction factor', 1., 0.)
try:
    prior = inpt.get_value('prior',  'python code defining prior constraints on parameters', Fname('pg1018', '.py'))  
    sys.path.insert(0,'.')
    Prior = __import__(prior[:-3])
    prior = Prior.prior
except inp.InputError:
    prior = None
    print 'No prior will be applied.'

log       = inpt.get_value('log',  'log file for storage of mcmc output', Fname('pg1018', '.log', Fname.NEW))
append    = log.exists() and inpt.get_value('append',  'append output to this file (otherwise overwrite)?', True)

if not append:
    cmin    = inpt.get_value('cmin',   'expected minimum chi**2', 100., 0.)
    mfile   = inpt.get_value('model',  'initial timing model', Fname('pg1018', '.mod'))
    covar   = inpt.get_value('covar',  'covariance file', Fname('pg1018', '.lmq'))
    sfac    = inpt.get_value('sfac',  'jump scale factor', 0.5, 0.)
    nstore  = inpt.get_value('nstore', 'how often to store results', 10, 1)

plot   = inpt.get_value('plot',  'do you want to plot the initial fit?', True)
ntrial = inpt.get_value('ntrial', 'number of trials to carry out', 100000, 1)
best   = inpt.get_value('best', 'name of file for storing best model', Fname('nnser_best', '.mod', Fname.NEW))

# OK, now do some work.

# load data
data    = np.loadtxt(rvs)
times   = data[:,0]
vels    = data[:,1]
verrs   = smfac*np.sqrt(data[:,2]**2 + sigma*sigma)
verrs[data[:,2] <= 0] *= -1

if not append:

    # Read and check the model
    (head, vvar, pars, model) = orbits.read_model(mfile)
     
    try:
        orbits.check_rvpar(model)
    except Exception, err:
        print err
        exit(1)

    pinit = orbits.fixpar(model)
    pbest = pinit.copy()

    # Read file of covariances
    jump = mcmc.Jump(covar)
    jump.trim(vvar)
    jump *= sfac

    csq    = orbits.rvchisq(pinit, times, vels, verrs)
    cbest  = csq
    chain  = mcmc.Fchain(model, nstore, jump, ['chisq','lprior','lpost'], csq, log)
    #  must recover names in right order
    vvar   = chain.names[:-3]

else:

    # An MCMC log file to append to has been specified. Will first use this to set the model and jump distribution. 
    head = ['# Best model after MCMC run.\n']

    chain = mcmc.Fchain(log)
    pars  = chain.model.keys()
    print 'read chain'
    if chain.vals is None or len(chain.vals) == 0 or len(chain.vals.shape) != 2 or chain.vals.shape[1] < 1:
        print 'File =',log,'contains no results to start from!'
        exit(1)

    # Use model read in from header to define the starting model,
    # taking the last line of the file to update the variable parameters.
    # Very last parameter (chi**2) excluded in all cases.
    vvar   = chain.names[:-3]
    pinit  = orbits.fixpar(chain.model, zip(vvar, chain.vals[-1,:-3]))
    cmin   = chain.chmin
    nstore = chain.nstore
    csq    = chain.vals[-1,-3]
    chisqs = chain.vals[:,-3]
    cbest  = chisqs.min()
    pbest  = orbits.fixpar(chain.model, zip(vvar, chain.vals[chisqs == cbest,:-3][0]))
    jump   = chain.jump

# Reduced chi**2
cred = cmin/(len(times)-len(vvar))

print 'Initial chi**2 =',csq

# plot fit.
if plot:
    import pylab
    print 'Plotting best model, chi**2 =',cbest

    pylab.subplot(211)
    pylab.errorbar(times, vels, verrs, fmt='og', capsize=0)
    trange = times.max()-times.min()
    t1, t2 = times.min()-trange/10., times.max()+trange/10.
    ftime = np.linspace(t1, t2, 200)
    pred   = orbits.rvp(pbest, ftime)
    pylab.plot(ftime, pred, 'r--')

    pylab.subplot(2,1,2)
    pred   = orbits.rvp(pbest, times)
    pylab.errorbar(times, vels-pred, verrs, fmt='og', capsize=0)
    pylab.xlim(t1,t2)
    print 'Kill plot window to proceed'
    pylab.show()

# Now to MCMC things.
cinit = csq
p     = pinit.copy()
nbest = 0

# ln(prior)
if prior is None:
    lpold = orbits.lprior(pinit)
else:
    lpold = orbits.lprior(pinit) + prior(pinit)

n, nchange = 0, 0

while n < ntrial:
    n      += 1

    ptest  = orbits.mod_rvpar(jump.trial(p))
    if orbits.check_rvmod(ptest):

        if prior is None:
            lptest = orbits.lprior(ptest)
        else:
            lptest = orbits.lprior(ptest) + prior(ptest)


        ctest  = orbits.rvchisq(ptest, times, vels, verrs)

        if np.random.uniform() < np.exp(lptest - lpold - (ctest-csq)/cred/2.):
            p         = ptest
            csq       = ctest
            lpold     = lptest
            nchange  += 1

        if csq < cbest:
            cbest = csq
            pbest = p.copy()
            print '\nNew best chi**2 =',cbest
            for name in vvar:
                print 'Parameter: %-10s = %24.15f' % (name,pbest[name])

    if n % 10000 == 0: print 'Reached',n,'acceptance rate =',nchange/float(max(1,n))

    if n % nstore == 0: 
        vals = [p[vi] for vi in vvar] + [csq,lpold,csq-2.*lpold]
        chain.add_line(vals)

print '\nBest chi**2 =',cbest

# Write out best model
fmo = open(best, 'w')
for h in head:
    fmo.write(h)
fmo.write('\n')
for par in pars:
    if par in vvar:
        fmo.write(par + ' = ' + str(pbest[par]) + ' v\n')
    else:
        fmo.write(par + ' = ' + str(pbest[par]) + ' f\n')
fmo.close()
