#!/usr/bin/env python

import sys
import numpy as np
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

usage = """
usage: %s times prior log [nskip] append (cmin model covar sfac nstore) plot [emax] ntrial best

fits eclipse times with a model of an ephemeris plus deviations caused by third bodies.
The program needs to be adapted for any given system; this version is set for the NN Ser
times listed by Beuermann et al (2010). It starts from a model that needs to be defined
by the user. It then minimises a set of user defined parameters first using the simplex 
method and then Levenburg-Marquardt. Finally an MCMC run takes place

Arguments:

times   -- file of eclipse times with cycle number, time and error as columns. Optionally
           at the top of the file there can be a line of the form '# offset = 1234' (NB
           the spaces must be there) which is an integer that will be subtracted from 
           the cycle numbers in the file to reduce the correlation between the T0 and 
           period terms of the ephemeris.

prior  -- python file which implements a user-defined prior. This file, if specified, must contain a function called 'prior'  
          which returns ln(prior probability) [this is in addition to an internal Jeffries prior on the orbital separations i.e. 1/a]. 
          The function is fed the parameter names and values via a dictionary, its only argument. The essential use of this is
          to control unconstrained problems where to get any sort of sensible answer you have to impose arbitrary constraints
          on parameters. You should normally try not to use it, but sometimes you won't get a result without it. 

log     -- name of log file to store MCMC results in. 

nskip   -- number of point groups in file to skip. Horrible attempted workaround for excess memory usage
           on loading old log file. Defaults to zeo if not explicitly set.
 
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
             a1 = 12.3 f
             period1 = 5600. v
             .
             .
             etc

  covar   -- Covariance file. Use genlmq.py to start, covar.py once you have a log file to work from. This is used
             to define the MCMC jumps.

  sfac    -- Scale factor to scale jump sizes from raw values. Needs experimentation; should be aiming for ~25%
             acceptance.

  nstore  -- how often to store models. Not worth storing more often than correlation length.

plot    -- make a plot of the best fit so far (or not) [this could just be your initial model]

emax    -- maximum error in seconds for residuals plot

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
inpt.register('times',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('prior',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('log',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nskip',   inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('append',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('cmin',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('model',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('covar',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('sfac',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nstore',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('plot',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('emax',    inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('ntrial',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('best',    inp.Input.LOCAL, inp.Input.PROMPT)

times     = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))
try:
    prior = inpt.get_value('prior',  'python code defining prior constraints on parameters', Fname('ippeg', '.py'))  
    sys.path.insert(0,'.')
    Prior = __import__(prior[:-3])
    prior = Prior.prior
except inp.InputError:
    prior = None
    print 'No prior will be applied.'

log       = inpt.get_value('log',  'log file for storage of mcmc output', Fname('nnser', '.log', Fname.NEW))
inpt.set_default('nskip',0)
nskip     = inpt.get_value('nskip', 'number of point groups to skip', 0, 0)
append    = log.exists() and inpt.get_value('append',  'append output to this file (otherwise overwrite)?', True)

if not append:
    cmin    = inpt.get_value('cmin',   'expected minimum chi**2', 100., 0.)
    mfile   = inpt.get_value('model',  'initial timing model', Fname('nnser', '.mod'))
    covar   = inpt.get_value('covar',  'covariance file', Fname('nnser', '.lmq'))
    sfac    = inpt.get_value('sfac',  'jump scale factor', 0.5, 0.)
    nstore  = inpt.get_value('nstore', 'how often to store results', 10, 1)

plot   = inpt.get_value('plot', 'do you want to plot the initial fit?', True)
emax   = inpt.get_value('emax', 'maximum errorbar for point to appear in residuals plot (seconds)', 10.)
ntrial = inpt.get_value('ntrial', 'number of trials to carry out', 100000, 1)
best   = inpt.get_value('best', 'name of file for storing best model', Fname('nnser_best', '.mod', Fname.NEW))

# OK, now do some work.

# load data
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

if not append:

    # Read and check the model
    (head, vvar, pars, model) = orbits.read_model(mfile)
     
    try:
        orbits.check_tpar(model)
    except Exception, err:
        print 'check_tpar problem:',err
        exit(1)

    pinit = orbits.fixpar(model)
    pbest = pinit.copy()

    # Read file of covariances
    jump = mcmc.Jump(covar)
    jump *= sfac

    csq    = orbits.tchisq(pinit, cycle, time, etime)

    cbest  = csq
    chain  = mcmc.Fchain(model, nstore, jump, ['chisq','lprior','lpost'], cmin, log)

    #  must recover names in right order
    vvar   = chain.names[:-3]

    # cut down covariances to match model
    jump.trim(vvar)

else:

    # An MCMC log file to append to has been specified. Will first use this to set the model and jump distribution. 
    head = ['# Best model after MCMC run.\n']

    chain = mcmc.Fchain(log, nskip)
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
cred = cmin/(len(time)-len(vvar))

print 'Initial chi**2 =',csq

# plot fit.
if plot:
    import matplotlib.pyplot as plt
    print 'Plotting best model, chi**2 =',cbest
    SEC_DAY = subs.DAY
    
    t0     = pbest['t0']
    period = pbest['period']

    plt.subplot(2,1,1)
    plt.errorbar(cycle+offset, SEC_DAY*(time - orbits.ephmod(cycle,t0,period)), SEC_DAY*etime, fmt='og', capsize=0)
    fcycle = np.linspace(cycle.min(),cycle.max(),400)
    pred   = orbits.etimes(pbest, fcycle) - orbits.ephmod(fcycle,t0,period)

    plt.plot(fcycle+offset, SEC_DAY*pred, 'r--')
    plt.xlim(cycle.min()+offset-1000., cycle.max()+offset+10000.)
    plt.subplot(2,1,2)
    ok = SEC_DAY*etime < emax
    plt.errorbar(cycle[ok]+offset, SEC_DAY*(time[ok] - orbits.etimes(pbest,cycle[ok])), SEC_DAY*etime[ok], fmt='og', capsize=0)
    plt.xlim(cycle.min()+offset-1000., cycle.max()+offset+10000.)
    print 'Kill plot window to proceed'
    plt.show()


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

    ptest  = orbits.mod_tpar(jump.trial(p))
    if orbits.check_tmod(ptest):

        if prior is None:
            lptest = orbits.lprior(ptest)
        else:
            lptest = orbits.lprior(ptest) + prior(ptest)

        try:

            ctest  = orbits.tchisq(ptest, cycle, time, etime)

            if np.random.uniform() < np.exp(lptest - lpold - (ctest-csq)/cred/2.):
                p         = ptest
                csq       = ctest
                lpold     = lptest
                nchange  += 1

            if csq < cbest:
                cbest = csq
                pbest = p
                print '\nNew best chi**2 =',cbest
                for name in vvar:
                    print 'Parameter: %-10s = %24.15f' % (name,pbest[name])

        except RuntimeError, err:
            print 'RuntimeError trapped:',err

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
