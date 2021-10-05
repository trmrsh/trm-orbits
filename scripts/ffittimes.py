#!/usr/bin/env python

import sys
import numpy as np
import emcee
import trm.orbits as orbits
import trm.mcmc as mcmc
import trm.subs as subs
import trm.subs.input as inp
from   trm.subs import Fname

# make print statements flush immediately for better
# monitoring of often long runs.

usage = """
usage: %s times prior log [nskip] append (cmin model method (nwalker stretch [nsub]) 
covar sfac nstore)) (nthread) ntrial [beta] repacc plot [emax wbest] 

fits eclipse times with a model of an ephemeris plus deviations caused by
third bodies.  This uses affine-invariant MCMC sampling provided vy the emcee
package. This is fancy, hence the extra 'f' at the start.

Arguments:

times -- file of eclipse times with cycle number, time and error as
         columns. Optionally at the top of the file there can be a line of the
         form '# offset = 1234' (NB the spaces must be there) which is an
         integer that will be subtracted from the cycle numbers in the file to
         reduce the correlation between the T0 and period terms of the
         ephemeris.

prior -- python file which implements a user-defined prior. This file, if
         specified, must contain a function called 'prior' which returns
         ln(prior probability) [this is in addition to an internal Jeffries
         prior on the orbital separations i.e. 1/a].  The function is fed the
         parameter names and values via a dictionary, its only argument. The
         essential use of this is to control unconstrained problems where to
         get any sort of sensible answer you have to impose arbitrary
         constraints on parameters. You should normally try not to use it, but
         sometimes you won't get a result without it.

log   -- name of log file to store MCMC results in. 

nskip -- number of point groups in file to skip. This is here to avoid memory
         issues when using multiple threads in batch mode. Defaults to zero if 
         not explicitly set. Can be set negative in which case it indicates the
         number of groups to keep. "-1" is probably the most useful value other 
         than 0. The reason for not having it permanently set to -1 is in order 
         to increase the chances of hitting a low chi**2.

append  -- True to append an old file as opposed to creating one from scratch


if append:

  cmin    -- minimum chi**2 achievable. Needed to define the jump probability by effectively 
             scaling the uncertainties so that the reduced chi**2 = 1. You need to experiment 
             before knowing what this should be typically.

  model --   an initial model file with names of parameters, equals sign, value
             and v or f for variable or fixed. Only some parameter names are
             possible. They are t0, period, quad, a#, period#, e#, epoch#,
             lperi#, iangle#, Omega#, type# where # = integer and all with a
             given # must be present. Example (take care to separate with
             spaces):

             t0 = 55000.3 v
             period = 0.13 v
             a1 = 12.3 f
             period1 = 5600. v
             mass = 1.5 v
             integ = 1 f
             .
             .
             etc

  method -- 'm' uses standard Metropolis-Hastings gaussian jumps while 'a' uses the affine method.

  if method == 'a':

    nwalker -- This is for the affine MCMC. Its almost as if nwalker chains
               are started. This must be >= 2*nvar where nvar is the number of
               variable parameters, but typically quite a bit bigger still. It
               must be a multiple of 2.

    stretch -- stretch factor. Usual value 2. Large reduces acceptance but may
               improve coverage of parameter space.

    nsub    -- number of chunks to divide each nstore cycle into to reduce memory problems. 
               This also is related to the parameter "dbeta" below.

  covar --   Covariance file. If method='m' this is used the generate the
             jumps. If method='a' it is used initially to define the walkers,
             but it used no more after this.

  sfac --    Scale factor to scale jump sizes from raw values. In M-H case you
             need to adjust it until the acceptance rate is around 25%. In the
             affine case it should be fairly small just to generate the
             initial set of walkers.

  nstore  -- how often to store models. Not worth storing more often than correlation length.

if method == 'a':

  nthread  -- number of threads for parallel processing.

ntrial  -- number of trials

beta    -- annealing factor. If you want to explore for other minima set this to
           a small value. Range 0 to 1. This multiplies the log(post)
           value. "Annealing schedules" have to be applied by hand.  as a
           series of increases in beta. This must be specifically set and
           will always default to 1.

dbeta   -- amount to increase beta after each of the nsub chunks of a store cycle. 
           Defaults to 0. If set then beta is written out at the end of each store
           cycle as well as the values, while it is less than 1. This allows you to 
           implement a crude "annealing schedule". By making it change for each of 
           nsub chunks, the change is smoother overall. This value should not be too 
           large because it carries with it some risk of leaving poor models stranded.
           The use of dbeta combined with nsub adds some overhead since the posterior
           probabilities of all walkers must be re-computed for each new beta which
           happens on nsub occasions whereas they can be recycled if debeta = 0.

repacc  -- True/False to report acceptance fraction

plot    -- make a plot of the best fit so far (or not) [this could just be your initial model]

emax    -- maximum error in seconds for residuals plot

wbest   -- True to write of walker when the model improves
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
inpt.register('method',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nwalker', inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('stretch', inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nsub',    inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('covar',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('sfac',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nstore',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nthread', inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('ntrial',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('beta',    inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('dbeta',   inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('repacc',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('plot',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('emax',    inp.Input.LOCAL, inp.Input.HIDE)
inpt.register('wbest',   inp.Input.LOCAL, inp.Input.HIDE)

# load the data
times   = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))

# load the data
cycle,time,etime,offset,soften = orbits.load_times(times)

# prior or not
try:
    prior = inpt.get_value('prior',  'python code defining prior constraints on parameters', \
                               Fname('ippeg', '.py'))  
    sys.path.insert(0,'.')
    Prior = __import__(prior[:-3])
    prior = Prior.prior
except inp.InputError:
    prior = None
    print 'No prior will be applied.'

log     = inpt.get_value('log',  'log file for storage of mcmc output', \
                             Fname('nnser', '.log', Fname.NEW))
inpt.set_default('nskip',0)
nskip     = inpt.get_value('nskip', 'number of point groups to skip', 0)
append    = log.exists() and \
    inpt.get_value('append',  'append output to this file (otherwise overwrite)?', True)

if not append:
    cmin    = inpt.get_value('cmin',   'expected minimum chi**2', 100., 0.)
    mfile   = inpt.get_value('model',  'initial timing model', Fname('nnser', '.mod'))
    head, vvar, pars, model = orbits.read_model(mfile)     
    
    try:
        orbits.check_tpar(model)
    except Exception, err:
        print 'check_tpar problem:',err
        exit(1)
    pinit = orbits.fixpar(model)
    pbest = pinit.copy()

    method  = inpt.get_value('method',   'method to use, "m" for Metroplis or "a" for affine', \
                             'm', lvals=['a','m'])

    if method == 'a':
        nwalker = inpt.get_value('nwalker', 'number of walkers', max(1000, 2*len(vvar)), \
                                     max(1,2*len(vvar)), multipleof=2)
        stretch = inpt.get_value('stretch', 'stretch factor', 2., 1.)
        nsub    = inpt.get_value('nsub', 'number of substeps per store cycle (to save memory)', 10, 1)
    else:
        nwalker = 1
        stretch = 0.
        nsub    = 1

    covar   = inpt.get_value('covar',  'covariance file', Fname('nnser', '.lmq'))
    sfac    = inpt.get_value('sfac',  'jump scale factor', 0.5, 0.)
    jump    = mcmc.Jump(covar)
    jump.trim(vvar)
    jump   *= sfac
    nstore  = inpt.get_value('nstore', 'how often to store results', 10, 1, multipleof=nsub)

    # OK all inputs done with
    chain  = mcmc.Fchain(model, nstore, method, jump, nwalker, stretch, \
                             ['chisq','lprior','lpost'], cmin, log)

    # recover names in right order
    vvar   = chain.names[:-3]

    # cut down covariances to match model
    jump.trim(vvar)

else:

    chain = mcmc.Fchain(log, nskip)
    pars  = chain.model.keys()
    print 'Read',len(chain),'points in from',log
    if chain.vals is None or len(chain.vals) == 0 or \
            len(chain.vals.shape) != 2 or chain.vals.shape[1] < 1:
        print 'File =',log,'contains no results to start from!'
        exit(1)

    # Use model read in from header to define the starting model,
    # taking the last line of the file to update the variable parameters.
    # last three parameters (chi**2 etc) excluded in all cases.
    vvar    = chain.vpars()
    pinit   = orbits.fixpar(chain.model, zip(vvar, chain.vals[-1,:chain.nvpars()]))
    model   = chain.model
    nstore  = chain.nstore
    method  = chain.method
    nwalker = chain.nwalker
    stretch = chain.stretch
    cmin    = chain.chmin
    chisqs  = chain.chisq()
    csq     = chisqs[-1]
    print 'Minimum chi**2 model from file =',chain.cmin()
    
    pbest   = orbits.fixpar(chain.model, zip(vvar, chain.vals[chisqs.argmin(),:chain.nvpars()]))
    jump    = chain.jump

    if len(chain) % nwalker != 0:
        print 'The number of lines =',len(chain.vals),'was not a multiple of nwalker =',nwalker
        exit(1)

    if method == 'a':
        nsub    = inpt.get_value('nsub', 'number of substeps per store cycle (to save memory)', 10, 1)
        if nstore % nsub != 0:
            print 'nsub must divide exactly into nstore'
            exit(1)

if method == 'a':
    nthread = inpt.get_value('nthread', 'number of threads', 1, 1, 16)

ntrial  = inpt.get_value('ntrial', 'number of trials to carry out', 10000, max(nstore, 1))
inpt.set_default('beta', 1.)
beta    = inpt.get_value('beta', 'annealing factor', 1., 0., 1.)
inpt.set_default('dbeta', 0.)
dbeta   = inpt.get_value('dbeta', 'increase in annealing factor per sub-chunk', 0., 0., 1.)
repacc  = inpt.get_value('repacc', 'report acceptance fraction?', True)
plot    = inpt.get_value('plot', 'do you want to plot the initial fit?', True)
emax    = inpt.get_value('emax', 'maximum errorbar for point to appear in residuals plot (seconds)', 10.)
inpt.set_default('wbest', False)
wbest   = inpt.get_value('wbest', 'write out walkers with best model?', False)

# a couple of helper functions

def lnprior(p, prior):
    """
    Computes ln(prior prob) given a model dictionary and prior
    function (which can be None)
    """

    # compute ln(prior prob)
    if prior is None:
        lnpri = orbits.lprior(p)
    else:
        lnpri = orbits.lprior(p) + prior(p)
    return lnpri

def lnpost(p, prior, cycle, time, etime, cmin):
    """
    Computes ln(posterior prob) given a model dictionary, a prior
    function (which can be None), data and the minimum chi**2 expected
    for error bar re-scaling. Returns: ln(post), ln(prior), chisq. If
    anything fails comes back with -Inf,0,Inf
    """

    if not orbits.check_tmod(p):
        return (float('-Inf'),0.,float('Inf'))

    # compute ln(prior prob)
    lnpri = lnprior(p, prior)

    # compute chisquared (= -2*ln(likelihood))
    try:
        chisq = orbits.tchisq(p, cycle, time, etime)

        # scale chisq according to the preset cmin value
        cscale = ((len(time)-len(vvar))/cmin)*chisq

        return (lnpri - cscale/2., lnpri, chisq)
    except RuntimeError, err:
        return (float('-Inf'),0.,float('Inf'))

class Lpprob(object):
    """
    This returns the ln of posterior probability as required by emcee
    in 'function object' form given vector x of variable values. It is 
    assumed that the variables in x run in the same order as in vvar, 
    the names of the variables. 'model' is a dictionary of all parameter 
    values. 'prior' is the prior probability routine. 'cycle', 'time' and 
    'etime' are the data. 'cmin' is the minimum chi-squared predicted which
    is used to re-scale error bars. Any failures are converted into very 
    low probabilities.
    """

    def __init__(self, pinit, vvar, prior, cycle, time, etime, cmin, method, beta):
        self.p       = pinit
        self.vvar    = vvar
        self.prior   = prior
        self.cycle   = cycle
        self.time    = time
        self.etime   = etime
        self.cmin    = cmin
        self.method  = method
        self.beta    = beta

    def __call__(self, x):
        if method == 'a':
            # update model dictionary
            self.p.update(dict(zip(self.vvar, x)))
            lnpo, lnpri, chisq = lnpost(self.p, self.prior, self.cycle, self.time, self.etime, self.cmin)
        else:
            lnpo, lnpri, chisq = lnpost(x, self.prior, self.cycle, self.time, self.etime, self.cmin)
        return self.beta*lnpo


# plot fit.
if plot:
    import matplotlib.pyplot as plt

    lpost, lpri, chisq = lnpost(pbest, prior, cycle, time, etime, cmin)

    print 'Plotting model with chi**2 =',chisq
    SEC_DAY = subs.DAY
    
    t0     = pbest['t0']
    period = pbest['period']

    plt.subplot(211)
    plt.errorbar(cycle+offset, SEC_DAY*(time - orbits.ephmod(cycle,t0,period)), SEC_DAY*etime, fmt='og', capsize=0)
    fcycle = np.linspace(cycle.min(),cycle.max(),400)
    pred   = orbits.etimes(pbest, fcycle) - orbits.ephmod(fcycle,t0,period)

    plt.plot(fcycle+offset, SEC_DAY*pred, 'r--')
    plt.xlim(cycle.min()+offset, cycle.max()+offset)

    plt.subplot(212)
    ok = SEC_DAY*etime < emax
    plt.errorbar(cycle[ok]+offset, SEC_DAY*(time[ok] - orbits.etimes(pbest,cycle[ok])), \
                     SEC_DAY*etime[ok], fmt='og', capsize=0)
    plt.xlim(cycle.min()+offset, cycle.max()+offset)

    print 'Kill plot window to proceed'
    plt.show()


# create the posterior probability function
lnprob = Lpprob(pinit, vvar, prior, cycle, time, etime, cmin, method, beta)

if method == 'a':

    chain.add_info('# beta = ' + str(beta) + '\n')

    if chain.vals is None or not len(chain.vals):
        # Generate the walkers. We use the jump distribution 
        # to make a cloud of (ok) vectors
        walkers = []
        for i in xrange(nwalker):
            ok = False
            while not ok:
                ptest = jump.trial(pinit)
                ok    = orbits.check_tmod(ptest)
            walkers.append(np.array([ptest[vnam] for vnam in vvar]))
            print 'walker i --> ',lnprob(walkers[-1])
    else:
        # Generate walkers from the file just read
        walkers = chain.vals[-nwalker:,:-3]

    sampler = emcee.EnsembleSampler(nwalker, len(vvar), lnprob,
                                    a=stretch, threads=nthread)

    lnpmax = -1.e30

    # Do the hard work
    lnps = None
    rs   = None
    ntot = 0
    for i in xrange(ntrial/nstore):

        if i > 0 and beta < 1.0 and dbeta > 0.:
            chain.add_info('# beta = ' + str(beta) + '\n')

        # go through in steps to reduce memory requirement
        mrate = 0.
        for n in xrange(nsub):
            # When beta changes, we must recompute the log(posterior) probs 
            # for all walkers to make sure that we are comparing like with 
            # when carrying out the MCMC stuff. This adds a bit of an overhead
            # but hopefully not too large if nsub is big enough
            if dbeta > 0.:
                lnps = None
            walkers, lnps, rs = sampler.run_mcmc(walkers, nstore // nsub,
                                                 rs, lnps)

            ntot  += nwalker * (nstore // nsub)
            mrate += sampler.acceptance_fraction.mean()

            # check to see if we have improved the model
            improve = False
            for walker, lnp  in zip(walkers, lnps):
                p  = orbits.fixpar(model, zip(vvar, walker))
                if lnp/beta > lnpmax:
                    pbest  = orbits.fixpar(model, zip(vvar, walker))
                    lnpmax = lnp/beta
                    lnpri  = lnprior(pbest, prior)
                    chisq  = 2.*cmin*(lnpri - lnp/beta)/(len(time)-len(vvar))
                    print 'New best model has ln(post) =',lnp/beta,', chi**2 =',chisq
                    improve = True

            if improve and wbest:
                # Write the current position to a file, one line per walker, computing the
                # chisq and lnpriors
                for walker, lnp  in zip(walkers, lnps):
                    p     = orbits.fixpar(model, zip(vvar, walker))
                    lnpri = lnprior(p, prior)
                    lnpt  = lnp/beta
                    chisq = 2.*cmin*(lnpri - lnpt)/(len(time)-len(vvar))
                    chain.add_line(np.concatenate((walker, [chisq, lnpri, lnpt])),nomem=True)


            sampler.reset()

            # update beta if need be. Avoid updating on the last step to ensure that
            # the beta is correct for the output step that comes soon 
            if n < nsub-1 and beta < 1.0 and dbeta > 0.:
                beta += dbeta
                beta  = min(1.0, beta)
                lnprob.beta = beta


        if repacc:
            mrate /= nsub
            print 'Group',i+1,'acceptance rate =',mrate
            sys.stdout.flush()

        if not wbest:
            # Write the current position to a file, one line per walker, computing the
            # chisq and lnpriors
            for walker, lnp  in zip(walkers, lnps):
                p     = orbits.fixpar(model, zip(vvar, walker))
                lnpri = lnprior(p, prior)
                lnpt  = lnp/beta
                chisq = 2.*cmin*(lnpri - lnpt)/(len(time)-len(vvar))
                chain.add_line(np.concatenate((walker, [chisq, lnpri, lnpt])),nomem=True)
            
        if beta < 1.0 and dbeta > 0.:
            beta += dbeta
            beta  = min(1.0, beta)
            lnprob.beta = beta

else:

    p = pinit.copy()
    nt, nc, n = 0, 0, 0
    lnprob0 = lnprob(p)

    while n < ntrial:
        n   += 1
        nt  += 1   

        ptest  = orbits.mod_tpar(jump.trial(p))
        if orbits.check_tmod(ptest):
            lnprobt = lnprob(ptest)

            if lnprobt > lnprob0 or np.random.uniform() < np.exp(lnprobt-lnprob0):
                lnprob0 = lnprobt
                p       = ptest
                nc     += 1

        if n % nstore == 0: 
            if repacc:
                print 'Reached',n,'acceptance rate =',nc/float(nt)
                sys.stdout.flush()
                nt, nc = 0, 0

            lnpri = lnprior(p, prior)
            lnpt  = lnprob0
            chisq = 2.*cmin*(lnpri - lnpt)/(len(time)-len(vvar))
            vals  = [p[vi] for vi in vvar] + [chisq,lnpri,lnpt]
            chain.add_line(vals)

