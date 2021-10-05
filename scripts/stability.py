#!/usr/bin/env python

import sys, signal, os
import math as m
import numpy as np
from multiprocessing import Pool
from trm import subs, mcmc, orbits
import trm.subs.input as inp
from trm.subs import Fname

usage = """
usage: %s inlog outlog tint efactor rescape stoerm nstep acc nprint nthread [nbatch]

Calculates the stability of a set of orbits by integration, using mulitple
threads. It works by attempting to integrate the orbits for a user-defined
time interval and recording how long the orbit actually survives. The orbits
read in are divided amongst multiple threads for speed. It tries as far as
possible to minimise memory requirements by only reading from disk as necessary.

Arguments:
 
 inlog   -- input log file of orbits. This will be used to generate a new log file with the
            same orbits but with additional stability info tacked on.

 outlog  -- output log file of orbits. If it is an old one, it will be appended to (assuming
            that it has the right structure). It will be assumed to be in step with the input
            log so that however many entries it starts with will be skipped in the input file.

 tint    -- total integration time, years, -ve to go backwards in time.

 efactor -- max ratio KE/PE of outgoing most distant object to allow. Integration stopped on reaching
            this. 

 rescape -- max radius from barycentre (in AU) to allow. Integration stops on reaching this.

 stoerm  -- use Stoermer's rule for the integration or not. Stoermer's rule, which applies
            for 2nd-order conservative equations, is twice as fast as the alternative which 
            is maintained for comparison.

 nstep   -- maximum number of steps per integration. This is a safety device to guard against an integration
            taking forever with tiny timesteps. If too many reach this number before getting to tint, you
            probably have a problem.

 acc     -- accuracy parameter for integration. Smaller takes longer. Typical 1.e-12

 nprint  -- how often to report progress.

 nthread -- number of threads.

 nbatch  -- For nthread > 1, this is the number of orbits to treat at a time.
            Since all results of the nbatch orbits need to be stored in memory
            before they are written to disk, this basically allows control over
            the amount of memory being used and how frequently results are 
            written to disk. It should be a number large enough enough that no
            one integration should rival the time taken by the other nbatch-1 or
            some of the threads will be idle.

The end result is a new log file with five columns added which are:

 ttime   -- total time taken, either until tint was reached or a planet was kicked out.

 ntest   -- Flag to say what has caused integration to stop early. 0: nothing, it went all
            the way to tinit or nstep or there was an error (see ierr). Single digit 
            number > 0: object has exceeded preset KE/PE ratio efactor. 2-digit number
            ending with 0: first two digits indicate corresponding objects have collided.
            4 digit number ending 000: first digit shows that corresponding object has
            exceeded rescape.

 eratio  -- KE/PE ratio of outermost object.

 npoint  -- number of steps carried out.

 ierr    -- error flag. 0 = OK, 1 = stepsize underflow in bsstep.

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
inpt.register('tint',    inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('efactor', inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('rescape', inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('stoerm',  inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('nstep',   inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('acc',     inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nprint',  inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nthread', inp.Input.LOCAL, inp.Input.PROMPT)
inpt.register('nbatch',  inp.Input.LOCAL, inp.Input.PROMPT)

inlog   = inpt.get_value('inlog', 'input log file containing orbits', Fname('nnser', '.log', Fname.OLD))
outlog  = inpt.get_value('outlog', 'output log file containing orbits and stability data', Fname('nnser_stab', '.log', Fname.NEW))
tint    = inpt.get_value('tint', 'number of years to integrate orbits', -1000000.)
efactor = inpt.get_value('efactor', 'max KE/PE ratio to allow', 1.2)
rescape = inpt.get_value('rescape', 'max radius to allow (AU)', 15.)
stoerm  = inpt.get_value('stoerm', "use Stoermer's rule for integration?", True)
nstep   = inpt.get_value('nstep', 'maximum number of steps per integration', 2000000, 1)
acc     = inpt.get_value('acc', 'integration accuracy parameter', 1.e-12, 1.e-20)
nprint  = inpt.get_value('nprint', 'how often to report progress', 20, 1)
nthread = inpt.get_value('nthread', 'number of threads', 1, 1)
if nthread > 1:
    nbatch  = inpt.get_value('nbatch', 'batch size', 2000, 1)

# load the input log file as an iterator to save on memory
ichain = mcmc.IChain(inlog)

if os.path.exists(outlog):

    # load an old file without blowing too much memory.
    ochain = mcmc.Fchain(outlog,-1)
    if ochain.vals is None or len(ochain.vals) == 0 or \
            len(ochain.vals.shape) != 2 or ochain.vals.shape[1] < 1:
        print 'File =',outlog,'contains no results to start from!'
        exit(1)
    pars   = ochain.model.keys()
    print 'Read',len(ochain),'points in from',outlog

    if set(ichain.names) > set(ochain.names):
        print 'ERROR: input log file has variables not present in the output file'
        exit(1)

    if 'ttime' not in ochain.names:
        print 'ERROR: Failed to find ttime amongst the parameters'
        exit(1)

else:

    # create from scratch
    ochain = mcmc.Fchain(ichain.model, ichain.nstore, ichain.method, ichain.jump, \
                             ichain.nwalker, ichain.stretch, \
                             ['chisq','lprior','lpost','ttime','ntest','eratio','npoint','ierr'], \
                             ichain.chmin, outlog)

class Stab(object):
    """
    Function object that can operate on a model in the form of a vector
    of values to then carry out N-body integration. Basically store all
    the parameters needed as attributes. It returns the original vector
    of values with stability info tacked on.
    """

    def __init__(self, nstep, tint, vnams, pinit, efactor, rescape, stoerm, acc):
        self.nstep   = nstep
        self.tint    = 365.25*tint
        self.vnams   = vnams
        self.pinit   = pinit
        self.efactor = efactor
        self.rescape = rescape
        self.stoerm  = stoerm
        self.acc     = acc

    def __call__(self, mvec):
        """
        Given a model vectors, computes stability info.
        Returns model vector with stability data tacked on.
        """
        # generate dictionary
        p = dict(self.pinit)
        for key, val in zip(self.vnams, mvec):
            p[key] = val
                
        lrvm = orbits.ptolrvm(p)

        ttime, ntest, eratio, npoint, ierr, tnext = \
            orbits.integrate(lrvm, self.nstep, tstop=abs(self.tint), acc=self.acc, reverse=self.tint < 0, efactor=self.efactor, \
                                 rescape=self.rescape, stoerm=self.stoerm)
        return (list(mvec) + [ttime/365.25, ntest, eratio, npoint, ierr])


# initialisation. stab is the callable object
nstart = ochain.nmods
stab   = Stab(nstep, tint, ichain.names,  orbits.fixpar(ochain.model), efactor, rescape, stoerm, acc)

if nthread == 1:

    for i, vals in ichain:

        if i >= nstart:

            # generate dictionary
            result = stab(vals)
                
            if (i+1) % nprint == 0:
                print 'Reached orbit',i+1
            
            # write out result
            ochain.add_line(result,nomem=True)

else:
                
    # multi-threaded version. 
    # Create a pool
    pool = Pool(nthread)

    # OK wind through input file, filling up the list of models 
    # for each process in turn.
    alist = []
    nb = 0
    for i, vals in ichain:

        if i >= nstart:
            alist.append(vals)
            if len(alist) == nbatch:
                results = pool.map(stab, alist)

                # write out the results
                for result in results:
                    ochain.add_line(result,nomem=True)
                            
                # initialise for next batch
                alist = []

                if nb % nprint == 0:
                    print 'Reached orbit',i+1
                nb += 1

    if len(alist):
        # Deal with left-overs
        results = pool.map(stab, alist)

        # write out the results
        for result in results:
            ochain.add_line(result,nomem=True)

