#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore")


import pylab, os
import scipy.stats as stats
import scipy.optimize as optimize
import numpy as np
import trm.orbits as orbits
import trm.mcmc as mcmc

from optparse import OptionParser

usage = """usage: %prog [options] rvfile period sigma

Generates an MCMC run to try to constrain eccentricity of an essentially circular orbit. It uses an 
approximation that only hold for small eccentricities, so don't use for truly eccentric orbits.

Arguments:

 rvfile  -- file of radial velocities times, vels, velocity errors. It should end .rv (will be added if not found). The root
            part will be used to generate other files root.log and root.pdf containing MCMC results and a plot of best-fit circular
            and eccentric orbits.
 period  -- orbital period (same units as times in rvfile). Must be accurate enought to hit the right alias.
 sigma   -- extra uncertainty to add, km/s, to give chi**2 = 1
"""

parser = OptionParser(usage)
parser.add_option("-p", "--prod", dest="prod", default=100000, type="int",\
                  help="number of trials for production run (default = 500000)")
parser.add_option("-n", "--nstore", dest="nstore", default=20, type="int",\
                  help="how often to store MCMC results (default = 20)")

(options, args) = parser.parse_args()

if len(args) != 3:
    print 'Require 3 arguments. Use -h for help'
    exit(1)

rvfile  = args[0]
period  = float(args[1])
sigma   = float(args[2])

if not rvfile.endswith('.rv'):
    rvfile += '.rv'

if not os.path.isfile(rvfile):
    print 'Could not find radial elocity file =',rvfile
    exit(1)

# files produced
resfile  = rvfile[:-3] + '.res'
mcmcfile = rvfile[:-3] + '.log'
pdffile  = rvfile[:-3] + '.pdf'

NPROD  = options.prod
#NSCALE = options.scale
NSTORE = options.nstore

data   = np.loadtxt(rvfile)

t      = data[:,0]
v      = data[:,1]
ve     = data[:,2]
ve     = np.sqrt(ve**2 + sigma**2)

# Functions defining a series of models:
#
# Idea is to get reasonable parameters to kick off with before fitting the period. The eccentric model is only
# valid for low eccentricities and is based on fitting harmonics

def model1(p, t, toff, period):
    """
    Circular, fixed period
    """
    gamma  = p[0]
    a1     = p[1]
    b1     = p[2]
    phase  = 2.*np.pi*(t-toff)/period
    return gamma + a1*np.cos(phase) + b1*np.sin(phase)

def model2(p, t, toff):
    """
    Circular, free period
    """
    gamma  = p[0]
    a1     = p[1]
    b1     = p[2]
    period = p[3]
    phase  = 2.*np.pi*(t-toff)/period
    return gamma + a1*np.cos(phase) + b1*np.sin(phase)

def model3(p, t, toff):
    """
    Small eccentricity, free period
    """
    gamma  = p[0]
    a1     = p[1]
    b1     = p[2]
    a2     = p[3]
    b2     = p[4]
    period = p[5]
    phase  = 2.*np.pi*(t-toff)/period
    return gamma + a1*np.cos(phase) + b1*np.sin(phase) + a2*np.cos(2.*phase) + b2*np.sin(2.*phase)

# corresponding scaled residuals for Lev-Marq
def resid1(p, t, v, ve, toff, period):
    """
    Circular, fixed period
    """
    return (v-model1(p, t, toff, period))/ve

def resid2(p, t, v, ve, toff):
    """
    Circular, free period
    """
    return (v-model2(p, t, toff))/ve

def resid3(p, t, v, ve, toff):
    """
    Small eccentricity, free period
    """
    return (v-model3(p, t, toff))/ve

# corresponding chi**2
def chisq1(p, t, v, ve, toff, period):
    """
    Circular, fixed period
    """
    return (resid1(p, t, v, ve, toff, period)**2).sum()

def chisq2(p, t, v, ve, toff):
    """
    Circular, free period
    """
    return (resid2(p, t, v, ve, toff)**2).sum()

def chisq3(p, t, v, ve, toff):
    """
    Small eccentricity, free period
    """
    return (resid3(p, t, v, ve, toff)**2).sum()

# offset time
toff = t.mean()

# first a circular orbit fit without letting the period change, simplex
p0   = [0., -2., 2.]
p1 = optimize.fmin(chisq1, p0, (t,v,ve,toff,period), disp=False)

olog = open(resfile, 'w')

print 'Minimum chi**2 (circular, simplex, no period change) =',chisq1(p1, t, v, ve, toff, period),', period =',period
olog.write('Minimum chi**2 (circular, simplex, no period change) = ' + str(chisq1(p1, t, v, ve, toff, period)) + ', period = ' + str(period) + '\n')

# second fit allowing period to change
p2 = np.concatenate((p1, [period,]))
p3 = optimize.fmin(chisq2, p2, (t,v,ve,toff), disp=False)

cmin = chisq2(p3, t, v, ve, toff)
print 'Minimum chi**2 (circular, simplex, with period change) =',cmin,', period =',p3[-1]
olog.write('Minimum chi**2 (circular, simplex, with period change) = ' + str(cmin) + ', period = ' + str(p3[-1]) + '\n')

# third fit allowing period to change, with Lev-Marq
p4 = optimize.leastsq(resid2, p3, (t,v,ve,toff))[0]

cmin_circ = chisq2(p4, t, v, ve, toff)
print 'Minimum chi**2 (circular, levmarq, with period change) =',cmin_circ,', period =',p4[-1]
olog.write('Minimum chi**2 (circular, levmarq, with period change) = ' + str(cmin_circ) + ', period = ' + str(p4[-1]) + '\n')

# fourth fit allowing for a small eccentricity
p5 = np.concatenate((p4[:3],[0.,0.,p4[-1]]))
(p6,cov) = optimize.leastsq(resid3, p5, (t,v,ve,toff),full_output=True)[:2]

cmin_ecc = chisq3(p6, t, v, ve, toff)
print 'Minimum chi**2 (eccentric, levmarq, with period change) =',cmin_ecc,', period =',p6[-1]
olog.write('Minimum chi**2 (eccentric, levmarq, with period change) = ' + str(cmin_ecc) + ', period = ' + str(p6[-1]) + '\n')

# OK, now fits and errors
cfphase = np.linspace(0.,1.,200)
pfcirc  = np.concatenate((p4[:-1],[1.,]))
vfcirc  = model2(pfcirc, cfphase, 0.)
cphase  = (t - toff) / p4[-1]
cphase -= np.floor(cphase)
comc    = v - model2(pfcirc, cphase, 0.)
efphase = np.linspace(0.,1.,200)
pfecc   = np.concatenate((p6[:-1],[1.,]))
vfecc   = model3(pfecc, efphase, 0.)
ephase  = (t - toff) / p6[-1]
ephase -= np.floor(ephase)
eomc    = v - model3(pfecc, ephase, 0.)

y1 = min((v-ve).min(), vfcirc.min(), vfecc.min()) - 10.
y2 = max((v+ve).max(), vfcirc.max(), vfecc.max()) + 10.

add = ve.mean()
yr1 = min((comc-ve).min(), (eomc-ve).min()) - add
yr2 = max((comc+ve).max(), (eomc+ve).max()) + add

pylab.subplot(2,2,1)
pylab.plot(cfphase, vfcirc)
pylab.errorbar(cphase, v, ve, fmt='.g', capsize=0)
pylab.ylim(y1,y2)
pylab.ylabel('Velocity (km/s)')
pylab.title('Circular')
pylab.subplot(2,2,3)
pylab.errorbar(cphase, v - model2(pfcirc, cphase, 0.), ve, fmt='.g', capsize=0)
pylab.ylim(yr1,yr2)
pylab.xlabel('Orbital phase')
pylab.ylabel('O - C velocity (km/s)')
pylab.subplot(2,2,2)
pylab.plot(efphase, vfecc)
pylab.errorbar(ephase, v, ve, fmt='.g', capsize=0)
pylab.title('Eccentric')
pylab.ylim(y1,y2)
pylab.subplot(2,2,4)
pylab.errorbar(ephase, v - model3(pfecc, ephase, 0.), ve, fmt='.g', capsize=0)
pylab.ylim(yr1,yr2)
pylab.xlabel('Orbital phase')
pylab.suptitle(rvfile[:-3])
pylab.savefig(pdffile)

# Now on with rest

df1 = len(v) - 4
df2 = len(v) - 6
F = (cmin_circ/df1)/(cmin_ecc/df2)
print '\nBest-fit eccentricity =',np.sqrt((p6[3]**2+p6[4]**2)/(p6[1]**2+p6[2]**2))
olog.write('\nBest-fit eccentricity = ' + str(np.sqrt((p6[3]**2+p6[4]**2)/(p6[1]**2+p6[2]**2))) + '\n')
print 'Measured F-ratio =',F
olog.write('Measured F-ratio = ' + str(F) + '\n')
chance = stats.f.sf(F, df1, df2)
print 'This value or greater can occur by chance',round(1000.*chance)/10.,'% of the time'
olog.write('This value or greater can occur by chance ' + str(round(1000.*chance)/10.) + '% of the time' + '\n')
if chance < 0.05:
    print 'Eccentricity significant at 95% level\n'
    olog.write('Eccentricity significant at 95% level\n\n')
else:
    print 'Eccentricity not significant at 95% level\n'
    olog.write('Eccentricity not significant at 95% level\n\n')

# create jump distribution from levmarq covariances
sigma  = np.sqrt(np.diag(cov))
corr   = ((cov/sigma).transpose()/sigma).transpose()
names  = ('gamma', 'a1', 'b1', 'a2', 'b2', 'period')
jump   = mcmc.Jump(names, sigma, corr)

# creat initial dictionary
p = dict(zip(names, p6))

# These are the variable ones with initial sigmas
vvar = ('gamma', 'a1', 'b1', 'a2', 'b2', 'period')

def chisq(p,t,v,ve,toff):
    pv = [p['gamma'], p['a1'], p['b1'], p['a2'], p['b2'], p['period']]
    return chisq3(pv, t, v, ve, toff)

csq    = chisq(p,t,v,ve,toff)
cbest  = csq
pbest  = p
nbest  = 0
print 'Initial chi**2 of MCMC production run =',csq,'\n'

## scale to get better acceptance 
#print 'Now determining scale factor for 25% acceptance ....'
#
#while n < NSCALE:
#    n      += 1
#    ntotal += 1
#    ntot   += 1
#
#    ptest = jump.trial(p)
#    ctest = chisq(ptest,t,v,ve,toff)
#
#    if ctest < csq or np.random.uniform() < np.exp(-(ctest-csq)/2.):
#        p         = ptest
#        csq       = ctest
#        nsuccess += 1
#        nchange  += 1
#
#    if csq < cbest:
#        cbest = csq
#        pbest = p
#        print 'New best chi**2 =',cbest
#
#    if nsuccess == 200:
#        jump *= 4.*nsuccess/float(ntotal)
#        nsuccess = 0
#        ntotal   = 0
#
#    if n % 10000 == 0: print 'Reached',n,'acceptance rate =',nchange/float(max(1,ntot))

# Production run
nsuccess, n, ntotal, nchange, ntot = 0, 0, 0, 0, 0
p = pbest
fchain = mcmc.Fchain(p, NSTORE, jump, ['e','chisq'], len(t), mcmcfile)

while n < NPROD:
    n      += 1
    ntot   += 1

    ptest = jump.trial(p)
    ctest = chisq(ptest,t,v,ve,toff)

    if ctest < csq or np.random.uniform() < np.exp(-(ctest-csq)/2.):
        p         = ptest
        csq       = ctest
        nchange  += 1

    if csq < cbest:
        cbest = csq
        pbest = p
        print 'New best chi**2 (somewhat unexpectedly) =',cbest

    if n % NSTORE == 0: 
        vals = [p[vi] for vi in vvar] + [np.sqrt((p['a2']**2+p['b2']**2)/(p['a1']**2+p['b1']**2)),csq]
        fchain.add_line(vals)

    if n % 10000 == 0: print 'Reached',n,'acceptance rate =',nchange/float(max(1,ntot))


e = np.sort(fchain.vals[:,-2])
print '\n68% of eccentricity values are <',e[int(round(0.68*len(e)))]
print '95% of eccentricity values are <',e[int(round(0.95*len(e)))]
print '99% of eccentricity values are <',e[int(round(0.99*len(e)))]

olog.write('68% of eccentricity values are <' + str(e[int(round(0.68*len(e)))]) + '\n')
olog.write('95% of eccentricity values are <' + str(e[int(round(0.95*len(e)))]) + '\n')
olog.write('99% of eccentricity values are <' + str(e[int(round(0.99*len(e)))]) + '\n')
olog.close()

print '\nMCMC chain stored in',mcmcfile
print 'Summary results stored in',resfile
print 'Hardcopy figure saved to',pdffile

print 'You should be wary if (a) the chi**2 dropped significantly during the MCMC run,'
print '(b) the acceptance rate was very low, or (c) the best-fit eccentricity is large.'
