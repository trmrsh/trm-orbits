#!/usr/bin/env python

usage = """
usage: %s chain xindex yindex nchop [ngroup] times [x1 x2 y1 y2 xoffset yoffset]

Allows selection of particular points from a 2D scatter plot and subsequent display 
of the corresponding orbit fit. The idea is to display a scatter plot, then use the mouse 
to click on points (must be unique within 5 points on the screen) and then the data and fit
will be displayed. 

chain   -- MCMC orbits
xindex  -- name or index of parameter to plot along X axis
yindex  -- name or index of parameter to plot along Y axis
nchop   -- number of point groups to remove, negative to just take last -nchop groups.
ngroup  -- index of particular group to select
times   -- file of eclipse times that the MCMC file was generated from
x1      -- lower X limit
x2      -- upper X limit
y1      -- lower Y limit
y2      -- upper Y limit
xoffset -- offset to subtract from X, defaults to 0.
yoffset -- offset to subtract from Y, defaults to 0.
""" 

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from trm import subs, orbits, mcmc
import trm.subs.input as inp
from   trm.subs import Fname

# generate arguments
inpt = inp.Input('PYTHON_ORBITS_ENV', '.pyorbits', sys.argv)

inpt.register('chain',   inp.Input.LOCAL,  inp.Input.PROMPT)
inpt.register('xindex',  inp.Input.LOCAL,  inp.Input.PROMPT)
inpt.register('yindex',  inp.Input.LOCAL,  inp.Input.PROMPT)
inpt.register('nchop',   inp.Input.GLOBAL, inp.Input.PROMPT)
inpt.register('ngroup',  inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('times',   inp.Input.LOCAL,  inp.Input.PROMPT)
inpt.register('x1',      inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('x2',      inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('y1',      inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('y2',      inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('xoffset', inp.Input.LOCAL,  inp.Input.HIDE)
inpt.register('yoffset', inp.Input.LOCAL,  inp.Input.HIDE)

chain   = inpt.get_value('chain', 'log of mcmc output', Fname('nnser', '.log', Fname.OLD))
xindex  = inpt.get_value('xindex', 'index of X axis variable', 'a1')
yindex  = inpt.get_value('yindex', 'index of Y axis variable', 'a2')
nchop   = inpt.get_value('nchop', 'number of point groups to remove from the start', 0)
inpt.set_default('ngroup', 0)
ngroup  = inpt.get_value('ngroup', 'group number to plot', 0)
times   = inpt.get_value('times', 'file of eclipses times', Fname('nnser', '.dat'))
x1      = inpt.get_value('x1',  'left-hand plot limit', 0.)
x2      = inpt.get_value('x2',  'right-hand plot limit', 0.)
y1      = inpt.get_value('y1',  'bottom plot limit', 0.)
y2      = inpt.get_value('y2',  'top plot limit', 0.)
inpt.set_default('xoffset', 0.)
xoffset = inpt.get_value('xoffset',  'offset to subtract from X values', 0.)
inpt.set_default('yoffset', 0.)
yoffset = inpt.get_value('yoffset',  'offset to subtract from Y values', 0.)

# OK now do some work

# load chain
chain = mcmc.Chain(chain)

try:
    xind = int(xindex) - 1
except:
    xind = chain.names.index(xindex) 

try:
    yind = int(yindex) - 1
except:
    yind = chain.names.index(yindex) 

if ngroup == 0:
    if nchop < 0 and -nchop*chain.nwalker < len(chain):
        chain.chop(len(chain)+nchop*chain.nwalker)
else:
    if ngroup == -1:
        chain.vals = chain.vals[-chain.nwalker:,]
    else:
        if ngroup > 0:
            ngroup -= 1
        chain.vals = chain.vals[chain.nwalker*ngroup:chain.nwalker*(ngroup+1),]

x = chain.vals[:,xind] - xoffset
y = chain.vals[:,yind] - yoffset

print 'Will plot',len(x),'points.'

# load the data
cycle,time,etime,offset,soften = orbits.load_times(times)

def limits(x):
    x1 = x.min()
    x2 = x.max()
    xr = x2 - x1
    x1 -= xr/10.
    x2 += xr/10.
    return (x1,x2)

if x1 == x2:
    x1,x2 = limits(x)
if y1 == y2:
    y1,y2 = limits(y)

if xoffset < 0.:
    xlabel = chain.names[xind] + ' + ' + str(-xoffset)
elif xoffset > 0.:
    xlabel = chain.names[xind] + ' - ' + str(xoffset)
else:
    xlabel = chain.names[xind]

if yoffset < 0.:
    ylabel = chain.names[yind] + ' + ' + str(-yoffset)
elif yoffset > 0.:
    ylabel = chain.names[yind] + ' - ' + str(yoffset)
else:
    ylabel = chain.names[yind]

fig   = plt.figure()
ax    = fig.add_subplot(111)

scatt, = ax.plot(x,y,'g.',picker=5)
ax.set_xlim(x1,x2)
ax.set_ylim(y1,y2)
ax.set_xlabel(xlabel)
ax.set_ylabel(xlabel)

def onpick(event):
    """
    Defines the action on clicking -- note the use of globals here
    """
    if event.artist != scatt: return True

    N = len(event.ind)
    if N == 0:
        print 'No point selected'
        return True
    elif N > 1:
        print 'More than one point selected'
        return True

    dataind = event.ind[0]
    print 'Selected point (index,x,y) = ',dataind,x[dataind],y[dataind]
    vvar = chain.vpars()
    pd = orbits.fixpar(chain.model,zip(vvar,chain.vals[dataind,:len(vvar)]))
    t0 = pd['t0']
    period = pd['period']
    figi = plt.figure()
    ax1  = figi.add_subplot(211)
    ax1.errorbar(cycle+offset, subs.DAY*(time - orbits.ephmod(cycle,t0,period)), 
                subs.DAY*etime, fmt='og', capsize=0)
    fcycle = np.linspace(cycle.min(),cycle.max(),400)
    pred   = orbits.etimes(pd, fcycle, stoerm=True) - orbits.ephmod(fcycle,t0,period)

    ax1.plot(fcycle+offset, subs.DAY*pred, 'r--')
    ax1.set_xlim(cycle.min()+offset-1000., cycle.max()+offset+10000.)

    # plot residuals
    ax2  = figi.add_subplot(212,sharex=ax1)
    pred = orbits.etimes(pd, cycle, stoerm=True)
    ax2.errorbar(cycle+offset, subs.DAY*(time - pred), subs.DAY*etime, fmt='og', capsize=0)

    chisq = (((time-pred)/etime)**2).sum()
    sumw  = (1/etime**2).sum()
    nvar  = len(vvar)
    ndata = len(etime)
    print 'Chisq =',chisq,', residuals =',np.sqrt(ndata/(ndata-nvar)*chisq/sumw)

    figi.show()
    return True

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()
