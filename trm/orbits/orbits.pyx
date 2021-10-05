import math
from libc.math cimport sin, cos

from scipy.integrate import ode

import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

# Gauss' gravitational constant sqrt(G*MSUN), AU^(3/2) day^-1
KGAUSS = 0.01720209895

# G*MSUN, AU^3 day^-2 (KGAUSS**2)
GMGAUSS = KGAUSS*KGAUSS

class NbodyF:

    def __init__(self, mass):
        """
        Defines the function object for N-body integration.

        mass   : (array)
            array of masses. Can be a single element in which case the orbit of a test particle
            will be computed.
        """
        self.mass = mass


    def __call__(self, t, y):
        """
        Computes the right-hand sides of the ODEs needed for N-body
        integration.  ODEs have form dy_i/dt = f_i(t, y) where represent all
        the variables and y_i the i-th one. So this routine evaluates the f_i.
        On input, t cy contains the coordinates & velocities for each object
        i.e. for two objects one would have a 12 element array of the form:

           x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2

        Arguments::

           t : (float)
             the time

           y : (array)
             array of values of variables at time t.

        The routine returns derivatives dy/dt
        """

        NMASS = len(self.mass)
        dydt = np.empty(6*NMASS)

        # Set derivatives of coordinates equal to velocities and zero the accelerations
        for n in range(NMASS):
            dydt[6*n:6*n+3] = y[6*n+3:6*n+6]
            dydt[6*n+3:6*n+6] = 0.

        if NMASS > 1:

            # Now for the accelerations.
            for i1 in range(NMASS):

                ind1 = 6*i1
                jnd1 = ind1 + 3
                x1, y1, z1 = y[ind1:jnd1]

                for i2 in range(i1+1,NMASS):

                    ind2 = 6*i2
                    jnd2 = ind2 + 3
                    x2, y2, z2 = y[ind2:jnd2]
                    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
                    rcu = (dx**2 + dy**2 + dz**2)**1.5

                    # Now the accelerations due to the other objects upon the
                    # object in question with a bit of extra stuff to ensure
                    # we don't miss out any terms by always counting forward
                    # from the object under consideration.

                    # Generate velocity derivatives (in AU/day**2)
                    AFAC = GMGAUSS/rcu

                    # acceleration of i1 due to i2 (attractive towards i2)
                    AFAC1 = AFAC*self.mass[i2]

                    dydt[jnd1  ] += AFAC1*dx
                    dydt[jnd1+1] += AFAC1*dy
                    dydt[jnd1+2] += AFAC1*dz

                    # acceleration of i2 due to i1 (note sign swap)
                    AFAC2 = AFAC*self.mass[i1]
                    dydt[jnd2  ] -= AFAC2*dx
                    dydt[jnd2+1] -= AFAC2*dy
                    dydt[jnd2+2] -= AFAC2*dz
        else:

            # For just one component we pretend it is a zero mass particle
            # orbiting the origin where there is a mass equal to that supplied
            # through the mass vector.
            dx, dy, dz = y[:3]
            rcu = (dx**2 + dy**2 + dz**2)**1.5

            # acceleration/r
            AFAC = -GMGAUSS*self.mass[0]/rcu

            # acceleration components
            dydt[3] = AFAC*dx;
            dydt[4] = AFAC*dy;
            dydt[5] = AFAC*dz;

        return dydt

def integrate(lrvm, times, tstop=1.e8, nkeep=0, tmax=10000., acc=1.e-10,
              reverse=False, nmax=1000000, efactor=1.1, rescape=10.):

    """N-body integration routine given starting positions velocities and
    masses. It works on the basis that the first particle dominates in
    mass. Checks are made for collisions and escape that stop the
    integration. Note that the initial conditions are assumed to apply at
    t=0. If you set the second argument to be a set of times, only collision
    checks are made every 5 steps to reduce time spent integrating hopeless
    cases.  In very close approaches, the number of steps between points can
    increase dramatically and this should be prevented by these checks which
    use the object sizes specified in lrvm. The checks are only made every so
    often for speed.

    Arguments:

       lrvm   : list of Rvm objects
          one element for each particle defining the conditions at t=0.

       times  : array | int

          an array of times (in days) or the maximum number of integration
          steps to attempt.  If it is an array of times, they must increase
          monotonically. The next variables, apart from 'ttry', 'acc' and
          'interact', are only relevant if 'times' is an integer.

       tstop  : float [if times is an int]
          time at which to stop integration (days)

       nkeep  : int [if times is an int]
          how often to store results (0 not to bother)

       tmax   : float [if times is an int]
          maximum time step to allow (days). Useful if you want fine stepped results

       acc    : float
          fractional accuracy parameter

       reverse: bool [in times is an int]
          integrate backwards in time if True

       nmax   : int
          maximum number of steps to take per times (if times is an array).
          Safety valve

       ncheck : int
          frequency of checks for close approach, escape, energy. Note that
          only close approach checks are made when an array of times is
          submitted, meaning the next two parameters are irrelevant in this
          case. If 'times' is set to an integer, then close approach, maximum
          radius and energy ratio are all checked.  ncheck=5 by default to
          reduce the extra computation per step.

       efactor: float
          ratio |KE/PE| of most distant particle to count as escaped.

       rescape: float
          radius from centre of mass (AU) to count as escaped

    Returns: arr

       arr     : array
          data array of times, positions, velocities [AU, AU/day]

    old stuff below:

       ntest   : int
          integer flag to indicate problems: 0 is OK. Up to four digits
          otherwise. If first is set, then an object has exceeded the escape
          radius. If the next two are, two of the objects have collided. If
          the last then an object has violated the energy test. See also
          orbits.trans_code for translating this into a message.

       eratio  : float
          ratio KE/PE of the most distant particle

       nstore  : int
          number of points stored (if nkeep > 0)



    NB Last two parameters are only returned if you specified nkeep > 0.

    """

    # work out positions and velocities at t=0
    mass = []
    y0 = []
    for rvm in lrvm:
        mass.append(rvm.m)
        y0 += [rvm.r.x, rvm.r.y, rvm.r.z, rvm.v.x, rvm.v.y, rvm.v.z]
    mass = np.array(mass)
    y0 = np.array(y0)

    # create the function for the ODE integrator
    nbf = NbodyF(mass)
    nbody = ode(nbf).set_integrator('lsoda', atol=acc, nsteps=nmax)

    if isinstance(times, np.ndarray):

        if not np.all(times[1:] > times[:-1]):
            raise ValueError('times no monotonically increasing')

        # initialise t=0 phase space position
        nbody.set_initial_value(y0, 0)

        # get to start
        y0 = nbody.integrate(times[0])

        # re-initialise
        nbody.set_initial_value(y0, times[0])

        # make space for data
        arr = np.empty((len(times), 1+6*len(lrvm)))

        # save first point
        arr[0,0] = times[0]
        arr[0,1:] = y0

        # Integrate to all the other points
        for ntime, time in enumerate(times[1:]):
            # integrate to time = time
            y0 = nbody.integrate(time)
            if not nbody.successful():
                raise ValueError(
                    'orbit integration failed on time number {:d} = {:f}'.format(
                        ntime+2, time
                    )
                )
            # store the results
            arr[ntime+1,0] = time
            arr[ntime+1,1:] = y0

        return arr

    else:
        raise NotImplementedError('Have no coded nsteps integrator')


def mean2ecc(np.ndarray[DTYPE_t, ndim=1] manom, double e, double acc=1.e-12):
    """Solves Kepler's equation. M = E - e*sin(E) for E given M where M is the
    "mean anomolay" aka orbital phase, but measured in radians, and E is the
    eccentric anomaly also in radians. NB This routine has been speeded up
    with Cython and will only accept a double precision 1D numpy.ndarray for
    manom on input. Use mean2eccS for single floats.

    Arguments::

       manom  : (array)
           mean anomoly (radians)

       e      : (float)
           eccentricity

       acc    : (float)
           precision of results (radians)

    Returns: eccentric anomolies (radians)
    """

    cdef unsigned int necc = len(manom)
    cdef np.ndarray[DTYPE_t, ndim=1] ecc = np.empty((necc), dtype=DTYPE)
    cdef unsigned int i

    for i in range(necc):
        ecc[i] = mean2eccS(manom[i], e, acc)
    return ecc

cpdef double mean2eccS(double manom, double e, double acc=1.e-12):
    """Solves Kepler's equation. M = E - e*sin(E) for E given M where M is the
    "mean anomoly" aka orbital phase, but measured in radians, and E is the
    eccentric anomaly also in radians.

    Arguments::

       manom  : (float)
           mean anomoly [radians]

       e      : (float)
           eccentricity

       acc    : (float)
           precision of results [radians]

    Returns: eccentric anomoly [radians]

    """

    cdef double man, lo, hi, span, ecc, f, df, temp, dx

    # translate to 0 to 2*pi
    man = manom % (2.*math.pi)

    # set lower upper limits
    if man < math.pi:
        lo, hi = 0., math.pi
    else:
        lo, hi = math.pi, 2.*math.pi

    # initial guess
    span = hi-lo
    ecc = man

    f = ecc - e*sin(ecc) - man
    df = 1. - e*cos(ecc)

    while 1:

        if (((ecc-hi)*df-f)*((ecc-lo)*df-f) >= 0.) | (abs(2.*f) > abs(span*df)):
            # Binary chop
            temp = lo
            dx = 0.5*(hi-lo)
            ecc = lo + dx

        else:
            # Newton-Raphson
            dx = f / df
            temp = ecc
            ecc -= dx

        if temp == ecc or abs(dx) < acc:
            break

        f = ecc - e*sin(ecc) - man
        df = 1.-e*cos(ecc)

        # move the limits
        if f < 0.:
            lo = ecc
        else:
            hi = ecc

        span = hi - lo

    return ecc
