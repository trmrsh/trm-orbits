#!/usr/bin/env python

"""sub-module implementing particular hierarchical Kepler orbits, in
particular triple orbits and two forms of quadruple system.

"""

from math import sin, cos, sqrt, pi
import numpy as np
from .core import true2mean, mean2true

def position(nu,OMEGA,omega,i,e,a):
    """Returns the (x,y,z) position(s) in an observer-centred Cartesian system
    equivalent to a particular orbit. The z-axis points towards the observer,
    the x-axis parallel to the line of nodes and the y-axis follows a
    right-handed convention relative to the other two.

    Arguments::

      nu    : (float / array)
         true anomaly or anomalies (radians)

      OMEGA : (float)
         longitude of ascending node (radians)

      omega : (float)
         argument of periapsis (radians)

      i     : (float)
         orbital inclination (radians)

      e     : (float)
         eccentricity

      a     : (float)
         semi-major axis.

    Returns (x,y,z) position(s). x, y and z are arrays if nu is an array or
    floats if nu is a float.

    """

    omf = nu + omega
    comf = np.cos(omf)
    somf = np.sin(omf)
    r = (a*(1 - e**2)) / (1 + e * np.cos(nu))
    X = r*(-sin(OMEGA)*comf - (cos(i)*cos(OMEGA))*somf)
    Y = r*(+cos(OMEGA)*comf - (cos(i)*sin(OMEGA))*somf)
    Z = -r*sin(i)*somf
    return (X,Y,Z)

def velocity(nu,OMEGA,omega,i,e,a,P):
    """Returns the (vx,vy,vz) velocity(s) equivalent to a particular orbit. z
    measured along the line of sight towards the observer.

    Arguments::

      nu    : (float / array)
         true anomaly or anomalies (radians)

      OMEGA : (float)
         longitude of ascending node (radians)

      omega : (float)
         argument of periapsis (radians)

      i     : (float)
         orbital inclination (radians)

      e     : (float)
         eccentricity

      a     : (float)
         semi-major axis [AU]

      P     : (float)
         orbital period [days].

    Returns (vx,vy,vz) velocity(s) in AU / day. vx, vy and vz are arrays if nu is
    an an array or floats if nu is a float.

    """

    # pre-compute some stuff
    n = 2.*pi/P
    omf = nu + omega
    comf = np.cos(omf)
    somf = np.sin(omf)
    f = a*n / sqrt(1-e**2)
    sinf = f*(somf+e*sin(omega))
    cosf = f*(comf+e*cos(omega))

    vx = +sinf*sin(OMEGA)-cosf*cos(i)*cos(OMEGA)
    vy = -sinf*cos(OMEGA)-cosf*cos(i)*sin(OMEGA)
    vz = -cosf*sin(i)

    return (vx, vy, vz)

def position2(nu,OMEGA,omega,i,e,a1,a2):
    """Returns the (x,y,z) position(s) in an observer-centred Cartesian system
    equivalent to a particular orbit. The z-axis points towards the observer,
    the x-axis parallel to the line of nodes and the y-axis follows a
    right-handed convention relative to the other two. This version returns
    two sets of values for two stars in opposite sides of the same orbit (slightly
    faster than computing each one separately). "omega" is assumed to apply to a1.

    Arguments::

      nu    : (float / array)
         true anomaly or anomalies (radians)

      OMEGA : (float)
         longitude of ascending node (radians)

      omega : (float)
         argument of periapsis (radians)

      i     : (float)
         orbital inclination (radians)

      e     : (float)
         eccentricity

      a1    : (float)
         semi-major axis of star corresponding to omega

      a2    : (float)
         semi-major axis of other star, which effectively has omega --> omega + pi

    Returns (x1,y1,z1,x2,y2,z2)
    """

    omf = nu + omega
    comf = np.cos(omf)
    somf = np.sin(omf)
    r = (1 - e**2) / (1 + e * np.cos(nu))
    X = r*(-sin(OMEGA)*comf - (cos(i)*cos(OMEGA))*somf)
    Y = r*(+cos(OMEGA)*comf - (cos(i)*sin(OMEGA))*somf)
    Z = r*(-sin(i)*somf)
    return (a1*X,a1*Y,a1*Z,-a2*X,-a2*Y,-a2*Z)

def velocity2(nu,OMEGA,omega,i,e,a1,a2,n):
    """Returns the (vx,vy,vz) velocities equivalent to a particular orbit. z
    measured along the line of sight towards the observer.

    Arguments::

      nu    : (float / array)
         true anomaly or anomalies (radians)

      OMEGA : (float)
         longitude of ascending node (radians)

      omega : (float)
         argument of periapsis (radians)

      i     : (float)
         orbital inclination (radians)

      e     : (float)
         eccentricity

      a1    : (float)
         semi-major axis of star corresponding to omega [AU]

      a2    : (float)
         semi-major axis of other star, which effectively has omega --> omega + pi [AU]

      n     : (float)
         orbital angular velocity [rad/day]

    Returns (vx1,vy1,vz1,vx2,vy2,vz2) velocities in AU/day

    """

    # pre-compute some stuff
    omf = nu + omega
    comf = np.cos(omf)
    somf = np.sin(omf)
    f = n / sqrt(1-e**2)
    sinf = f*(somf+e*sin(omega))
    cosf = f*(comf+e*cos(omega))

    vx = +sinf*sin(OMEGA)-cosf*cos(i)*cos(OMEGA)
    vy = -sinf*cos(OMEGA)-cosf*cos(i)*sin(OMEGA)
    vz = -cosf*sin(i)

    return (a1*vx, a1*vy, a1*vz, -a2*vx, -a2*vy, -a2*vz)

def triplePos(t, a1, a2, a3, ab, eb1, eb2,
              omegab1, omegab2, Pb1, Pb2,
              ib1, ib2, OMEGAb1, OMEGAb2,
              t0b1, t0b2, ttype, acc=1.e-12):

    """This calculates the positions of 3 stars at a series of times with the
    orbits modelled as a hierarchy of Kepler 2-body orbits (an
    approximation). Stars 1 & 2 form the first inner binary (1 / 2,
    binary 1), Stars (1+2) and star3 form the second outer binary ((1+2) /
    3, binary 2). 

    This applies a first-order correction for light-travel time by correcting
    all positions according to the velocities times a light travel time
    referenced to the CoM.

    Arguments::

       t       : (array)
           times [days]

       a1      : (float)
           semi-major axis of star 1 around binary 1's CoM [solar]

       a2      : (float)
           semi-major axis of star 2 around binary 1's CoM [solar]

       a3      : (float)
           semi-major axis of star 3 around system CoM [solar]

       ab      : (float)
           semi-major axis of binary 1 rel. to system CoM [solar]

       eb1     : (float)
           eccentricity of binary 1

       eb2     : (float)
           eccentricity of binary 2

       omegab1 : (float)
           argument of periapsis of binary 1 [degrees]

       omegab2 : (float)
           argument of periapsis of binary 2 [degrees]

       Pb1     : (float)
           period of binary 1 [days]

       Pb2     : (float)
           period of binary 2 [days]

       ib1     : (float)
           inclination of binary 1 [degrees]

       ib2     : (float)
           inclination of binary 2 [degrees]

       OMEGAb1 : (float)
           longitude of ascending node of binary 1 [degrees]

       OMEGAb2 : (float)
           longitude of ascending node of binary 2 [degrees]

       t0b1    : (float)
           zeropoint of binary 1 [days]

       t0b2    : (float)
           zeropoint of binary 2 [days]

       ttype  : (int)
           1 == t0 is time at periastron, 2 == t0 is time of eclipse

       acc    : (float)
           Accuracy parameter for solving Kepler's equation

    Returns ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3)) where each of x1,y1,z1,x2 etc
    is a vector equivalent to the input array of times.

    """

    # Calculate mean angular velocities [rad/day]
    nb1 = 2.*np.pi/Pb1
    nb2 = 2.*np.pi/Pb2

    # speed of light, AU/day
    c = 173.1446327

    # offsets to the true anomolies if T0 defined by conjunction phases.
    mzerob1 = 0.
    mzerob2 = 0.
    if ttype == 2:
        mzerob1 = true2mean(3.*np.pi/2.-omegab1, eb1)
        mzerob2 = true2mean(3.*np.pi/2.-omegab2, eb2)

    # Basic method: computes positions & velocities of all stars once. Use
    # the velocities to apply first order corrections to the positions got
    # light-travel time.

    # Calculate mean anomalies of all three binaries
    mb1 = nb1*(t - t0b1) + mzerob1
    mb2 = nb2*(t - t0b2) + mzerob2

    # Calculate corresponding true anomalies
    nub1 = mean2true(mb1,eb1,acc)
    nub2 = mean2true(mb2,eb2,acc)

    # Calculate positions & velocities of the CoMs of:

    #  1) star 1 & 2 rel to binary 1 CoM
    x1,y1,z1,x2,y2,z2 = position2(nub1,OMEGAb1,omegab1,ib1,eb1,a1,a2)
    vx1,vy1,vz1,vx2,vy2,vz2 = velocity2(nub1,OMEGAb1,omegab1,ib1,eb1,a1,a2,nb1)

    #  2) binary 1 [1+2] & 3 rel to system CoM
    xb1,yb1,zb1,x3,y3,z3 = position2(nub2,OMEGAb2,omegab2,ib2,eb2,ab,a3)
    vxb1,vyb1,vzb1,vx3,vy3,vz3 = velocity2(nub2,OMEGAb2,omegab2,ib2,eb2,ab,a3,nb2)

    # Correct positions and velocities of stars 1 and 2 so that they are with
    # respect to an inertial frame, i.e. the system CoM (star 3 already is)

    # star 1
    x1 += xb1
    y1 += yb1
    z1 += zb1
    vx1 += vxb1
    vy1 += vyb1
    vz1 += vzb1

    # star 2
    x2 += xb1
    y2 += yb1
    z2 += zb1
    vx2 += vxb1
    vy2 += vyb1
    vz2 += vzb1

    # Now have inertial frame positions and velocities of all stars in AU and
    # AU / day. If z > 0, then a star is closer to Earth than the system CoM,
    # so we see it 'advanced' relative to the CoM by z/c, so we add v*z/c to
    # the positions.

    ltt = z1/c
    x1 += vx1*ltt
    y1 += vy1*ltt
    z1 += vz1*ltt

    ltt = z2/c
    x2 += vx2*ltt
    y2 += vy2*ltt
    z2 += vz2*ltt

    ltt = z3/c
    x3 += vx3*ltt
    y3 += vy3*ltt
    z3 += vz3*ltt

    return ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3))

def quad1Pos(t, a1, a2, a3, a4, ab1, ab2, eb1, eb2, eb3,
             omegab1, omegab2, omegab3, Pb1, Pb2, Pb3,
             ib1, ib2, ib3, OMEGAb1, OMEGAb2, OMEGAb3,
             t0b1, t0b2, t0b3, ttype, acc=1.e-12):

    """This calculates the positions of 4 stars at a series of times with
    the orbits modelled as a hierarchy of Kepler 2-body orbits (an
    approximation). Stars 1 & 2 form the first, innermost, binary (1 /
    2, binary 1), Stars (1+2) and star 4 forms the second, middle
    binary ((1+2) / 4, binary 2). Finally stars (1+2)+4 and star 3
    forms the third outermost binary ((1+2)+4) / 3, binary 3).

    This applies a first-order correction for light-travel time by
    correcting all positions according to the velocities times a light
    travel time referenced to the CoM.

    Arguments::

       t       : (array)
           times [days]

       a1      : (float)
           semi-major axis of star 1 around binary 1's CoM [solar]

       a2      : (float)
           semi-major axis of star 2 around binary 1's CoM [solar]

       a3      : (float)
           semi-major axis of star 3 around system CoM [solar]

       a4      : (float)
           semi-major axis of star 4 around binary 2's CoM [solar]

       ab1     : (float)
           semi-major axis of binary 1 rel. to binary 2's CoM [solar]

       ab2     : (float)
           semi-major axis of binary 2 rel. to system CoM [solar]

       eb1     : (float)
           eccentricity of binary 1

       eb2     : (float)
           eccentricity of binary 2

       eb3     : (float)
           eccentricity of binary 3

       omegab1 : (float)
           argument of periapsis of binary 1 [degrees]

       omegab2 : (float)
           argument of periapsis of binary 2 [degrees]

       omegab3 : (float)
           argument of periapsis of the ((1+2)+4) / 3 orbit [degrees]

       Pb1     : (float)
           period of binary 1 [days]

       Pb2     : (float)
           period of binary 2 [days]

       Pb3     : (float)
           period of binary 3 [days]

       ib1     : (float)
           inclination of binary 1 [degrees]

       ib2     : (float)
           inclination of binary 2 [degrees]

       ib3     : (float)
           inclination of binary 3 [degrees]

       OMEGAb1 : (float)
           longitude of ascending node of binary 1 [degrees]

       OMEGAb2 : (float)
           longitude of ascending node of binary 2 [degrees]

       OMEGAb3 : (float)
           longitude of ascending node of binary 3 [degrees]

       t0b1    : (float)
           zeropoint of binary 1 [days]

       t0b2    : (float)
           zeropoint of binary 2 [days]

       t0b3     : (float)
           zeropoint of binary 3 [days]


       ttype  : (int)
           1 == t0 is time at periastron, 2 == t0 is time of eclipse

       acc    : (float)
           Accuracy parameter for solving Kepler's equation

    Returns ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4)) where each of
    x1,y1,z1,x2 etc is a vector equivalent to the input array of times.

    """

    # Calculate mean angular velocities [rad/day]
    nb1 = 2.*np.pi/Pb1
    nb2 = 2.*np.pi/Pb2
    nb3 = 2.*np.pi/Pb3

    # speed of light, AU/day
    c = 173.1446327

    # offsets to the true anomolies if T0 defined by conjunction phases.
    mzerob1 = 0.
    mzerob2 = 0.
    mzerob3 = 0.
    if ttype == 2:
        mzerob1 = true2mean(3.*np.pi/2.-omegab1, eb1)
        mzerob2 = true2mean(3.*np.pi/2.-omegab2, eb2)
        mzerob3 = true2mean(3.*np.pi/2.-omegab3, eb3)

    # Basic method: computes positions & velocities of all stars once. Use
    # the velocities to apply first order corrections to the positions got
    # light-travel time.

    # Calculate mean anomalies of all three binaries
    mb1 = nb1*(t - t0b1) + mzerob1
    mb2 = nb2*(t - t0b2) + mzerob2
    mb3 = nb3*(t - t0b3) + mzerob3

    # Calculate corresponding true anomalies
    nub1 = mean2true(mb1,eb1,acc)
    nub2 = mean2true(mb2,eb2,acc)
    nub3 = mean2true(mb3,eb3,acc)

    # Calculate positions & velocities of the CoMs of:

    #  1) star 1 & 2 rel to binary 1 CoM
    x1,y1,z1,x2,y2,z2 = position2(nub1,OMEGAb1,omegab1,ib1,eb1,a1,a2)
    vx1,vy1,vz1,vx2,vy2,vz2 = velocity2(nub1,OMEGAb1,omegab1,ib1,eb1,a1,a2,nb1)

    #  2) binary 1 [1+2] & 4 rel to binary 2 CoM
    xb1,yb1,zb1,x4,y4,z4 = position2(nub2,OMEGAb2,omegab2,ib2,eb2,ab1,a4)
    vxb1,vyb1,vzb1,vx4,vy4,vz4 = velocity2(nub2,OMEGAb2,omegab2,ib2,eb2,ab1,a4,nb2)

    #  3) binary 2 [(1+2)+4] and star 3 rel to binary 3 CoM
    xb2,yb2,zb2,x3,y3,z3 = position2(nub3,OMEGAb3,omegab3,ib3,eb3,ab2,a3)
    vxb2,vyb2,vzb2,vx3,vy3,vz3 = velocity2(nub3,OMEGAb3,omegab3,ib3,eb3,ab2,a3,nb3)

    # Correct positions and velocities of stars 1, 2 and 4 so that they are with respect
    # to an inertial frame, i.e. the system CoM (star 3 already is)

    # First get total offset to b1
    xb1 += xb2
    yb1 += yb2
    zb1 += zb2
    vxb1 += vxb2
    vyb1 += vyb2
    vzb1 += vzb2

    # star 1
    x1 += xb1
    y1 += yb1
    z1 += zb1
    vx1 += vxb1
    vy1 += vyb1
    vz1 += vzb1

    # star 2
    x2 += xb1
    y2 += yb1
    z2 += zb1
    vx2 += vxb1
    vy2 += vyb1
    vz2 += vzb1

    # star 4
    x4 += xb2
    y4 += yb2
    z4 += zb2
    vx4 += vxb2
    vy4 += vyb2
    vz4 += vzb2

    # Now have inertial frame positions and velocities of all stars in AU and
    # AU / day. If z > 0, then a star is closer to Earth than the system CoM,
    # so we see it 'advanced' relative to the CoM by z/c, so we add v*z/c to the
    # positions.

    ltt = z1/c
    x1 += vx1*ltt
    y1 += vy1*ltt
    z1 += vz1*ltt

    ltt = z2/c
    x2 += vx2*ltt
    y2 += vy2*ltt
    z2 += vz2*ltt

    ltt = z3/c
    x3 += vx3*ltt
    y3 += vy3*ltt
    z3 += vz3*ltt

    ltt = z4/c
    x4 += vx4*ltt
    y4 += vy4*ltt
    z4 += vz4*ltt

    return ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4))

def quad2Pos(t,a1,a2,a3,a4,ab1,ab2,eb1,eb2,eb3,
             omegab1,omegab2,omegab3,Pb1,Pb2,Pb3,
             ib1,ib2,ib3,OMEGAb1,OMEGAb2,OMEGAb3,
             t0b1,t0b2,t0b3,ttype,acc=1.e-12):

    """This carries out the same as 'triplePosition' but for a hierarchical
    quadruple modelled as two 'inner' binaries orbiting each other within an
    'outer' binary. The inner binaries are composed of stars 1 & 2 (binary 1,
    b1) and 3 & 4 (binary 2, b2) in an outer binary (labelled 'b3'). This can
    be represented in an obvious way by (1+2)+(3+4): each '+' is an orbit
    while the parentheses define the hierarchy.

    This only corrects for light travel time of the 'outer' orbit of the
    binary's CoM.

    Arguments::

       t : array
           times [days]

       a1 : float
           semi-major axis star 1 rel. to binary 1's CoM [AU]

       a2 : float
           semi-major axis star 2 rel. to binary 1's CoM [AU]

       a3 : float
           semi-major axis star 3. If mode==1, this is rel. to binary 2's
           CoM. If mode==2, it is rel. to system CoM [AU]

       a4 : float
           semi-major axis star 4. If mode==1 this is rel. to binary 2's
           CoM. If mode==2, it is rel to CoM of stars (1+2)+4 [AU]

       ab1 : float
           semi-major axis of binary 1. mode==1 this is rel. to system CoM,
           mode==2 this is rel. to CoM of stars (1+2)+4 [AU]

       ab2 : float
           semi-major axis of binary 2 rel. to system CoM (mode==1);
           semi-major of 1+2+4 triple rel. to system CoM, (mode==2) [AU]

       eb1 : float
           eccentricity of binary 1 (1+2 orbit)

       eb2 : float
           eccentricity of binary 2. For mode==1, this is the second binary
           that forms a quadruple with binary 1 (i.e. the 3+4 orbit). For
           mode==2, it is the outer orbit of the 'inner' triple, i.e. the
           (1+2) / 4 orbit

       eb3 : float
           eccentricity of (1+2) / (3+4) orbit (mode==1) or the ((1+2)+4) / 3
           orbit (mode==2)

       omegab1 : float
           argument of periapsis of binary 1 (1+2 orbit) [radians]

       omegab2 : float
           argument of periapsis of binary 2. For mode==1, this is the second
           binary that forms a quadruple with binary 1 (i.e. the 3+4
           orbit). For mode==2, it is the outer orbit of the 'inner' triple,
           i.e. the (1+2) / 4 orbit [radians]

       omegab3 : float
           argument of periapsis of (1+2) / (3+4) orbit (mode==1) or the
           ((1+2)+4) / 3 orbit (mode==2) [radians]

       Pb1 : float
           period of binary 1 (1+2 orbit) [days]

       Pb2 : float
           period of binary 2. For mode==1, this is the second binary
           that forms a quadruple with binary 1 (i.e. the 3+4 orbit). For
           mode==2, it is the outer orbit of the 'inner' triple, i.e. the
           (1+2) / 4 orbit [days]

       Pb3 : float
           period of (1+2) / (3+4) orbit (mode==1) or the ((1+2)+4) / 3
           orbit (mode==2) [days]

       ib1 : float
           inclination of binary 1 (1+2 orbit) [radians]

       ib2 : float
           inclination of binary 2. For mode==1, this is the second binary
           that forms a quadruple with binary 1 (i.e. the 3+4 orbit). For
           mode==2, it is the outer orbit of the 'inner' triple, i.e. the
           (1+2) / 4 orbit [radians]

       ib3  : float
           inclination of (1+2) / (3+4) orbit (mode==1) or the ((1+2)+4) / 3
           orbit (mode==2) [radians]

       OMEGAb1 : float
           longitude of ascending node of binary 1 (1+2 orbit) [radians]

       OMEGAb2 : float
           longitude of ascending node of binary 2. For mode==1, this is the
           second binary that forms a quadruple with binary 1 (i.e. the 3+4
           orbit). For mode==2, it is the outer orbit of the 'inner' triple,
           i.e. the (1+2) / 4 orbit [radians]

       OMEGAb3 : float
           longitude of ascending node of (1+2) / (3+4) orbit (mode==1) or the
           ((1+2)+4) / 3 orbit (mode==2) [radians]

       t0b1 : float
           zeropoint of binary 1 (1+2 orbit) [days]

       t0b2 : float
           zeropoint of binary 2. For mode==1, this is the second binary
           that forms a quadruple with binary 1 (i.e. the 3+4 orbit). For
           mode==2, it is the outer orbit of the 'inner' triple, i.e. the
           (1+2) / 4 orbit [days]

       t0b3 : float
           zeropoint of (1+2) / (3+4) orbit (mode==1) or the ((1+2)+4) / 3
           orbit (mode==2) [days]

       ttype : int
           1 == t0 is time at periastron, 2 == t0 is time of eclipse

       mode : int
           1 == (1+2)+(3+4) orbits, 2 == ((1+2)+4)+3

       acc : float
           Accuracy parameter for solving Kepler's equation

    Returns ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4)) where each of
    x1,y1,z1,x2 etc is a vector equivalent to the input array of times.

    """

    # speed of light, AU/day
    c = 173.1446327

    mzerob1 = 0.
    mzerob2 = 0.
    mzerob3  = 0.
    if ttype == 2:
        mzerob1 = true2mean(3.*np.pi/2.-omegab1, eb1)
        mzerob2 = true2mean(3.*np.pi/2.-omegab2, eb2)
        mzerob3 = true2mean(3.*np.pi/2.-omegab3, eb3)

    # Calculate Mean and True Anomaly for quadruple orbit only
    mb3 = (2*np.pi/Pb3)*(t - t0b3) + mzerob3
    nub3 = mean2true(mb3,eb3,acc)

    # Calculate positions of the CoMs of each binary
    xb1,yb1,zb1 = position(nub3,OMEGAb3,omegab3,ib3,eb3,ab1)
    xb2,yb2,zb2 = position(nub3,OMEGAb3,omegab3+np.pi,ib3,eb3,ab2)

    # Compute mean & true anomalies with LTT corrections (for
    # quadruple orbit only)
    mb3b1 = (2*np.pi/Pb3)*(t + (zb1/c - t0b3)) + mzerob3
    mb3b2 = (2*np.pi/Pb3)*(t + (zb2/c - t0b3)) + mzerob3
    mb1 = (2*np.pi/Pb1)*(t + (zb1/c - t0b1)) + mzerob1
    mb2 = (2*np.pi/Pb2)*(t + (zb2/c - t0b2)) + mzerob2

    nub3b1 = mean2true(mb3b1,eb3,acc)
    nub3b2 = mean2true(mb3b2,eb3,acc)
    nub1 = mean2true(mb1,eb1,acc)
    nub2 = mean2true(mb2,eb2,acc)

    # Recalculate positions of binary CoMs
    xb1,yb1,zb1 = position(nub3b1,OMEGAb3,omegab3,ib3,eb3,ab1)
    xb2,yb2,zb2 = position(nub3b2,OMEGAb3,omegab3+np.pi,ib3,eb3,ab2)

    x1,y1,z1 = position(nub1,OMEGAb1,omegab1,ib1,eb1,a1)
    x2,y2,z2 = position(nub1,OMEGAb1,omegab1+np.pi,ib1,eb1,a2)
    x3,y3,z3 = position(nub2,OMEGAb2,omegab2,ib2,eb2,a3)
    x4,y4,z4 = position(nub2,OMEGAb2,omegab2+np.pi,ib2,eb2,a4)

    x1 += xb1
    y1 += yb1
    z1 += zb1

    x2 += xb1
    y2 += yb1
    z2 += zb1

    x3 += xb2
    y3 += yb2
    z3 += zb2

    x4 += xb2
    y4 += yb2
    z4 += zb2

    return ((x1,y1,z1),(x2,y2,z2),(x3,y3,z3),(x4,y4,z4))
