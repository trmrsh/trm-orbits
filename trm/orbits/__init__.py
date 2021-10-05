#!/usr/bin/env python

from __future__ import print_function

"""a module to calculate things to do with elliptical orbits. It includes a
class to define orbits, various routines to deal with elliptical orbits as
well (e.g. computation of various anomolies and the like), and a sub-module,
'mods', that implements hierarchical triple and quadruple Keplerian orbits.

Orbit basics
============

Useful things it is hard to remember with orbits: important planes to imagine
are the orbital plane and the plane of the sky. They intersect along a line
called the 'line of nodes'. At each end of this line orbiting objects either
come towards or away from us. The one where they go away is called the
'ascending node'.  The position angle of the ascending node is one of the
angles that defines the orientation of the orbit on the sky. It matters for
visual binaries but is indeterminate for others. Also for visual binaries, the
orbital inclination ranges from 0 to 180. 0 to 90 is reserved for binaries
that rotate counter-clockwise on the sky. The 'argument of periapsis' (omega)
is the angle measured from the ascending node to the periastron in the
direction of the orbit.

The periastron becomes ill-defined for small eccentricities thus the routines
have some options to reference epoch to the ascending node instead as this
helps MCMC runs.

Some routines construct orbits from a dictionary of parameters. This should be
defined as follows (this is a complete list and not all are needed for all
routines):

Ephemeris parameters:

 t0      -- zero point of binary ephemeris, days
 period  -- orbital period, days
 quad    -- quadratic term, days (assumed = 0 if not set)

General parameters:

 coord   -- coordinate system, 'Jacobi', 'Astro' or 'Marsh'
 tstart  -- reference time at which the rvms are computed (days)
 mass0   -- mass of object 0 (the binary for eclipsers) (solar)
 rint0   -- interaction radius for object 0 (AU)
 integ   -- True to use N-body integration, False for Kepler orbits
 gamma   -- systemic velocity (AU/day)

Planet parameters (the '1' becomes '2' for planet 2 etc):

 mass1    -- mass (solar)
 a1       -- semi-major axis (AU)
 epoch1   -- epoch at periastron if eperi1 is True, at ascending node if False
 e1       -- eccentricity if eomega1 is True
 omega1   -- argument of periapsis (rads) if eomega1 is True
 sqeco1   -- sqrt(e)*cos(omega) if eomega1 is False
 sqeso1   -- sqrt(e)*sin(omega) if eomega1 is False
 iangle1  -- orbital inclination (rads)
 Omega1   -- longitude of ascending node (rads)
 eperi1   -- flag, see above
 eomega1  -- flag, see above

The module provides three (approximate) parameterisations of N-body
orbits. One uses 'Jacobi' coordinates, one astrocentric ('Astro') coordinates,
the other Jacobi coordinates with a small change of the relationship between
semi-major axis and orbital frequency designed to reflect a hierarchical
((star,planet1),planet2) ordering. These last are called 'Marsh' for
short. 'Jacobi' coordinates are hierarchical, taking a set of objects in
groups, first m0, then m0+m1, then m0+m1+m2 etc. Successive coordinate vectors
point from the centre of mass of one group to the next object to be included.
e.g. the first points from 0 to 1, the second points from the CoM of (0+1) to
2, etc. 'k' factors referred to in the documentation are calculated as
m1/(m0+m1), m2/(m0+m1+m2) etc and allow one to get the motion of object 0 in
reflex to the others.

'Astro' coordinates are simply referenced to the centre of mass of object 0.
'k' is calculated as m1/(m0+m1), m2/(m0+m2), in the case of ptolorb, but k =
m1/m0 in the case of prorvm (internal use only) since the vectors are scaled
to barycentric by factors of m0/m0+m1 etc. They are the same as Jacobi in the
case of 2 bodies.

Experiments I have carried out suggest relatively little to choose between
them.

"""

from .core import *
from . import model
