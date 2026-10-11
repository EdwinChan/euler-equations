"""
Solve numerically the Euler equations for compressible adiabatic ideal
gas in one dimension starting from general initial conditions using
finite-volume methods, specifically, Godunov and MUSCL\u2013Hancock schemes.

[wu]         | primitive and conserved variables
[wu]i        | initial state
[wu][esr]    | exact, numerical, and reference solutions
[wudvpe][lr] | reconstructions on left and right sides of interfaces
g            | adiabatic constant
x[lu]        | extent of simulation domain
x[vf]        | volume- and face-centered cell coordinates
xd           | cell width
[nm]         | physical and ghost cell count
t            | time
[dvp][lr]    | initial left and right states of Riemann problem

Reference:
Toro, E.F. 2009, Riemann Solvers and Numerical Methods for Fluid
  Dynamics (3rd ed.; Berlin: Springer)
"""

import math
import numbers
import sys
import warnings

import numpy

# small number
sml_num = math.sqrt(sys.float_info.epsilon)

# variable count
NV = 3
# names of primitive variables
prim_names = ['density', 'velocity', 'pressure']
# indices into primitive and conserved variable arrays
DN, VL, PR = range(NV)
DN, MO, EN = range(NV)
# indices into wave arrays
LF, CN, RT = range(NV)

def sliding_window(array, shift, axis):
  # ruff: ignore[PLC0415]  # pylint: disable-next=import-outside-toplevel
  from numpy.lib.stride_tricks import sliding_window_view

  array = numpy.asarray(array)
  axis = axis if axis >= 0 else axis + array.ndim
  array = sliding_window_view(array, array.shape[axis] - shift, axis)
  return numpy.moveaxis(array, [axis, -1], [0, axis+1])

def prim_to_cons(w, g):
  u = numpy.empty_like(w)
  u[DN] = w[DN]
  u[MO] = w[DN]*w[VL]
  u[EN] = w[DN]*w[VL]**2/2 + w[PR]/(g-1)
  return u

def cons_to_prim(u, g):
  w = numpy.empty_like(u)
  w[DN] = u[DN]
  w[VL] = u[MO]/u[DN]
  w[PR] = (u[EN] - u[MO]**2/(2*u[DN])) * (g-1)
  return w

def eigv(v, h, c):
  k = numpy.ones((NV, NV, *v.shape))
  k[LF][MO] = v-c
  k[LF][EN] = h-v*c
  k[CN][MO] = v
  k[CN][EN] = v**2/2
  k[RT][MO] = v+c
  k[RT][EN] = h+v*c
  return k

def spd_phys(w, g):
  return numpy.abs(w[VL]) + numpy.sqrt(g*w[PR]/w[DN])

def flx_phys(w, u):
  f = numpy.empty_like(w)
  f[DN] = u[MO]
  f[MO] = u[MO]*w[VL] + w[PR]
  f[EN] = (u[EN]+w[PR])*w[VL]
  return f

def make_grid(xl, xu, n, m, periodic, uniform):
  # ruff: ignore[PLC0415]  # pylint: disable-next=redefined-outer-name, import-outside-toplevel
  import numpy.random

  if xl > xu:
    raise ValueError('spatial limits are misordered')
  if not (isinstance(n, numbers.Integral) and n > 0):
    raise ValueError('physical cell count must be a positive integer')
  if not (isinstance(m, numbers.Integral) and m > 0):
    raise ValueError('ghost cell count must be a positive integer')

  xf = numpy.linspace(xl, xu, n+1)

  if not uniform:
    rng = numpy.random.default_rng()
    d = (xu-xl)/(n*3)
    xf[1:-1] += rng.uniform(-d, d, size=n-1)

  if periodic:
    xf = numpy.hstack([xf[-m-1:-1]-xu+xl, xf, xf[1:m+1]-xl+xu])
  else:
    d = (xu-xl)/n * numpy.arange(1, m+1)
    xf = numpy.hstack([xl-d[::-1], xf, xu+d])

  xv = (xf[:-1] + xf[1:]) / 2
  xd = xf[1:] - xf[:-1]

  return xv, xf, xd

def bc_none(w, m):
  _ = w, m  # pylint; unused-argument

def bc_periodic(w, m):
  w[:, :m] = w[:, -m*2:-m]
  w[:, -m:] = w[:, m:m*2]

def bc_transmit(w, m):
  w[:, :m] = w[:, m*2-1:m-1:-1]
  w[:, -m:] = w[:, -m-1:-m*2-1:-1]

def bc_reflect(w, m):
  bc_transmit(w, m)
  w[VL, :m] *= -1
  w[VL, -m:] *= -1

def lim_minmod(l, r):
  return numpy.select([l*r<0, abs(l)<abs(r), True], [0, l, r])

def lim_mc(l, c, r):
  d = numpy.copysign(numpy.minimum.reduce([abs(l), abs(c), abs(r)]), c)
  d *= (l*c >= 0) & (c*r >= 0)
  return d

def lim_char(w, u, ud, g, lim):
  d, v, p, e = w[DN], w[VL], w[PR], u[EN]
  h = (e+p)/d
  c = numpy.sqrt(g*p/d)
  # right eigenvectors of linearized waves, equation (3.13)
  k = eigv(v, h, c)
  # amplitudes of linearized waves
  y = numpy.empty_like(ud)
  y[:, CN] = (g-1)/c**2 * ((h-v**2)*ud[:, DN] + v*ud[:, MO] - ud[:, EN])
  y[:, LF] = 1/(2*c) * ((v+c)*ud[:, DN] - ud[:, MO]) - y[:, CN]/2
  y[:, RT] = ud[:, DN] - y[:, LF] - y[:, CN]
  # slope-limited amplitudes of linearized waves
  y = lim(*y)
  return numpy.sum(y[:, numpy.newaxis] * k, axis=0)

def rcn_trim(wl, ul, wr, ur):
  # convert from [wu]l and [wu]r being states on right and left sides of cells
  # to states on left and right sides of interfaces
  wl, wr = wl[:, :-1], wr[:, 1:]
  ul, ur = ul[:, :-1], ur[:, 1:]
  return wl, ul, wr, ur

def rcn_debug(w, wl, wr, xv, xf, n, m):
  # ruff: ignore[PLC0415]  # pylint: disable-next=redefined-outer-name, import-outside-toplevel, reimported
  from matplotlib import pyplot

  # pylint: disable-next=using-constant-test
  if False:
    i = DN
    pyplot.plot(xv, w[i], marker='x')
    for x1, x2, y1, y2 in zip(
        xf[m-1:n+m+1], xf[m:n+m+2], wr[i], wl[i], strict=True):
      pyplot.plot([x1, x2], [y1, y2], color='red', alpha=1/3)
    for x in xf:
      pyplot.axvline(x, color='black', alpha=0.1)
    pyplot.show()

def rcn_linear_prim(w, wd, g, xv, xf, xd, td):
  # slope at time step start, equation (14.33)
  wl = w[:, 1:-1] + wd * (xf[2:-1] - xv[1:-1])
  wr = w[:, 1:-1] + wd * (xf[1:-2] - xv[1:-1])
  ul = prim_to_cons(wl, g)
  ur = prim_to_cons(wr, g)
  # evolution by half time step, equation (14.34)
  ue = td/(2*xd[1:-1]) * (flx_phys(wl, ul) - flx_phys(wr, ur))
  ul -= ue
  ur -= ue
  wl = cons_to_prim(ul, g)
  wr = cons_to_prim(ur, g)
  return wl, ul, wr, ur

def rcn_linear_cons(u, ud, g, xv, xf, xd, td):
  # slope at time step start
  ul = u[:, 1:-1] + ud * (xf[2:-1] - xv[1:-1])
  ur = u[:, 1:-1] + ud * (xf[1:-2] - xv[1:-1])
  wl = cons_to_prim(ul, g)
  wr = cons_to_prim(ur, g)
  # evolution by half time step, equation (14.34)
  ue = td/(2*xd[1:-1]) * (flx_phys(wl, ul) - flx_phys(wr, ur))
  ul -= ue
  ur -= ue
  wl = cons_to_prim(ul, g)
  wr = cons_to_prim(ur, g)
  return wl, ul, wr, ur

def rcn_flat(w, u, g, xv, xf, xd, n, m, td):
  _ = g, xd, td  # pylint; unused-argument

  # first-order in space over smooth regions
  # (evolution by half time step does nothing)
  wl = wr = w[:, m-1:n+m+1]
  ul = ur = u[:, m-1:n+m+1]
  rcn_debug(w, wl, wr, xv, xf, n, m)
  return rcn_trim(wl, ul, wr, ur)

def rcn_minmod_prim(w, u, g, xv, xf, xd, n, m, td):
  _ = u  # pylint; unused-argument

  # second-order in space over smooth regions
  lr = sliding_window((w[:, 1:] - w[:, :-1]) / (xv[1:] - xv[:-1]), 1, 1)
  wd = lim_minmod(*lr)
  wl, ul, wr, ur = rcn_linear_prim(w, wd, g, xv, xf, xd, td)
  wl, wr = wl[:, m-2:n+m], wr[:, m-2:n+m]
  ul, ur = ul[:, m-2:n+m], ur[:, m-2:n+m]
  rcn_debug(w, wl, wr, xv, xf, n, m)
  return rcn_trim(wl, ul, wr, ur)

def rcn_minmod_char(w, u, g, xv, xf, xd, n, m, td):
  # second-order in space over smooth regions
  lr = sliding_window((u[:, 1:] - u[:, :-1]) / (xv[1:] - xv[:-1]), 1, 1)
  ud = lim_char(w[:, 1:-1], u[:, 1:-1], lr, g, lim_minmod)
  wl, ul, wr, ur = rcn_linear_cons(u, ud, g, xv, xf, xd, td)
  wl, wr = wl[:, m-2:n+m], wr[:, m-2:n+m]
  ul, ur = ul[:, m-2:n+m], ur[:, m-2:n+m]
  rcn_debug(w, wl, wr, xv, xf, n, m)
  return rcn_trim(wl, ul, wr, ur)

def rcn_mc_prim(w, u, g, xv, xf, xd, n, m, td):
  _ = u  # pylint; unused-argument

  # second-order in space over smooth regions
  l = (w[:, 1:-1] - w[:, :-2]) / (xv[1:-1] - xf[1:-2])
  c = (w[:, 2:]   - w[:, :-2]) / (xv[2:]   - xv[:-2])
  r = (w[:, 1:-1] - w[:, 2:])  / (xv[1:-1] - xf[2:-1])
  wd = lim_mc(l, c, r)
  wl, ul, wr, ur = rcn_linear_prim(w, wd, g, xv, xf, xd, td)
  wl, wr = wl[:, m-2:n+m], wr[:, m-2:n+m]
  ul, ur = ul[:, m-2:n+m], ur[:, m-2:n+m]
  rcn_debug(w, wl, wr, xv, xf, n, m)
  return rcn_trim(wl, ul, wr, ur)

def rcn_mc_char(w, u, g, xv, xf, xd, n, m, td):
  # second-order in space over smooth regions
  lcr = numpy.empty((3, *w[:, 1:-1].shape))
  lcr[0] = (u[:, 1:-1] - u[:, :-2]) / (xv[1:-1] - xf[1:-2])
  lcr[1] = (u[:, 2:]   - u[:, :-2]) / (xv[2:]   - xv[:-2])
  lcr[2] = (u[:, 1:-1] - u[:, 2:])  / (xv[1:-1] - xf[2:-1])
  ud = lim_char(w[:, 1:-1], u[:, 1:-1], lcr, g, lim_mc)
  wl, ul, wr, ur = rcn_linear_cons(u, ud, g, xv, xf, xd, td)
  wl, wr = wl[:, m-2:n+m], wr[:, m-2:n+m]
  ul, ur = ul[:, m-2:n+m], ur[:, m-2:n+m]
  rcn_debug(w, wl, wr, xv, xf, n, m)
  return rcn_trim(wl, ul, wr, ur)

def spd_roe_1(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  el, er = ul[EN], ur[EN]
  hl, hr = (el+pl)/dl, (er+pr)/dr
  # Roe-average characteristic velocities, equations (10.49) to (10.51)
  a = numpy.sqrt(numpy.maximum(sml_num, dl))
  b = numpy.sqrt(numpy.maximum(sml_num, dr))
  ve = (a*vl + b*vr) / (a+b)
  he = (a*hl + b*hr) / (a+b)
  ce = numpy.sqrt(numpy.maximum(0, (g-1)*(he-ve**2/2)))
  return ve-ce, None, ve+ce

def spd_roe_2(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  el, er = ul[EN], ur[EN]
  hl, hr = (el+pl)/dl, (er+pr)/dr
  # Roe-average characteristic velocities
  a = numpy.sqrt(numpy.maximum(sml_num, dl))
  b = numpy.sqrt(numpy.maximum(sml_num, dr))
  ve = (a*vl + b*vr) / (a+b)
  ce = (a*(hl-vl**2/2) + b*(hr-vr**2/2)) / (a+b)
  ce += a*b/(2*(a+b)**2) * (vl-vr)**2
  ce = numpy.sqrt(numpy.maximum(0, (g-1)*ce))
  return ve-ce, None, ve+ce

def spd_toro(wl, ul, wr, ur, g):
  _ = ul, ur  # pylint; unused-argument

  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  dl = numpy.maximum(sml_num, dl)
  dr = numpy.maximum(sml_num, dr)
  pl = numpy.maximum(sml_num, pl)
  pr = numpy.maximum(sml_num, pr)
  cl = numpy.sqrt(g*pl/dl)
  cr = numpy.sqrt(g*pr/dr)
  # middle-state pressure estimates
  def pe_pvrs():
    # primitive-variable Riemann solver, equations (10.61) and (10.62)
    return (pl+pr)/2 - (dl+dr)*(cl+cr)*(vr-vl)/8
  def pe_trrs():
    # two-rarefaction Riemann solver, equations (9.31) and (9.32)
    z = (g-1)/(2*g)
    return (
      numpy.maximum(0, cl+cr - (g-1)/2*(vr-vl)) /
      (cl*pl**-z + cr*pr**-z))**(1/z)
  def pe_tsrs():
    # two-shock Riemann solver, equations (9.31), (9.41), and (9.42)
    pe = numpy.maximum(0, pe_pvrs())
    al, bl = 2/((g+1)*dl), (g-1)/(g+1)*pl
    ar, br = 2/((g+1)*dr), (g-1)/(g+1)*pr
    gl = numpy.sqrt(al/(pe+bl))
    gr = numpy.sqrt(ar/(pe+br))
    return (gl*pl+gr*pr - (vr-vl)) / (gl+gr)
  def pe_anrs():
    # adaptive non-iterative Riemann solver, figure (9.4)
    pe = pe_pvrs()
    pn, px = numpy.minimum(pl, pr), numpy.maximum(pl, pr)
    # ruff: ignore[PLR2004]
    return numpy.select(
      [(px/pn<2) & (pn<pe) & (pe<px), pe<pn, True], [pe, pe_trrs(), pe_tsrs()])
  pe = pe_anrs()
  # pressure-based left and right characteristic velocities, equations (10.59)
  # and (10.60), or equations (4.52), (4.55), (4.59), and (4.62)
  # (if pressure estimate is exact, then characteristic velocities are also
  # exact)
  sl = vl - cl * numpy.sqrt(1 + (g+1)/(2*g) * numpy.maximum(0, pe/pl-1))
  sr = vr + cr * numpy.sqrt(1 + (g+1)/(2*g) * numpy.maximum(0, pe/pr-1))
  # center characteristic velocity, equation (10.37)
  # (if pressure estimate is exact, then characteristic velocities are ordered;
  # if additionally there are two shocks, then characteristic velocities are
  # also exact)
  a, b = dl*(sl-vl), dr*(sr-vr)
  sm = (pr-pl + a*vl-b*vr) / (a-b)
  if not (numpy.all(sl < sm) and numpy.all(sm < sr)):
    warnings.warn('characteristic velocities are misordered', stacklevel=2)
  return sl, sm, sr

def flx_roe(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  el, er = ul[EN], ur[EN]
  hl, hr = (el+pl)/dl, (er+pr)/dr
  cl = numpy.sqrt(numpy.maximum(0, g*pl/dl))
  cr = numpy.sqrt(numpy.maximum(0, g*pr/dr))
  ud = ur-ul
  # Roe-average characteristic velocities, equation (11.60)
  a = numpy.sqrt(numpy.maximum(sml_num, dl))
  b = numpy.sqrt(numpy.maximum(sml_num, dr))
  ve = (a*vl + b*vr) / (a+b)
  he = (a*hl + b*hr) / (a+b)
  ce = numpy.sqrt(numpy.maximum(sml_num, (g-1)*(he-ve**2/2)))
  # right eigenvectors of linearized waves, equation (11.59)
  k = eigv(ve, he, ce)
  # amplitudes of linearized waves, equation (11.69)
  y = numpy.empty_like(ud)
  y[CN] = (g-1)/ce**2 * ((he-ve**2)*ud[DN] + ve*ud[MO] - ud[EN])
  y[LF] = 1/(2*ce) * ((ve+ce)*ud[DN] - ud[MO]) - y[CN]/2
  y[RT] = ud[DN] - y[LF] - y[CN]
  # physical fluxes
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  # characteristic velocities, equation (11.58)
  s = numpy.empty_like(wl)
  s[LF] = ve-ce
  s[CN] = ve
  s[RT] = ve+ce
  # Roe flux, equations (11.29) and (11.59)
  f = (fl+fr - numpy.sum((y * abs(s))[:, numpy.newaxis] * k, axis=0)) / 2
  # left entropy fix, equations (11.136) and (11.137)
  dz = dl+y[LF]
  vz = (dl*vl + y[LF]*(ve-ce)) / (dl+y[LF])
  pz = (g-1) * (el + y[LF]*(he-ve*ce) - dz*vz**2/2)
  cz = numpy.sqrt(numpy.maximum(0, g*pz/dz))
  # left entropy fix, equation (11.119)
  szl, szr = vl-cl, vz-cz
  # left entropy fix, equations (11.120), (11.128), and (11.129)
  # ruff: ignore[SIM300]
  i = (szl<0) & (0<szr)
  if numpy.any(i):
    szl, szr = szl[i], szr[i]
    szl *= (szr-s[LF][i]) / (szr-szl)
    f[:, i] = fl[:, i] + y[LF][i] * szl * k[LF][:, i]
  # right entropy fix, equation (11.139)
  dz = dr-y[RT]
  vz = (dr*vr - y[RT]*(ve+ce)) / (dr-y[RT])
  pz = (g-1) * (er - y[RT]*(he+ve*ce) - dz*vz**2/2)
  cz = numpy.sqrt(numpy.maximum(0, g*pz/dz))
  # right entropy fix, equation (11.130)
  szl, szr = vz+cz, vr+cr
  # right entropy fix, equations (11.131), (11.133), and (11.134)
  # ruff: ignore[SIM300]
  i = (szl<0) & (0<szr)
  if numpy.any(i):
    szl, szr = szl[i], szr[i]
    szr *= (s[RT][i]-szl) / (szr-szl)
    f[:, i] = fr[:, i] - y[RT][i] * szr * k[RT][:, i]
  return f

def flx_hll(wl, ul, wr, ur, g):
  sl, _, sr = spd_toro(wl, ul, wr, ur, g)
  # HLL flux, equation (10.21)
  sl = numpy.minimum(-sml_num, sl)
  sr = numpy.maximum( sml_num, sr)
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  return (sr*fl - sl*fr + sl*sr*(ur-ul)) / (sr-sl)

def flx_hlle(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  cl = numpy.sqrt(numpy.maximum(0, g*pl/dl))
  cr = numpy.sqrt(numpy.maximum(0, g*pr/dr))
  # characteristic velocities
  # (agrees with other version over smooth regions)
  sl, _, sr = spd_roe_1(wl, ul, wr, ur, g)
  sl = numpy.minimum(sl, vl-cl)
  sr = numpy.maximum(sr, vr+cr)
  # HLLE flux, equation (10.21)
  sl = numpy.minimum(-sml_num, sl)
  sr = numpy.maximum( sml_num, sr)
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  return (sr*fl - sl*fr + sl*sr*(ur-ul)) / (sr-sl)

def flx_hlle_bad(wl, ul, wr, ur, g):
  _ = g  # pylint; unused-argument

  # characteristic velocities, equations (10.52) to (10.54)
  # (supposedly agrees with other version over smooth regions)
  sl, _, sr = spd_roe_2(wl, ul, wr, ur, 2)
  # HLLE flux, equation (10.21)
  sl = numpy.minimum(-sml_num, sl)
  sr = numpy.maximum( sml_num, sr)
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  return (sr*fl - sl*fr + sl*sr*(ur-ul)) / (sr-sl)

def flx_hllc_1(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  el, er = ul[EN], ur[EN]
  sl, sm, sr = spd_toro(wl, ul, wr, ur, g)
  # intermediate state, equation (10.39)
  o = numpy.ones_like(dl)
  uml = dl * (sl-vl)/(sl-sm) * [o, sm, el/dl+(sm-vl)*(sm+pl/(dl*(sl-vl)))]
  umr = dr * (sr-vr)/(sr-sm) * [o, sm, er/dr+(sm-vr)*(sm+pr/(dr*(sr-vr)))]
  # HLLC flux, equation (10.38)
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  fml = fl + sl*(uml-ul)
  fmr = fr + sr*(umr-ur)
  return numpy.select([sl>0, sm>0, sr>0, True], [fl, fml, fmr, fr])

def flx_hllc_2(wl, ul, wr, ur, g):
  dl, dr = wl[DN], wr[DN]
  vl, vr = wl[VL], wr[VL]
  pl, pr = wl[PR], wr[PR]
  sl, sm, sr = spd_toro(wl, ul, wr, ur, g)
  # HLLC flux, equation (10.41)
  fl = flx_phys(wl, ul)
  fr = flx_phys(wr, ur)
  m = numpy.zeros_like(wl)
  m[MO] = 1
  m[EN] = sm
  fml = (sm*(sl*ul-fl) + sl*m*(pl+dl*(sl-vl)*(sm-vl))) / (sl-sm)
  fmr = (sm*(sr*ur-fr) + sr*m*(pr+dr*(sr-vr)*(sm-vr))) / (sr-sm)
  return numpy.select([sl>0, sm>0, sr>0, True], [fl, fml, fmr, fr])

bcs = {
  'none':     bc_none,
  'periodic': bc_periodic,
  'transmit': bc_transmit,
  'reflect':  bc_reflect}
rcns = {
  'flat':   {'prim': rcn_flat,        'char': rcn_flat},
  'minmod': {'prim': rcn_minmod_prim, 'char': rcn_minmod_char},
  'mc':     {'prim': rcn_mc_prim,     'char': rcn_mc_char}}
flxs = {
  'roe':  flx_roe,
  'hll':  flx_hll,
  'hlle': flx_hlle,
  'hllc': flx_hllc_1}

def evolve(
    w, g, xv, xf, xd, n, m, tl, tu, *,
    bc='none', rcn='flat', rcn_var='prim', flx='hllc', cfl=0.9):
  if not len(xv) == len(xf)-1 == len(xd) == n+m*2:
    raise ValueError('state has incorrect cell count')
  if tl > tu:
    raise ValueError('temporal limits are misordered')
  t, w = +tl, +w

  if m < {'flat': 1, 'minmod': 2, 'mc': 2}[rcn]:
    raise ValueError('not enough ghost cells for reconstruction')

  bc  = bcs [bc]
  rcn = rcns[rcn][rcn_var]
  flx = flxs[flx]

  while True:
    # apply boundary condition
    bc(w, m)

    # apply floor
    i = w[DN] < sml_num
    if numpy.any(i):
      warnings.warn('density reaches floor', stacklevel=2)
      w[DN, i] = sml_num
    i = w[PR] < sml_num
    if numpy.any(i):
      warnings.warn('pressure reaches floor', stacklevel=2)
      w[PR, i] = sml_num

    yield +t, w
    if t == tu:
      break

    # calculate time step
    td = numpy.min(xd / spd_phys(w, g)) * cfl
    # pylint: disable-next=no-else-raise
    if td < sml_num:
      raise ValueError('step is too small')
    elif td > tu-t:
      t, td = tu, tu-t
    else:
      t += td

    # convert to conserved variables
    u = prim_to_cons(w, g)

    # reconstruct interface states
    wl, ul, wr, ur = rcn(w, u, g, xv, xf, xd, n, m, td)

    # calculate numerical flux
    f = flx(wl, ul, wr, ur, g)

    # update variables
    u[:, m:-m] += (f[:, :-1] - f[:, 1:]) / xd[m:-m] * td
    w = cons_to_prim(u, g)

# ruff: disable[E402]  # pylint: disable=wrong-import-position, wrong-import-order
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot

import iterate
import riemann
from rungekutta import interpolate
# ruff: enable[E402]  # pylint: enable=wrong-import-position, wrong-import-order

def calc_error(we, ws, xd, m):
  d = numpy.abs(ws[DN, m:-m] - we[DN, m:-m])
  return {
    'L1': numpy.average(d, weights=xd[m:-m]),
    'L2': numpy.sqrt(numpy.average(d**2, weights=xd[m:-m]))}

def format_error(error):
  return (f'{key}: {value}' for key, value in error.items())

def plot_static(xs, ws, xr, wr, m):
  for i, title in zip([DN, VL, PR], prim_names, strict=True):
    pyplot.subplot(2, 2, i+1)
    pyplot.scatter(
      xs[m:-m], ws[i, m:-m],
      marker='o', facecolor='none', edgecolor='C0')
    pyplot.plot(xr, wr[i], color=str(2/3), linestyle='dotted')
    pyplot.title(title)

def plot_animated(xv, t, tw, m):
  fps = 30
  tw = interpolate(tw, t/(fps*5))

  lines = []
  for i, title in zip([DN, VL, PR], prim_names, strict=True):
    pyplot.subplot(2, 2, i+1)
    line, = pyplot.plot([], [])
    lines.append(line)
    pyplot.title(title)

  def draw(tw):
    _, w = tw
    for i, line in zip([DN, VL, PR], lines, strict=True):
      line.set_data(xv[m:-m], w[i, m:-m])
      line.axes.relim()
      line.axes.autoscale_view()

  return FuncAnimation(
    pyplot.gcf(), draw, tw,
    interval=1000/fps, repeat=False, cache_frame_data=False)

def demo_advect(
    g, n, *,
    shape, uniform_grid, show_plot, print_error, animate, **kwargs):
  m = 2
  t = 1
  # pylint: disable-next=use-dict-literal
  kwargs = dict(bc='periodic', **kwargs)

  # ruff: disable[E731]  # pylint: disable=unnecessary-lambda-assignment
  if shape == 'sine':
    pert = lambda x: math.cos(2*math.pi*x)
  elif shape == 'square':
    pert = lambda x: numpy.floor(x*2-1/2)%2*2-1
  elif shape == 'triangle':
    pert = lambda x: (numpy.floor(x*2)%2*2-1) * (x*2%1-1/2)
  elif shape == 'sawtooth':
    pert = lambda x: x*2%2-1
  else:
    raise ValueError('recognized shape')

  init = lambda x: [1+0.5*pert(x), 1, 1e-4]
  exct = lambda x: init(x-t)
  # ruff: enable[E731]  # pylint: enable=unnecessary-lambda-assignment

  xv, xf, xd = make_grid(-0.5, 0.5, n, m, periodic=True, uniform=uniform_grid)
  wi = numpy.transpose([init(x) for x in xv])
  tw = evolve(
    wi, g, xv, xf, xd, n, m, 0, t if not animate else math.inf, **kwargs)

  if not animate:
    we = numpy.transpose([exct(x) for x in xv])
    xr = numpy.linspace(-0.5, 0.5, 101)
    wr = numpy.transpose([exct(x) for x in xr])
    (_, _), (_, ws) = iterate.first_last(tw)
    if print_error:
      print(*format_error(calc_error(we, ws, xd, m)), sep='  ')
    # ruff: ignore[ERA001]; debugging
    # print(
    #   numpy.sum((prim_to_cons(wi, g)*xd)[:, m:-m], axis=1),
    #   numpy.sum((prim_to_cons(ws, g)*xd)[:, m:-m], axis=1))
    plot_static(xv, ws, xr, wr, m)
  else:
    _ = plot_animated(xv, t, tw, m)

  if show_plot:
    pyplot.show()

def demo_linear_wave(
    g, n, *,
    uniform_grid, show_plot, print_error, animate, **kwargs):
  m = 2
  t = 1
  # pylint: disable-next=use-dict-literal
  kwargs = dict(bc='periodic', **kwargs)

  k = 2*math.pi
  # ruff: disable[E731]  # pylint: disable=unnecessary-lambda-assignment
  pert = lambda x: 1e-8 * math.cos(k*x)

  init = lambda x: [1+pert(x), pert(x), 1/g+pert(x)]
  exct = lambda x: init(x-t)
  # ruff: enable[E731]  # pylint: enable=unnecessary-lambda-assignment

  xv, xf, xd = make_grid(-0.5, 0.5, n, m, periodic=True, uniform=uniform_grid)
  wi = numpy.transpose([init(x) for x in xv])
  tw = evolve(
    wi, g, xv, xf, xd, n, m, 0, t if not animate else math.inf, **kwargs)

  if not animate:
    we = numpy.transpose([exct(x) for x in xv])
    xr = numpy.linspace(-0.5, 0.5, 101)
    wr = numpy.transpose([exct(x) for x in xr])
    (_, _), (_, ws) = iterate.first_last(tw)
    if print_error:
      print(*format_error(calc_error(we, ws, xd, m)), sep='  ')
    # ruff: ignore[ERA001]; debugging
    # print(
    #   numpy.sum((prim_to_cons(wi, g)*xd)[:, m:-m], axis=1),
    #   numpy.sum((prim_to_cons(ws, g)*xd)[:, m:-m], axis=1))
    plot_static(xv, ws, xr, wr, m)
  else:
    _ = plot_animated(xv, t, tw, m)

  if show_plot:
    pyplot.show()

def demo_riemann(
    dl, vl, pl, dr, vr, pr, g, xl, xu, n, t, *,
    uniform_grid, show_plot, print_error, animate, **kwargs):
  m = 2
  # pylint: disable-next=use-dict-literal
  kwargs = dict(bc='transmit', **kwargs)

  init = riemann.init(dl, vl, pl, dr, vr, pr)
  exct = riemann.soln(dl, vl, pl, dr, vr, pr, g)

  xv, xf, xd = make_grid(xl, xu, n, m, periodic=False, uniform=uniform_grid)
  wi = numpy.transpose([init(x) for x in xv])
  tw = evolve(
    wi, g, xv, xf, xd, n, m, 0, t if not animate else math.inf, **kwargs)

  if not animate:
    we = numpy.transpose([exct(x/t) for x in xv])
    xr = numpy.asarray(exct.make_grid(xl/t, xu/t))
    wr = numpy.transpose([exct(x) for x in xr])
    xr *= t
    (_, _), (_, ws) = iterate.first_last(tw)
    if print_error:
      print(*format_error(calc_error(we, ws, xd, m)), sep='  ')
    if uniform_grid and dl == dr and vl == -vr and pl == pr and xl == -xu:
      assert numpy.allclose(ws[DN, :],  ws[DN, ::-1])
      assert numpy.allclose(ws[VL, :], -ws[VL, ::-1])
      assert numpy.allclose(ws[PR, :],  ws[PR, ::-1])
    plot_static(xv, ws, xr, wr, m)
  else:
    _ = plot_animated(xv, t, tw, m)

  if show_plot:
    pyplot.show()

def main():
  # ruff: ignore[PLC0415]  # pylint: disable-next=import-outside-toplevel
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'problem', choices=[
      'sine', 'square', 'triangle', 'sawtooth',
      'linear-wave', 'single-shock',
      'sod', 'einfeldt', 'strong-shock', 'three-discontinuities',
      'stationary-contact-discontinuity',
      'stationary-isolated-contact-discontinuity',
      'moving-isolated-contact-discontinuity'])
  parser.add_argument(
    '--adiabatic-index', type=float, default=7/5)
  parser.add_argument(
    '--cell-count', type=int, default=100)
  parser.add_argument(
    '--nonuniform-grid', dest='uniform_grid', action='store_false')
  parser.add_argument(
    '--reconstruction', dest='rcn', choices=rcns, default='minmod')
  parser.add_argument(
    '--reconstruction-variables', dest='rcn_var',
    choices=['prim', 'char'], default='prim')
  parser.add_argument(
    '--riemann-solver', dest='flx', choices=flxs, default='hllc')
  parser.add_argument(
    '--courant', dest='cfl', type=float, default=0.9)
  parser.add_argument(
    '--hide-plot', dest='show_plot', action='store_false')
  parser.add_argument(
    '--animate', action='store_true')
  parser.add_argument(
    '--print-error', action='store_true')
  args = vars(parser.parse_args())
  if not args['show_plot'] and args['animate']:
    parser.error('--hide-plot and --animate cannot be used together')
  if args['print_error'] and args['animate']:
    parser.error('--print_error and --animate cannot be used together')

  g = args.pop('adiabatic_index')
  n = args.pop('cell_count')
  problem = args.pop('problem')

  # ruff: ignore[PLC0415]  # pylint: disable-next=import-outside-toplevel
  from solve import brentq as solve_eqn

  if problem in ['sine', 'square', 'triangle', 'sawtooth']:
    demo_advect(g, n, shape=problem, **args)
  elif problem == 'linear-wave':
    demo_linear_wave(g, n, **args)
  elif problem == 'single-shock':
    m = 2
    dr, pr = 1, 1
    l = 2*(m**2-1)/(m*(g+1))
    dl = dr * m/(m-l)
    pl = pr * ((2*m**2-1)*g+1)/(g+1)
    vl = l * math.sqrt(g*pr/dr)
    demo_riemann(
      dl, vl, pl, dr, 0, pr, g,
      -0.25, 0.75, n, 0.2, **args)

  # test problems, section 10.8
  elif problem == 'sod':
    # Sod shock tube |[1978JCoPh..27....1S]|
    demo_riemann(
      1, 0.75, 1, 0.125, 0, 0.1, g,
      -0.3, 0.7, n, 0.2, **args)
  elif problem == 'einfeldt':
    # strong rarefaction |[1991JCoPh..92..273E]|
    demo_riemann(
      1, -2, 0.4, 1, 2, 0.4, g,
      -0.5, 0.5, n, 0.15, **args)
  elif problem == 'strong-shock':
    demo_riemann(
      1, 0, 1e3, 1, 0, 1e-2, g,
      -0.5, 0.5, n, 0.012, **args)
  elif problem == 'three-discontinuities':
    demo_riemann(
      5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950, g,
      -0.4, 0.6, n, 0.035, **args)
  elif problem == 'stationary-contact-discontinuity':
    v = solve_eqn(
      lambda v: riemann.soln(1, v, 1e3, 1, v, 1e-2, g).char_velo[1], -50, 0)
    demo_riemann(
      1, v, 1e3, 1, v, 1e-2, g,
      -0.8, 0.2, n, 0.012, **args)
  elif problem == 'stationary-isolated-contact-discontinuity':
    demo_riemann(
      1.4, 0, 1, 1, 0, 1, g,
      -0.5, 0.5, n, 2, **args)
  elif problem == 'moving-isolated-contact-discontinuity':
    demo_riemann(
      1.4, 0.1, 1, 1, 0.1, 1, g,
      -0.4, 0.6, n, 2, **args)

if __name__ == '__main__':
  main()
