"""
Solve analytically the Euler equations for compressible adiabatic ideal
gas in one dimension middleting from initial conditions of Riemann
problems.

[dvp][lr] | left and right states
g         | adiabatic constant
t         | time

Reference:
Toro, E.F. 2009, Riemann Solvers and Numerical Methods for Fluid
  Dynamics (3rd ed.; Berlin: Springer)
"""

import bisect
import math
import sys

import numpy

from solve import brentq as solve_eqn

def init(dl, vl, pl, dr, vr, pr):
  return lambda x: (dl, vl, pl) if x < 0 else (dr, vr, pr)

def soln(dl, vl, pl, dr, vr, pr, g):
  # side states refer to left and right states
  # subscripts l, r, and s refer to left, right, and side states
  # subscript m refers to middle states

  # velocity difference between side and middle states, equations (4.6) to
  # (4.8)
  def f(ds, ps, p):
    # decide if shock or rarefaction wave separates side and middle states
    # (that is, if states are connected by Hugoniot locus or integral curve)
    # pylint: disable-next=no-else-return
    if p > ps:
      a, b = 2/((g+1)*ds), (g-1)/(g+1)*ps
      return (p-ps) * math.sqrt(a/(p+b))
    # ruff: ignore[RET505]
    else:
      cs = math.sqrt(g*ps/ds)
      return 2*cs/(g-1) * ((p/ps)**((g-1)/(2*g))-1)

  # middle-state pressure, equation (4.5)
  pm = solve_eqn(
    lambda p: (vl-f(dl, pl, p)) - (vr+f(dr, pr, p)), 0, 10**10)
  # middle-state velocity, equation (4.9)
  vm = vl-f(dl, pl, pm)

  def solve_wave(ds, vs, ps, s):
    cs = math.sqrt(g*ps/ds)
    # decide if shock or rarefaction wave separates side and middle states
    # (that is, if states are connected by Hugoniot locus or integral curve)
    # pylint: disable-next=no-else-return
    if (a := pm/ps) > 1:
      # middle-state density, equations (4.50) and (4.57)
      b = (g-1)/(g+1)
      dm = ds * (a+b)/(a*b+1)
      # characteristic velocity of shock, equations (4.52) and (4.59)
      sh = vs + s*cs * math.sqrt(1+(g+1)/(2*g)*(a-1))
      return sh, math.nan, dm
    # ruff: ignore[RET505]
    else:
      # middle-state density, equations (4.53) and (4.60)
      dm = ds * a**(1/g)
      # head and tail characteristic velocities of rarefaction wave, equations
      # (4.54), (4.55), (4.61), and (4.62)
      sh = vs + s*cs
      st = vm + s*math.sqrt(g*pm/dm)
      return sh, st, dm

  wl = solve_wave(dl, vl, pl, -1)
  wr = solve_wave(dr, vr, pr, 1)

  def result(x):
    if x < vm:
      ds, vs, ps, s = dl, vl, pl, -1
      sh, st, dm = wl
    else:
      ds, vs, ps, s = dr, vr, pr, 1
      sh, st, dm = wr

    # pylint: disable-next=no-else-return
    if s*(x-sh) > 0:
      # side state
      return ds, vs, ps
    # ruff: ignore[RET505]
    elif s*(x-st) > 0:
      # rarefaction wave, equations (4.56) and (4.63)
      cs = math.sqrt(g*ps/ds)
      a = 2/(g+1)
      b = (a - s*(g-1)/(g+1)*(vs-x)/cs)**(2/(g-1))
      d = ds * b
      v = a * ((g-1)/2*vs - s*cs + x)
      p = ps * b**g
      return d, v, p
    else:
      # middle state
      return dm, vm, pm

  def make_grid(vl, vu, n=100):
    e = math.sqrt(sys.float_info.epsilon)
    # ruff: ignore[E731]  # pylint: disable-next=unnecessary-lambda-assignment
    pair = lambda x: [x*(1-math.copysign(e, x))-e, x*(1+math.copysign(e, x))+e]
    result = []
    sh, st, _ = wl
    if math.isnan(st):
      result += pair(sh)
    else:
      result += numpy.linspace(sh, st, n).tolist()
    result += pair(vm)
    sh, st, _ = wr
    if math.isnan(st):
      result += pair(sh)
    else:
      result += numpy.linspace(st, sh, n).tolist()
    i = bisect.bisect_right(result, vl)
    j = bisect.bisect_left (result, vu)
    return [vl, *result[i:j], vu]

  result.make_grid = make_grid
  result.char_velo = [wl[1::-1], vm, wr[:2]]
  return result

def demo(dl, vl, pl, dr, vr, pr, g, t):
  # ruff: ignore[PLC0415]  # pylint: disable-next=import-outside-toplevel
  from matplotlib import pyplot

  wi = init(dl, vl, pl, dr, vr, pr)
  ws = soln(dl, vl, pl, dr, vr, pr, g)

  e = math.sqrt(sys.float_info.epsilon)
  xi = numpy.asarray([-1/(2*t), -e, e, 1/(2*t)])
  xs = numpy.asarray(ws.make_grid(xi[0], xi[-1]))

  wi = numpy.transpose([wi(xi) for xi in xi])
  ws = numpy.transpose([ws(xs) for xs in xs])

  for i, title in enumerate(['density', 'velocity', 'pressure']):
    pyplot.subplot(2, 2, i+1)
    pyplot.plot(xi*t, wi[i], color=str(2/3), linestyle='dotted')
    pyplot.plot(xs*t, ws[i])
    pyplot.title(title)

  pyplot.show()

def main():
  # ruff: ignore[PLC0415]  # pylint: disable-next=import-outside-toplevel
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument(
    'problem', choices=[
      'single-shock',
      'sod', 'einfeldt', 'strong-shock-1', 'strong-shock-2',
      'three-discontinuities'])
  parser.add_argument(
    '--adiabatic-index', type=float, default=7/5)
  args = vars(parser.parse_args())

  g = args['adiabatic_index']
  problem = args['problem']

  if problem == 'single-shock':
    m = 2
    dr, pr = 1, 1
    l = 2*(m**2-1)/(m*(g+1))
    dl = dr * m/(m-l)
    pl = pr * ((2*m**2-1)*g+1)/(g+1)
    vl = l * math.sqrt(g*pr/dr)
    demo(
      dl, vl, pl, dr, 0, pr, g, 0.1)

  # test problems, section 4.3.3
  elif problem == 'sod':
    # Sod shock tube |[1978JCoPh..27....1S]|
    demo(1, 0, 1, 0.125, 0, 0.1, g, 0.25)
  elif problem == 'einfeldt':
    # strong rarefaction |[1991JCoPh..92..273E]|
    demo(1, -2, 0.4, 1, 2, 0.4, g, 0.15)
  elif problem == 'strong-shock-1':
    demo(1, 0, 1e3, 1, 0, 1e-2, g, 0.012)
  elif problem == 'strong-shock-2':
    demo(1, 0, 1e-2, 1, 0, 1e2, g, 0.035)
  elif problem == 'three-discontinuities':
    demo(5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950, g, 0.035)

  # ruff: ignore[ERA001]; failing due to extreme parameters
  # demo(1, -0.3, 1e-6, 1, 0.3, 0.01, g, 0.5)

if __name__ == '__main__':
  main()
