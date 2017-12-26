#"""
#Part C for FEM Project torus
#3D torus with zero source term f(x) = 0 using FEniCS
#computing mass loss and save to file
#problem
# ut - alpha*Laplace(u) = f 
# u(0) = p on boundary (R-sqrt(x1^2 + x2^2))^2 <= r^2
# u(0) = 0 if not on boundary
#p = 10, R = 0.5, r = 0.2, T = 20 for 3D mesh
#using 3d mesh sphere1 sphere2
#optimization that achieve total dose of 
# M = {10, 15, 30} at 
# t = {5, 7, 30}
#"""


from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import time as time
import scipy.optimize as optimize
from scipy.optimize import minimize
import scipy.interpolate

# Create mesh and define function space

mesh = Mesh("sphere1.xml")
Q = FunctionSpace(mesh, "CG", 1)

# define parameters
T = 50  #final time
h = mesh.hmin()
dt = h

alpha = 0.01

# create subdomain for Dirichlet boundary
class DirichletBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary

def my_optimization(data):
	p = data[0]
	big_R = data[1]
	small_r = data[2]

	#set up boundary condition
	g = Constant(0.0)
	bc = DirichletBC(Q, g, DirichletBoundary())

	#Define initial condition
	indata = Expression("pow(big_R-sqrt(pow(x[0],2)+pow(x[1],2)),2)+pow(x[2],2)<=pow(small_r,2) ? p : 0", degree = 3, small_r = small_r, big_R = big_R, p = p) #enter initial condition
	u0 = Function(Q)
	u0 = interpolate(indata, Q)

	# Define variational problem
	u = TrialFunction(Q)
	v = TestFunction(Q)	
	f = Constant(0.0)

	#copy initial data
	u_initial = Function(Q)
	u_initial = interpolate(indata, Q)

	#form finite element
	a = u*v*dx + alpha*dt*dot(grad(u),grad(v))*dx
	L = (u0 + dt*f)*v*dx

	u = u0
	t = 0.0

	save_t = []
	save_mass =[]
	i = 0


	while t < T:

 		t += dt
 		save_t.append(t)
		solve(a == L, u, bc)
		M = (u_initial - u)*dx
		mass = assemble(M)
		save_mass.append(mass)
		u0.assign(u)
		
	mass_interpolate = scipy.interpolate.interp1d(save_t, save_mass)
	#key time {5, 7, 30}
	#compare with target {10, 15, 30}
	M5 = mass_interpolate(5.0)
	M7 = mass_interpolate(7.0)
	M30 = mass_interpolate(30.0)

	F = (M5 - 10.0)**2 + (M7 - 15.0)**2 + (M30 - 30.0)**2

	return F

data = [20.0, 0.5, 0.1]
res = minimize(my_optimization, data, method='nelder-mead', options={'xtol':1e-3, 'disp':True})
print(res)

# Optimization terminated successfully.
#          Current function value: 0.712622
#          Iterations: 56
#          Function evaluations: 108
#  final_simplex: (array([[ 40.96337657,   0.49755718,   0.29778305],
#        [ 40.96397918,   0.49751391,   0.29779825],
#        [ 40.96411376,   0.49752591,   0.29779792],
#        [ 40.96307954,   0.49749975,   0.29779801]]), array([ 0.71262219,  0.7126222 ,  0.71262227,  0.71262238]))
#            fun: 0.71262218915890196
#        message: 'Optimization terminated successfully.'
#           nfev: 108
#            nit: 56
#         status: 0
#        success: True
#              x: array([ 40.96337657,   0.49755718,   0.29778305])
# #results:


