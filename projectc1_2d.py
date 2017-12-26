#"""
#Part C for FEM Project torus
#2D torus with zero source term f(x) = 0 using FEniCS
#problem
# ut - alpha*Laplace(u) = f 
# u(0) = p on boundary |R-sqrt(x1^2 + x2^2)| <= r
# u(0) = 0 if not on boundary
#p = 10, R = 0.5, r = 0.2, T = 20 for 2D mesh
#using 2d mesh circle1, circle2, circle3
#"""

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
import time as time

# Create mesh and define function space

mesh = Mesh("circle1.xml")
Q = FunctionSpace(mesh, "CG", 1)

# define parameters
T = 20  #final time
h = mesh.hmin()
dt = h

alpha = 0.01
#setting the boundary parameter
p = 10 
big_R = 0.5
small_r = 0.2

# create subdomain for Dirichlet boundary
class DirichletBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary

#set up boundary condition
g = Constant(0.0)
bc = DirichletBC(Q, g, DirichletBoundary())

#Define initial condition
indata = Expression("pow(big_R-sqrt(pow(x[0],2)+pow(x[1],2)),2)<=pow(small_r,2) ? p : 0", degree = 2, small_r = small_r, big_R = big_R, p = p) #enter initial condition
u0 = Function(Q)
u0 = interpolate(indata, Q)


# Define variational problem
u = TrialFunction(Q)
v = TestFunction(Q)
f = Constant(0.0)

a = u*v*dx + alpha*dt*dot(grad(u),grad(v))*dx
L = (u0 + dt*f)*v*dx

u = u0
t = 0.0

#save vtkfile for visualization
vtkfile = File('vtkfile/circle1.pvd')
# Compute solution
while t < T:
 	t += dt
	solve(a == L,u,bc)
	vtkfile << (u,t)
	u0.assign(u)

