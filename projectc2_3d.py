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
#"""


from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np

# Create mesh and define function space

mesh = Mesh("sphere1.xml")
Q = FunctionSpace(mesh, "CG", 1)

# define parameters
T = 50  #final time
h = mesh.hmin()
dt = h

alpha = 0.01
p = 40 #setting the boundary parameter
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
indata = Expression("pow(big_R-sqrt(pow(x[0],2)+pow(x[1],2)),2)+pow(x[2],2)<=pow(small_r,2) ? p : 0", degree = 3, small_r = small_r, big_R = big_R, p = p) #enter initial condition
u0 = Function(Q)
u0 = interpolate(indata, Q)
print(u0)

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
result_file = open("set3_mesh1.txt","w")

while t < T:
 	t += dt
	#u0.t = t
	solve(a == L, u, bc)
	#plot(u)
	M = (u_initial - u)*dx
	mass = assemble(M)
	#plot(mesh)
	#interactive()
	result_file.write(str(t))
	result_file.write(";")
	result_file.write(str(mass))
	result_file.write("\n")
	
	u0.assign(u)

result_file.close()

