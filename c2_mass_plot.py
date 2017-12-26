#plot the mass loss

import numpy as np
import matplotlib.pyplot as pyplot

def readfile(myfile):
  #this function readfile datafile
  
  datafile = open(myfile,"r")
  time = []
  mass_loss = []
  for row in datafile:
    vars = [float(j) for j in row.split(";")]
    time.append(vars[0])
    mass_loss.append(vars[1])

  return [time, mass_loss]

result1 = readfile("set1_mesh1.txt")
result2 = readfile("set2_mesh1.txt")
result3 = readfile("set3_mesh1.txt")

pyplot.figure(figsize = (10,10))
pyplot.plot(result1[0], result1[1], label = "p = 10")
pyplot.plot(result2[0], result2[1], label = "p = 20")
pyplot.plot(result3[0], result3[1], label = "p = 40")

pyplot.xlabel("time(s)", color = "black", fontsize=14)
pyplot.ylabel("massloss", color = "black", fontsize=14)
pyplot.title("mass_loss for torus", fontsize=20)
pyplot.legend()

pyplot.savefig('mass_loss_mesh1.png')
