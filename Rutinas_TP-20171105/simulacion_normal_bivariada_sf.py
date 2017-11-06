#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulacion de variables aleatorias normales correlacionadas
Metodo de Box-Muller y luego factorizacion de Cholesky de la matriz de Cov.
Version sencilla sin usar funciones y sin modulos numericos
Por Ignacio Bello
"""
#Imports
from __future__ import division
import random
import math
import matplotlib.pyplot as pl

#--------------------------------------------------------------
#Programa principal
#--------------------------------------------------------------

#Datos
n_sim = int(1e4) #Numero de simulaciones
mu1 = 0.0
mu2 = 0.0
var1 = 1.0
var2 = 1.0
rho = 0.8

#limites para histogramas - Modificar a gusto
aa = [-4., -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4.]
corregir_lims = True #Si la muestra escapa a los limites los amplia
eps = 1e-3

#Inicializacion de listas
uu = [0]*n_sim
vv = [0]*n_sim
zz1 = [0]*n_sim
zz2 = [0]*n_sim
xx1 = [0]*n_sim
xx2 = [0]*n_sim
n_bins = len(aa)-1
frec1 = [0]*n_bins
frec2 = [0]*n_bins
hist1 = [0]*n_bins
hist2 = [0]*n_bins

#Simulacion
for i in range(n_sim):
    uu[i] = random.random() #Uniformes (se podrian leer desde archivo)
    vv[i] = random.random() #Uniformes (se podrian leer desde archivo)

for i in range(n_sim):
    #Normales(0, 1) independientes por Box-Muller (ej. 4.18)
    r = math.sqrt(-2*math.log(uu[i]))
    theta = 2 * math.pi * vv[i]
    zz1[i] = r * math.cos(theta)
    zz2[i] = r * math.sin(theta)
    #Normal bivariada con descomposicion casera de Cholesky
    a = [[math.sqrt(var1), 0], 
         [rho * math.sqrt(var2), math.sqrt(var2 - rho**2 * var2)]]    
    xx1[i] = mu1 + a[0][0] * zz1[i]
    xx2[i] = mu2 + a[1][0] * zz1[i] + a[1][1] * zz2[i]

#Funciones histograma y distribucion empirica
#Conteo de frecuencia absoluta
for i in range(n_sim):        
    if corregir_lims:
        if xx1[i] < aa[0]:
            aa[0] = xx1[i]
        elif xx1[i] >= aa[-1]:
            aa[-1] = xx1[i] + eps
    for j in range(n_bins):
        if aa[j] <= xx1[i] and xx1[i]<aa[j+1]:
            frec1[j] += 1

for i in range(n_sim):        
    if corregir_lims:
        if xx2[i] < aa[0]:
            aa[0] = xx2[i]
        elif xx2[i] >= aa[-1]:
            aa[-1] = xx2[i] + eps
    for j in range(n_bins):
        if aa[j] <= xx2[i] and xx2[i]<aa[j+1]:
            frec2[j] += 1

#Calculo de funcion histograma
for i in range(n_bins):
    hist1[i] = frec1[i] / (n_sim * (aa[i+1] - aa[i]))
    hist2[i] = frec2[i] / (n_sim * (aa[i+1] - aa[i]))

#Salida por pantalla
print('Limites: ' + str(aa))
#print('Frec.1: ' + str(frec1))
#print('Hist.1: ' + str(hist1))
print('Frec.2: ' + str(frec2))
print('Hist.2: ' + str(hist2))

#Funcion distribucion empirica
xx1_ordenado = sorted(xx1)
xx2_ordenado = sorted(xx2)
yy = range(n_sim)
for i in range(len(yy)):
    yy[i] = yy[i]/n_sim

#Se agrega el ultimo escalon para que dibuje bien la Fde
xx1_ordenado.append(xx1_ordenado[-1])
xx2_ordenado.append(xx2_ordenado[-1])
yy.append(1.)

#figura1 = pl.figure()
#pl.step(xx1_ordenado, yy, label='Fde', lw=2)
#pl.grid()
#pl.legend(loc='best')
#pl.xlabel('x1')
#pl.ylabel('Fde')
#pl.title('n_sim = '+str(n_sim))
#pl.xlim(-3., 3.)
#figura1.show()    
#
figura2 = pl.figure()
pl.step(xx2_ordenado, yy, label='Fde', lw=2)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x2')
pl.ylabel('Fde')
pl.title('n_sim = '+str(n_sim))
pl.xlim(-3., 3.)
figura2.show()    
#
#figura3 = pl.figure()
#pl.hist(xx1, bins = aa, normed=True, label='hist', lw=2)
#pl.grid()
#pl.legend(loc='best')
#pl.xlabel('x1')
#pl.ylabel('hist')
#pl.title('n_sim = '+str(n_sim))
#figura3.show()
#
figura4 = pl.figure()
pl.hist(xx2, bins = aa, normed=True, label='hist', lw=2)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x1')
pl.ylabel('hist')
pl.title('n_sim = '+str(n_sim))
figura4.show()

#Grafico en el plano de los puntos (x1, x2) simulados
figura5 = pl.figure()
pl.plot(xx1, xx2, '.', label='x1, x2', lw=1)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x1')
pl.ylabel('x2')
pl.xlim(-4, 4)
pl.ylim(-4, 4)
pl.title('Normal bivariada, rho = '+ str(rho) + ' n_sim = '+str(n_sim))
figura5.show()