#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulacion de variables aleatorias normales correlacionadas
Metodo de Box-Muller y luego factorizacion de Cholesky de la matriz de Cov.
Version sencilla usando funciones, sin modulos numericos
Por Ignacio Bello
"""
#Imports
from __future__ import division
import random
import math
import matplotlib.pyplot as pl

#--------------------------------------------------------------
#Funciones
#--------------------------------------------------------------

def f_hist(datos, lims, devolver_frecuencia=False):
    """
    Devuelve lista con el valor de f_hist correspondiente a cada intervalo

    La funcion se puede definir: 
    f_hist(x) = f_hist[i] * 1{ lims[i] <= x < lims[i+1] }
    
    para graficarla conviene usar el comando hist de matplotlib
    """
    #incializacion
    n_bins = len(lims) - 1
    frec_abs = [0]*n_bins
    hist = [0]*n_bins
    #conteo frecuencia absoluta
    for i in range(len(datos)):
        for j in range(n_bins):
            if lims[j] <= datos[i] and datos[i] < lims[j+1]:
                frec_abs[j] += 1
    #calculo de funcion histograma
    for j in range(n_bins):
        hist[j] = frec_abs[j] / (n_sim * (lims[j+1] - lims[j]))
    #salida de resultados
    if devolver_frecuencia:
        return frec_abs, hist
    else:
        return hist

def fde(datos):
    """Devuevle listas xx e yy para graficar una fde con step(xx,yy)"""
    n_datos = len(datos)    
    xx = sorted(datos)
    yy = range(n_datos)
    for i in range(n_datos):
        yy[i] = yy[i]/n_datos
    #se agrega un punto extra para que cierre el dibujo    
    xx.append(xx[-1])
    yy.append(1.)
    return xx, yy
    

def box_muller(u1, u2):
    """Devuelve z1, z2 normales estandar independientes"""
    r = math.sqrt(-2*math.log(u1))
    theta = 2 * math.pi * u2
    z1 = r * math.cos(theta)
    z2 = r * math.sin(theta)
    return z1, z2
    
def norm_biv(z1, z2, mu1, mu2, var1, var2, rho):
    """Devuelve x1, x2 normales correlacionadas"""    
    #Descomposicion cholesky matrix 2x2    
    a = [[math.sqrt(var1), 0], 
         [rho * math.sqrt(var2), math.sqrt(var2 - rho**2 * var2)]]
    x1 = mu1 + a[0][0] * zz1[i]
    x2 = mu2 + a[1][0] * zz1[i] + a[1][1] * zz2[i]
    return x1, x2

def phi(x, mu=0, var=1):
    return 1/math.sqrt(2*var*math.pi) * math.exp(- (x-mu)**2 / (2*var))

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
eps = 1e-8

#Inicializacion de listas
uu = [0]*n_sim
vv = [0]*n_sim
zz1 = [0]*n_sim
zz2 = [0]*n_sim
xx1 = [0]*n_sim
xx2 = [0]*n_sim
n_bins = len(aa)-1

#Simulacion 
for i in range(n_sim):
    uu[i] = random.random() #Uniformes (se podrian leer desde archivo)
    vv[i] = random.random() #Uniformes (se podrian leer desde archivo)

for i in range(n_sim):
    zz1[i], zz2[i] = box_muller(uu[i], vv[i])
    xx1[i], xx2[i] = norm_biv(zz1[i], zz2[i], mu1, mu2, var1, var2, rho)

#Funciones histograma y distribucion empirica
if corregir_lims:
    aa[0] = min(aa[0], min(xx1), min(xx2))
    aa[-1] = max(aa[-1], max(xx1)+eps, max(xx2)+eps)
                                #indice -1 en python llama al ultimo elemento

frec_x1, hist_x1 = f_hist(xx1, aa, True)
frec_x2, hist_x2 = f_hist(xx2, aa, True)

print('Limites: ' + str(aa))
#print('Frec. X1: ' + str(frec_x2))
#print('Hist. X1: ' + str(hist_x2))
print('Frec. X2: ' + str(frec_x2))
print('Hist. X2: ' + str(hist_x2))

xx1_fde, yy1 = fde(xx1)
xx2_fde, yy2 = fde(xx2)

#figura1 = pl.figure()
#pl.step(xx1_fde, yy1, label='Fde', lw=2)
#pl.grid()
#pl.legend(loc='best')
#pl.xlabel('x1')
#pl.ylabel('Fde')
#pl.title('n_sim = '+str(n_sim))
#pl.xlim(-3., 3.)
#figura1.show()    

figura2 = pl.figure()
pl.step(xx2_fde, yy2, label='Fde', lw=2)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x2')
pl.ylabel('Fde')
pl.title('n_sim = '+str(n_sim))
pl.xlim(-3., 3.)
figura2.show()    

#figura3 = pl.figure()
#pl.hist(xx1, bins = aa, normed=True, label='hist', lw=2)
#pl.grid()
#pl.legend(loc='best')
#pl.xlabel('x1')
#pl.ylabel('hist')
#pl.title('n_sim = '+str(n_sim))
#figura3.show()

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