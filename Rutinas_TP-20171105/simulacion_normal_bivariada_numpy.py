#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simulacion de variables aleatorias normales correlacionadas
Metodo de Box-Muller y luego factorizacion de Cholesky de la matriz de Cov.
Version con uso de numpy
Por Ignacio Bello
"""
#Imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl


#--------------------------------------------------------------
#Funciones
#--------------------------------------------------------------

def f_hist(datos, lims, devolver_frecuencia=False):
    """
    Devuelve array con el valor de f_hist correspondiente a cada intervalo

    La funcion se puede definir: 
    f_hist(x) = f_hist[i] * 1{ lims[i] <= x < lims[i+1] }
    
    para graficarla conviene usar el comando hist de matplotlib
    """
    #incializacion
    n_bins = lims.size - 1
    frec_abs = np.zeros(n_bins)
    #conteo frecuencia absoluta
    for i in range(n_bins):
        frec_abs[i] = np.sum( (lims[i] <= datos) * (datos < lims[i+1]) )
    #calculo de funcion histograma
    hist = frec_abs  / (n_sim * (lims[1:] - lims[:-1]))
    #salida de resultados
    if devolver_frecuencia:
        return frec_abs, hist
    else:
        return hist

def fde(datos):
    """Devuevle arrays xx e yy para graficar una fde con step(xx,yy)"""
    n_datos = datos.size
    xx = datos.copy()
    xx.sort()
    yy = np.linspace(0, 1, n_datos+1)
    xx = np.hstack([xx, xx[-1]])
        #se agrega un punto extra para que cierre el dibujo    
    return xx, yy
    

def box_muller(UU):
    """
    Devuelve ZZ array (n, 2) de normales estandar independientes
    UU debe ser array (n, 2)
    """
    r = np.sqrt(-2*np.log(UU[:, 0]))
    theta = 2 * np.pi * UU[:, 1]
    ZZ = r * np.array([np.cos(theta), np.sin(theta)])
    return ZZ.T
    
def norm_biv(ZZ, Mu, Cov):
    """
    Devuelve XX array (n, 2) de normales correlacionadas
    ZZ debe ser array (n, 2)    
    Mu debe ser array (2,1), Cov array (2,2)
    """    
    #Descomposicion cholesky matrix 2x2    
    A = np.linalg.cholesky(Cov)
    XX = (A.dot(ZZ.T)+Mu).T
    return XX

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
aa = np.array([-4.5, -3., -2., -1., -0.5, 0., 0.5, 1., 2., 3., 4.5])
corregir_lims = True #Si la muestra escapa a los limites los amplia
eps = 1e-8

#Simulacion 
UU = np.random.rand(n_sim, 2) #(se podrian leer desde archivo)
ZZ = box_muller(UU)
Mu = np.array([[mu1], [mu2]])
Cov = np.array([ [var1, rho*(var1*var2)**0.5], 
                 [rho*(var1*var2)**0.5, var2] ])
XX = norm_biv(ZZ, Mu, Cov)

#Funciones histograma y distribucion empirica
if corregir_lims:
    aa[0] = np.min([aa[0], XX.min()])
    aa[-1] = np.max([aa[-1], XX.max() + eps])

frec_x1, hist_x1 = f_hist(XX[:,0], aa, True)
frec_x2, hist_x2 = f_hist(XX[:,1], aa, True)

print('Limites: ' + str(aa))
#print('Frec. X1: ' + str(frec_x2))
#print('Hist. X1: ' + str(hist_x2))
print('Frec. X2: ' + str(frec_x2))
print('Hist. X2: ' + str(hist_x2))

xx1_fde, yy1 = fde(XX[:,0])
xx2_fde, yy2 = fde(XX[:,1])

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
#pl.hist(XX[:,0], bins = aa, normed=True, label='hist', lw=2)
#pl.grid()
#pl.legend(loc='best')
#pl.xlabel('x1')
#pl.ylabel('hist')
#pl.title('n_sim = '+str(n_sim))
#figura3.show()

figura4 = pl.figure()
pl.hist(XX[:,1], bins = aa, normed=True, label='hist', lw=2)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x1')
pl.ylabel('hist')
pl.title('n_sim = '+str(n_sim))
figura4.show()

figura5 = pl.figure()
pl.plot(XX[:,0], XX[:,1], '.', label='x1, x2', lw=1)
pl.grid()
pl.legend(loc='best')
pl.xlabel('x1')
pl.ylabel('x2')
pl.xlim(-4.5, 4.5)
pl.ylim(-4.5, 4.5)
pl.title('Normal bivariada, rho = '+ str(rho) + ' n_sim = '+str(n_sim))
figura5.show()