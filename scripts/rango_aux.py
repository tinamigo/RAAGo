#!/usr/bin/python3

"""
Copyright 2020 Agustín Santiago Gutiérrez

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from math import sqrt, exp, hypot, pi
from scipy.stats import norm


# NOTA: Esta implementacion modela una VERSION SIMPLIFICADA del rango, que solamente tiene algunas cosas en cuenta.
#       La implementacion completa de rango esta en C++, y es el codigo de la AGA (American Go Association) con cambios minimos (komi, etc).
#        En particular, en esta version simplificada:
#           -- No se estan calculando los nuevos desvios finales (sigma), solo los nuevos ratings (mu).
#           -- La "nueva probabilidad de victoria" se calcula por lo tanto con los desvios originales, y lo correcto seria usar los nuevos.
#           -- Lo anterior cambiaria mucho el resultado de la win-prob final cuando el partido fue "batacazo"
#           -- Se asume que la partida es sin ventaja, sin implementar las correciones de handicap.
#           -- Se asume que jugador 1 y jugador 2 juegan un torneo / evento rankeado (superliga) en el que participan en EXACTAMENTE un partido (entre ellos)
#           -- Por lo tanto, los numeros calculados son solo aproximados si los jugadores juegan mas partidos en el torneo (los cambios de rating "no se suman" simplemente)
#           -- Se asume que el jugador 1 es quien gana el partido
#           -- Las variaciones de rango calculadas deberian ser correctas, asumiendo que se cumplen todas las hipotesis anteriores

#sigma_px = 1.0568465 # Numero magico del modelo: en definitiva estima que el desvio estandar en la performance de un jugador es 0.7473
sigma_px_const = 1.43614
#[Comentario de Martín] ^^^^^ define el beta, o sea el desvio por rendimiento / performance

def match_win_prob(mu1, mu2):
    return norm.cdf((mu1 - mu2)/sigma_px_const)

def density(x, mu, sigma):
    nx = (x-mu)/sigma
    return exp(-0.5 * nx*nx) / (sigma * sqrt(2.0 * pi))

def new_ratings(mu1, mu2, sigma1, sigma2):
    # Funcion de verosimilitud bayesiana: buscamos los mu que maximizan esto, y esos son los nuevos ratings.
    # En un torneo aago, la funcion que se optimiza es de n variables (el mu de cada uno de los n jugadores), y consisten en la multiplicacion
    # de TODOS los factores "match_win_prob" correspondientes a cada partido, y a TODOS los factores "density" correspondientes a cada jugador.
    def f(nmu1, nmu2):
        return match_win_prob(nmu1, nmu2) * density(nmu1, mu1, sigma1) * density(nmu2, mu2, sigma2)
    # Doble busqueda ternaria
    # Asumimos que la funcion de verosimilitud es unimodal en ambas direcciones (quizas sea de hecho concava?)
    # Empiricamente parece funcionar bien el calculo... asi que debe ser XD
    def besty(x):
        a = -30.0
        b = 10.0
        while (b-a > 0.0005):
            c1 = a + (b-a)/3.0
            c2 = a + 2.0*(b-a)/3.0
            if (f(x,c1) > f(x,c2)):
                b = c2
            else:
                a = c1
        return 0.5 * (a+b)
    xa = -30.0
    xb = 10.0
    while xb-xa > 0.0005:
        c1 = xa + (xb-xa)/3.0
        c2 = xa + 2.0*(xb-xa)/3.0
        if (f(c1,besty(c1)) > f(c2,besty(c2))):
            xb = c2
        else:
            xa = c1
    bestx = 0.5 * (xa+xb)
    return bestx, besty(bestx)

def closegap(x):
    if x > 0.0:
        return x - 1.0
    else:
        return x + 1.0

def creategap(x):
    if x > 0.0:
        return x + 1.0
    else:
        return x - 1.0

def age_sigma(sigma, days):
    return hypot(sigma, 0.0005 * days)

def win_chance(mu1, mu2, sigma1, sigma2):
    STEPS = 51
    assert(STEPS % 2 == 1)
    KSIGMAS = 6 # Integramos hasta KSIGMAS desvios
    ret = 0.0
    A = -STEPS//2
    B = A + STEPS
    gap1 = 2 * KSIGMAS * sigma1 / float(STEPS)
    gap2 = 2 * KSIGMAS * sigma2 / float(STEPS)
    totalp = 0.0
    for i in range(A, B):
        for j in range(A,B):
            nmu1 = mu1 + i * gap1
            nmu2 = mu2 + j * gap2
            p = density(nmu1, mu1, sigma1) * density(nmu2, mu2, sigma2)
            totalp += p
            ret += p * match_win_prob(nmu1, nmu2)
    assert(abs(totalp * gap1 * gap2 - 1.0) < 0.01) # La probabilidad total tiene que dar 1!
    return ret * gap1 * gap2

def show_data(mu1, mu2, sigma1, sigma2, age_ranking_1_in_days, age_ranking_2_in_days):
    sigma1 = age_sigma(sigma1, age_ranking_1_in_days)
    sigma2 = age_sigma(sigma2, age_ranking_2_in_days)
    print("Previous rating for player 1 : {:.3f} ± {:.3f}".format(mu1, sigma1))
    print("Previous rating for player 2 : {:.3f} ± {:.3f}".format(mu2, sigma2))
    mu1 = closegap(mu1)
    mu2 = closegap(mu2)
    nmu1, nmu2 = new_ratings(mu1, mu2, sigma1 , sigma2)
    print("Estimated new rating for player 1: {:.3f} (+ {:.3f})".format(creategap(nmu1), nmu1 - mu1))
    print("Estimated new rating for player 2: {:.3f} (- {:.3f})".format(creategap(nmu2), mu2 - nmu2))
    print("Estimated old probability for player 1 win: {}".format(win_chance(mu1, mu2, sigma1, sigma2)))
    print("Estimated new probability for player 1 win: {}".format(win_chance(nmu1, nmu2, sigma1, sigma2))) # Se deberian usar los desvios finales, que son distintos... pero no los calculamos aca.


#show_data(2, 2, 1.0 , 1.0, 0, 0)
#show_data(3.748, -1.136, 0.219 , 0.465, 0, 0)
#show_data(-1.136, 3.748, 0.465, 0.219 , 0, 0)
#0.381


###################################################################################################################################################
###################################################################################################################################################
#### desde acá programado por Martín Amigo en base a lo de arriba, completando en base al paper de AGA https://www.usgo.org/sites/default/files/pdf/AGARatings-Math.pdf
###################################################################################################################################################
###################################################################################################################################################
# reminder: el jugador 1 es el que gana
# la diferencia con lo de arriba es que agrego handicap y komi, y algunos detalles relativos a eso que aparecen en las formulas del paper y arriba no


#esto calcula la evidencia: la integral del likelihood
# lo hace con una grilla. calcula los valores de handicap para cada punto, los multiplica por el ancho del rectangulo (en realidad multiplica al final)
# y los suma
def win_chance_hk(mu1, mu2, sigma1, sigma2, handicap, komi, parameters, black_win):
    STEPS = 51
    assert(STEPS % 2 == 1)
    KSIGMAS = 6 # Integramos hasta KSIGMAS desvios
    ret = 0.0
    A = -STEPS//2
    B = A + STEPS
    gap1 = 2 * KSIGMAS * sigma1 / float(STEPS)
    gap2 = 2 * KSIGMAS * sigma2 / float(STEPS)
    totalp = 0.0
    for i in range(A, B):
        for j in range(A,B):
            nmu1 = mu1 + i * gap1
            nmu2 = mu2 + j * gap2
            p = density(nmu1, mu1, sigma1) * density(nmu2, mu2, sigma2)
            totalp += p
            mwp = match_win_prob_hk(nmu1, nmu2, handicap, komi, parameters, black_win, sigma1, sigma2)
            ret += p * mwp
            # print("match_win_prob_hk")
            # print(mwp)

    assert(abs(totalp * gap1 * gap2 - 1.0) < 0.01) # La probabilidad total tiene que dar 1!
    return ret * gap1 * gap2

# probabilidad de ganar de un jugador con mu1 contra uno de mu2, con ese handicap y komi (asumiendo el peso correspondiente a omicron para handicap)
def match_win_prob_hk(mu1, mu2, handicap, komi, parameters, black_win, sigma1, sigma2):
    # print("Params")
    # print(mu1, mu2, handicap, komi, parameters, sigma1, sigma2)
    # print("Numerador")
    # print(mu1 - mu2 - d(handicap, komi, parameters))
    # print("Denominador")
    # print(sigma_px(handicap, komi, sigma1, sigma2))
    if black_win == 'False': #mu1 es white, mu2 es black. el handicap se suma a mu2
        return norm.cdf((mu1 - mu2 - d(handicap, komi, parameters))/sigma_px(handicap, komi, sigma1, sigma2))
    elif black_win == 'True': #mu1 es black, mu2 es white. el handicap se suma a mu1
        return norm.cdf((mu1 + d(handicap, komi, parameters) - mu2)/sigma_px(handicap, komi, sigma1, sigma2))
    else:
        print("ERROR: black_win no es True ni False")

def d(handicap, komi, parameters):
    if handicap == 0 or handicap == 1:
        return (0.580 - (0.0757*komi))
    else:
        return (parameters[0] *  handicap + parameters[1] - (0.0757*komi))

def sigma_px(handicap, komi, sigma1, sigma2):
    if handicap == 0 or handicap == 1:
        return (1.0649 - (0.0021976*komi) + 0.00014984*(komi**2))
    else:
        return ((-0.0035169*komi) + b(handicap))
    #return sqrt(sigma1**2 + 1 + sigma2**2 + 1)
    #los uno vienen del beta

def b(handicap):
    if handicap == 2:
        return (1.13672)
    elif handicap == 3:
        return (1.18795)
    elif handicap == 4:
        return (1.22841)
    elif handicap == 5:
        return (1.27457)
    elif handicap == 6:
        return (1.31978)
    elif handicap == 7:
        return (1.35881)
    elif handicap == 8:
        return (1.39782)
    elif handicap == 9:
        return (1.43614)
    else:
        print("ERROR: HANDICAP MAYOR A 9 O NEGATIVO")

#win_chance_hk(-13.855, 1.77653, 0.457907, 0.240014, 9.0, 0.5, [1,0, 0.0005])

# for handicap_weight in [1,-1,3,-3]:
#     komi = 0.5
#     handicap = 9
#     parameters = [handicap_weight, 0, 0.0005]
#     print("Peso de handicap: " + str(handicap_weight))
#     print("Caso 1:")
#     mu1 = -20.4490
#     mu2 = 3.69153
#     sigma1 = 1
#     sigma2 = 1
#     black_win = 'True'
#     ev_result = win_chance_hk(mu1, mu2, sigma1, sigma2, handicap, komi, parameters, black_win)
#     print("Evidencia: " + str(ev_result))

# print("Ines Perado")
# print(win_chance(-20,5,1,1))
# print("Esperado")
# print(win_chance(5,-20,1,1))
