# COVID simulation

# Novice compartmental model with time-delay ODE, including incubation, quarantine, hospitalization, super spreader, quarantine leak, immunity, etc.
# The parameters for the COVID-19 are generally referenced from other papers

# -----Most parameters regarding medical containments are solely based on estimation and fittings------

# Typically I assume under governmental control, the parameters of contanct rate 'beta_e' and quarantine rate 'k0' for the exposed flocks can significally change. One can apply the logistic function for the parameter modification under certain policies.

# It is highly recommended that Markov change Monte Carlo (MCMC) is applied on different nodes for a more precise forcast


from __future__ import division
import numpy as np
from pylab import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from pylab import cos, linspace, subplots
from ddeint import ddeint

def logistic(t,start,duration):
    """logistic function, e.g. Fermi-Dirac statistics
    from 'start', costs 'duration' days"""
    return 1-1/(np.exp((t-start-duration/2)/(duration/8))+1)

#parameters
time = np.arange(0,18,1)
data = np.array([1,2,5,10,18,23,38,82,128,188,264,321,382,503,614,804,959,1135])     # actual data from 27Feb
n = 17424978    # susceptible individuals
beta_0 = 0.8    # contact rate
gamma_e1 = 1/4  # daily regular heal rate
gamma_e2 = 1/10 # daily super-spreader heal rate
gamma_e3 = 1/3  # recover rate for hospitalization
be = 0.995      # regular infected proportion
Lambda = 482    # daily birth rate
mu = 0.000024   # daily population death rate
mu_d = 0.034    # death rate of non-hospitalized
mu_d1 = 0.01    # death rate of hospitalized
sigma = 1/6     # latent period converting rate
pro = 0.17      # latent period infectous ratio
m1 = 0.3        # decay const 1
m2 = 0.3        # decay const 2
effi = 0.01     # leak rate of isolation, 179/804 on 13Mar
theta = 0.02    # immune waning rate
xi = 0.0002     # immune recover rate
k0 = 0.05        # quarantine rate for exposed
k1 = 0.84       # quarantine rate for infectious
alpha = 0.14    # hospitalization rate for quarantined
phi = 0.14      # hospit rate for infectious, 4/15 on 4Mar, 2/44 on 5Mar, 5/46 on 6Mar, 115/804 on 13 Mar
eini = 250      # initial exposed number
delaye = 6      # incubation time
delayR = 10     # recover duration
delayi1 = 10    # heal duration
delayi2 = 1     # hospitalization delay
delayP = 1      # hospital. delay
delayP2 = 10    # heal duration
delayQ = 4      # quarantine to hospitalization time
delayH = 10     # heal duration

k_prime = 0.06
beta_prime = 0.08
# modification after policies

#basic reproduction number estimation
D1 = mu + k0 + sigma + xi
D2 = k1 + mu + phi + gamma_e1
D3 = mu + alpha + xi
D4 = mu + gamma_e3
R0 = (sigma*beta_0)/((mu + xi + sigma)*(mu + gamma_e1 + mu_d)) #Basic reproduction number w/o control
Rc = (beta_0*pro)/D1 + (beta_0*sigma)/(D1*D2) + (k0*beta_0*effi)/(D1*D3) + (sigma*k1*beta_0*effi)/(D1*D2*D3) + (sigma*phi*beta_0*effi)/(D1*D2*D4) + (k0*alpha*beta_0*effi)/(D1*D3*D4)
Rc_prime = (1-beta_prime/beta_0)*((beta_0*pro)/(D1+k_prime) + (beta_0*sigma)/(D1*D2) + ((k0+k_prime)*beta_0*effi)/((D1+k_prime)*D3) + (sigma*k1*beta_0*effi)/((D1+k_prime)*D2*D3) + (sigma*phi*beta_0*effi)/((D1+k_prime)*D2*D4) + (k0*alpha*beta_0*effi)/(D1*D3*D4))#Basic reproduction number w/ control, disregard super spreader


def model(Y,t,de,dr,di1,di2,dp,dp2,dq,dh):
    """ODE groups for balance equations. See compartmental model."""
    S,E,Q,I,P,H,R = Y(t)
    """corresponding to susp., expos., quaran., infec., super spreader, hospit., recov."""
    Rdr = Y(t-dr)[6]
    Ede = Y(t-de)[1]
    Idi1 = Y(t-di1)[3]
    Idi2 = Y(t-di2)[3]
    Pdp = Y(t-dp)[4]
    Pdp2 = Y(t-dp2)[4]
    Qdq = Y(t-dq)[2]
    Hdh = Y(t-dh)[5]
    """t-delay ODE"""
    k = k0+k_prime*logistic(t,19,7)
    beta_e = beta_0-beta_prime*logistic(t,19,7)
    
    dsdt = Lambda - mu*S - beta_e*np.exp(-((m1*I+m2*(Q+H))/n))*(I+P+effi*(H+Q)+pro*E)*S/n + theta*Rdr
    dedt = 2500000*np.exp(-5000*t**2)*t + beta_e*np.exp(-((m1*I+m2*(Q+H))/n))*(I+P+effi*(H+Q)+pro*E)*S/n - (mu+xi)*E - (k+ sigma)*Ede
    """inital condition of 250 import cases"""
    dqdt = k*Ede + k1*Idi2 - (xi + mu)*Q - alpha*Qdq
    didt = be*sigma*Ede - mu*I - (gamma_e1 + mu_d)*Idi1 - (phi + k1)*Idi2
    dpdt = (1 - be)*sigma*Ede - mu*P - (gamma_e2 + mu_d)*Pdp2 - (k1 + phi)*Pdp
    dhdt = alpha*Qdq + phi*Idi2 + (phi + k1)*Pdp - mu*H - (gamma_e3 + mu_d1)*Hdh
    drdt = gamma_e1*Idi1 + gamma_e2*Pdp2 + gamma_e3*Hdh + xi*(Q + E) - mu*R - theta*Rdr
    """balance equations"""
    return array([dsdt,dedt,dqdt,didt,dpdt,dhdt,drdt])

g = lambda t : array([n,0,0,0,0,0,0]) # initial value

nmax = 2000
tt = np.linspace(0,40,nmax) #time
yy = ddeint(model,g,tt,fargs=(delaye,delayR,delayi1,delayi2,delayP,delayP2,delayQ,delayH,))
# solving the ODEs
yy[np.where(yy<0)] = 0

heal, = plot(tt, yy[:,2]+yy[:,5],c='peru', lw=2) #plot the quarantine and hospitalizations
syndrom_heal, = plot(tt,yy[:,2]+yy[:,3]+yy[:,4]+yy[:,5],c='r',lw=2) 
# plot the quarantine, hospitalizations and the rest of illed patients
all_, = plot(tt, yy[:,1]+yy[:,2]+yy[:,3]+yy[:,4]+yy[:,5],c='m', lw=2)
# all unrecovered patients
scatter = plt.scatter(time, data, c='c')
# actual data

plt.text(0, yy[nmax-1,2]/0.8, r'$R_0=%.2f$''\n' r'$R_c \approx %.2f \rightarrow %.2f$'%(R0,Rc,Rc_prime))
plt.legend([all_, heal, syndrom_heal,scatter], ["All infected","Quarantine+Hospitalization", "All syndromatic","Actual data"])
xticks(np.arange(6,40,step = 15), ('Mar', '15', 'Apr', '15', 'May'))

plt.title("Forecast of future Netherland\nIf the measures on 13 Mar work a little")

plt.show()
