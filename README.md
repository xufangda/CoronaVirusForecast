# CoronaVirusForecast

This is specifically simulated for the Netherland on 15 Mar 2020, trying to forecast the outbreak after strict measures being applied on 13 Mar

Novice compartmental model with time-delay ODE, including incubation, quarantine, hospitalization, super spreader, quarantine leak, immunity, etc.

The parameters for the COVID-19 are generally referenced from other papers

-----Most parameters regarding medical containments are solely based on estimation and fittings------

Typically I assume under governmental control, the parameters of contanct rate 'beta_e' and quarantine rate 'k0' for the exposed flocks can significally change. One can apply the logistic function for the parameter modification under certain measures.

It is highly recommended that Markov change Monte Carlo (MCMC) is applied on different nodes for a more precise forecast
