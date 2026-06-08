import numpy as np 
def pi_star(r:float,mu:float,sigma:float,gamma:float)->float:
    #Then the pi_star value would be
    return (mu - r) / (gamma * sigma**2) #value to invest in risky asset


