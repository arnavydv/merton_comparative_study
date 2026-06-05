import torch
def terminal_condition(number_of_points,T,w_max,w_min):
    T=torch.ones(number_of_points)*T
    w_max_points=number_of_points//2
    w_min_points=number_of_points//2
    W_max_p=torch.ones(w_max_points)*w_max
    w_min_p=torch.ones(w_min_points)*w_min
    return T,W_max_p,w_min_p   
def crra(wealth,gamma):
    term=1-gamma
    return wealth**term / term