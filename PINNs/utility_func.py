import math 
def CRRA(wealth:float,gamma:float)->float:
    if gamma==1:
        return math.log(wealth)
    term=1-gamma
    return (wealth**term)/term
