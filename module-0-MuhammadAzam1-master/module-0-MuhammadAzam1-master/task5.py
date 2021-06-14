import visdom
from project.datasets import Simple, Split, Xor
 
vis = visdom.Visdom()

def classify_Simple(pt):
    "Classify based on x position"
    if pt[0] < 0.5:
        return 1.0
    else:
        return 0.0

def classify_Split(pt):
    if pt[0] > 0.20 and pt[0] < 0.80:
       return 0.0 
    else: 
        return 1.0  

def classify_XOR(pt):
    if pt[0] < 0.5 and pt[1] < 0.5:
        return 0.0
    elif pt[0] <= 0.5 and pt[1] >= 0.5:
        return 1.0
    elif pt[0] >= 0.5 and pt[1] <= 0.5: 
        return 1.0
    elif pt[0] > 0.5 and pt[1] > 0.5:
        return 0.0

N = 100
Simple(N, vis=True).graph("Simple", model=classify_Simple)
#Split(N, vis=True).graph("Split", model=classify_Split)
#Xor(N, vis=True).graph("Xor", model=classify_XOR)