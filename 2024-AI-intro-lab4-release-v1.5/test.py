from autograder import scorer
import numpy as np
while True:
    n=float(input())
    print(scorer(np.exp(-n),0.90,0.0,0.95))