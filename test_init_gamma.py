from init_gamma0 import get_cosserat_gamma0

from txt_reader import positions1,positions2, rotations1, rotations2

import numpy as np

print("positions1 shape:", positions1.shape)
print("positions2 shape:", positions2.shape)
# Define start and end positions

i=54

start = positions2[i]
end = positions1[i]



R1 = rotations2[i]
R2 = rotations1[i]


# Get the Cosserat gamma0
gamma0 = get_cosserat_gamma0(start, end, 
                             R1=R1, R2=R2,
                             n_elem=49, E=3e7, plot=True,print_=True)
print("Gamma0:", gamma0)
