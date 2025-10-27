import numpy as np 

f0s = 10

l1s = np.array([343,  0, 1.4])
l2s = np.array([500,  287, 1800])
l3s = np.array([1000,  291, 1900])
l4s = np.array([1800,  989, 2000])
l5s = np.array([2522,  1300, 2200])
l6s = np.array([3500,  1575, 2400])

l1b = np.array([1200,750,1800])
l2b = np.array([1600,900,1900])
l3b = np.array([2000,1100,2000])
l4b = np.array([2500,1400,2100])
l5b = np.array([3000,1700,2200])
l6b = np.array([3500,2000,2300])

lam1s = l1s/f0s 
lam2s = l2s/f0s
lam3s = l3s/f0s
lam4s = l4s/f0s
lam5s = l5s/f0s
lam6s = l6s/f0s


z1ss = l1s[-1] * l1s[1]
z2ss = l2s[-1] * l2s[1]
z3ss = l3s[-1] * l3s[1]
z4ss = l4s[-1] * l4s[1]
z5ss = l5s[-1] * l5s[1]
z6ss = l6s[-1] * l6s[1]

z1sp = l1s[-1] * l1s[0]
z2sp = l2s[-1] * l2s[0]
z3sp = l3s[-1] * l3s[0]
z4sp = l4s[-1] * l4s[0]
z5sp = l5s[-1] * l5s[0]
z6sp = l6s[-1] * l6s[0]

rs12s = (z2ss - z1ss)/(z2ss + z1ss)
rs23s = (z3ss - z2ss)/(z3ss + z2ss)
rs34s = (z4ss - z3ss)/(z4ss + z3ss)
rs45s = (z5ss - z4ss)/(z5ss + z4ss)
rs56s = (z6ss - z5ss)/(z6ss + z5ss)

rp12s = (z2sp - z1sp)/(z2sp + z1sp)
rp23s = (z3sp - z2sp)/(z3sp + z2sp)
rp34s = (z4sp - z3sp)/(z4sp + z3sp)
rp45s = (z5sp - z4sp)/(z5sp + z4sp)
rp56s = (z6sp - z5sp)/(z6sp + z5sp)


# --------------------------
z1bs = l1s[-1] * l1s[1]
z2bs = l2s[-1] * l2s[1]
z3bs = l3s[-1] * l3s[1]
z4bs = l4s[-1] * l4s[1]

z1bp = l1s[-1] * l1s[0]
z2bp = l2s[-1] * l2s[0]
z3bp = l3s[-1] * l3s[0]
z4bp = l4s[-1] * l4s[0]

