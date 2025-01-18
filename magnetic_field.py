import ppigrf

from datetime import datetime

r     = 6500 # kilometers from center of Earht
theta = 30   # colatitude in degrees
phi   = 4    # degrees east (same as lon)

date = datetime(2021, 1, 1)
Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, date) # returns radial, south, east

print(f"Br: {Br} nT, Btheta: {Btheta} nT, Bphi: {Bphi} nT")


