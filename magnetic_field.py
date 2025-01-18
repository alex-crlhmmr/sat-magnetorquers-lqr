import ppigrf
import numpy as np

from datetime import datetime

r     = 6500 # kilometers from center of Earth
theta = 30   # colatitude in degrees
phi   = 4    # degrees east (same as lon)

date = datetime(2021, 1, 1) # year, month, day
Br, Btheta, Bphi = ppigrf.igrf_gc(r, theta, phi, date) # returns radial, south, east

#print(f"Br: {Br} nT, Btheta: {Btheta} nT, Bphi: {Bphi} nT")

def magnetic_field(lat,lon,alt,date):
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt, date) # returns east, north, up
    return np.array([Be, Bn, Bu])