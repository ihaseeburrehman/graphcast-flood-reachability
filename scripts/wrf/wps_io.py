"""Minimal reader for WPS intermediate files (format 5, lat-lon grids)."""
import struct
import numpy as np


def read(path):
    b = open(path, "rb").read(); i = 0; recs = []
    while i < len(b):
        n = struct.unpack(">i", b[i:i + 4])[0]; recs.append(b[i + 4:i + 4 + n]); i += 8 + n
    out, grid = {}, None
    for k in range(0, len(recs), 5):
        h = recs[k + 1]
        field = h[60:69].decode().strip()
        xlvl, nx, ny, iproj = struct.unpack(">f3i", h[140:156])
        _, la0, lo0, dla, dlo, _ = struct.unpack(">8s5f", recs[k + 2][:28])
        out[(field, round(xlvl))] = np.frombuffer(recs[k + 4], ">f4").reshape(ny, nx).astype(np.float64)
        grid = (la0 + dla * np.arange(ny), lo0 + dlo * np.arange(nx))
    return out, grid
