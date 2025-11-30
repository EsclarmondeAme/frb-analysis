# marching_cubes_numpy.py
# pure-python + numpy marching cubes
# uses compressed tables from mc_tables.py

import numpy as np
from mc_tables import EDGE_TABLE, TRI_TABLE

# vertex order per cube corner
CUBE_CORNERS = np.array([
    [0,0,0],
    [1,0,0],
    [1,1,0],
    [0,1,0],
    [0,0,1],
    [1,0,1],
    [1,1,1],
    [0,1,1]
], dtype=np.float32)

# edge -> corners
EDGE_INDEX = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def interpolate(p1, p2, v1, v2, iso):
    if abs(iso-v1) < 1e-12: return p1
    if abs(iso-v2) < 1e-12: return p2
    if abs(v1-v2) < 1e-12:  return p1
    t = (iso - v1) / (v2 - v1)
    return p1 + t * (p2 - p1)


def marching_cubes(volume, iso=0.0):
    Nx, Ny, Nz = volume.shape
    vertices = []
    faces = []

    for x in range(Nx-1):
        for y in range(Ny-1):
            for z in range(Nz-1):

                # read cube
                cube_vals = np.zeros(8)
                for i, corner in enumerate(CUBE_CORNERS):
                    cx, cy, cz = (corner + [x,y,z]).astype(int)
                    cube_vals[i] = volume[cx,cy,cz]

                # build cube index
                cube_index = 0
                for i, v in enumerate(cube_vals):
                    if v < iso:
                        cube_index |= (1 << i)

                mask = EDGE_TABLE[cube_index]
                if mask == 0:
                    continue

                # compute edge intersections
                edge_vertex = [None]*12
                for e in range(12):
                    if mask & (1 << e):
                        c1, c2 = EDGE_INDEX[e]
                        p1 = CUBE_CORNERS[c1] + [x,y,z]
                        p2 = CUBE_CORNERS[c2] + [x,y,z]
                        v1 = cube_vals[c1]
                        v2 = cube_vals[c2]
                        edge_vertex[e] = interpolate(p1, p2, v1, v2, iso)

                # create triangles
                tri_list = TRI_TABLE[cube_index]
                for i in range(0, len(tri_list), 3):
                    e1, e2, e3 = tri_list[i:i+3]
                    if edge_vertex[e1] is None: continue
                    if edge_vertex[e2] is None: continue
                    if edge_vertex[e3] is None: continue

                    v1 = edge_vertex[e1]
                    v2 = edge_vertex[e2]
                    v3 = edge_vertex[e3]

                    idx1 = len(vertices); vertices.append(v1)
                    idx2 = len(vertices); vertices.append(v2)
                    idx3 = len(vertices); vertices.append(v3)

                    faces.append([idx1, idx2, idx3])

    return np.array(vertices, dtype=np.float32), np.array(faces, dtype=np.int32)


if __name__ == "__main__":
    # demo: generate a sphere
    N = 32
    x,y,z = np.mgrid[-1:1:complex(0,N), -1:1:complex(0,N), -1:1:complex(0,N)]
    vol = x**2 + y**2 + z**2

    V, F = marching_cubes(vol, iso=0.5)
    print("vertices:", V.shape)
    print("faces:", F.shape)
    np.save("vertices.npy", V)
    np.save("faces.npy", F)

    print("saved: vertices.npy, faces.npy")
