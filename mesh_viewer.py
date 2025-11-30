import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_mesh(verts_file="vertices.npy", faces_file="faces.npy"):
    verts = np.load(verts_file)
    faces = np.load(faces_file)
    return verts, faces


def show_mesh(vertices, faces):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # build triangles for plotting
    tris = vertices[faces]

    mesh = Poly3DCollection(tris, alpha=0.7)
    mesh.set_edgecolor("k")
    mesh.set_facecolor((0.4, 0.7, 1.0, 0.6))

    ax.add_collection3d(mesh)

    # auto scale
    max_range = (vertices.max(axis=0) - vertices.min(axis=0)).max()
    mid = vertices.mean(axis=0)
    ax.set_xlim(mid[0] - max_range/2, mid[0] + max_range/2)
    ax.set_ylim(mid[1] - max_range/2, mid[1] + max_range/2)
    ax.set_zlim(mid[2] - max_range/2, mid[2] + max_range/2)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) == 3:
        verts_file = sys.argv[1]
        faces_file = sys.argv[2]
    else:
        verts_file = "vertices.npy"
        faces_file = "faces.npy"

    print(f"loading: {verts_file}, {faces_file}")
    v, f = load_mesh(verts_file, faces_file)
    print(f"vertices shape: {v.shape}")
    print(f"faces shape: {f.shape}")

    show_mesh(v, f)


if __name__ == "__main__":
    main()
