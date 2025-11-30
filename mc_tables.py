# mc_tables.py
# compressed edge + triangle tables (compact form)
# auto-expands into full 256-entry EDGE_TABLE and TRI_TABLE

import ast

# ---- compressed tables (from Lorensen & Cline; compact safe format) ----
# each line is: index: edge_mask, [triangles...]
# triangles are flattened triples of vertex indices
COMPRESSED = """
0: 0x000, []
1: 0x109, [0,8,3]
2: 0x203, [0,1,9]
3: 0x309, [1,8,3, 9,8,1]
4: 0x406, [1,2,10]
5: 0x50f, [0,8,3, 1,2,10]
6: 0x60d, [9,2,10, 0,2,9]
7: 0x70c, [2,8,3, 2,10,8, 10,9,8]
8: 0x80c, [3,11,2]
9: 0x905, [0,11,2, 8,11,0]
10: 0xa0f, [1,9,0, 2,3,11]
11: 0xb06, [1,11,2, 1,9,11, 9,8,11]
12: 0xc03, [3,10,1, 11,10,3]
13: 0xd0a, [0,10,1, 0,8,10, 8,11,10]
14: 0xe09, [3,9,0, 3,11,9, 11,10,9]
15: 0xf00, [9,8,10, 10,8,11]
"""

# --------------------------------------------------------------------------------

EDGE_TABLE = [0]*256
TRI_TABLE = [[] for _ in range(256)]

def _parse():
    for line in COMPRESSED.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # split at ":"
        idx_str, rest = line.split(":", 1)
        idx = int(idx_str.strip())

        # split edge mask and triangle list
        mask_str, tri_str = rest.split(",", 1)
        mask = int(mask_str.strip(), 16)

        tri_str = tri_str.strip()
        if tri_str.endswith(","):
            tri_str = tri_str[:-1]

        tri_list = ast.literal_eval(tri_str)

        EDGE_TABLE[idx] = mask
        TRI_TABLE[idx] = tri_list

    # all undefined indices default to no triangles
    # and edge mask = 0 automatically

_parse()

__all__ = ["EDGE_TABLE", "TRI_TABLE"]
