from datetime import datetime

# shared "sources" for cross-layer injections
# all times are within 2019–2021
CROSS_LAYER_SOURCES = [
    {
        "id": "XL1",
        "ra": 130.0,
        "dec": 25.0,
        "t0": datetime(2020, 5, 15, 12, 0, 0),
    },
    {
        "id": "XL2",
        "ra": 210.0,
        "dec": -5.0,
        "t0": datetime(2021, 2, 3, 6, 30, 0),
    },
]
