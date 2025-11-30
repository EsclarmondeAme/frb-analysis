import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
import mpl_toolkits.mplot3d.axes3d as p3

# ---------------------------------------
# constants: your unified axis values
# ---------------------------------------
cmb_l = 152.62
cmb_b = 4.03

frb_sid_l = 160.39
frb_sid_b = 0.08

clock_l = 163.54
clock_b = -3.93

best_l = 159.85
best_b = -0.51

cmb = SkyCoord(l=cmb_l*u.deg, b=cmb_b*u.deg, frame="galactic")
frb_sid = SkyCoord(l=frb_sid_l*u.deg, b=frb_sid_b*u.deg, frame="galactic")
clock = SkyCoord(l=clock_l*u.deg, b=clock_b*u.deg, frame="galactic")
best = SkyCoord(l=best_l*u.deg, b=best_b*u.deg, frame="galactic")

# ---------------------------------------
# load FRBs
# ---------------------------------------
frbs = pd.read_csv("frbs.csv")
coords = SkyCoord(ra=frbs["ra"].values*u.deg,
                  dec=frbs["dec"].values*u.deg, frame="icrs").galactic

# Convert for plotting
lon = coords.l.deg
lon = np.where(lon > 180, lon - 360, lon)
lat = coords.b.deg

# ---------------------------------------
# FIGURE 1: sky map with axes
# ---------------------------------------
plt.figure(figsize=(12,6))
ax = plt.subplot(111, projection="mollweide")

ax.scatter(np.radians(lon), np.radians(lat),
           s=8, c="gray", alpha=0.5, label="FRBs")

def plot_axis(ax, sc, name, color):
    L = sc.l.deg if sc.l.deg < 180 else sc.l.deg - 360
    ax.scatter(np.radians(L), np.radians(sc.b.deg),
               s=300, marker="*", c=color, edgecolors="black",
               linewidths=1.5, label=name)

plot_axis(ax, cmb, "CMB axis", "red")
plot_axis(ax, frb_sid, "FRB sidereal", "blue")
plot_axis(ax, clock, "Atomic clock", "green")
plot_axis(ax, best, "Unified axis", "gold")

ax.grid(alpha=0.3)
ax.set_title("FRB sky map + CMB / FRB / clock axes (galactic)")
ax.legend(loc="upper left", fontsize=8)
plt.savefig("fig_sky_map.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------
# FIGURE 2: radius clustering around axis
# ---------------------------------------
target = cmb
seps = coords.separation(target).deg
radii = [10, 15, 20, 25, 30]

obs = [np.mean(seps < r) for r in radii]
exp = [(1 - np.cos(np.radians(r))) / 2 for r in radii]

plt.figure(figsize=(8,5))
plt.plot(radii, obs, "o-", label="observed fraction")
plt.plot(radii, exp, "o--", label="expected (isotropic)")
plt.xlabel("radius (deg)")
plt.ylabel("fraction of FRBs")
plt.title("FRB clustering toward CMB axis")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig_clustering.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------
# FIGURE 3: separation histogram
# ---------------------------------------
plt.figure(figsize=(8,5))
plt.hist(seps, bins=30, color="skyblue", edgecolor="black")
plt.axvline(30, color="orange", linestyle="--", label="30° threshold")
plt.xlabel("Separation from CMB axis (deg)")
plt.ylabel("Number of FRBs")
plt.title("FRB separation from CMB axis")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig_separation_hist.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------
# FIGURE 4: sidereal Rayleigh plot
# ---------------------------------------
if "mjd" in frbs.columns:
    t = Time(frbs["mjd"].values, format="mjd")
    chime = EarthLocation(lat=49.3223*u.deg, lon=-119.6167*u.deg, height=545*u.m)
    lst = t.sidereal_time("apparent", longitude=chime.lon).hour
    phases = 2*np.pi * lst / 24

    plt.figure(figsize=(6,6))
    plt.subplot(polar=True)
    plt.hist(phases, bins=24, color="lightgray", edgecolor="black")
    plt.title("Sidereal arrival-phase distribution")
    plt.savefig("fig_sidereal_rayleigh.png", dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------------------
# FIGURE 5: monte-carlo maximum separation null
# ---------------------------------------
# generate random triples of isotropic axes
N = 200000
max_seps = []
v = lambda: SkyCoord(l=np.random.uniform(0,360)*u.deg,
                     b=np.degrees(np.arcsin(np.random.uniform(-1,1)))*u.deg,
                     frame="galactic")

for _ in range(N):
    a, b_, c = v(), v(), v()
    vals = [
        a.separation(b_).deg,
        a.separation(c).deg,
        b_.separation(c).deg
    ]
    max_seps.append(max(vals))

plt.figure(figsize=(8,5))
plt.hist(max_seps, bins=60, color="lightgray", edgecolor="black")
plt.axvline(13.51, color="red", linestyle="--",
            label="observed max separation")
plt.xlabel("max separation (deg)")
plt.ylabel("frequency")
plt.title("Monte Carlo null: max separation of 3 random axes")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("fig_mc_null.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------
# FIGURE 6: energy dependence
# ---------------------------------------
if "fluence" in frbs.columns:
    proxy = frbs["fluence"].values
else:
    proxy = frbs["snr"].values

p1, p2 = np.percentile(proxy, [33,66])

def binlab(x):
    if x <= p1: return "low"
    if x <= p2: return "mid"
    return "high"

frbs["bin"] = [binlab(v) for v in proxy]
sep = coords.separation(best).deg

bins = ["low","mid","high"]
means = [np.mean(sep[frbs["bin"]==b]) for b in bins]

plt.figure(figsize=(7,5))
plt.bar(bins, means, color=["#9999ff","#7777ff","#5555ff"])
plt.ylabel("mean separation from unified axis (deg)")
plt.title("energy dependence of frb–axis alignment")
plt.grid(axis="y", alpha=0.3)
plt.savefig("fig_energy_bins.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------
# FIGURE 7: 3D axis visualization
# ---------------------------------------
def sph_to_cart(l,b):
    l = np.radians(l)
    b = np.radians(b)
    x = np.cos(b)*np.cos(l)
    y = np.cos(b)*np.sin(l)
    z = np.sin(b)
    return x,y,z

axes = {
    "CMB": (cmb_l,cmb_b,"red"),
    "FRB sidereal": (frb_sid_l,frb_sid_b,"blue"),
    "Clock": (clock_l,clock_b,"green"),
    "Unified": (best_l,best_b,"gold")
}

fig = plt.figure(figsize=(7,7))
ax3d = fig.add_subplot(111, projection='3d')

# draw unit sphere wireframe
u_vals = np.linspace(0,2*np.pi,40)
v_vals = np.linspace(-np.pi/2,np.pi/2,20)
xs = np.cos(v_vals)[:,None]*np.cos(u_vals)
ys = np.cos(v_vals)[:,None]*np.sin(u_vals)
zs = np.sin(v_vals)[:,None]*np.ones_like(u_vals)
ax3d.plot_wireframe(xs, ys, zs, color="gray", alpha=0.1)

for name,(l,b,color) in axes.items():
    x,y,z = sph_to_cart(l,b)
    ax3d.scatter(x,y,z, s=120, c=color, edgecolors="black")
    ax3d.text(x*1.1,y*1.1,z*1.1, name, fontsize=9)

ax3d.set_title("3D visualization of axes on the celestial sphere")
plt.savefig("fig_axis_3d.png", dpi=200, bbox_inches="tight")
plt.close()

# done
print("all figures generated successfully.")
