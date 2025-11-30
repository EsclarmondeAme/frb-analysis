import tabula
import pandas as pd

# Extract all tables from the PDF
tables = tabula.read_pdf("nist_ptb_clock.pdf", pages="all", multiple_tables=True)

# Look for the table with MJD and fractional frequency
target = None
for t in tables:
    cols = [c.lower() for c in t.columns if isinstance(c, str)]
    if any("mjd" in c for c in cols) and any("fraction" in c or "freq" in c for c in cols):
        target = t
        break

if target is None:
    raise RuntimeError("Could not find a clock table in the PDF.")

# Clean up
df = target.rename(columns=lambda x: x.strip())
df.to_csv("nist_ptb_clock.csv", index=False)

print("done — saved nist_ptb_clock.csv")
