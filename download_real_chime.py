import pandas as pd
import requests
import json

print("Attempting to download REAL CHIME/FRB Catalog...")
print("=" * 60)

# Try multiple sources
sources_tried = []

# Source 1: TNS (Transient Name Server) FRB reports
try:
    print("\n[1/4] Trying TNS FRB database...")
    # TNS has some FRBs but requires registration - skip for now
    sources_tried.append("TNS - requires auth")
    print("  ✗ Requires authentication")
except Exception as e:
    sources_tried.append(f"TNS - {e}")

# Source 2: FRBCAT (historical catalog)
try:
    print("\n[2/4] Trying FRBCAT...")
    url = "http://frbcat.org/download"
    frbs = pd.read_csv(url, low_memory=False)
    
    if len(frbs) > 10:
        print(f"  ✓ Success! Downloaded {len(frbs)} FRBs from FRBCAT")
        
        # Clean and save
        # FRBCAT has columns like: frb_name, utc, ra, dec, dm, flux, width
        if 'utc' in frbs.columns or 'UTC' in frbs.columns:
            frbs.to_csv('chime_real_frbs.csv', index=False)
            print(f"\n✓ Saved to 'chime_real_frbs.csv'")
            print("\nFirst few FRBs:")
            print(frbs.head())
            print("\nColumns:")
            print(list(frbs.columns))
            exit(0)
        
except Exception as e:
    sources_tried.append(f"FRBCAT - {e}")
    print(f"  ✗ Failed: {e}")

# Source 3: Direct CHIME/FRB API (might be down)
try:
    print("\n[3/4] Trying CHIME/FRB API...")
    url = "https://www.chime-frb.ca/api/catalog"
    response = requests.get(url, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        if len(data) > 10:
            frbs = pd.DataFrame(data)
            frbs.to_csv('chime_real_frbs.csv', index=False)
            print(f"  ✓ Success! Downloaded {len(frbs)} FRBs")
            print("\nFirst few:")
            print(frbs.head())
            exit(0)
            
except Exception as e:
    sources_tried.append(f"CHIME API - {e}")
    print(f"  ✗ Failed: {e}")

# Source 4: Astronomical archives
try:
    print("\n[4/4] Trying VizieR astronomical database...")
    # VizieR catalog: J/ApJS/257/59 (CHIME/FRB Catalog 1)
    url = "https://vizier.cds.unistra.fr/viz-bin/votable?-source=J/ApJS/257/59/table1&-out.max=10000"
    
    # This returns VOTable format - pandas can read it
    frbs = pd.read_xml(url)
    
    if len(frbs) > 10:
        print(f"  ✓ Success! Downloaded {len(frbs)} FRBs from VizieR")
        frbs.to_csv('chime_real_frbs.csv', index=False)
        print("\nFirst few:")
        print(frbs.head())
        print("\nColumns:")
        print(list(frbs.columns))
        exit(0)
        
except Exception as e:
    sources_tried.append(f"VizieR - {e}")
    print(f"  ✗ Failed: {e}")

# If we get here, all sources failed
print("\n" + "=" * 60)
print("❌ Could not download real CHIME data from any source")
print("\nSources tried:")
for s in sources_tried:
    print(f"  • {s}")

print("\n" + "=" * 60)
print("ALTERNATIVE: Use published CHIME/FRB Catalog data")
print("=" * 60)
print("\nThe CHIME/FRB collaboration has published catalogs:")
print("  • Catalog 1: 536 FRBs (2019-2021)")
print("  • https://www.chime-frb.ca/catalog")
print("\nWe can manually create a dataset from their published tables.")
print("Or search for downloadable versions on:")
print("  • https://www.canfar.net/storage/list/AstroDataCitationDOI")
print("  • https://arxiv.org/abs/2106.04352 (paper with data)")

print("\nShould I:")
print("  A) Create sample based on published CHIME rates/properties")
print("  B) Try alternate download method")
print("  C) Proceed with analysis using what we have")