import gzip

F = "COD0MGXFIN_20240360000_01D_30S_CLK.CLK.gz"

with gzip.open(F, "rt", errors="ignore") as f:
    count = 0
    header_passed = False

    for line in f:
        line = line.rstrip()

        # detect end of header
        if "END OF HEADER" in line:
            header_passed = True
            continue

        # after header, print first 80 non-empty lines
        if header_passed:
            if line.strip():
                print(line)
                count += 1
            if count >= 80:
                break
