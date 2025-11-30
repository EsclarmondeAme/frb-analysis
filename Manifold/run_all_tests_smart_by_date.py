import os
import subprocess
import csv
import re
from datetime import datetime

# CONFIG
LOG_FILE = "validated_test_log.csv"
CSV_ARG = "frbs_unified.csv"
IGNORE_LIST = {
    "run_all_tests_smart_by_date.py",
    "setup.py"
}

def needs_csv_argument(file_path):
    """Check if a script uses sys.argv[1] or argparse to expect a CSV."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().lower()
            if "sys.argv[1]" in content or "argparse" in content:
                return True
            if "pd.read_csv(sys.argv[1]" in content:
                return True
        return False
    except:
        return False

def get_test_scripts():
    all_files = os.listdir()
    scripts = [
        f for f in all_files
        if f.endswith(".py")
        and f not in IGNORE_LIST
        and os.path.getsize(f) > 0
    ]
    return sorted(scripts, key=lambda f: os.path.getctime(os.path.abspath(f)))

def run_script(script, use_csv=False):
    cmd = ["python", script]
    if use_csv:
        cmd.append(CSV_ARG)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900  # 15 min max per script
        )
        return {
            "script": script,
            "timestamp": datetime.fromtimestamp(os.path.getctime(script)).isoformat(),
            "status": "SUCCESS" if result.returncode == 0 else "FAIL",
            "return_code": result.returncode,
            "csv_used": use_csv,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except Exception as e:
        return {
            "script": script,
            "timestamp": datetime.fromtimestamp(os.path.getctime(script)).isoformat(),
            "status": "ERROR",
            "return_code": None,
            "csv_used": use_csv,
            "stdout": "",
            "stderr": str(e)
        }

def main():
    test_scripts = get_test_scripts()
    print(f"Found {len(test_scripts)} test scripts.\n")
    
    log_entries = []

    for script in test_scripts:
        use_csv = needs_csv_argument(script)
        print(f"→ Running: {script} {'(with CSV)' if use_csv else ''}")
        result = run_script(script, use_csv)
        log_entries.append(result)
    
    with open(LOG_FILE, mode="w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "script", "timestamp", "status", "return_code", "csv_used", "stdout", "stderr"
        ])
        writer.writeheader()
        writer.writerows(log_entries)
    
    print(f"\n✅ All tests complete. Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
