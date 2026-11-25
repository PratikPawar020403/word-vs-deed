import sys, os
sys.path.insert(0, "fake")
os.environ["CONVERT_OR_HOLD_IMPORT_ONLY"] = "1"

import run_pilot_6row

print("Testing run_pilot_6row with fake SDK...")
run_pilot_6row.main()
print("Pilot run finished successfully with fake SDK!")
