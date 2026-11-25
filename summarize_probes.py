import json

with open('live_probe_results.json', 'r') as f:
    base = json.load(f)

with open('probe_prefixed_results.json', 'r') as f:
    prefixed = json.load(f)

print(f"Base probed: {len(base)}")
print(f"Prefixed probed: {len(prefixed)}")

print("\n--- ALL WORKING MODELS ---")
all_results = base + prefixed
seen = set()
unique_working = []
for r in all_results:
    if r['status'] == '200' and r['slug'] not in seen:
        seen.add(r['slug'])
        unique_working.append(r)

for r in unique_working:
    print(f"{r['slug']:<45} | {r['family']:<10} | native={r['native_schema']} | fallback={r['fallback_only']} | {r['notes']}")

print("\n--- NON-WORKING SUMMARY ---")
by_status = {}
for r in all_results:
    st = r['status']
    by_status[st] = by_status.get(st, 0) + 1
print("Status counts:", by_status)
