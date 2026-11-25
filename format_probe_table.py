import json

with open('live_probe_results.json', 'r') as f:
    base = json.load(f)
with open('probe_prefixed_results.json', 'r') as f:
    pref = json.load(f)

all_probes = base + pref
# Deduplicate by slug
seen = {}
for r in all_probes:
    seen[r['slug']] = r

print(f"Total unique slugs tested: {len(seen)}")

# Group by family
by_fam = {}
for slug, r in seen.items():
    fam = r['family']
    by_fam.setdefault(fam, []).append(r)

for fam in ["Anthropic", "xAI", "DeepSeek", "Google", "OpenAI", "IBM", "Qwen", "Mistral", "Meta"]:
    if fam not in by_fam:
        continue
    items = by_fam[fam]
    w = [i for i in items if i['status'] == '200']
    print(f"\n### {fam} (Tested: {len(items)}, Responding: {len(w)})")
    for i in sorted(items, key=lambda x: (x['status'] != '200', x['slug'])):
        print(f"- `{i['slug']}` | {i['status']} | native={i['native_schema']} | fallback={i['fallback_only']} | {i['notes']}")
