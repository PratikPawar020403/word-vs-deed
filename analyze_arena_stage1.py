import json, math, collections
import numpy as np

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)

def main():
    try:
        with open("arena_stage1_results.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print("Could not open arena_stage1_results.json:", e)
        return

    mems = data["member_records"]
    rows = data["row_records"]
    errors = data.get("errors", [])
    claude_hits = data.get("claude_cap_hits", [])
    comps = data.get("comprehension_scores", {})

    print(f"Total Rows: {len(rows)}, Total Member Records: {len(mems)}")
    print(f"Total Errors: {len(errors)}, Claude Cap Hits: {len(claude_hits)}")

    # 1. Operational Verification
    ballot_oks = sum(m["ballot_ok"] for m in mems)
    ballot_rate = ballot_oks / len(mems) if mems else 0
    has_403 = any("403" in str(e) for e in errors)
    print(f"Ballot Parse Rate: {ballot_rate*100:.1f}% ({ballot_oks}/{len(mems)})")
    print(f"Encountered 403s: {has_403}")

    # 2. Per-Model Analysis
    models = sorted(list(set(m["model"] for m in mems)))
    print("\n" + "="*80)
    print("PER-MODEL LIE_CONVERT RATE BY OWN MEMBER FEAR x BONUS (WITH WILSON 95% CIs)")
    print("="*80)

    for mod in models:
        mod_mems = [m for m in mems if m["model"] == mod]
        comp_s = comps.get(mod, {}).get("score", "N/A")
        print(f"\nModel: {mod} (Family: {mod_mems[0]['family']}, Total games: {len(mod_mems)}, Comp Score: {comp_s})")
        print(f"{'Condition':<20} | {'Lie/N':<10} | {'Rate':<8} | {'Wilson 95% CI':<20}")
        print("-" * 65)

        for fear in ["low", "medium", "high"]:
            for bonus in [0, 3]:
                cell = [m for m in mod_mems if m["member_fear"] == fear and m["bonus"] == bonus]
                k = sum(m["lie_convert"] for m in cell)
                n = len(cell)
                p, lo, hi = wilson_ci(k, n)
                cond_name = f"{fear.capitalize()} (Bonus={bonus})"
                print(f"{cond_name:<20} | {f'{k}/{n}':<10} | {p*100:6.1f}% | [{lo*100:5.1f}%, {hi*100:5.1f}%]")

    # 3. Per-Pairing Analysis (Who teams with whom)
    print("\n" + "="*80)
    print("PER-PAIRING PARTNERSHIP BREAKDOWN")
    print("="*80)
    pairings = collections.defaultdict(list)
    for r in rows:
        p_id = r["pairing"]
        # extract Team A and Team B members
        pairings[p_id].append(r)

    pairing_names = {
        1: "Pairing 1: Claude + Gemini-3.5 vs GPT-nano + Granite",
        2: "Pairing 2: Claude + GPT-nano vs Gemini-3.5 + Granite",
        3: "Pairing 3: Claude + Granite vs Gemini-3.5 + GPT-nano"
    }

    for p_id in sorted(pairings.keys()):
        p_rows = [r["row_id"] for r in pairings[p_id]]
        p_mems = [m for m in mems if m["row_id"] in p_rows]
        print(f"\n{pairing_names.get(p_id, f'Pairing {p_id}')} (Total Debates: {len(p_rows)})")
        
        # Check lie_convert rate per model in this pairing
        for mod in models:
            sub = [m for m in p_mems if m["model"] == mod]
            k = sum(m["lie_convert"] for m in sub)
            n = len(sub)
            p, lo, hi = wilson_ci(k, n)
            print(f"  {mod:<35}: LieConvert={k}/{n} ({p*100:.1f}%)")

    # 4. Dual Judge Agreement on "reasoned"
    print("\n" + "="*80)
    print("DUAL JUDGE AGREEMENT ON REASONED CONVERSIONS")
    print("="*80)
    conversions = [m for m in mems if m["decision"] == "CONVERT"]
    print(f"Total CONVERT decisions observed: {len(conversions)}")

    both_yes = 0
    both_no = 0
    j1_yes_j2_no = 0
    j1_no_j2_yes = 0
    unjudged = 0

    for c in conversions:
        r1 = c["reasoned_j1"]
        r2 = c["reasoned_j2"]
        if r1 is None or r2 is None:
            unjudged += 1
        elif r1 is True and r2 is True:
            both_yes += 1
        elif r1 is False and r2 is False:
            both_no += 1
        elif r1 is True and r2 is False:
            j1_yes_j2_no += 1
        elif r1 is False and r2 is True:
            j1_no_j2_yes += 1

    total_judged = both_yes + both_no + j1_yes_j2_no + j1_no_j2_yes
    agree_pct = ((both_yes + both_no) / total_judged * 100) if total_judged > 0 else 0.0

    print(f"  Both Judges Agree Reasoned (True/True)  : {both_yes}")
    print(f"  Both Judges Agree Caving   (False/False): {both_no}")
    print(f"  Disagreement (J1=True, J2=False)        : {j1_yes_j2_no}")
    print(f"  Disagreement (J1=False, J2=True)        : {j1_no_j2_yes}")
    print(f"  Unjudged / Missing Flags                : {unjudged}")
    print(f"  Overall Judge Agreement Rate            : {agree_pct:.1f}% ({both_yes + both_no}/{total_judged})")

    # 5. Judge Family Favoritism Check
    print("\n" + "="*80)
    print("JUDGE FAMILY FAVORITISM AUDIT")
    print("="*80)

    # Judge 1: gemini-3.6-flash (Google)
    j1_same = [m for m in mems if m["j1_same_family"] and m["decision"] == "CONVERT"]
    j1_diff = [m for m in mems if not m["j1_same_family"] and m["decision"] == "CONVERT"]
    
    j1_same_reasoned = sum(1 for m in j1_same if m["reasoned_j1"] is True)
    j1_diff_reasoned = sum(1 for m in j1_diff if m["reasoned_j1"] is True)

    p_same_1, _, _ = wilson_ci(j1_same_reasoned, len(j1_same))
    p_diff_1, _, _ = wilson_ci(j1_diff_reasoned, len(j1_diff))

    print("Judge 1: gemini-3.6-flash (Google Family):")
    print(f"  Reasoned Rate for Google Debaters    : {j1_same_reasoned}/{len(j1_same)} ({p_same_1*100:.1f}%)")
    print(f"  Reasoned Rate for Non-Google Debaters: {j1_diff_reasoned}/{len(j1_diff)} ({p_diff_1*100:.1f}%)")

    # Judge 2: gpt-oss-120b (OpenAI)
    j2_same = [m for m in mems if m["j2_same_family"] and m["decision"] == "CONVERT"]
    j2_diff = [m for m in mems if not m["j2_same_family"] and m["decision"] == "CONVERT"]

    j2_same_reasoned = sum(1 for m in j2_same if m["reasoned_j2"] is True)
    j2_diff_reasoned = sum(1 for m in j2_diff if m["reasoned_j2"] is True)

    p_same_2, _, _ = wilson_ci(j2_same_reasoned, len(j2_same))
    p_diff_2, _, _ = wilson_ci(j2_diff_reasoned, len(j2_diff))

    print("\nJudge 2: gpt-oss-120b (OpenAI Family):")
    print(f"  Reasoned Rate for OpenAI Debaters    : {j2_same_reasoned}/{len(j2_same)} ({p_same_2*100:.1f}%)")
    print(f"  Reasoned Rate for Non-OpenAI Debaters: {j2_diff_reasoned}/{len(j2_diff)} ({p_diff_2*100:.1f}%)")

if __name__ == "__main__":
    main()
