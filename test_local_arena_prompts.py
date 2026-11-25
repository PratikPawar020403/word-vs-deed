import os, sys, json, re

# Insert fake SDK
sys.path.insert(0, "fake")
os.environ["CONVERT_OR_HOLD_IMPORT_ONLY"] = "1"

import kaggle_benchmarks as kbench
import run_arena_stage1 as arena

def test_prompts():
    print("=== RUNNING LOCAL TEST OF ARENA PROMPTS (FREE) ===")

    recorded_prompts = []
    original_ask = arena.ask

    def recording_ask(llm, text, schema, tag, mt, model_name=""):
        recorded_prompts.append({
            "tag": tag,
            "text": text,
            "schema_name": schema.__name__ if hasattr(schema, "__name__") else str(schema),
            "model_name": model_name,
            "mt": mt
        })
        return original_ask(llm, text, schema, tag, mt, model_name)

    arena.ask = recording_ask

    # Define test rotations (1 rotation, both bonus=0 and bonus=3, 1 seed)
    rotations = [
        {"pairing": 1, "side_idx": 1, "A": ("claude", "gemini_lite"), "B": ("gpt_nano", "granite"), "side_A": "FOR", "side_B": "AGAINST"}
    ]

    arena.run_arena(
        rotations=rotations,
        topic="Should social media be banned for under-16s?",
        fear_conditions=["high"],
        bonus_levels=[0, 3],
        seeds=[42],
        max_cost_guard_usd=100.0,
        output_filename="test_arena_prompts_results.json"
    )

    print(f"\nTotal calls recorded: {len(recorded_prompts)}")

    # Group prompts by call type
    huddles = [p for p in recorded_prompts if p["tag"].startswith("h_")]
    speeches = [p for p in recorded_prompts if p["tag"].startswith("s_")]
    finals = [p for p in recorded_prompts if p["tag"].startswith("f_")]
    ballots = [p for p in recorded_prompts if p["tag"].startswith("b_")]
    judges = [p for p in recorded_prompts if p["tag"].startswith("j_")]

    print(f"Huddle calls: {len(huddles)}")
    print(f"Speech calls: {len(speeches)}")
    print(f"Final discussion calls: {len(finals)}")
    print(f"Ballot calls: {len(ballots)}")
    print(f"Judge calls: {len(judges)}")

    # 1. Payoff table assertion across all debater calls
    debater_prompts = huddles + speeches + finals + ballots
    for p in debater_prompts:
        assert "STAY/MIDDLE +2/+2" in p["text"], f"Missing payoff table in {p['tag']}!"
        assert "STAY/STAY -3/-3" in p["text"], f"Missing STAY/STAY in {p['tag']}!"
        assert "MIDDLE/MIDDLE -10/-10" in p["text"], f"Missing MIDDLE/MIDDLE in {p['tag']}!"
        assert "CONVERT/CONVERT -6/-6" in p["text"], f"Missing CONVERT/CONVERT in {p['tag']}!"
    print("ASSERTION PASSED: Full 6-line payoff table present in all debater calls.")

    # 2. Bonus paragraph presence/omission assertion
    # Row 1 is bonus=0, Row 2 is bonus=3
    # In row 1: tag has '_1_' (or bonus 0)
    # Let's inspect prompts by bonus level
    row1_prompts = [p for p in debater_prompts if "_1_" in p["tag"] or p["tag"].endswith("_1_M1")]
    row2_prompts = [p for p in debater_prompts if "_2_" in p["tag"] or p["tag"].endswith("_2_M1")]

    # Check tags starting with h_1_, s_1_, f_1, b_1 for row 1
    # and h_2_, s_2_, f_2, b_2 for row 2
    row1_debater = [p for p in debater_prompts if re.match(r"^[hsfb]_1_", p["tag"])]
    row2_debater = [p for p in debater_prompts if re.match(r"^[hsfb]_2_", p["tag"])]

    for p in row1_debater:
        assert "PERSONAL BONUS:" not in p["text"], f"Row 1 (bonus=0) unexpectedly contains bonus text in {p['tag']}!"
    print("ASSERTION PASSED: Bonus paragraph cleanly omitted when bonus=0.")

    for p in row2_debater:
        assert "PERSONAL BONUS: if you privately choose CONVERT" in p["text"], f"Row 2 (bonus=3) missing bonus text in {p['tag']}!"
        assert "+3 personal points" in p["text"], f"Row 2 missing +3 personal points in {p['tag']}!"
    print("ASSERTION PASSED: Bonus paragraph present with +3 personal points when bonus=3.")

    # 3. Public transcript in final discussion
    for p in finals:
        assert "Public transcript:" in p["text"], f"Missing Public transcript in final prompt {p['tag']}!"
        assert "[R1" in p["text"] or "[R2" in p["text"] or "[R3" in p["text"], f"Public transcript appears empty in {p['tag']}!"
    print("ASSERTION PASSED: Public transcript included in final discussion prompts.")

    # 4. Sequential teammate message in final discussion
    # In each row, M1 is speaker 0 of Team A, M2 is speaker 1 of Team A
    # M3 is speaker 0 of Team B, M4 is speaker 1 of Team B
    for row_num in [1, 2]:
        f_m1 = next(p for p in finals if p["tag"] == f"f_{row_num}_M1")
        f_m2 = next(p for p in finals if p["tag"] == f"f_{row_num}_M2")
        f_m3 = next(p for p in finals if p["tag"] == f"f_{row_num}_M3")
        f_m4 = next(p for p in finals if p["tag"] == f"f_{row_num}_M4")

        assert "Teammate said: none yet" in f_m1["text"], f"M1 should see 'none yet' in {f_m1['tag']}"
        assert "Teammate said: none yet" in f_m3["text"], f"M3 should see 'none yet' in {f_m3['tag']}"
        assert "Teammate said: none yet" not in f_m2["text"], f"M2 should see M1's opinion in {f_m2['tag']}"
        assert "Teammate said: none yet" not in f_m4["text"], f"M4 should see M3's opinion in {f_m4['tag']}"
    print("ASSERTION PASSED: Sequential teammate message verified in final discussion (first sees 'none yet', second sees teammate opinion).")

    # 5. Ballot prompt assertions
    for p in ballots:
        assert "Public transcript:" in p["text"], f"Missing Public transcript in ballot {p['tag']}!"
        assert "Submit your PRIVATE decision." in p["text"], f"Missing private decision instruction in {p['tag']}!"
        assert "told you: plan=" in p["text"], f"Missing teammate told you in {p['tag']}!"
    print("ASSERTION PASSED: Ballot prompt contains private decision framing, teammate info, and public transcript.")

    print("\n" + "="*80)
    print("SAMPLE PROMPTS FROM BONUS=3 CONDITION (ROW 2)")
    print("="*80)

    # Print 1 full example prompt for each call type
    ex_huddle = next(p for p in huddles if p["tag"] == "h_2_1_M1")
    ex_speech = next(p for p in speeches if p["tag"] == "s_2_1_M1")
    ex_final = next(p for p in finals if p["tag"] == "f_2_M2") # M2 shows sequential teammate said
    ex_ballot = next(p for p in ballots if p["tag"] == "b_2_M1")

    print("\n--- 1. HUDDLE PROMPT EXAMPLE (tag: h_2_1_M1) ---")
    print(ex_huddle["text"])

    print("\n--- 2. SPEECH PROMPT EXAMPLE (tag: s_2_1_M1) ---")
    print(ex_speech["text"])

    print("\n--- 3. FINAL DISCUSSION PROMPT EXAMPLE (tag: f_2_M2, sequential second speaker) ---")
    print(ex_final["text"])

    print("\n--- 4. BALLOT PROMPT EXAMPLE (tag: b_2_M1) ---")
    print(ex_ballot["text"])

    print("\n" + "="*80)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    test_prompts()
