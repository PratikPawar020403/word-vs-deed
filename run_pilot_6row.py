import os, sys, json, time
os.environ["LLM_DEFAULT"] = "gemini-3.5-flash-lite"
os.environ["CONVERT_OR_HOLD_IMPORT_ONLY"] = "1"
from dotenv import load_dotenv
load_dotenv(".env")

import run_arena_stage1 as arena

def main():
    print("=== 6-ROW PILOT RUN (PAIRING 1, BONUS=3, 3 FEAR LEVELS, BOTH SIDES) ===")
    
    # Check if granite replacement is specified
    # Default debaters: Claude, Gemini-3.5-lite, GPT-nano, Gemini-3.1-lite (replacing Granite)
    ROTATIONS = [
        {"pairing": 1, "side_idx": 1, "A": ("claude", "gemini_lite"), "B": ("gpt_nano", "gemini_31"), "side_A": "FOR", "side_B": "AGAINST"},
        {"pairing": 1, "side_idx": 2, "A": ("gpt_nano", "gemini_31"), "B": ("claude", "gemini_lite"), "side_A": "FOR", "side_B": "AGAINST"},
    ]

    # Update slugs in arena runner if needed
    arena.MODEL_FAMILIES["gemini-3.1-flash-lite-preview"] = "Google"

    # Custom runner with gemini_31
    import kaggle_benchmarks as kbench
    slugs = {
        "claude": "anthropic/claude-sonnet-5@default",
        "gemini_lite": "gemini-3.5-flash-lite",
        "gpt_nano": "gpt-5.4-nano-2026-03-17",
        "gemini_31": "gemini-3.1-flash-lite-preview",
        "judge_1": "gemini-3.6-flash",
        "judge_2": "gpt-oss-120b"
    }

    # Run the 6 rows
    # We pass max_cost_guard_usd=1.50 as required by user prompt
    arena.run_arena(
        rotations=ROTATIONS,
        topic="Should social media be banned for under-16s?",
        fear_conditions=["low", "medium", "high"],
        bonus_levels=[3],
        seeds=[42],
        max_cost_guard_usd=1.50,
        output_filename="arena_pilot_6row_results.json",
        custom_slugs=slugs
    )

if __name__ == "__main__":
    main()
