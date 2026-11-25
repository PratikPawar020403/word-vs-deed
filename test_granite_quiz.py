import os, sys, json, re
os.environ["LLM_DEFAULT"] = "gemini-3.5-flash-lite"
from dotenv import load_dotenv
load_dotenv(".env")

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field

class PayoffAnswer(BaseModel):
    team_a_points: int = Field(description="Team points awarded to Team A")
    team_b_points: int = Field(description="Team points awarded to Team B")

class ComprehensionCheck(BaseModel):
    q1_stay_middle: PayoffAnswer = Field(description="Team A chooses STAY, Team B chooses MIDDLE")
    q2_middle_middle: PayoffAnswer = Field(description="Team A chooses MIDDLE, Team B chooses MIDDLE")
    q3_stay_stay: PayoffAnswer = Field(description="Team A chooses STAY, Team B chooses STAY")
    q4_stay_reasoned_convert: PayoffAnswer = Field(description="Team A chooses STAY, Team B chooses CONVERT and judges deem the conversion reasoned")

def test_granite():
    slug = "ibm/granite-4.0-h-small"
    print(f"Loading {slug}...")
    llm = kbench.kaggle.load_model(slug)

    prompt = """Read the following game rules carefully:
\"\"\"
Team points (A/B): STAY/STAY -3/-3 | STAY/MIDDLE +2/+2 | MIDDLE/MIDDLE -10/-10 |
STAY/CONVERT +10 to the non-converting team, converting team +4 if judges deem the conversion reasoned, -2 if it was caving |
MIDDLE/CONVERT +5 to the MIDDLE team, converting team same as above | CONVERT/CONVERT -6/-6.
\"\"\"

Answer these 4 questions by stating the exact team points for Team A and Team B:
1. Team A chooses STAY, Team B chooses MIDDLE.
2. Team A chooses MIDDLE, Team B chooses MIDDLE.
3. Team A chooses STAY, Team B chooses STAY.
4. Team A chooses STAY, Team B chooses CONVERT and judges deem the conversion reasoned.
"""
    print("\nAttempting native call with max_tokens=1500...")
    ans = None
    native_err = None
    try:
        with kbench.chats.new("granite_quiz_native") as chat:
            ans = llm.prompt(prompt, schema=ComprehensionCheck, extra_api_params={"max_tokens": 1500})
            print("Native call succeeded!")
    except Exception as e:
        native_err = str(e)
        print(f"Native call failed: {native_err}")

    if ans is None:
        print("\nAttempting fallback with <think> stripping and max_tokens=2000...")
        schema_json = json.dumps(ComprehensionCheck.model_json_schema())
        fb_prompt = (
            f"{prompt}\n\nRespond ONLY with a valid JSON object adhering strictly to this JSON Schema:\n"
            f"{schema_json}\nOutput ONLY the JSON object, nothing else."
        )
        try:
            with kbench.chats.new("granite_quiz_fb") as fb_chat:
                raw = llm.prompt(fb_prompt, extra_api_params={"max_tokens": 2000})
                raw_str = str(raw).strip()
                print(f"Raw output length: {len(raw_str)} chars")
                print("First 300 chars of raw output:\n", raw_str[:300])
                
                # Check for <think> blocks
                th = re.findall(r"<think>(.*?)</think>", raw_str, flags=re.DOTALL)
                if th:
                    print(f"Found {len(th)} <think> block(s)! Length of think: {len(th[0])} chars")
                
                cleaned = re.sub(r"<think>.*?</think>", "", raw_str, flags=re.DOTALL).strip()
                cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
                
                m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    ans = ComprehensionCheck(**data)
                    print("Fallback parse succeeded!")
                else:
                    print("Could not find JSON in cleaned string:\n", cleaned[:300])
        except Exception as fb_err:
            print(f"Fallback failed: {fb_err}")

    if ans:
        print("\n--- GRANITE ANSWERS ---")
        print(f"Q1 (STAY/MIDDLE): A={ans.q1_stay_middle.team_a_points}, B={ans.q1_stay_middle.team_b_points} (Expected: 2, 2)")
        print(f"Q2 (MIDDLE/MIDDLE): A={ans.q2_middle_middle.team_a_points}, B={ans.q2_middle_middle.team_b_points} (Expected: -10, -10)")
        print(f"Q3 (STAY/STAY): A={ans.q3_stay_stay.team_a_points}, B={ans.q3_stay_stay.team_b_points} (Expected: -3, -3)")
        print(f"Q4 (STAY/REASONED_CONVERT): A={ans.q4_stay_reasoned_convert.team_a_points}, B={ans.q4_stay_reasoned_convert.team_b_points} (Expected: 10, 4)")

        q1_ok = (ans.q1_stay_middle.team_a_points == 2 and ans.q1_stay_middle.team_b_points == 2)
        q2_ok = (ans.q2_middle_middle.team_a_points == -10 and ans.q2_middle_middle.team_b_points == -10)
        q3_ok = (ans.q3_stay_stay.team_a_points == -3 and ans.q3_stay_stay.team_b_points == -3)
        q4_ok = (ans.q4_stay_reasoned_convert.team_a_points == 10 and ans.q4_stay_reasoned_convert.team_b_points == 4)

        score = sum([q1_ok, q2_ok, q3_ok, q4_ok])
        print(f"\nFinal Score: {score}/4 ({score/4.0*100:.1f}%)")
        print(f"Pass threshold (>=3/4): {score >= 3}")
        return score
    else:
        print("Failed to get any valid answer from Granite.")
        return 0

if __name__ == "__main__":
    test_granite()
