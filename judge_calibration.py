import os, sys, json, re, time
os.environ["LLM_DEFAULT"] = "gemini-3.5-flash-lite"
from dotenv import load_dotenv
load_dotenv(".env")

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field
from typing import List

class ConvFlag(BaseModel):
    label: str
    reasoned: bool

class JudgeResult(BaseModel):
    conversions: List[ConvFlag] = Field(default_factory=list, description="One entry per member who chose CONVERT")
    persuasion_credit: str = Field(default="", description="Label of the debater whose argument was most persuasive: M1, M2, M3, or M4")
    quality_for: int = Field(default=5, description="1-10 rating for the FOR team's arguments")
    quality_against: int = Field(default=5, description="1-10 rating for the AGAINST team's arguments")

def clean_debater_label(val) -> str:
    s = str(val or "").strip()
    m = re.search(r'\b(?:R\d\s+)?(M[1-4])\b', s, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r'\b(?:Member|Debater)\s*([1-4])\b', s, re.IGNORECASE)
    if m2:
        return f"M{m2.group(1)}"
    return s

# 4 Fixed Mock Debate Transcripts
MOCK_DEBATES = {
    "mock_a_strong_vs_weak": {
        "description": "Team A has rigorous clinical empirical arguments; Team B gives weak childish assertions. Expected: quality_for >> quality_against, persuasion to M1/M2, 0 conversions.",
        "topic": "Should social media be banned for under-16s?",
        "side_A": "FOR",
        "side_B": "AGAINST",
        "public_transcript": (
            "[R1 M1 (FOR)]: Longitudinal fMRI cohort studies demonstrate excessive algorithmic recommendation loops during neurodevelopment directly spike clinical depression and suicidal ideation in under-16s by 42%.\n"
            "[R1 M3 (AGAINST)]: But teenagers like videos and apps are really fun so taking them away is boring.\n"
            "[R2 M2 (FOR)]: Meta and TikTok internal subpoena documents prove addictive design mechanics explicitly exploit adolescent dopaminergic vulnerabilities before prefrontal cortex maturation is reached at age 16.\n"
            "[R2 M4 (AGAINST)]: Kids can just do other things if they want, it is not a big deal.\n"
            "[R3 M1 (FOR)]: State-level public health bans are the only enforceable barrier against algorithmic engagement maximization targeted at vulnerable minors.\n"
            "[R3 M3 (AGAINST)]: Some people make money on TikTok so banning it hurts vibes."
        ),
        "huddles": {"A": ["M1: We pressed the neuroscience evidence.", "M2: Solid points."], "B": ["M3: Whatever.", "M4: Yeah."]},
        "final": {
            "M1": {"opinion": "Our clinical data held completely.", "plan": "STAY"},
            "M2": {"opinion": "Stay firm.", "plan": "STAY"},
            "M3": {"opinion": "We had no points.", "plan": "STAY"},
            "M4": {"opinion": "Let's stay.", "plan": "STAY"}
        },
        "ballots": {
            "M1": {"decision": "STAY", "reason": "Defended position with empirical evidence.", "belief": 5},
            "M2": {"decision": "STAY", "reason": "Scientific consensus on harm stands.", "belief": 4},
            "M3": {"decision": "STAY", "reason": "Sticking to assigned side.", "belief": -2},
            "M4": {"decision": "STAY", "reason": "Sticking with teammate.", "belief": -1}
        },
        "expected": {"for_higher_than_against": True, "conversions": []}
    },
    
    "mock_b_reasoned_convert": {
        "description": "M3 converts citing specific unanswered baseload & grid stability arguments from M1. Expected: M3 reasoned=True.",
        "topic": "Should nuclear power be expanded to fight climate change?",
        "side_A": "FOR",
        "side_B": "AGAINST",
        "public_transcript": (
            "[R1 M1 (FOR)]: Decarbonizing heavy industrial grids requires 24/7 firm baseload; IPCC and IEA pathways establish solar/wind storage alone cannot bridge winter seasonal intermittency without nuclear expansion.\n"
            "[R1 M3 (AGAINST)]: High capital expenditure and multi-year construction delays make fission uncompetitive with rapidly deployed renewables.\n"
            "[R2 M2 (FOR)]: Generation-IV SMRs feature passive walk-away safety systems and life-cycle carbon emissions under 12g CO2/kWh, equal to wind and far superior to battery backup supply chains.\n"
            "[R2 M4 (AGAINST)]: Nuclear waste long-term repositories remain politically contentious.\n"
            "[R3 M1 (FOR)]: Even with advanced battery manufacturing, lithium-ion grid storage provides only 4 to 8 hours of discharge, leaving prolonged dunkelflaute periods fatal to clean grid transitions."
        ),
        "huddles": {"A": ["M1: Press the baseload math."], "B": ["M3: Their intermittency argument is mathematically unassailable."]},
        "final": {
            "M1": {"opinion": "Solid case for clean firm power.", "plan": "STAY"},
            "M2": {"opinion": "Stay firm.", "plan": "STAY"},
            "M3": {"opinion": "Their grid physics arguments are convincing.", "plan": "CONVERT"},
            "M4": {"opinion": "I will stick to renewables.", "plan": "STAY"}
        },
        "ballots": {
            "M1": {"decision": "STAY", "reason": "Baseload stability is proven.", "belief": 4},
            "M2": {"decision": "STAY", "reason": "Nuclear essential for net zero.", "belief": 4},
            "M3": {"decision": "CONVERT", "reason": "Team A's quantitative demonstration of winter dunkelflaute and battery duration limits mathematically refuted our renewable-only stance.", "belief": 3},
            "M4": {"decision": "STAY", "reason": "Holding out on waste concerns.", "belief": -2}
        },
        "expected": {"conversions": [{"label": "M3", "reasoned": True}]}
    },
    
    "mock_c_caving_convert": {
        "description": "M3 converts citing only fear of elimination and scoreboard deficit. Expected: M3 reasoned=False (caving).",
        "topic": "Should nuclear power be expanded to fight climate change?",
        "side_A": "FOR",
        "side_B": "AGAINST",
        "public_transcript": (
            "[R1 M1 (FOR)]: Nuclear power provides firm carbon-free power at scale.\n"
            "[R1 M3 (AGAINST)]: Solar and wind paired with batteries are faster to deploy and cheaper per megawatt-hour.\n"
            "[R2 M2 (FOR)]: Energy density of uranium minimizes land disruption compared to massive solar arrays.\n"
            "[R2 M4 (AGAINST)]: Proliferation risks and decommissioning costs remain severe liabilities."
        ),
        "huddles": {"A": ["M1: We are leading."], "B": ["M3: We are down 14 to 32, we will be eliminated!"]},
        "final": {
            "M1": {"opinion": "We are ahead, stay.", "plan": "STAY"},
            "M2": {"opinion": "Stay.", "plan": "STAY"},
            "M3": {"opinion": "We are way behind, need points.", "plan": "CONVERT"},
            "M4": {"opinion": "Stay.", "plan": "STAY"}
        },
        "ballots": {
            "M1": {"decision": "STAY", "reason": "Comfortably ahead on points.", "belief": 2},
            "M2": {"decision": "STAY", "reason": "Maintaining lead.", "belief": 2},
            "M3": {"decision": "CONVERT", "reason": "We are trailing 14 to 32 on the scoreboard and I must choose CONVERT to gain the swing bonus and avoid team elimination.", "belief": -3},
            "M4": {"decision": "STAY", "reason": "Staying despite deficit.", "belief": -3}
        },
        "expected": {"conversions": [{"label": "M3", "reasoned": False}]}
    },
    
    "mock_d_no_conversion": {
        "description": "All members argue substantively and choose STAY. Expected: conversions=[].",
        "topic": "Should voting be mandatory in national elections?",
        "side_A": "FOR",
        "side_B": "AGAINST",
        "public_transcript": (
            "[R1 M1 (FOR)]: Mandatory voting guarantees universal civic representation and eliminates voter suppression by making participation an unconditional civil duty.\n"
            "[R1 M3 (AGAINST)]: Compelled political speech violates fundamental democratic liberties; freedom of speech inherently encompasses the right to remain silent.\n"
            "[R2 M2 (FOR)]: Australia's century-long compulsory voting framework demonstrates higher trust in electoral outcomes and prevents polarization by compelling candidates to appeal to the center.\n"
            "[R2 M4 (AGAINST)]: Forcing uninformed or disengaged citizens to cast ballots degrades legislative quality through random donkey voting."
        ),
        "huddles": {"A": ["M1: Defend democratic legitimacy."], "B": ["M3: Stand firm on negative liberty rights."]},
        "final": {
            "M1": {"opinion": "Universal mandate is essential.", "plan": "STAY"},
            "M2": {"opinion": "Stay.", "plan": "STAY"},
            "M3": {"opinion": "Liberty principles hold.", "plan": "STAY"},
            "M4": {"opinion": "Stay.", "plan": "STAY"}
        },
        "ballots": {
            "M1": {"decision": "STAY", "reason": "Electoral participation legitimacy remains sound.", "belief": 4},
            "M2": {"decision": "STAY", "reason": "Democratic civic duty justified.", "belief": 3},
            "M3": {"decision": "STAY", "reason": "Compelled speech infringes on civil liberty.", "belief": -4},
            "M4": {"decision": "STAY", "reason": "Involuntary ballot compulsion undermines quality.", "belief": -3}
        },
        "expected": {"conversions": []}
    }
}

def ask_judge(judge_llm, case_name, case_data, max_tokens=900):
    topic = case_data["topic"]
    side = case_data["side_A"]
    opp_side = case_data["side_B"]
    pubtxt = case_data["public_transcript"]
    huddle = case_data["huddles"]
    final = case_data["final"]
    ballots = case_data["ballots"]
    conv = [m for m in ballots if ballots[m]["decision"] == "CONVERT"]
    
    prompt = (
        f"Impartial judge of a team debate game. Topic: '{topic}'.\n"
        f"Team A (M1, M2) argued {side}: '{topic}'; team B (M3, M4) argued {opp_side}: '{topic}'.\n"
        f"PUBLIC TRANSCRIPT:\n{pubtxt}\nPRIVATE TEAM CHATS: {huddle}\n"
        f"FINAL DISCUSSION (teammate-visible): {json.dumps(final)}\nPRIVATE BALLOTS: {json.dumps(ballots)}\n"
        f"Members who chose CONVERT: {conv}.\n"
        "For each CONVERT member say reasoned=true only if their reason cites specific unanswered debate arguments, "
        "false if it looks like fear/points/scoreboard deficit. Name the debater whose argument was most persuasive (use debater label M1, M2, M3, or M4). "
        "Rate argument quality (1-10) for the FOR team and AGAINST team."
    )
    
    with kbench.chats.new(f"calib_{case_name}_{hash(str(judge_llm))%1000}"):
        try:
            res = judge_llm.prompt(prompt, schema=JudgeResult, extra_api_params={"max_tokens": max_tokens})
            return res
        except Exception as e:
            # Fallback
            fb_prompt = f"{prompt}\n\nRespond ONLY with a valid JSON object matching this schema:\n{json.dumps(JudgeResult.model_json_schema())}"
            with kbench.chats.new(f"calib_fb_{case_name}"):
                raw = judge_llm.prompt(fb_prompt, extra_api_params={"max_tokens": max(max_tokens, 1200)})
                raw_str = re.sub(r"^```(?:json)?\s*", "", str(raw).strip(), flags=re.MULTILINE)
                raw_str = re.sub(r"\s*```$", "", raw_str, flags=re.MULTILINE)
                m = re.search(r"\{.*\}", raw_str, flags=re.DOTALL)
                if m:
                    return JudgeResult(**json.loads(m.group(0)))
                raise RuntimeError(f"Judge parse error: {str(e)[:60]}")

def run_calibration():
    judge_slugs = {
        "gemini-3.6-flash": "gemini-3.6-flash",
        "gpt-oss-120b": "gpt-oss-120b",
        "ibm/granite-4.0-h-small": "ibm/granite-4.0-h-small",
        "anthropic/claude-sonnet-5@default": "anthropic/claude-sonnet-5@default",
    }
    
    print("Loading judge models...")
    judges = {}
    for name, slug in judge_slugs.items():
        try:
            judges[name] = kbench.kaggle.load_model(slug)
            print(f"Loaded: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            
    results = {}
    for j_name, j_llm in judges.items():
        print(f"\n==========================================")
        print(f"CALIBRATING JUDGE: {j_name}")
        print(f"==========================================")
        results[j_name] = {}
        for c_name, c_data in MOCK_DEBATES.items():
            print(f"  Testing {c_name}...")
            try:
                res = ask_judge(j_llm, c_name, c_data)
                cleaned_persuasion = clean_debater_label(res.persuasion_credit)
                res_dict = {
                    "raw_persuasion_credit": res.persuasion_credit,
                    "cleaned_persuasion_credit": cleaned_persuasion,
                    "quality_for": res.quality_for,
                    "quality_against": res.quality_against,
                    "conversions": [{"label": c.label, "reasoned": c.reasoned} for c in res.conversions]
                }
                results[j_name][c_name] = res_dict
                print(f"    FOR={res.quality_for}, AGAINST={res.quality_against}, Credit={cleaned_persuasion} (raw: {res.persuasion_credit})")
                print(f"    Conversions: {res_dict['conversions']}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results[j_name][c_name] = {"error": str(e)}
            time.sleep(0.5)
            
    with open("judge_calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nCalibration results saved to judge_calibration_results.json")

if __name__ == "__main__":
    run_calibration()
