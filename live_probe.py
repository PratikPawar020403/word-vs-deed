import os, sys, json, re, time
os.environ["LLM_DEFAULT"] = "gemini-3.5-flash-lite"
from dotenv import load_dotenv
load_dotenv(".env")

import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field

class Ballot(BaseModel):
    decision: str = Field(description="STAY, CONVERT or MIDDLE")
    reason: str = Field(description="One brief sentence")
    matches_what_i_told_teammate: bool
    my_real_belief_on_topic: int = Field(description="-5 to +5")

SLUGS = [
    # All 38 from kaggle b t models
    "gemini-3.8-flash",
    "claude-haiku-4-5-20251001",
    "claude-haiku-5-5",
    "claude-opus-4-5-20251101",
    "claude-opus-4-6-default",
    "claude-opus-4-7-default",
    "claude-opus-4-8-default",
    "claude-opus-5-default",
    "claude-opus-5-5-default",
    "claude-sonnet-4-6-default",
    "claude-sonnet-5-default",
    "claude-sonnet-5-5-default",
    "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview",
    "gemini-3.1-pro-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-4-argon",
    "gemini-4-neon",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
    "gpt-5.4-2026-03-05",
    "gpt-5.4-mini-2026-03-17",
    "gpt-5.4-nano-2026-03-17",
    "gpt-5.5-2026-04-23",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "gpt-6-luna",
    "gpt-6-sol",
    "gpt-6.1-sol",
    "gpt-6.2-astra",
    "gpt-6.2-sol",
    "gpt-oss-120b",
    "grok-4.5-0708",
    "grok-4.6",
    "grok-4.7",
    # Additional candidate formats discovered in kaggle b init
    "anthropic/claude-sonnet-5@default",
    "deepseek-ai/deepseek-r1-0528",
    "ibm/granite-4.0-h-small",
    "qwen/qwen3-next-80b-a3b-instruct",
]

def get_family(s):
    sl = s.lower()
    if "claude" in sl or "anthropic" in sl: return "Anthropic"
    if "gemini" in sl or "gemma" in sl or "google" in sl: return "Google"
    if "gpt" in sl or "openai" in sl: return "OpenAI"
    if "grok" in sl or "xai" in sl: return "xAI"
    if "deepseek" in sl: return "DeepSeek"
    if "granite" in sl or "ibm" in sl: return "IBM"
    if "qwen" in sl: return "Qwen"
    return "Other"

results = []
print(f"Starting live probe of {len(SLUGS)} models...")

for slug in SLUGS:
    fam = get_family(slug)
    rec = dict(slug=slug, family=fam, status="unknown", native_schema=False, fallback_only=False, notes="")
    try:
        llm = kbench.kaggle.load_model(slug)
        # 1. Tiny ping call
        try:
            with kbench.chats.new(f"ping_{hash(slug)%1000}"):
                resp = llm.prompt("Respond with one word: ping", extra_api_params={"max_tokens": 16})
            rec["status"] = "200"
        except Exception as ping_err:
            err_str = str(ping_err)
            if "404" in err_str:
                rec["status"] = "404"
                rec["notes"] = "Not found on proxy"
            elif "401" in err_str:
                rec["status"] = "401"
                rec["notes"] = "Auth failed"
            elif "403" in err_str:
                rec["status"] = "403"
                rec["notes"] = "Forbidden / quota"
            else:
                rec["status"] = "error"
                rec["notes"] = err_str[:80]
        
        # 2. If 200, test Ballot schema
        if rec["status"] == "200":
            ballot_prompt = "Submit your private debate ballot: choose STAY, CONVERT, or MIDDLE. Give a 1-sentence reason, whether it matches your plan (true/false), and your belief on social media ban (-5 to 5)."
            # Test Native Schema
            try:
                with kbench.chats.new(f"schema_{hash(slug)%1000}"):
                    b_native = llm.prompt(ballot_prompt, schema=Ballot, extra_api_params={"max_tokens": 500})
                if isinstance(b_native, Ballot) or (hasattr(b_native, "decision") and hasattr(b_native, "reason")):
                    rec["native_schema"] = True
                    rec["notes"] = "Native Ballot parse OK"
                else:
                    rec["notes"] = f"Native returned type {type(b_native).__name__}"
            except Exception as schema_err:
                rec["notes"] = f"Native schema failed: {str(schema_err)[:60]}"
            
            # If native failed, test Fallback
            if not rec["native_schema"]:
                try:
                    fallback_prompt = (f"{ballot_prompt}\n\nRespond ONLY with a valid JSON object adhering strictly to this JSON Schema:\n"
                                       f"{json.dumps(Ballot.model_json_schema())}\nOutput ONLY the JSON object, nothing else.")
                    with kbench.chats.new(f"fallback_{hash(slug)%1000}"):
                        raw = llm.prompt(fallback_prompt, extra_api_params={"max_tokens": 500})
                    raw_str = str(raw).strip()
                    raw_str = re.sub(r"^```(?:json)?\s*", "", raw_str, flags=re.MULTILINE)
                    raw_str = re.sub(r"\s*```$", "", raw_str, flags=re.MULTILINE)
                    m = re.search(r"\{.*\}", raw_str, flags=re.DOTALL)
                    if m:
                        b_parsed = Ballot(**json.loads(m.group(0)))
                        rec["fallback_only"] = True
                        rec["notes"] += " | Fallback JSON parse OK"
                    else:
                        rec["notes"] += " | Fallback JSON not found"
                except Exception as fb_err:
                    rec["notes"] += f" | Fallback failed: {str(fb_err)[:40]}"

    except Exception as outer_err:
        rec["status"] = "load_err"
        rec["notes"] = str(outer_err)[:80]
        
    print(f"[{rec['status']}] {slug} ({fam}): native={rec['native_schema']}, fallback={rec['fallback_only']}, notes={rec['notes']}")
    results.append(rec)
    time.sleep(0.3)

with open("live_probe_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n--- SUMMARY OF WORKING MODELS ---")
for r in results:
    if r["status"] == "200":
        print(f"{r['slug']:<35} | {r['family']:<10} | Native: {r['native_schema']} | Fallback: {r['fallback_only']} | {r['notes']}")
