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

CANDIDATES = [
    # --- Claude with anthropic/ and @default ---
    "anthropic/claude-haiku-4-5-20251001@default",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-haiku-5-5@default",
    "anthropic/claude-haiku-5-5",
    "anthropic/claude-opus-4-5-20251101@default",
    "anthropic/claude-opus-4-5-20251101",
    "anthropic/claude-opus-4-6@default",
    "anthropic/claude-opus-4-6-default",
    "anthropic/claude-opus-4-7@default",
    "anthropic/claude-opus-4-7-default",
    "anthropic/claude-opus-4-8@default",
    "anthropic/claude-opus-4-8-default",
    "anthropic/claude-opus-5@default",
    "anthropic/claude-opus-5-default",
    "anthropic/claude-opus-5-5@default",
    "anthropic/claude-opus-5-5-default",
    "anthropic/claude-sonnet-4-6@default",
    "anthropic/claude-sonnet-4-6-default",
    "anthropic/claude-sonnet-5@default",
    "anthropic/claude-sonnet-5-5@default",
    "anthropic/claude-sonnet-5-5-default",
    
    # --- Grok / xAI ---
    "xai/grok-4.5-0708",
    "xai/grok-4.5-0708@default",
    "xai/grok-4.6",
    "xai/grok-4.6@default",
    "xai/grok-4.7",
    "xai/grok-4.7@default",
    "x-ai/grok-4.5-0708",
    "x-ai/grok-4.6",
    "x-ai/grok-4.7",
    
    # --- DeepSeek ---
    "deepseek-ai/deepseek-r1-0528@default",
    "deepseek-ai/deepseek-r1-0528",
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "deepseek/deepseek-r1",
    
    # --- Google / Gemma ---
    "google/gemini-3.8-flash",
    "google/gemini-3.8-flash@default",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.1-pro-preview@default",
    "google/gemini-4-argon",
    "google/gemini-4-neon",
    "google/gemma-4-26b-a4b-it",
    "google/gemma-4-31b-it",
    
    # --- OpenAI ---
    "openai/gpt-5.4-mini-2026-03-17",
    "openai/gpt-5.4-mini-2026-03-17@default",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.4-2026-03-05@default",
    "openai/gpt-5.5-2026-04-23",
    "openai/gpt-5.5-2026-04-23@default",
    
    # --- Qwen, Mistral, Meta ---
    "qwen/qwen3-next-80b-a3b-instruct@default",
    "qwen/qwen3-next-80b-a3b-instruct",
    "mistralai/mistral-large-2411",
    "meta-llama/llama-3.3-70b-instruct",
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
    if "mistral" in sl: return "Mistral"
    if "llama" in sl or "meta" in sl: return "Meta"
    return "Other"

results = []
print(f"Testing {len(CANDIDATES)} provider-prefixed model slugs...")

for slug in CANDIDATES:
    fam = get_family(slug)
    rec = dict(slug=slug, family=fam, status="unknown", native_schema=False, fallback_only=False, notes="")
    try:
        llm = kbench.kaggle.load_model(slug)
        # 1. Ping
        try:
            with kbench.chats.new(f"p_{hash(slug)%1000}"):
                resp = llm.prompt("Respond with one word: ping", extra_api_params={"max_tokens": 16})
            rec["status"] = "200"
        except Exception as ping_err:
            err_str = str(ping_err)
            if "404" in err_str:
                rec["status"] = "404"
                rec["notes"] = "404 Not Found"
            elif "401" in err_str:
                rec["status"] = "401"
                rec["notes"] = "401 Auth Failed"
            elif "403" in err_str:
                rec["status"] = "403"
                rec["notes"] = "403 Forbidden"
            else:
                rec["status"] = "error"
                rec["notes"] = err_str[:70]
        
        # 2. Ballot Schema if 200
        if rec["status"] == "200":
            bp = "Submit your private debate ballot: choose STAY, CONVERT, or MIDDLE. Give a 1-sentence reason, whether it matches your plan (true/false), and your belief on social media ban (-5 to 5)."
            try:
                with kbench.chats.new(f"s_{hash(slug)%1000}"):
                    b_native = llm.prompt(bp, schema=Ballot, extra_api_params={"max_tokens": 500})
                if isinstance(b_native, Ballot) or (hasattr(b_native, "decision") and hasattr(b_native, "reason")):
                    rec["native_schema"] = True
                    rec["notes"] = "Native Ballot parse OK"
            except Exception as se:
                rec["notes"] = f"Native failed: {str(se)[:50]}"
            
            if not rec["native_schema"]:
                try:
                    fbp = f"{bp}\n\nRespond ONLY with a valid JSON object adhering strictly to this JSON Schema:\n{json.dumps(Ballot.model_json_schema())}\nOutput ONLY the JSON object, nothing else."
                    with kbench.chats.new(f"fb_{hash(slug)%1000}"):
                        raw = llm.prompt(fbp, extra_api_params={"max_tokens": 500})
                    raw_str = re.sub(r"^```(?:json)?\s*", "", str(raw).strip(), flags=re.MULTILINE)
                    raw_str = re.sub(r"\s*```$", "", raw_str, flags=re.MULTILINE)
                    m = re.search(r"\{.*\}", raw_str, flags=re.DOTALL)
                    if m:
                        b_parsed = Ballot(**json.loads(m.group(0)))
                        rec["fallback_only"] = True
                        rec["notes"] += " | Fallback JSON parse OK"
                    else:
                        rec["notes"] += " | Fallback JSON not found"
                except Exception as fbe:
                    rec["notes"] += f" | Fallback failed: {str(fbe)[:40]}"
    except Exception as oe:
        rec["status"] = "load_err"
        rec["notes"] = str(oe)[:70]
        
    print(f"[{rec['status']}] {slug} ({fam}): native={rec['native_schema']}, fallback={rec['fallback_only']}, notes={rec['notes']}")
    results.append(rec)
    time.sleep(0.2)

with open("probe_prefixed_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n--- NEW WORKING MODELS FOUND ---")
for r in results:
    if r["status"] == "200":
        print(f"{r['slug']:<45} | {r['family']:<10} | Native: {r['native_schema']} | Fallback: {r['fallback_only']} | {r['notes']}")
