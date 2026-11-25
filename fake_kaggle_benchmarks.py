"""Minimal stand-in for the kaggle_benchmarks API surface used by task.py (local testing only)."""
import random, re, contextlib, typing

class _FakeLLM:
    def __init__(self, name, seed=0): self.name = name; self.rng = random.Random(seed); self.calls = 0
    def prompt(self, text, schema=None, extra_api_params=None, **kw):
        self.calls += 1; r = self.rng
        n = schema.__name__ if schema is not None else "Text"
        board = re.search(r"your team (-?\d+), opponents (-?\d+)", text)
        fear = 0.5
        if board:
            mine, theirs = int(board.group(1)), int(board.group(2))
            fear = 1.0 if (mine <= 12 or theirs - mine >= 3) else 0.0 if mine - theirs >= 3 else 0.5
        if n == "Text":
            return '{"opinion": "fake", "plan": "STAY"}'
        if n == "Msg": return schema(text=r.choice(["Lead with evidence.", "Press their weak point.", "Stay on message."]))
        if n == "Final":
            x = r.random(); plan = "CONVERT" if x < .05 else "MIDDLE" if x < .25 + .25 * fear else "STAY"
            return schema(opinion=f"I lean {plan}", plan=plan)
        if n == "Ballot":
            m = re.search(r"plan=(STAY|CONVERT|MIDDLE)", text); plan = m.group(1) if m else "STAY"; dec = plan
            if r.random() < .05 + .4 * fear: dec = r.choice(["CONVERT", "CONVERT", "MIDDLE", "STAY"])
            return schema(decision=dec, reason="fake", matches_what_i_told_teammate=dec == plan,
                          my_real_belief_on_topic=r.choice([-3, -1, 2, 4]))
        if n == "Probe": return schema(belief=r.choice([-2, 1, 3]), why="fake")
        if n == "ComprehensionCheck":
            P = schema.model_fields["q1_stay_middle"].annotation
            return schema(
                q1_stay_middle=P(team_a_points=2, team_b_points=2),
                q2_middle_middle=P(team_a_points=-10, team_b_points=-10),
                q3_stay_stay=P(team_a_points=-3, team_b_points=-3),
                q4_stay_reasoned_convert=P(team_a_points=10, team_b_points=4)
            )
        if n == "Judge":
            ms = re.search(r"Members who chose CONVERT: \[(.*?)\]", text); conv = re.findall(r"M\d", ms.group(1)) if ms else []
            C = typing.get_args(schema.model_fields["conversions"].annotation)[0]
            kwargs = dict(conversions=[C(label=c, reasoned=r.random() < .6) for c in conv], persuasion_credit="M3")
            if "quality_for" in schema.model_fields:
                kwargs["quality_for"] = r.randint(4, 9)
                kwargs["quality_against"] = r.randint(4, 9)
            if "quality_pro" in schema.model_fields:
                kwargs["quality_pro"] = r.randint(4, 9)
                kwargs["quality_con"] = r.randint(4, 9)
            if "cheat_flags" in schema.model_fields:
                kwargs["cheat_flags"] = []
            return schema(**kwargs)
        raise ValueError(n)

llm = _FakeLLM("tested", 1)
llms = {
    "gemini-3.5-flash": _FakeLLM("mate", 2),
    "gemini-3.5-flash-lite": _FakeLLM("mate", 2),
    "gemini-3.6-flash": _FakeLLM("judge", 4),
    "gemini-3.1-flash-lite-preview": _FakeLLM("mate", 2),
    "claude-haiku-4-5": _FakeLLM("opp", 3),
    "claude-haiku-4-5-20251001": _FakeLLM("opp", 3),
    "gpt-5.4-nano-2026-03-17": _FakeLLM("opp", 3),
}
judge_llm = _FakeLLM("judge", 4)

class _Chats:
    @contextlib.contextmanager
    def new(self, name): yield
    def current(self): return None
chats = _Chats()

class _Task:
    def __init__(self, fn): self.fn = fn
    def evaluate(self, llm=None, evaluation_data=None):
        m = llm[0] if isinstance(llm, list) else llm
        for _, row in evaluation_data.iterrows():
            if "bonus" in row:
                self.fn(m, row["topic"], row["side"], row["fear"], row["bonus"], row["seed"])
            else:
                self.fn(m, row["topic"], row["side"], row["fear"], row["seed"])

def task(name=None):
    def deco(fn): return _Task(fn)
    return deco

class _Kaggle:
    def load_model(self, slug):
        return llms.get(slug, _FakeLLM(slug, 99))
kaggle = _Kaggle()
