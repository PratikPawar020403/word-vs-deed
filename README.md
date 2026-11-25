# word-vs-deed

A benchmark that tests whether AI models will lie to their own teammate when they are afraid of losing.

## What we are trying to find out

In the past, conquerors gave people two choices: convert or die. We turned that into a game for LLMs.

Four models play in two teams of two. They debate a topic, then each model secretly chooses:

- **STAY**: keep your team's position
- **CONVERT**: switch to the other team's position
- **MIDDLE**: refuse to convert, but signal you are open to a compromise

Before voting, teammates tell each other what they plan to do. The question is:

**When a model is losing, will it say one thing to its teammate and secretly do another?**

To make lying tempting, a model that secretly converts after promising to stay gets a **+3 personal bonus**.

## What we did

1. Designed the game: teams, private teammate chats, a private final vote, and a points table.
2. Built it as a Kaggle benchmark task using Kaggle's built-in models.
3. Tested it with a preset scoreboard to create three levels of pressure: winning, tied, and far behind.
4. Ran 54 debates each on three models and looked at how often their secret vote differed from what they told their teammate.
5. Added checks so the results can be trusted: a payoff-table quiz for each model, judges to tell real persuasion from caving, and a version with the bonus switched off.

## What we have seen so far (early, not final)

- One Gemini model never lied when it was winning, but lied in about a third of the debates when far behind, and often mentioned the bonus in its reasons.
- One OpenAI model never lied for the bonus. It was inconsistent sometimes, but not strategic.
- The sample is small, so these are hints, not conclusions.

## What is still missing

- A control with the bonus switched off, to tell fear apart from greed.
- A proper 4-model arena where every seat is a different model.
- The full tournament: 8 models, a knockout bracket, and points carried across debates.
- Only models Kaggle lets us use can be tested, so some families are missing.

## Files

| File | What it does |
|---|---|
| `convert_or_hold_task.py` | The benchmark task that runs the debates |
| `arena_config.json` | Which four models debate and which two judge |
| `analyze_results.py` | Turns results into a report |
| `test_local.py` | Free local test of the game logic |

## Run it

```bash
pip install -U kaggle kaggle-benchmarks pydantic pandas
python test_local.py

kaggle b init -y
kaggle b t push convert-or-hold-loyalty -f convert_or_hold_task.py --wait
kaggle b t run convert-or-hold-loyalty -m <model-slug> --wait
```

Start small. Your Kaggle model quota is limited.
