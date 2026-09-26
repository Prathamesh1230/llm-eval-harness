# LLM Eval Harness

A testing framework for RAG pipelines. Point it at a set of documents and
questions you know the answers to, and it tells you whether retrieval and
answer quality got better or worse.

Built because changing a chunk size or a prompt in a RAG system usually
means guessing at whether it helped. This measures it instead.

## What it does

Runs every question in a test set through a RAG pipeline, then scores the
result two ways:

**Retrieval** — was the right document in the top k, and at what rank.
Hit rate, MRR, and precision@k, computed with plain arithmetic. No model
involved, so the numbers are identical on every run.

**Answers** — faithfulness, relevance, and completeness, scored 1-5 by an
LLM judge against an anchored rubric.

Every run is saved to SQLite, so you can compare today's numbers against
last week's and see exactly which questions changed.

## The design decision that matters

The generator and the judge are different model families. Gemini writes the
answers, Groq grades them.

Models rate their own output higher than an independent grader does. If the
same model both answers and scores, the numbers are inflated by an amount
you can't measure. Splitting them removes that.

The judge also runs at temperature 0 with anchored rubrics — each score from
1 to 5 has a written definition. Unanchored scales drift toward 4 and stop
discriminating between good and mediocre.

## Running it

Needs Python 3.11+, a Gemini API key, and a Groq API key. Both have free
tiers.

```bash
git clone https://github.com/Prathamesh1230/llm-eval-harness
cd llm-eval-harness

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS/Linux

pip install -r requirements.txt
cp .env.example .env            # then add your keys
```

Put some documents in `data/docs/` and write questions in
`data/golden.yaml`, then:

```bash
python -m evalharness.cli run
python -m evalharness.cli runs
```

For the web interface:

```bash
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000 and click "Use sample documents" to try it
without setting anything up.

## CLI

```bash
evalharness run                          # evaluate the built-in pipeline
evalharness run -c configs/chunk_400.yaml
evalharness run --url https://your-api.com/chat
evalharness runs                         # list past runs
evalharness compare <run_a> <run_b>
evalharness failures <run_id>            # what broke and why
evalharness check <baseline> <current>   # exits 1 on regression — for CI
```

## Testing your own chatbot

The harness doesn't only test the pipeline that ships with it. Point it at
any chatbot that answers over HTTP:

```yaml
name: production
target:
  type: http
  url: https://your-company.com/api/chat
```

It expects a response shaped like this, though every field name is
configurable:

```json
{
  "answer": "Employees get 24 days of paid leave.",
  "sources": [{"doc_id": "hr_policy.pdf", "text": "Employees are entitled to..."}]
}
```

If your chatbot doesn't return sources, retrieval metrics are skipped and
the rest still works.

Target URLs are checked before any request is made. Anything resolving to
localhost, a private network, or a cloud metadata address is refused,
because on a public deployment the URL comes from a stranger. Set
`ALLOW_PRIVATE_TARGETS=true` when self-hosting and testing something
internal.

## Continuous integration

`.github/workflows/eval.yml` runs the harness on every pull request and
fails the build if a metric drops more than 5% against a stored baseline.
Add `GEMINI_API_KEY` and `GROQ_API_KEY` as repository secrets, and set
`BASELINE_RUN_ID` as a repository variable to turn on gating.

## Layout
