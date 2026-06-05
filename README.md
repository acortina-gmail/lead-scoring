# lead-scoring

ML models that score leads along the **commercial funnel** for OBS, so the sales
team can prioritise which leads to call first. Each funnel stage is its own
self-contained model (own training pipeline, serving API, infra and deploy
scripts) — they share the lead, not the code.

## Repository layout

```
lead-scoring/
└── ml/
    └── lead_scoring/                  # the lead-scoring funnel family
        ├── pre_lead/                  # ← stage 1: scores at form submit (LIVE)
        │   ├── src/leadscoring/       #   shared library (baked into both images)
        │   ├── pipelines/             #   Vertex AI training pipeline (KFP v2)
        │   ├── serving/               #   FastAPI scoring API (Cloud Run)
        │   ├── training/              #   training base image
        │   ├── deploy/                #   00 setup → 01 build → 02 train → 03 serve
        │   ├── terraform/             #   infra (bucket + Artifact Registry + APIs)
        │   ├── notebooks/             #   modelling analysis for this stage
        │   ├── README.md              #   ← full docs for this model
        │   └── DEPLOY.md              #   ← step-by-step deploy guide (ES)
        ├── contact/                   # (future) stage 2: after first contact
        └── interview/                 # (future) stage 3: after interview
```

The funnel idea: a lead fills a form (`pre_lead`), then is contacted, interviewed,
etc. Each later stage re-scores the same lead with the information available *at
that point* — so we track how a lead's likelihood evolves down the funnel.

## Why a folder per stage (and the `ml/` wrapper)

- **Self-contained models.** Each stage has its own deploy/terraform/pyproject, so it
  ships, retrains and (if needed) runs on a **different cloud** independently. The
  Cloud Build context is just that stage's folder — no cross-stage coupling.
- **Room to grow.** The `ml/` wrapper leaves the repo root free for **Dataform**
  definitions or non-funnel ML, should they live here too.
- **Shared code comes later.** When `contact/` arrives and the common scaffolding is
  obvious, lift it into `ml/lead_scoring/_shared/` — don't abstract on one model.

## Where to start

The first (and currently only) model is **[`ml/lead_scoring/pre_lead/`](ml/lead_scoring/pre_lead/README.md)** —
its README covers the architecture, the dev/prod environments, the monthly-retrain
promotion gate, and how to score; `DEPLOY.md` is the step-by-step GCP deploy guide.

## CI/CD (planned)

GitHub Actions (chosen over Cloud Build for multi-cloud portability — later funnel
stages may run on Azure if their data lives in Dynamics). One **path-filtered**
workflow per stage, with `dev`/`prod` **GitHub Environments** (manual approval on
`prod`). See the per-model README's "Environments & model promotion" section.
