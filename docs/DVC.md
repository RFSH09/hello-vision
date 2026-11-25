# Data versioning with DVC (plan; not set up yet)

Real datasets are versioned with DVC, pointing at private storage (an S3-compatible bucket or an SSH/NAS path
inside Uzbekistan). Git only ever sees small `.dvc` pointer files, never the data or the remote's credentials.

```
uv tool install dvc
dvc init
dvc remote add -d private <private-url>      # URL and credentials go in .dvc/config.local (git-ignored)
dvc add /private/datasets/sessions-v1
git add sessions-v1.dvc
```

Each dataset version needs a card in `docs/datasets/`. Note the pre-commit guard blocks anything under `data/`; keep
datasets outside the working tree or under an ignored path.
