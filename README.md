### feature-or-not.py

Invoke with:
```
make && \
docker run --rm \
  --volume /tmp:/tmp \
  --volume /path/to/velocity_enriched_prs.csv:/pull_requests.csv \
  codeclimate/feature-or-not
```

### bin scripts

1. Add `.env` (see `.env.example`)
1. Load all PRs for a repository: `bin/load-prs codeclimate/velocity`
1. Classify PRs as features or not manually: `bin/classify-prs`
1. Dump out the csv: `bin/dump-csv`
