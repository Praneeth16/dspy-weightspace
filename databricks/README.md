# Run on your Databricks workspace

Single-node GPU job cluster (`g5.2xlarge`, A10G 24 GB, GPU ML runtime 17.3). The notebook creates `<catalog>.<schema>` and a volume `artifacts`, writes probe
outputs, transcripts and the patch there, and logs the run to MLflow under `/Users/<you>/dspy-weightspace`.

```bash
PROFILE=<your profile>
ME=$(databricks current-user me --profile $PROFILE -o json | jq -r .userName)
databricks workspace mkdirs /Workspace/Users/$ME/dspy-weightspace --profile $PROFILE
databricks workspace import /Workspace/Users/$ME/dspy-weightspace/databricks \
  --file notebooks/databricks.py --format SOURCE --language PYTHON --overwrite --profile $PROFILE

# pick a Claude Foundation Model API endpoint
databricks serving-endpoints list --profile $PROFILE -o json | jq -r '.[].name' | grep claude

sed -e "s|__USER__|$ME|" -e "s|__REFLECTION_ENDPOINT__|databricks-claude-sonnet-5-5|" databricks/job.json > /tmp/job.json
RUN_ID=$(databricks jobs submit --no-wait --json @/tmp/job.json --profile $PROFILE | jq -r .run_id)
databricks jobs get-run $RUN_ID --profile $PROFILE | jq '.state'
# when TERMINATED / SUCCESS:
TASK=$(databricks jobs get-run $RUN_ID --profile $PROFILE | jq -r '.tasks[0].run_id')
databricks jobs get-run-output $TASK --profile $PROFILE | jq -r '.notebook_output.result' | jq .
```

Parameters (notebook widgets, also settable in `job.json`): `reflection_endpoint`, `catalog`, `schema`, `n_dev`,
`n_holdout`, `max_new_tokens`, `passes_per_round`, `max_metric_calls`.

`job.json` uses a classic single-node GPU cluster because serverless GPU was not available on the test workspace. If
yours has it, replace `new_cluster` with an `environment_key` + `environments` block (`client: "4"`).

Auth for the reflection LM inside the notebook comes from the notebook's own context; no token to paste.
