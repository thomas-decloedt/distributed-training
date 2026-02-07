# Next Steps

After `terraform apply` completes, follow this flow.

## 1. Run Distributed Training

From the project root (not `infra/`):

```bash
cd /path/to/distributed-training
./scripts/run_gcp.sh
```

This will:
- Build and push the training image via Cloud Build
- SSH to master and worker VMs
- Run 2-node DDP training on MNIST
- Upload model to `gs://{project_id}-distributed-training/models/latest/model.pt`
- Log metrics to Vertex AI Experiments

## 2. Check Metrics

After training completes:

- [Vertex AI Experiments](https://console.cloud.google.com/vertex-ai/experiments) → select project, region `us-central1`
- Open the `distributed-training` (or `distributed-mnist`) experiment
- View runs, loss curves, and hyperparameters

## 3. Test the Inference Endpoint

Endpoint requires auth (`gcloud auth login`). Use `make test-inference` or:

```bash
# Get URL
INFERENCE_URL=$(cd infra && terraform output -raw inference_url)
AUTH_HEADER="Authorization: Bearer $(gcloud auth print-identity-token)"

# Health check (works before training)
curl -H "$AUTH_HEADER" $INFERENCE_URL/health

# Predict (requires model; run training first)
curl -X POST $INFERENCE_URL/predict \
  -H "$AUTH_HEADER" \
  -H "Content-Type: application/json" \
  -d '{"image": [[0.0, ... (784 floats for 28x28 MNIST)]]}'
```

## 4. Tear Down (When Done)

```bash
cd infra && terraform destroy
```

Use `e2-small` or `e2-medium`; `terraform destroy` when finished to avoid costs.
