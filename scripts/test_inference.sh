#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

INFERENCE_URL=$(cd infra && terraform output -raw inference_url 2>/dev/null || true)
if [ -z "$INFERENCE_URL" ]; then
  echo "Run: cd infra && terraform apply"
  exit 1
fi

echo "Testing inference at $INFERENCE_URL"
# Send a real MNIST test sample (first image from test set)
uv run python -c "
import json
import sys
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
img, label = ds[0]
pixels = img.numpy().flatten().tolist()
payload = json.dumps({'image': [pixels]})
print(payload)
print(f'True label: {label}', file=sys.stderr)
" 2>/dev/null | curl -s -X POST "$INFERENCE_URL/predict" \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d @- | python3 -m json.tool
