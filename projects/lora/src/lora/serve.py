"""LoraCase implements CaseProtocol for the reusable serve app."""

from pathlib import Path

import torch
from common.serve_models import PredictRequest, PredictResponse
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


class LoraPredictRequest(PredictRequest):
    """Request: prompt or text for generation."""

    prompt: str = ""
    text: str = ""


class LoraPredictResponse(PredictResponse):
    """Response: generated output or error."""

    output: str = ""
    error: str | None = None


class LoraCase:
    RequestModel = LoraPredictRequest
    ResponseModel = LoraPredictResponse

    def load(self, path: str) -> PeftModel:
        if path.startswith("gs://"):
            local_dir = Path("/tmp/lora_adapters")
            local_dir.mkdir(parents=True, exist_ok=True)
            from google.cloud import storage

            parts = path[5:].split("/", 1)
            bucket_name = parts[0]
            prefix = parts[1].rstrip("/") + "/"
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            for blob in bucket.list_blobs(prefix=prefix):
                rel = blob.name[len(prefix) :]
                if not rel:
                    continue
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                blob.download_to_filename(str(dest))
            adapter_path = str(local_dir)
        else:
            adapter_path = path
        base = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model.eval()
        return model

    def predict(self, model: PeftModel, request: LoraPredictRequest) -> LoraPredictResponse:
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        prompt = request.prompt or request.text
        if not prompt:
            return LoraPredictResponse(error="No prompt or text provided")
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return LoraPredictResponse(output=output.strip())
