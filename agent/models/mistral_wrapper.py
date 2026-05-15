import os

CHECKPOINT_MAP = {
    "politifact": "/home/akumar/llm_fakenews/mistral_output/mistral_politifact_20260513_2357/checkpoint-555",
    "gossipcop":  "/home/akumar/llm_fakenews/mistral_output/mistral_gossipcop_20260514_0000/checkpoint-11625",
    "welfake":    "/home/akumar/llm_fakenews/mistral_output/mistral_welfake_20260514_0026/checkpoint-25052",
    "buzzfeed":   "/home/akumar/llm_fakenews/mistral_output/mistral_buzzfeed_20260514_0202/checkpoint-96",
    "liar2":      "/home/akumar/llm_fakenews/mistral_output/mistral_liar2_20260514_0203/checkpoint-12057",
    "fake_real":  "/home/akumar/llm_fakenews/mistral_output/mistral_fake_real_20260514_0935/checkpoint-2412",
}

# 0=FAKE, 1=REAL — matches your training script (verdict column)
LABELS = {0: "FAKE", 1: "REAL"}


class MistralLoRAWrapper:

    def __init__(self, model=None, tokenizer=None, stub=True):
        self.model = model
        self.tokenizer = tokenizer
        self.stub = stub
        if stub:
            print("[MistralLoRAWrapper] Running in STUB mode.")

    @classmethod
    def load_stub(cls):
        return cls(stub=True)

    @classmethod
    def load_real(cls, dataset: str = "politifact"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        checkpoint = CHECKPOINT_MAP.get(dataset)
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        base = "mistralai/Mistral-7B-v0.1"
        print(f"[MistralLoRAWrapper] Loading {dataset} checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base, num_labels=2,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model.eval()
        print(f"[MistralLoRAWrapper] Loaded. Labels: {LABELS}")
        return cls(model=model, tokenizer=tokenizer, stub=False)

    def predict(self, claim: str, evidence: str = "") -> dict:
        if self.stub:
            return self._stub_predict(claim)
        return self._real_predict(claim, evidence)

    def _stub_predict(self, claim: str) -> dict:
        import random
        claim_lower = claim.lower()
        fake_kw = ["hoax", "conspiracy", "fake", "false", "microchip", "deep state"]
        real_kw = ["study", "research", "confirmed", "official", "published"]
        fake_hits = sum(1 for k in fake_kw if k in claim_lower)
        real_hits = sum(1 for k in real_kw if k in claim_lower)
        if fake_hits > real_hits:
            label, conf = "FAKE", round(random.uniform(0.75, 0.95), 3)
        elif real_hits > fake_hits:
            label, conf = "REAL", round(random.uniform(0.75, 0.95), 3)
        else:
            label, conf = "FAKE", round(random.uniform(0.55, 0.75), 3)
        return {"label": label, "confidence": conf, "stub": True}

    def _real_predict(self, claim: str, evidence: str = "") -> dict:
        import torch
        text = f"Claim: {claim} Evidence: {evidence}" if evidence else f"Claim: {claim}"
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            idx = probs.argmax().item()

        label = LABELS[idx]
        conf  = float(probs[idx])
        return {
            "label":      label,
            "confidence": round(conf, 3),
            "all_probs":  {LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)},
            "stub":       False,
        }
