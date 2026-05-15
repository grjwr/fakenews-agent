"""
Mistral-7B LoRA wrapper for fake news classification.
- Stub mode: runs on CPU, no GPU needed, for pipeline development
- Real mode: loads LoRA adapter, requires GPU (H100)

Checkpoint map (per-dataset fine-tuned models):
  politifact : mistral_politifact_20260513_2357/checkpoint-555
  gossipcop  : mistral_gossipcop_20260514_0000/checkpoint-37578
  welfake    : mistral_welfake_20260514_0026/checkpoint-25052
  buzzfeed   : mistral_buzzfeed_20260514_0202/checkpoint-...
  liar2      : mistral_liar2_20260514_0203/checkpoint-...
  fake_real  : mistral_fake_real_20260514_0935/checkpoint-...
"""

import os

CHECKPOINT_MAP = {
    "politifact": "/home/akumar/llm_fakenews/mistral_output/mistral_politifact_20260513_2357/checkpoint-555",
    "gossipcop":  "/home/akumar/llm_fakenews/mistral_output/mistral_gossipcop_20260514_0000/checkpoint-37578",
    "welfake":    "/home/akumar/llm_fakenews/mistral_output/mistral_welfake_20260514_0026/checkpoint-25052",
    "buzzfeed":   "/home/akumar/llm_fakenews/mistral_output/mistral_buzzfeed_20260514_0202",
    "liar2":      "/home/akumar/llm_fakenews/mistral_output/mistral_liar2_20260514_0203",
    "fake_real":  "/home/akumar/llm_fakenews/mistral_output/mistral_fake_real_20260514_0935",
}

LABELS = {0: "REAL", 1: "FAKE"}


class MistralLoRAWrapper:

    def __init__(self, model=None, tokenizer=None, stub=True):
        self.model = model
        self.tokenizer = tokenizer
        self.stub = stub
        if stub:
            print("[MistralLoRAWrapper] Running in STUB mode — mock predictions only.")

    @classmethod
    def load_stub(cls):
        return cls(stub=True)

    @classmethod
    def load_real(cls, dataset: str = "politifact"):
        """Load real LoRA model — requires GPU."""
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel

        checkpoint = CHECKPOINT_MAP.get(dataset)
        if not checkpoint or not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found for dataset: {dataset}\nPath: {checkpoint}")

        base = "mistralai/Mistral-7B-v0.1"
        print(f"[MistralLoRAWrapper] Loading base model: {base}")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            base,
            num_labels=2,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print(f"[MistralLoRAWrapper] Loading LoRA adapter: {checkpoint}")
        model = PeftModel.from_pretrained(base_model, checkpoint)
        model.eval()
        return cls(model=model, tokenizer=tokenizer, stub=False)

    def predict(self, claim: str, evidence: str = "") -> dict:
        if self.stub:
            return self._stub_predict(claim)
        return self._real_predict(claim, evidence)

    def _stub_predict(self, claim: str) -> dict:
        """Keyword-based stub — deterministic, no GPU needed."""
        import random
        claim_lower = claim.lower()
        fake_kw = ["hoax", "conspiracy", "fake", "false", "crisis actor",
                   "deep state", "microchip", "plandemic", "lie", "cover-up"]
        real_kw = ["study", "research", "confirmed", "official", "published",
                   "university", "according to", "scientists", "evidence"]
        fake_hits = sum(1 for k in fake_kw if k in claim_lower)
        real_hits = sum(1 for k in real_kw if k in claim_lower)

        if fake_hits > real_hits:
            label, conf = "FAKE", round(random.uniform(0.75, 0.95), 3)
        elif real_hits > fake_hits:
            label, conf = "REAL", round(random.uniform(0.75, 0.95), 3)
        else:
            label, conf = "FAKE", round(random.uniform(0.55, 0.75), 3)

        return {
            "label":      label,
            "confidence": conf,
            "all_probs":  {"FAKE": conf, "REAL": 1-conf} if label=="FAKE"
                          else {"FAKE": 1-conf, "REAL": conf},
            "stub":       True,
        }

    def _real_predict(self, claim: str, evidence: str = "") -> dict:
        """Real inference — GPU required."""
        import torch
        text = f"Claim: {claim} Evidence: {evidence}" if evidence else f"Claim: {claim}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]
            idx = probs.argmax().item()

        label = LABELS[idx]
        conf  = float(probs.max())
        return {
            "label":      label,
            "confidence": round(conf, 3),
            "all_probs":  {LABELS[i]: round(float(p), 3) for i, p in enumerate(probs)},
            "stub":       False,
        }
