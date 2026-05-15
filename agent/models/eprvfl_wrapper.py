"""
EPRVFL stub wrapper — returns mock predictions for pipeline development.
Replace predict() body with real model inference once GPU nodes are up
and EPRVFL checkpoint is available.
"""
import random

class EPRVFLWrapper:
    LABELS = ["FAKE", "REAL"]

    def __init__(self, stub=True):
        self.stub = stub
        if stub:
            print("[EPRVFLWrapper] Running in STUB mode — mock predictions only.")

    @classmethod
    def load(cls, checkpoint_path: str = None):
        # TODO: replace with real loading once GPU is available
        # model = YourEPRVFLModel(...)
        # model.load_state_dict(torch.load(checkpoint_path))
        return cls(stub=True)

    def predict(self, claim: str, evidence: str = "") -> dict:
        if self.stub:
            # Deterministic stub based on simple keyword heuristics
            claim_lower = claim.lower()
            fake_keywords = ["hoax", "conspiracy", "fake", "false", "lie",
                             "microchip", "crisis actor", "deep state"]
            real_keywords = ["study", "research", "confirmed", "official",
                             "according to", "published", "university"]

            fake_hits = sum(1 for k in fake_keywords if k in claim_lower)
            real_hits = sum(1 for k in real_keywords if k in claim_lower)

            if fake_hits > real_hits:
                label, conf = "FAKE", round(random.uniform(0.70, 0.90), 3)
            elif real_hits > fake_hits:
                label, conf = "REAL", round(random.uniform(0.70, 0.90), 3)
            else:
                label, conf = "FAKE", round(random.uniform(0.50, 0.70), 3)

            return {
                "label": label,
                "confidence": conf,
                "all_probs": {"FAKE": 1-conf, "REAL": conf} if label=="REAL"
                             else {"FAKE": conf, "REAL": 1-conf},
                "stub": True,
            }
        # Real inference goes here (GPU required)
        raise NotImplementedError("Load real checkpoint first.")
