class EPRVFLWrapper:
    @staticmethod
    def load():
        return EPRVFLWrapper()
    
    def predict(self, claim, evidence=""):
        """Stub mode with improved heuristics"""
        fake_keywords = [
            "hoax", "fake", "microchip", "bill gates", "5g", "conspiracy",
            "witch hunt", "deep state", "illuminati", "chemtrails", "vaccine poison",
            "lizard people", "moon landing hoax", "flatearth", "new world order"
        ]
        real_keywords = [
            "study shows", "research", "confirmed", "official", "government",
            "report", "data", "evidence", "scientists", "according to",
            "published", "verified", "fact-checked", "analysis"
        ]
        
        claim_lower = claim.lower()
        
        # Count keyword matches
        fake_score = sum(1 for kw in fake_keywords if kw in claim_lower)
        real_score = sum(1 for kw in real_keywords if kw in claim_lower)
        
        # Determine verdict
        if fake_score > real_score:
            confidence = min(0.95, 0.6 + (fake_score * 0.1))
            return {"label": "FAKE", "confidence": confidence, "stub": True}
        elif real_score > fake_score:
            confidence = min(0.95, 0.6 + (real_score * 0.1))
            return {"label": "REAL", "confidence": confidence, "stub": True}
        else:
            # If tie, slightly favor REAL (more conservative)
            return {"label": "REAL", "confidence": 0.55, "stub": True}
