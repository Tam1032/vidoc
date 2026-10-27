"""Expense claims and the ledger of paid invoices (in memory, seeded from data/claims)."""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from vidoc.schema import Invoice

CLAIMS_DIR = Path(__file__).resolve().parent.parent / "data" / "claims"


@dataclass
class ClaimStore:
    claims: dict[str, dict]
    ledger: list[dict]
    extracted: dict[str, Invoice] = field(default_factory=dict)  # claim_id -> what the extractor read
    decisions: dict[str, dict] = field(default_factory=dict)     # claim_id -> final action taken

    @classmethod
    def load(cls, path: Path = CLAIMS_DIR) -> "ClaimStore":
        claims = {c["claim_id"]: c for c in (json.loads(l) for l in (path / "claims.jsonl").read_text().splitlines() if l)}
        ledger = json.loads((path / "ledger_seed.json").read_text())
        return cls(copy.deepcopy(claims), copy.deepcopy(ledger))

    def image_path(self, claim_id: str) -> Path:
        return CLAIMS_DIR / self.claims[claim_id]["image"]

    def public_claim(self, claim_id: str) -> dict:
        """What the agent may see: everything except the hidden ground-truth invoice."""
        c = self.claims[claim_id]
        return {k: v for k, v in c.items() if k != "invoice"} | {"status": self.decisions.get(claim_id, {}).get("action", "new")}
