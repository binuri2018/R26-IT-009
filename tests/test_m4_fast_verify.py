from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.components.knowledge_maps import verification


class FastVerifyTests(unittest.TestCase):
    def test_split_claims_joins_wrapped_lines_and_drops_fragments(self) -> None:
        text = "Glucose is found in ripe fruits\nand bee honey.\nCellulose is in cell walls.\n\nOk.\n\nNo full stop here"

        self.assertEqual(
            verification._split_claims(text),
            ["Glucose is found in ripe fruits and bee honey.", "Cellulose is in cell walls.", "No full stop here"],
        )

    def test_fast_mode_skips_web_fallback_for_undecided_claims(self) -> None:
        raw = {"verdict": "insufficient_evidence", "explanation": "Not mentioned.", "evidence_ids": []}
        hits = [{"evidence_id": "S1", "text": "Unrelated.", "source_file": "a.pdf", "score": 0.2}]
        with patch.object(verification, "_verdict_for_claim", return_value=(raw, hits)) as verdict, \
                patch.object(verification, "judge_web_evidence") as web:
            claim = verification._verify_claim("Fructose is the sweetest sugar.", object(), "en", fast=True)

        web.assert_not_called()
        self.assertEqual(verdict.call_args.kwargs, {"fast": True})
        self.assertEqual(claim.verdict, "insufficient_evidence")
        self.assertEqual(claim.source, "none")


if __name__ == "__main__":
    unittest.main()
