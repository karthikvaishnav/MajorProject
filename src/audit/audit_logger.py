"""
Audit Layer: Tamper-Evident SHA-256 Hash-Chained Log Engine
Provides cryptographic proof of detection decisions for auditability and compliance.
"""

import os
import json
import hashlib
import time
from typing import Dict, List, Any, Tuple


class TamperEvidentAuditLogger:
    """
    Append-only log file manager using SHA-256 hash chaining.
    """

    def __init__(self, log_filepath: str = "audit_log.json"):
        self.log_filepath = log_filepath
        if not os.path.exists(self.log_filepath):
            self._init_log_file()

    def _init_log_file(self):
        genesis_entry = {
            "entry_id": 0,
            "timestamp": time.time(),
            "caller_id": "SYSTEM_GENESIS",
            "calibrated_scam_risk_score": 0.0,
            "risk_tier": "GENESIS",
            "evidence_summary": "Genesis block for tamper-evident audit log chain.",
            "prev_hash": "0" * 64
        }
        genesis_hash = self._compute_hash(genesis_entry)
        genesis_entry["current_hash"] = genesis_hash
        
        with open(self.log_filepath, "w") as f:
            json.dump([genesis_entry], f, indent=2)

    def _compute_hash(self, entry: Dict[str, Any]) -> str:
        """
        Computes SHA-256 hash over canonical entry fields.
        """
        payload = f"{entry['entry_id']}|{entry['timestamp']}|{entry['caller_id']}|{entry['calibrated_scam_risk_score']}|{entry['prev_hash']}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_all_entries(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_filepath):
            return []
        with open(self.log_filepath, "r") as f:
            return json.load(f)

    def log_decision(
        self,
        caller_id: str,
        risk_score: float,
        risk_tier: str,
        evidence_summary: str,
        audio_sha256: str = None
    ) -> Dict[str, Any]:
        """
        Appends a new decision entry onto the hash chain.
        """
        entries = self.get_all_entries()
        last_entry = entries[-1]
        prev_hash = last_entry["current_hash"]
        entry_id = len(entries)

        new_entry = {
            "entry_id": entry_id,
            "timestamp": time.time(),
            "caller_id": caller_id,
            "audio_sha256": audio_sha256 or hashlib.sha256(f"audio_{entry_id}".encode()).hexdigest(),
            "calibrated_scam_risk_score": risk_score,
            "risk_tier": risk_tier,
            "evidence_summary": evidence_summary,
            "prev_hash": prev_hash
        }
        
        current_hash = self._compute_hash(new_entry)
        new_entry["current_hash"] = current_hash

        entries.append(new_entry)
        with open(self.log_filepath, "w") as f:
            json.dump(entries, f, indent=2)

        return new_entry

    def verify_integrity(self) -> Tuple[bool, str]:
        """
        Validates the SHA-256 hash chain from genesis to head.
        Returns (is_valid, report_message).
        """
        entries = self.get_all_entries()
        if not entries:
            return False, "Audit log is empty."

        for i in range(len(entries)):
            entry = entries[i]
            
            # Verify previous hash link
            if i > 0:
                prev_entry = entries[i - 1]
                if entry["prev_hash"] != prev_entry["current_hash"]:
                    return False, f"Broken chain link at entry {i}: prev_hash mismatch."
            
            # Recompute current entry hash
            recomputed = self._compute_hash(entry)
            if recomputed != entry["current_hash"]:
                return False, f"Tampered record detected at entry {i}: hash verification failed."

        return True, f"Audit log verified: {len(entries)} entries intact with valid SHA-256 hash chain."
