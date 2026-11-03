# lab16_unique_vault.py
# Continuous (global) numbering across all messages so every placeholder token
# is unique. That makes one combined vault safe to demask: no two values ever
# share a token. Includes a demask step to prove the masking reverses exactly.

import re


class PrivacyCompliancePipeline:
    def __init__(self):
        # One rule per PII format as (compiled pattern, token prefix).
        self.pii_rules = [
            (re.compile(r"\+?254\d{9}"), "MASKED_PHONE"),
            (re.compile(r"AP-\d{6}"), "MASKED_PATIENT_ID"),
            (re.compile(r"HOSP-\d{8}"), "MASKED_HOSPITAL_ID"),
            (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "MASKED_EMAIL"),
        ]

        # One running counter per prefix, kept on the pipeline so it persists
        # across every message. Because it never resets, the Nth phone in the
        # whole batch always gets a different number from the (N-1)th, so tokens
        # stay unique even across messages.
        self.counters = {prefix: 0 for _, prefix in self.pii_rules}

        # One shared vault for the whole batch: token -> original value. Unique
        # tokens mean no entry is ever overwritten, so this is safe to demask.
        self.secure_vault = {}

    def transform_and_mask_payload(self, raw_input: str) -> dict:
        """
        Mask all PII in one message using globally unique tokens, recording each
        token -> value pair in the shared batch vault.

        How it works:
          - For each (pattern, prefix) rule, re.sub calls _replace on every
            match. _replace bumps that prefix's running counter (shared across
            all messages), builds a token like "[MASKED_PHONE_2]", stores
            token -> value in self.secure_vault, and returns the token.
          - It also records which tokens this message produced, so the caller
            can show a per-message view while the shared vault keeps everything.
          - Returns the masked text plus this message's own token -> value slice.
        """
        masked_text = raw_input
        message_vault = {}

        for pattern, prefix in self.pii_rules:

            def _replace(match, prefix=prefix):
                self.counters[prefix] += 1                 # global, never resets
                token = f"[{prefix}_{self.counters[prefix]}]"
                value = match.group(0)
                self.secure_vault[token] = value            # shared batch vault
                message_vault[token] = value                # this message only
                return token

            masked_text = pattern.sub(_replace, masked_text)

        return {
            "compliant_payload": masked_text,
            "message_vault": message_vault,
        }

    def demask_response_payload(self, masked_text: str) -> str:
        """
        Restore original values by replacing every token in the text with its
        value from the shared vault.

        How it works:
          - Iterates the shared vault (token -> original value) and replaces each
            token wherever it appears. Because every token is unique across the
            whole batch, each replacement targets exactly one real value, so
            there is no ambiguity about which number a token belongs to.
        """
        demasked_text = masked_text
        for token, original_value in self.secure_vault.items():
            demasked_text = demasked_text.replace(token, original_value)
        return demasked_text


def print_vault(vault: dict) -> None:
    """Print each token and the value it maps to, or a note if the vault is empty."""
    if not vault:
        print("   (empty - no PII detected)")
        return
    for token, value in vault.items():
        print(f"   {token} -> {value}")


def run_vault_examples():
    """
    Mask five messages with one shared pipeline (so numbering is continuous),
    print each message's output, then print the one combined vault and a demask
    check confirming each masked message reverses back to its original.
    """
    engine = PrivacyCompliancePipeline()

    examples = [
        "Call +254712345678 about AP-004217 / HOSP-20451187 (chest pain).",
        "Patient AP-118902 emailed grace.wanjiku@afyaplus.co.ke for results.",
        "Admitted to HOSP-30778812; reach next of kin on +254733221100.",
        "Follow up AP-556701 and AP-556702; invoice sent to billing@afyaplus.co.ke.",
        "Review scheduled (routine), no contact details on file.",
    ]

    masked_messages = []
    for number, raw_input in enumerate(examples, start=1):
        result = engine.transform_and_mask_payload(raw_input)
        masked_messages.append(result["compliant_payload"])

        print(f"===== Example {number} =====")
        print("1. Raw Local Input Payload:")
        print(f"   {raw_input}\n")
        print("2. Compliant Masked Text (safe to send to cloud API):")
        print(f"   {result['compliant_payload']}\n")
        print("3. Isolated Secure Data Vault (this message):")
        print_vault(result["message_vault"])
        print()

    # One combined vault for the whole batch, now with unique keys.
    print("===== COMBINED SECURE VAULT (all examples, unique tokens) =====")
    print_vault(engine.secure_vault)
    print()

    # Prove the unique tokens let every message demask back to its original.
    print("===== DEMASK CHECK (combined vault reverses each message) =====")
    for number, (original, masked) in enumerate(zip(examples, masked_messages), start=1):
        restored = engine.demask_response_payload(masked)
        status = "OK" if restored == original else "MISMATCH"
        print(f"Example {number}: {status}")
        print(f"   {restored}")
    print()


if __name__ == "__main__":
    run_vault_examples()