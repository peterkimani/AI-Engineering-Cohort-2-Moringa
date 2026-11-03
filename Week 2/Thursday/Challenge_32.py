# lab16_hospital_id_mask.py
# Core skill (from Lab 26): one precise regex rule per PII format.
# Output style (from Lab 16): raw input -> compliant masked text -> secure vault.
# Each match becomes a UNIQUE numbered token recorded in the vault, so the
# masking stays reversible and repeated values never overwrite one another.

import re


class PrivacyCompliancePipeline:
    def __init__(self):
        # One rule per PII format as (compiled pattern, token prefix). Keeping
        # the rules in a list means adding a new PII type is a one-line change
        # and never affects the existing rules.
        #   \+?254\d{9}  -> Kenyan phone: optional "+", "254", then 9 digits
        #   AP-\d{6}     -> AfyaPlus ID: literal "AP-" then exactly 6 digits
        #   HOSP-\d{8}   -> hospital ID: literal "HOSP-" then exactly 8 digits
        self.pii_rules = [
            (re.compile(r"\+?254\d{9}"), "MASKED_PHONE"),
            (re.compile(r"AP-\d{6}"), "MASKED_PATIENT_ID"),
            (re.compile(r"HOSP-\d{8}"), "MASKED_HOSPITAL_ID"),
        ]

    def transform_and_mask_payload(self, raw_input: str) -> dict:
        """
        Replace every PII match with a unique placeholder token and record a
        private token -> original-value mapping (the vault).

        How it works:
          - masked_text starts as a copy of the input; secure_vault starts empty.
          - For each (pattern, prefix) rule, a counter starts at 0. re.sub runs
            the inner _replace on every match: it increments the counter, builds
            a unique token like "[MASKED_PHONE_1]", stores token -> matched value
            in the vault, and returns the token as the replacement. Numbering by
            the counter keeps each occurrence distinct, so two phone numbers
            become _1 and _2 rather than overwriting each other in the vault.
          - Rules run in sequence and each matches only its own format, so text
            matching no rule (the clinical description) passes through unchanged.
        """
        masked_text = raw_input
        secure_vault = {}

        for pattern, prefix in self.pii_rules:
            counter = {"n": 0}  # mutable so the nested function can increment it

            # Default args pin this iteration's prefix/counter into the closure,
            # so a rule defined inside the loop does not capture later values.
            def _replace(match, prefix=prefix, counter=counter):
                counter["n"] += 1
                token = f"[{prefix}_{counter['n']}]"
                secure_vault[token] = match.group(0)
                return token

            masked_text = pattern.sub(_replace, masked_text)

        return {
            "compliant_payload": masked_text,   # safe to send to the cloud
            "secure_vault": secure_vault,        # keep this local only
        }


def run_masking_demo():
    """
    Print the three stages of the privacy layer for one sample message: the raw
    local input, the masked text safe to send to a cloud model, and the isolated
    vault that stays on the local machine.
    """
    engine = PrivacyCompliancePipeline()

    raw_input = "Call +254712345678 about AP-004217 / HOSP-20451187 (chest pain)."

    result = engine.transform_and_mask_payload(raw_input)

    print("1. Raw Local Input Payload:")
    print(f"   {raw_input}\n")
    print("2. Compliant Masked Text (safe to send to cloud API):")
    print(f"   {result['compliant_payload']}\n")
    print("3. Isolated Secure Data Vault (stays local):")
    print(f"   {result['secure_vault']}")


if __name__ == "__main__":
    run_masking_demo()