# lab3_4_privacy_masking.py
# A privacy pipeline that strips personally identifiable information (PII) out
# of text BEFORE it is sent to a cloud LLM, then restores it locally after the
# model responds. The pattern keeps real patient contact details on your own
# machine while still letting the cloud model process the request. This is the
# "mask, send, unmask" approach to using a third-party API with sensitive data.

import os                                  # standard library (available for env use)
import re                                  # regular expressions, used to find phone/email PII
from dotenv import load_dotenv             # loads variables (e.g. OPENAI_API_KEY) from a .env file
from langchain_openai import ChatOpenAI    # the OpenAI chat model wrapper

# Load .env into the environment so the LLM client can find the API key.
load_dotenv()


class PrivacyCompliancePipeline:
    def __init__(self):
        # Pre-compile the two regex patterns once when the pipeline is created,
        # so they are reused on every call rather than recompiled each time.
        #
        # phone_pattern matches Kenyan mobile numbers:
        #   (?:\+254|0) - a non-capturing group for the prefix, either the
        #                 international "+254" or the local leading "0"
        #   [17]        - the first digit after the prefix (Kenyan mobiles
        #                 start 7 or 1 after the prefix)
        #   \d{8}       - exactly eight more digits
        # So it matches both "0712345678" and "+254712345678".
        self.phone_pattern = re.compile(r'(?:\+254|0)[17]\d{8}')

        # email_pattern matches a standard email address:
        #   [a-zA-Z0-9._%+-]+ - the local part (before the @)
        #   @                 - the literal @ sign
        #   [a-zA-Z0-9.-]+    - the domain name
        #   \.[a-zA-Z]{2,}    - a dot and a top-level domain of 2+ letters
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def transform_and_mask_payload(self, raw_patient_input: str) -> dict:
        """
        Replace PII in the input text with placeholder tokens, and keep a
        private mapping of each token back to the real value.

        How it works:
          - Starts with masked_text as a copy of the input and an empty
            extracted_pii_vault dict.
          - Finds every phone number with the phone regex. For each match,
            builds a unique token like "[MASKED_PHONE_1]" (the number comes
            from enumerate, so matches are numbered 1, 2, 3...), stores
            token -> real value in the vault, and replaces the real value
            with the token in the text.
          - Repeats the same process for emails, producing "[MASKED_EMAIL_n]"
            tokens.
          - Returns a dict with the fully masked text and the vault. The text
            is safe to send out; the vault stays local.
        """
        masked_text = raw_patient_input
        extracted_pii_vault = {}

        # Scan text and swap out phone numbers for unique tracking masks.
        phone_matches = self.phone_pattern.findall(masked_text)
        for index, match in enumerate(phone_matches):
            # index+1 makes the tokens human-friendly (start at 1, not 0).
            token_id = f"[MASKED_PHONE_{index+1}]"
            extracted_pii_vault[token_id] = match          # remember the real value
            masked_text = masked_text.replace(match, token_id)  # hide it in the text

        # Scan text and swap out email strings for unique tracking masks.
        email_matches = self.email_pattern.findall(masked_text)
        for index, match in enumerate(email_matches):
            token_id = f"[MASKED_EMAIL_{index+1}]"
            extracted_pii_vault[token_id] = match
            masked_text = masked_text.replace(match, token_id)

        return {
            "compliant_payload": masked_text,     # safe to send to the cloud
            "secure_vault": extracted_pii_vault,  # keep this local only
        }

    def demask_response_payload(self, raw_model_output: str, secure_vault: dict) -> str:
        """
        Restore the original PII in the model's response using the saved vault.

        How it works:
          - Starts with demasked_text as a copy of the model output.
          - Iterates over every token -> original_value pair in the vault and
            replaces each token back with its original value in the text.
          - Returns the restored text. This only works if the model preserved
            the tokens exactly; any token the model altered will not be
            restored.
        """
        demasked_text = raw_model_output
        for token_id, original_value in secure_vault.items():
            demasked_text = demasked_text.replace(token_id, original_value)
        return demasked_text


def run_secure_llm_call():
    """
    End-to-end demo of the mask, send, unmask flow.

    How it works:
      1. Builds a pipeline and a sample patient message containing a phone
         number and an email.
      2. Masks the message, producing text safe to send plus a local vault.
      3. Sends ONLY the masked text to the cloud LLM, which rewrites it into a
         formal triage log entry without ever seeing the real contact details.
      4. De-masks the model's response locally, re-inserting the real values,
         and prints the final view.
    """
    print("--- Executing Secure Compliance Masking Sequence ---")
    compliance_engine = PrivacyCompliancePipeline()

    raw_input = "Hello, my name is David. My phone number is 0712345678 and my email is david@example.com. I need a urgent dental checkup window scheduled."
    print(f"1. Raw Local Input Payload:\n {raw_input}\n")

    # Mask the data BEFORE it leaves the local machine.
    pipeline_result = compliance_engine.transform_and_mask_payload(raw_input)
    print(f"2. Compliant Masked Text (Sent to Cloud API):\n {pipeline_result['compliant_payload']}\n")
    print(f"3. Isolated Secure Data Vault (Stays inside Local Servers): {pipeline_result['secure_vault']}\n")

    # Create the live LLM client. temperature=0.0 keeps the rewrite deterministic.
    print("--- Dispatching Masked String to Cloud LLM Interface ---")
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

    # The model only ever sees the masked text, so it cannot learn the real
    # phone number or email.
    model_response = llm.invoke(f"Rewrite this healthcare request into a formal medical triage log entry: {pipeline_result['compliant_payload']}")
    print(f"\n4. Raw Cloud Model Response Output:\n {model_response.content}\n")

    # Restore the real values locally using the vault we kept aside.
    final_output = compliance_engine.demask_response_payload(model_response.content, pipeline_result['secure_vault'])
    print(f"5. Final Local De-masked Application View:\n {final_output}")


if __name__ == "__main__":
    run_secure_llm_call()