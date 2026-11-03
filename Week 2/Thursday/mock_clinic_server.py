# mock_clinic_server.py
# A minimal MCP server that pretends to be a small clinic back end. It exposes
# a few tools over stdio so an MCP client can discover and call them. Written
# for mcp 2.x, where the server helper is MCPServer (it was FastMCP in 1.x).
# Each plain Python function below becomes an MCP tool, and its docstring
# becomes the tool description the client sees.

from mcp.server.mcpserver import MCPServer

# Create the server instance. The name is a human-readable label reported
# during the initialization handshake.
mcp = MCPServer("mock-clinic")

# In-memory stand-in for a real database, so the server is self-contained.
DOCTORS = [
    {"id": 1, "name": "Dr. Achieng", "specialty": "General Practice"},
    {"id": 2, "name": "Dr. Kamau", "specialty": "Pediatrics"},
    {"id": 3, "name": "Dr. Wanjiru", "specialty": "Cardiology"},
]


@mcp.tool()
def list_doctors(specialty: str = "") -> list[dict]:
    """
    Return the clinic's doctors, optionally filtered by specialty.

    How it works:
      - If `specialty` is empty, returns the full DOCTORS list unchanged.
      - Otherwise does a case-insensitive substring match: it lowercases
        both the requested specialty and each doctor's specialty and keeps
        only the doctors whose specialty contains the requested text.
    """
    if not specialty:
        return DOCTORS
    needle = specialty.lower()
    return [doc for doc in DOCTORS if needle in doc["specialty"].lower()]


@mcp.tool()
def book_appointment(doctor_id: int, patient_name: str, day: str) -> dict:
    """
    Pretend to book an appointment and return a confirmation record.

    How it works:
      - Looks up the doctor by matching `doctor_id` against the DOCTORS list.
      - If no doctor matches, returns a dict with an "error" key.
      - If found, returns a confirmation dict echoing the patient name, the
        chosen day, and the doctor's name. No real persistence happens.
    """
    doctor = next((d for d in DOCTORS if d["id"] == doctor_id), None)
    if doctor is None:
        return {"error": f"No doctor found with id {doctor_id}"}
    return {
        "status": "confirmed",
        "patient": patient_name,
        "doctor": doctor["name"],
        "day": day,
    }


@mcp.tool()
def triage_symptom(symptom: str) -> dict:
    """
    Give a rough urgency rating for a reported symptom.

    How it works:
      - Lowercases the symptom text for matching.
      - Checks it against two hard-coded keyword sets: `emergency` words map
        to urgency "high", `moderate` words map to urgency "medium".
      - The first matching set wins; if nothing matches, defaults to "low".
      - Returns a dict with the urgency level and a short advice string.
        This is a toy rule-based classifier for the lab, not medical advice.
    """
    text = symptom.lower()
    emergency = {"chest pain", "bleeding", "unconscious", "difficulty breathing"}
    moderate = {"fever", "vomiting", "persistent cough", "rash"}

    if any(word in text for word in emergency):
        return {"urgency": "high", "advice": "Seek emergency care immediately."}
    if any(word in text for word in moderate):
        return {"urgency": "medium", "advice": "Book an appointment within 24 hours."}
    return {"urgency": "low", "advice": "Monitor symptoms and rest."}


@mcp.tool()
def clinic_hours() -> dict:
    """
    Return the clinic's opening hours as a static dict.

    How it works:
      - No input and no computation; returns a fixed schedule so the client
        has a zero-argument tool to exercise during discovery.
    """
    return {
        "weekdays": "08:00 to 18:00",
        "saturday": "09:00 to 13:00",
        "sunday": "closed",
    }


# Entry point. When run as a script (which is how the client launches it),
# start the server over stdio so it talks to the parent client process via
# standard input/output.
if __name__ == "__main__":
    mcp.run(transport="stdio")