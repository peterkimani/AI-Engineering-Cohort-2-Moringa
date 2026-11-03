# server_config.py
# (The "lab3_1_server_config" step.) Builds the parameters used to launch the
# MCP server as a subprocess. Kept in its own module so other files can import
# it WITHOUT running a client, which is the whole point of separating config
# from connection logic.

import os
import sys
from pathlib import Path
from mcp import StdioServerParameters

# Absolute path to the server script, resolved from THIS file's folder so the
# location does not depend on which directory you launch from.
SERVER_SCRIPT = Path(__file__).parent / "mock_clinic_server.py"


def configure_remote_server() -> StdioServerParameters:
    """
    Build and return the configuration for launching the MCP server.

    How it works:
      - Reads the launch command from the MCP_SERVER_CMD environment variable.
        If it is unset, it falls back to sys.executable (the current venv's
        python), so the server always starts with a valid interpreter.
      - Sets args to the absolute path of mock_clinic_server.py.
      - Returns a StdioServerParameters object describing how to start the
        server over stdio. It only describes the launch; it does not run it.
    """
    command = os.getenv("MCP_SERVER_CMD") or sys.executable
    return StdioServerParameters(
        command=command,
        args=[str(SERVER_SCRIPT)],
    )