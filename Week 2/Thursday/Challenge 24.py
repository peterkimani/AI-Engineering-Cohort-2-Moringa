# Challenge 24.py
# (The "lab3_2_transport_stream" step.) Verifies the required environment flags
# are set, then opens the stdio transport to the MCP server. Fails fast with a
# clear message if any flag is missing, so the client never connects with
# incomplete configuration.

import asyncio
import os
from dotenv import load_dotenv            # loads MCP_SERVER_CMD, OPENAI_API_KEY, etc. from .env
from mcp.client.stdio import stdio_client
from server_config import configure_remote_server

# Load variables from a .env file into the environment BEFORE the flag check,
# so flags defined there are visible to os.getenv below.
load_dotenv()

# The environment flags the client requires before it is allowed to connect.
REQUIRED_FLAGS = ["MCP_SERVER_CMD","OPENAI_API_KEY"]


def verify_required_flags():
    """
    Confirm every required environment flag is set before connecting.

    How it works:
      - For each name in REQUIRED_FLAGS, reads it with os.getenv, which returns
        None when a variable is unset. `not os.getenv(f)` is therefore True for
        both a missing and an empty value.
      - Collects the failing names into `missing`.
      - If `missing` is non-empty, raises SystemExit, which stops the program
        immediately and prints exactly which flags are absent (fail fast).
      - If empty, returns normally and the caller may proceed.
    """
    missing = [flag for flag in REQUIRED_FLAGS if not os.getenv(flag)]
    if missing:
        raise SystemExit(f"Missing env flags: {missing}")


async def verify_transport_stream():
    """
    Verify configuration, then open the stdio communication streams.

    How it works:
      1. Calls verify_required_flags() first, so an incomplete environment
         aborts before any server process is launched.
      2. Builds the launch configuration via configure_remote_server().
      3. Opens stdio_client, which boots the server subprocess and yields the
         read and write streams, whose types are printed to confirm the pipe
         was allocated. The async context manager cleans up on exit.
    """
    print("--- Step 2: Opening Standard I/O Communication Streams ---")

    # Fail fast: refuse to continue if any required flag is missing.
    verify_required_flags()

    target_config = configure_remote_server()

    # Establish the transport line. This boots the server script in the background.
    async with stdio_client(target_config) as (read_stream, write_stream):
        print("Communication pipe successfully allocated.")
        print(f"Read stream object initialized: {type(read_stream)}")
        print(f"Write stream object initialized: {type(write_stream)}")


if __name__ == "__main__":
    asyncio.run(verify_transport_stream())