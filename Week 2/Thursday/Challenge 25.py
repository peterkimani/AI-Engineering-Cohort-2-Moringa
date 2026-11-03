# Challenge 25.py
# (The "lab3_3_client_session" step.) Opens a session to the MCP server and
# runs the initialize handshake, but wraps the whole attempt so a missing or
# unstartable server is reported cleanly instead of crashing with a traceback.

import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from server_config import configure_remote_server


async def run_protocol_handshake():
    """
    Connect to the MCP server and perform the protocol handshake, handling
    the case where the server cannot be reached.

    How it works:
      - Builds the launch configuration via configure_remote_server().
      - Wraps the entire connection attempt (booting the server subprocess,
        opening the ClientSession, and the initialize handshake) in a
        try/except block.
      - On success, prints a confirmation and returns True.
      - On ANY failure, the except block prints one readable line and returns
        False. Catching Exception also covers the ExceptionGroup the MCP stack
        raises when the server subprocess dies at launch, so no unhandled
        stack trace ever reaches the user.
    """
    print("--- Step 3: Executing Protocol Handshake (Initialize) ---")
    target_config = configure_remote_server()

    try:
        async with stdio_client(target_config) as (read_stream, write_stream):
            # Open a structural protocol session over the active raw text streams
            async with ClientSession(read_stream, write_stream) as session:
                # Perform the critical protocol greeting handshake
                await session.initialize()
                print("Handshake Completed! Client and Server have successfully negotiated feature sync.")
                return True
    except Exception as e:
        # Clean, explanatory failure instead of a raw traceback.
        print(f"Could not reach MCP server: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(run_protocol_handshake())