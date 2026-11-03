# lab15_mcp_client.py
# A minimal MCP (Model Context Protocol) client that connects to a local
# MCP server over stdio and prints out the tools the server exposes.

import asyncio          # Provides the async event loop used to drive the client
import sys              # Gives access to sys.executable, the current Python interpreter
from pathlib import Path  # Builds a reliable absolute path to the server script

# MCP client primitives:
#   ClientSession         - manages the protocol-level conversation with the server
#   StdioServerParameters - describes how to launch/connect to a stdio-based server
from mcp import ClientSession, StdioServerParameters

# stdio_client opens a transport that talks to the server over its
# standard input/output streams (the server runs as a child process)
from mcp.client.stdio import stdio_client

# Absolute path to the server, resolved from THIS file's own folder, so the
# client works no matter which directory you launch it from.
SERVER_SCRIPT = Path(__file__).parent / "mock_clinic_server.py"

# Configuration for launching the MCP server as a subprocess. This tells the client how to launch the server. 
server_params = StdioServerParameters(
    # sys.executable is the venv python running this client, so the server
    # starts with the same packages available.
    command=sys.executable,
    args=[str(SERVER_SCRIPT)],
)


async def main():
    """
    Connect to the MCP server, initialize a session, and list its tools.

    Steps:
      1. Launch the server subprocess and open a stdio transport, which
         yields a read stream and a write stream for exchanging messages.
      2. Wrap those streams in a ClientSession to handle the MCP protocol.
      3. Perform the initialization handshake with the server.
      4. Request the server's available tools and print each tool's
         name and description.

    No calculations are performed here; the function only performs
    discovery (listing tools) and prints the results.
    """
    # Open the stdio transport to the server. This starts the subprocess
    # and returns (read, write) streams for communication. The async
    # context manager ensures the streams/subprocess are cleaned up on exit.
    async with stdio_client(server_params) as (read, write):
        # Create a protocol session over the read/write streams. The
        # context manager ensures the session is properly closed afterward.
        async with ClientSession(read, write) as session:
            # Perform the MCP handshake (exchange capabilities, protocol
            # version, etc.). Must be called before any other requests.
            await session.initialize()

            # Ask the server for the list of tools it exposes.
            # Returns an object whose `.tools` attribute is the tool list.
            tools = await session.list_tools()

            # Print each discovered tool's name and description.
            print("Discovered tools:")
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description}")


# Entry point: run the async main() coroutine on the event loop until
# it completes.
asyncio.run(main())