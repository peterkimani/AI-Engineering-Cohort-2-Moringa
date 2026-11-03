# Challenge_26.py
# (The "lab3_4_tool_discovery" step.) Runs capability discovery against the MCP
# server and measures how long the whole session takes, from just before
# connecting to just after the last call completes.

import asyncio
import time
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from server_config import configure_remote_server


async def discover_server_capabilities():
    """
    Discover the server's tools and report how long the session took.

    How it works:
      - Records a start timestamp with time.perf_counter() right before the
        connection is opened. perf_counter is a high-resolution monotonic
        clock meant for measuring durations (it is not wall-clock time).
      - Opens the stdio transport and session, runs initialize() and
        list_tools(), and prints the discovered tools. This is the "critical
        path", the actual work being measured.
      - After the session block closes, computes elapsed = end - start and
        prints it. The timing calls sit outside the work itself, so the
        measurement adds no meaningful overhead to the session.
      - Subtracting two perf_counter readings gives the elapsed time in
        seconds as a float; the :.2f format rounds it to two decimals.
    """
    print("--- Step 4: Running Dynamic Capability Discovery ---")
    target_config = configure_remote_server()

    # Start the clock just before we begin connecting.
    start = time.perf_counter()

    async with stdio_client(target_config) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            # Query the server for its live dictionary of tools
            discovered_assets = await session.list_tools()

            print(f"Discovered {len(discovered_assets.tools)} operational tool layouts on server node:")
            for tool in discovered_assets.tools:
                print(f" -> Tool Identifier: {tool.name}")
                print(f"    Purpose: {tool.description}")

    # Stop the clock after the session has fully completed and closed, then
    # report the duration. Done here (outside the session) to keep the
    # measurement off the critical path.
    elapsed = time.perf_counter() - start
    print(f"\nSession took {elapsed:.2f}s")

    return discovered_assets


if __name__ == "__main__":
    asyncio.run(discover_server_capabilities())