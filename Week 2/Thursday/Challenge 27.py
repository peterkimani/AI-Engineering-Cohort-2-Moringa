# Challenge_27.py
# (Searching discovered tools by name.) Connects to the MCP server, discovers
# its tools, then searches that list for one specific tool and reports whether
# it was found, before any attempt to invoke it.

import asyncio  # event loop used to run the async client (await-based I/O)

# MCP client building blocks:
#   ClientSession - wraps the raw streams and speaks the MCP protocol
#                   (handshake, list_tools, call_tool, etc.)
from mcp import ClientSession

# stdio_client - transport that launches the server as a subprocess and
#                talks to it over standard input/output
from mcp.client.stdio import stdio_client

# configure_remote_server - our helper (from server_config.py) that returns
#                            the StdioServerParameters describing how to launch
#                            the server
from server_config import configure_remote_server

# The tool name we want to locate in the server's advertised tool list.

TARGET_TOOL = "asset_depreciation"


async def find_tool_by_name():
    """
    Discover the server's tools and search the list for a specific one by name.

    How it works:
      - Opens the session and calls list_tools() to get the full advertised
        list (the discovery step from Lab 15).
      - Prints every discovered tool name (requirement: list all tools).
      - Searches with next((t for t in tools.tools if t.name == TARGET_TOOL),
        None). The generator yields each tool whose .name matches; next()
        returns the FIRST match, or the default None if there is no match. This
        is a single-pass search that stops at the first hit.
      - Reports the result: if a tool was found it prints its description; if
        None, it reports the tool is unavailable. Only a found tool would be
        safe to invoke afterwards.
    """
    print("--- Step 5: Searching Discovered Tools by Name ---")

    # Build the launch configuration (interpreter + server script path).
    target_config = configure_remote_server()

    # Launch the server subprocess and open the stdio transport. The async
    # context manager yields the read/write streams and guarantees the server
    # is shut down when the block exits.
    async with stdio_client(target_config) as (read_stream, write_stream):
        # Wrap the raw streams in a protocol session. This context manager
        # closes the session cleanly on exit.
        async with ClientSession(read_stream, write_stream) as session:
            # Mandatory MCP handshake; must run before any other request.
            await session.initialize()

            # Discover all tools the server exposes. The result is a response
            # object whose .tools attribute holds the actual list of tools.
            tools = await session.list_tools()

            # Requirement 1: list every discovered tool by name.
            print(f"Discovered {len(tools.tools)} tools:")
            for tool in tools.tools:               # iterate the tool objects
                print(f"  - {tool.name}")           # each has a .name attribute

            # Requirement 2: search the discovered list for the named tool.
            # next(...) walks the generator and returns the first tool whose
            # name equals TARGET_TOOL. The second argument, None, is the
            # default returned when nothing matches, which prevents the
            # StopIteration error next() would otherwise raise on no match.
            target = next((t for t in tools.tools if t.name == TARGET_TOOL), None)

            # Requirement 3: report whether it was found.
            if target:  # a tool object is truthy; None is falsy
                # .description is the tool's docstring; .strip() trims the
                # leading/trailing whitespace so the printout is tidy.
                print(f"\nFound '{TARGET_TOOL}'. Description:\n  {target.description.strip()}")
            else:
                # Not-found path: clean report instead of an error or a crash.
                print(f"\nTool '{TARGET_TOOL}' is not available on this server.")

            # Hand the found tool (or None) back to the caller, in case a later
            # step wants to invoke it.
            return target


# Entry point: run the async function on the event loop until it finishes.
# Guarded by __name__ so importing this file does NOT launch a session.
if __name__ == "__main__":
    asyncio.run(find_tool_by_name())