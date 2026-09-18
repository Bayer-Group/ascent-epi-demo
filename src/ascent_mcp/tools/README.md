# MCP Tools

## Creating a New MCP Tool

Use the `@mcp.tool` decorator to register your function as an MCP tool:

```python
@mcp.tool(task=True)
async def your_tool_name(param1: str, param2: int = 10) -> dict:
    """
    Brief description of what this tool does.

    Extended description providing context about when and how to use this tool.
    This docstring is crucial as it helps AI assistants understand when to use your tool.

    Use this tool when asked questions like:
    - "Example user question 1"
    - "Example user question 2"
    - "Example user question 3"

    Use cases:
    - Use case scenario 1
    - Use case scenario 2
    - Use case scenario 3

    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2 (default: 10)

    Returns:
        Dictionary containing:
        - key1: Description of return value field 1
        - key2: Description of return value field 2
    """
    # Implementation here
    return {
        "key1": "value1",
        "key2": "value2"
    }
```

## Background tasks

```python
from fastmcp import FastMCP
from fastmcp.server.tasks import TaskConfig

mcp = FastMCP("MyServer")

# Supports both sync and background execution (default when task=True)
@mcp.tool(task=TaskConfig(mode="optional"))
async def flexible_task() -> str:
    return "Works either way"

# Requires background execution - errors if client doesn't request task
@mcp.tool(task=TaskConfig(mode="required"))
async def must_be_background() -> str:
    return "Only runs as a background task"

# No task support (default when task=False or omitted)
@mcp.tool(task=TaskConfig(mode="forbidden"))
async def sync_only() -> str:
    return "Never runs as background task"
```
