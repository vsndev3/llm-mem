import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client

from . import config

logger = logging.getLogger(__name__)


class LlmMemClient:
    def __init__(self, config_path: str | None = None, bank_name: str = "default"):
        self.config_path = config_path or str(config.CONFIG_PATH)
        self.bank = bank_name
        self._session: ClientSession | None = None
        self._read = None
        self._write = None
        self._stdio_cm = None

    async def start(self):
        args = []
        if self.config_path:
            args.extend(["--config", self.config_path])

        env = os.environ.copy()
        env.setdefault("RUST_LOG", "error")
        env.setdefault("LLAMACPP_LOG", "error")
        env.setdefault("RUST_BACKTRACE", "0")
        env.pop("LD_DEBUG", None)

        server_params = StdioServerParameters(
            command=str(config.BINARY_PATH),
            args=args,
            env=env,
        )

        self._stdio_cm = stdio_client(server_params)
        self._read, self._write = await self._stdio_cm.__aenter__()
        self._session = ClientSession(self._read, self._write)
        await self._session.__aenter__()
        await self._session.initialize()
        logger.info("MCP client connected (bank=%s)", self.bank)

    async def stop(self):
        if self._session:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
        if self._stdio_cm:
            try:
                await self._stdio_cm.__aexit__(None, None, None)
            except Exception:
                pass
        logger.info("MCP client stopped")

    async def _call(self, tool_name: str, params: dict[str, Any] | None = None) -> Any:
        if self._session is None:
            raise RuntimeError("Client not started")
        params = params or {}

        import json as _json

        try:
            result = await self._session.call_tool(tool_name, params)
        except RuntimeError:
            result = await self._raw_call_sync(tool_name, params)
            return result

        content = result.content[0] if result.content else None
        if content and hasattr(content, "text"):
            try:
                return _json.loads(content.text)
            except (_json.JSONDecodeError, ValueError):
                return content.text
        return content

    async def _raw_call_sync(self, tool_name: str, params: dict[str, Any]) -> Any:
        import json as _json

        raw_result = await self._session.send_request(
            mcp_types.CallToolRequest(
                params=mcp_types.CallToolRequestParams(
                    name=tool_name,
                    arguments=params,
                ),
            ),
            mcp_types.CallToolResult,
        )
        content = raw_result.content[0] if raw_result.content else None
        if content and hasattr(content, "text"):
            try:
                return _json.loads(content.text)
            except (_json.JSONDecodeError, ValueError):
                return content.text
        return str(raw_result)

    async def system_status(self) -> dict:
        return await self._call("system_status")

    async def create_bank(self, name: str, description: str = "") -> dict:
        self.bank = name
        return await self._call("create_memory_bank", {"name": name, "description": description})

    async def store_content(
        self,
        content: str,
        memory_type: str = "conversational",
        topics: list[str] | None = None,
        context: list[str] | None = None,
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
            "bank": self.bank,
        }
        if topics:
            params["topics"] = topics
        if context:
            params["context"] = context
        if user_id:
            params["user_id"] = user_id
        if agent_id:
            params["agent_id"] = agent_id
        return await self._call("add_content_memory", params)

    async def store_intuitive(
        self,
        messages: list[dict[str, str]],
        memory_type: str = "conversational",
        user_id: str | None = None,
        agent_id: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {
            "messages": messages,
            "memory_type": memory_type,
            "bank": self.bank,
        }
        if user_id:
            params["user_id"] = user_id
        if agent_id:
            params["agent_id"] = agent_id
        return await self._call("add_intuitive_memory", params)

    async def query(
        self,
        query: str,
        k: int = 5,
        memory_type: str | None = None,
        pyramid_mode: str | None = None,
        graph_traversal: dict | None = None,
        similarity_threshold: float | None = None,
        granularity: str | None = None,
        excerpt_max_chars: int | None = None,
    ) -> dict:
        params: dict[str, Any] = {
            "query": query,
            "k": k,
            "bank": self.bank,
        }
        if memory_type:
            params["memory_type"] = memory_type
        if similarity_threshold is not None:
            params["similarity_threshold"] = similarity_threshold
        if pyramid_mode and pyramid_mode != "none":
            params["pyramid_config"] = {"mode": pyramid_mode}
        if graph_traversal:
            params["graph_traversal"] = graph_traversal
        if granularity:
            params["granularity"] = granularity
        if excerpt_max_chars is not None:
            params["excerpt_max_chars"] = excerpt_max_chars

        return await self._call("query_memory", params)

    async def trigger_abstraction(self, target_layer: int = 1) -> dict:
        return await self._call("trigger_abstraction", {"target_layer": target_layer})

    async def start_abstraction_pipeline(self) -> dict:
        return await self._call("start_abstraction_pipeline")

    async def stop_abstraction_pipeline(self) -> dict:
        return await self._call("stop_abstraction_pipeline")

    async def cleanup_bank(self, bank_name: str) -> dict:
        return await self._call("cleanup_resources", {
            "target": "banks",
            "name": bank_name,
            "confirm": "I confirm this data will be permanently lost",
        })


@asynccontextmanager
async def mcp_session(config_path: str | None = None, bank_name: str = "default"):
    client = LlmMemClient(config_path=config_path, bank_name=bank_name)
    await client.start()
    try:
        yield client
    finally:
        await client.stop()


def format_turn(role: str, content: str, speaker: str | None = None) -> str:
    label = speaker or role.capitalize()
    return f"[{label}]: {content}"


def format_session(messages: list[dict], session_idx: int, timestamp: str = "") -> str:
    lines = [f"--- Session {session_idx + 1} ---"]
    if timestamp:
        lines.append(f"Date: {timestamp}")
    for msg in messages:
        role = msg.get("role", "unknown")
        name = msg.get("name")
        label = name or role.capitalize()
        lines.append(f"[{label}]: {msg.get('content', '')}")
    return "\n".join(lines)
