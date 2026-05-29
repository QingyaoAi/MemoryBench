"""TencentDB Agent Memory baseline.

Wraps the TencentDB-Agent-Memory Gateway sidecar (Node.js/TypeScript) via the
Python hermes-plugin client.  Architecture:

  TencentDBAgent
    ├── GatewaySupervisor  ── spawns Node.js gateway as a subprocess
    ├── MemoryTencentdbSdkClient  ── HTTP calls to the gateway
    └── LlmFactory  ── Python LLM for answer generation

Memory pipeline (runs inside the gateway):
  L0 Conversation → L1 Atom (structured facts) → L2 Scenario → L3 Persona

Ingestion uses the gateway /seed endpoint; retrieval uses /recall.

Prerequisites (run once in the baseline directory):
    cd baselines/TencentDB-Agent-Memory
    npm install
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from src.agent.base_agent import BaseAgent
from src.llms import LlmFactory

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import the Python hermes-plugin client/supervisor shipped with the baseline.
# The package __init__.py imports Hermes-specific deps not present here, so we
# load client.py and supervisor.py directly via importlib while injecting a
# minimal fake package so relative imports inside supervisor.py work correctly.
# ---------------------------------------------------------------------------
import importlib.util as _ilu
import types as _types

_PLUGIN_PKG = "memory.memory_tencentdb"
_PLUGIN_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../baselines/TencentDB-Agent-Memory/hermes-plugin/memory/memory_tencentdb",
    )
)

def _load_tdai_module(mod_name: str, filename: str):
    """Load a single .py file as part of the tdai plugin package."""
    full_name = f"{_PLUGIN_PKG}.{mod_name}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    spec = _ilu.spec_from_file_location(
        full_name,
        os.path.join(_PLUGIN_DIR, filename),
        submodule_search_locations=[],
    )
    mod = _ilu.module_from_spec(spec)
    mod.__package__ = _PLUGIN_PKG
    sys.modules[full_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Ensure the parent packages exist in sys.modules so relative imports resolve.
for _pkg in ("memory", _PLUGIN_PKG):
    if _pkg not in sys.modules:
        _fake = _types.ModuleType(_pkg)
        _fake.__package__ = _pkg
        _fake.__path__ = []
        sys.modules[_pkg] = _fake

_client_mod = _load_tdai_module("client", "client.py")
_supervisor_mod = _load_tdai_module("supervisor", "supervisor.py")
# Expose on the fake package so relative imports inside supervisor work.
sys.modules[_PLUGIN_PKG].client = _client_mod  # type: ignore[attr-defined]

MemoryTencentdbSdkClient = _client_mod.MemoryTencentdbSdkClient
GatewaySupervisor = _supervisor_mod.GatewaySupervisor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TencentDBAgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm", "anthropic"] = Field(
        default="openai",
        description="LLM provider for answer generation.",
    )
    llm_config: dict = Field(default_factory=dict)

    memory_cache_dir: str = Field(
        default="./tencentdb_index",
        description="Directory used as the gateway's data store (SQLite + JSON files).",
    )
    retrieve_k: int = Field(default=5)

    # --- Gateway HTTP server ---
    gateway_port: int = Field(
        default=8420,
        description=(
            "Port for the Node.js gateway HTTP server.  "
            "Change this when running multiple experiments in parallel."
        ),
    )
    gateway_host: str = Field(default="127.0.0.1")

    # --- Gateway LLM (for L1 memory extraction inside the gateway) ---
    gateway_llm_base_url: str = Field(
        default="http://localhost:12366/v1",
        description="Base URL for the LLM used by the gateway's L1 extraction pipeline.",
    )
    gateway_llm_api_key: str = Field(default="noop")
    gateway_llm_model: str = Field(
        default="Qwen/Qwen3-8B",
        description="Model name for the gateway's L1 extraction pipeline.",
    )

    # Recall strategy passed to the gateway
    recall_strategy: str = Field(
        default="hybrid",
        description="Memory recall strategy: 'keyword', 'embedding', or 'hybrid'.",
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TencentDBAgent(BaseAgent):
    """MemoryBench agent backed by the TencentDB Agent Memory Gateway sidecar."""

    def __init__(self, config: TencentDBAgentConfig = TencentDBAgentConfig()):
        self.config = config

        # Resolve the gateway data directory from memory_cache_dir.
        data_dir = os.path.abspath(os.path.join(config.memory_cache_dir, "tdai_data"))
        os.makedirs(data_dir, exist_ok=True)
        self._data_dir = data_dir

        # Build the command to launch the gateway.
        baseline_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../baselines/TencentDB-Agent-Memory")
        )
        tsx_bin = os.path.join(baseline_dir, "node_modules", ".bin", "tsx")
        server_ts = os.path.join(baseline_dir, "src", "gateway", "server.ts")

        if not os.path.exists(tsx_bin):
            raise RuntimeError(
                f"tsx not found at {tsx_bin}. "
                "Run 'npm install' inside baselines/TencentDB-Agent-Memory first."
            )

        gateway_cmd = f"{tsx_bin} {server_ts}"

        # Inject gateway configuration via env vars BEFORE supervisor copies os.environ.
        os.environ.setdefault("TDAI_GATEWAY_PORT", str(config.gateway_port))
        os.environ.setdefault("TDAI_GATEWAY_HOST", config.gateway_host)
        os.environ.setdefault("TDAI_DATA_DIR", data_dir)
        os.environ.setdefault("TDAI_LLM_BASE_URL", config.gateway_llm_base_url)
        os.environ.setdefault("TDAI_LLM_API_KEY", config.gateway_llm_api_key)
        os.environ.setdefault("TDAI_LLM_MODEL", config.gateway_llm_model)
        # Always override port/host/data_dir since they may differ per instance.
        os.environ["TDAI_GATEWAY_PORT"] = str(config.gateway_port)
        os.environ["TDAI_GATEWAY_HOST"] = config.gateway_host
        os.environ["TDAI_DATA_DIR"] = data_dir
        os.environ["TDAI_LLM_BASE_URL"] = config.gateway_llm_base_url
        os.environ["TDAI_LLM_API_KEY"] = config.gateway_llm_api_key
        os.environ["TDAI_LLM_MODEL"] = config.gateway_llm_model

        self._supervisor = GatewaySupervisor(
            host=config.gateway_host,
            port=config.gateway_port,
            gateway_cmd=gateway_cmd,
        )
        available = self._supervisor.ensure_running()
        if not available:
            raise RuntimeError(
                "TencentDB gateway failed to start. "
                "Check that Node.js ≥22.16 is installed and 'npm install' was run in "
                "baselines/TencentDB-Agent-Memory."
            )

        self._client: MemoryTencentdbSdkClient = self._supervisor.client

        # Python LLM for answer generation (separate from the gateway's LLM).
        self.llm = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config,
        )

    # ---- helpers -----------------------------------------------------------

    @staticmethod
    def _messages_to_seed_sessions(
        messages: List[Dict[str, str]],
        session_key: str,
    ) -> List[Dict]:
        """Convert a flat messages list into seed-format sessions.

        The seed API expects:
          [{"sessionKey": "...", "conversations": [[msg, msg], [msg, msg], ...]}]

        Each inner list is one "round" (typically one user+assistant pair).
        """
        rounds: List[List[Dict[str, str]]] = []
        current_round: List[Dict[str, str]] = []

        for msg in messages:
            current_round.append({"role": msg["role"], "content": msg["content"]})
            # Close a round after each assistant message.
            if msg["role"] == "assistant":
                if current_round:
                    rounds.append(current_round)
                    current_round = []

        # Flush any trailing user-only messages.
        if current_round:
            rounds.append(current_round)

        if not rounds:
            return []

        return [{"sessionKey": session_key, "conversations": rounds}]

    # ---- memory interface --------------------------------------------------

    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ) -> None:
        """Seed a full dialog into the gateway (L0 → L1 extraction)."""
        session_key = f"mb_{conversation_idx}"
        sessions = self._messages_to_seed_sessions(messages, session_key)
        if not sessions:
            return
        try:
            self._client.seed(
                data={"sessions": sessions},
                session_key=session_key,
                auto_fill_timestamps=True,
            )
        except Exception as exc:
            logger.warning("TencentDB seed failed for conversation %s: %s", conversation_idx, exc)

    def retrieve_memory(self, content: str, k: Optional[int] = None) -> str:
        """Return a context string from the gateway's /recall endpoint."""
        try:
            result = self._client.recall(
                query=content,
                session_key="mb_retrieval",
            )
            return result.get("context", "")
        except Exception as exc:
            logger.warning("TencentDB recall failed: %s", exc)
            return ""

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        lang: Literal["en", "zh"] = "en",
        retrieve_k: Optional[int] = None,
    ) -> str:
        """Retrieve memory context then generate an answer with the Python LLM."""
        question = messages[-1]["content"]
        context = self.retrieve_memory(question, k=retrieve_k or self.config.retrieve_k)

        if context:
            if lang == "zh":
                user_prompt = (
                    f"相关记忆：\n{context}\n\n用户输入：\n{question}\n\n"
                    "请根据提供的记忆，准确、自然地回答用户的输入。"
                )
            else:
                user_prompt = (
                    f"Context:\n{context}\n\nUser: \n{question}\n\n"
                    "Based on the context provided, respond naturally and appropriately "
                    "to the user's input above."
                )
            messages[-1]["content"] = user_prompt

        return self.llm.generate_response(messages=messages)

    # ---- persistence -------------------------------------------------------

    def save_memories(self) -> None:
        """The gateway persists to SQLite automatically — this is a no-op."""
        pass

    def load_memories(self) -> None:
        """Ensure the gateway is running so it can serve its persisted SQLite data."""
        self._supervisor.ensure_running()

    # ---- lifecycle ---------------------------------------------------------

    def shutdown(self) -> None:
        """Gracefully stop the gateway sidecar process."""
        try:
            self._supervisor.shutdown()
        except Exception as exc:
            logger.debug("Error shutting down TencentDB gateway: %s", exc)

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
