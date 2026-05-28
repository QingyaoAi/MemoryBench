"""LIGHT: a cognitively-inspired three-layer memory system.

Based on the LIGHT framework introduced in the BEAM repo
(https://github.com/mohammadtavakoli78/BEAM). LIGHT augments an LLM with three
complementary memory modules:

  - Episodic memory  - long-term vector store of dialogue chunks with LLM-
                       generated summaries; retrieved by semantic similarity.
  - Working memory   - the most recent N turns kept verbatim.
  - Scratchpad       - an LLM-maintained running summary that compresses
                       salient facts, persistent user instructions, and recent
                       context after every new exchange.

At inference time all three are concatenated into the prompt. This file is a
fresh implementation in the MemoryBench agent/solver style — it follows the
same algorithmic recipe as BEAM/light.py but does not depend on LangChain.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Deque, Dict, List, Literal, Optional, Union

import faiss
import numpy as np
import torch
from openai import OpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.agent.base_agent import BaseAgent
from src.llms import LlmFactory


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
_SCRATCHPAD_UPDATE_PROMPT = """You maintain a compact running summary of an ongoing conversation.
Update it with the new exchange below.
Keep: durable facts about the user, persistent instructions, the user's current goal,
and recent context that would help answer follow-up questions. Drop chit-chat.
Stay under 400 words. Return ONLY the updated summary, no preamble.

Previous summary:
{scratchpad}

New exchange:
{exchange}
"""

_EPISODIC_SUMMARY_PROMPT = """Summarise the dialogue chunk below in 2-4 sentences,
keeping named entities, facts, and the user's questions/instructions.
Return ONLY the summary, no preamble.

Dialogue:
{chunk}
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class LightAgentConfig(BaseModel):
    llm_provider: Literal["openai", "vllm", "anthropic"] = Field(
        default="openai",
        description="LLM provider used both for chat completion and for summarisation.",
    )
    llm_config: dict = Field(default_factory=dict)
    embedder_provider: Literal["openai", "vllm", "huggingface"] = Field(default="vllm")
    embedder_model: str = Field(default="Qwen/Qwen3-Embedding-0.6B")
    embedding_dim: int = Field(default=1024)
    embedder_base_url: Optional[str] = Field(default=None)
    embedder_api_key: Optional[str] = Field(default="")
    memory_cache_dir: str = Field(default="./light_index")
    retrieve_k: int = Field(default=5)
    working_memory_size: int = Field(
        default=100,
        description="Max recent turns kept verbatim in working memory.",
    )
    scratchpad_max_chars: int = Field(
        default=4000,
        description="Hard cap on scratchpad length after each LLM update.",
    )
    enable_summary: bool = Field(
        default=True,
        description=(
            "When True, the scratchpad and per-chunk episodic summaries are produced "
            "by the LLM. Set False to use a cheap truncation fallback (much faster, "
            "weaker retrieval)."
        ),
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class LightAgent(BaseAgent):
    EPISODIC_INDEX_FILE = "episodic.faiss"
    EPISODIC_META_FILE = "episodic_meta.json"
    WORKING_FILE = "working.json"
    SCRATCH_FILE = "scratchpad.txt"

    def __init__(self, config: LightAgentConfig = LightAgentConfig()):
        self.config = config
        cache = config.memory_cache_dir
        os.makedirs(cache, exist_ok=True)

        self.episodic_index_path = os.path.join(cache, self.EPISODIC_INDEX_FILE)
        self.episodic_meta_path = os.path.join(cache, self.EPISODIC_META_FILE)
        self.working_path = os.path.join(cache, self.WORKING_FILE)
        self.scratch_path = os.path.join(cache, self.SCRATCH_FILE)

        # Episodic memory
        self.episodic_index = faiss.IndexFlatL2(config.embedding_dim)
        self.episodic_meta: List[Dict] = []

        # Working memory — FIFO of recent turns.
        self.working_memory: Deque[str] = deque(maxlen=config.working_memory_size)

        # Scratchpad
        self.scratchpad: str = ""

        # Embedder
        if config.embedder_provider == "huggingface":
            self.embedder = SentenceTransformer(
                config.embedder_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        else:
            self.embedder_client = OpenAI(
                api_key=config.embedder_api_key or "noop",
                base_url=config.embedder_base_url,
            )

        # Chat LLM (also used for summarisation when enable_summary)
        self.llm = LlmFactory.create(
            provider_name=config.llm_provider,
            config=config.llm_config,
        )

    # ---- embedding -----------------------------------------------------
    def _embed(self, text: str) -> np.ndarray:
        size = len(text)
        # Shrink-and-retry like EmbedderAgent does, to dodge token-limit errors.
        for part in range(100, 0, -1):
            slice_end = max(1, int(size * (part + 1) // 100))
            try:
                if self.config.embedder_provider == "huggingface":
                    vec = self.embedder.encode(
                        [text[:slice_end]],
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )[0]
                else:
                    vec = self.embedder_client.embeddings.create(
                        input=[text[:slice_end]],
                        model=self.config.embedder_model,
                    ).data[0].embedding
                return np.asarray(vec, dtype=np.float32)
            except Exception:
                continue
        return np.zeros(self.config.embedding_dim, dtype=np.float32)

    # ---- summarisation -------------------------------------------------
    def _llm_complete(self, prompt: str) -> str:
        try:
            return self.llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            ) or ""
        except Exception:
            return ""

    def _update_scratchpad(self, exchange_text: str) -> None:
        if not self.config.enable_summary:
            merged = (self.scratchpad + "\n" + exchange_text).strip()
            self.scratchpad = merged[-self.config.scratchpad_max_chars:]
            return
        prompt = _SCRATCHPAD_UPDATE_PROMPT.format(
            scratchpad=self.scratchpad or "(empty)",
            exchange=exchange_text[:4000],
        )
        updated = self._llm_complete(prompt).strip()
        self.scratchpad = (updated or self.scratchpad)[: self.config.scratchpad_max_chars]

    def _summarise_chunk(self, chunk_text: str) -> str:
        if not self.config.enable_summary:
            return chunk_text[:512]
        prompt = _EPISODIC_SUMMARY_PROMPT.format(chunk=chunk_text[:4000])
        out = self._llm_complete(prompt).strip()
        return out or chunk_text[:512]

    # ---- episodic API --------------------------------------------------
    def add_memory(
        self,
        content: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add one chunk to episodic memory.

        The chunk is summarised by the LLM (or truncated, if `enable_summary`
        is off). The summary is what we embed for retrieval — that gives
        denser, more queryable vectors than the raw turns.
        """
        if doc_id is None:
            doc_id = f"doc_{len(self.episodic_meta)}"
        summary = self._summarise_chunk(content)
        vec = self._embed(summary)
        try:
            self.episodic_index.add(np.asarray([vec], dtype=np.float32))
        except Exception:
            time.sleep(1)
            self.episodic_index.add(np.asarray([vec], dtype=np.float32))
        self.episodic_meta.append({
            "doc_id": doc_id,
            "content": content,
            "summary": summary,
            "metadata": metadata or {},
        })

    def delete_memory(self, doc_id: str) -> None:
        """Lazy delete: flip a flag, skip during retrieval."""
        for m in self.episodic_meta:
            if m["doc_id"] == doc_id:
                m["deleted"] = True
                return

    # ---- ingest a full conversation ------------------------------------
    def add_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        conversation_idx: Union[int, str] = 0,
    ) -> None:
        """Ingest one full dialogue:

          working   <- every turn verbatim, FIFO-capped
          episodic  <- one chunk per (user, assistant) pair, embedded by summary
          scratchpad<- single LLM update covering the whole conversation
        """
        conv_id = str(conversation_idx)

        for msg in messages:
            self.working_memory.append(f"{msg['role'].upper()}: {msg['content']}")

        # Episodic: pair-up turns.
        pair: List[Dict[str, str]] = []
        for idx, msg in enumerate(messages):
            pair.append(msg)
            if len(pair) == 2 or idx == len(messages) - 1:
                chunk_text = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in pair
                )
                self.add_memory(
                    content=chunk_text,
                    doc_id=f"conv_{conv_id}_pair_{idx}",
                    metadata={"conversation_idx": conv_id},
                )
                pair = []

        # Scratchpad: one update per conversation.
        conv_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in messages
        )
        self._update_scratchpad(conv_text)

    # ---- retrieval -----------------------------------------------------
    def retrieve_memory(self, content: str, k: Optional[int] = None) -> str:
        """Concatenate scratchpad + working memory + top-k episodic hits."""
        k = k or self.config.retrieve_k

        episodic_hits: List[str] = []
        if self.episodic_index.ntotal > 0:
            vec = self._embed(content)
            _, idxs = self.episodic_index.search(
                np.asarray([vec], dtype=np.float32),
                min(k * 2, self.episodic_index.ntotal),
            )
            seen = set()
            for i in idxs[0]:
                if i < 0 or i >= len(self.episodic_meta):
                    continue
                m = self.episodic_meta[i]
                if m.get("deleted") or m["doc_id"] in seen:
                    continue
                seen.add(m["doc_id"])
                episodic_hits.append(m["content"])
                if len(episodic_hits) >= k:
                    break

        sections: List[str] = []
        if self.scratchpad:
            sections.append(f"# Running Summary\n{self.scratchpad}")
        if self.working_memory:
            sections.append("# Recent Turns\n" + "\n".join(self.working_memory))
        if episodic_hits:
            sections.append(
                "# Episodic Retrieval\n" + "\n---\n".join(episodic_hits)
            )
        return "\n\n".join(sections)

    # ---- inference -----------------------------------------------------
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        lang: Literal["en", "zh"] = "en",
        retrieve_k: Optional[int] = None,
    ) -> str:
        question = messages[-1]["content"]
        context = self.retrieve_memory(question, k=retrieve_k)

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

    # ---- persistence ---------------------------------------------------
    def save_memories(self) -> None:
        faiss.write_index(self.episodic_index, self.episodic_index_path)
        with open(self.episodic_meta_path, "w", encoding="utf-8") as f:
            json.dump(self.episodic_meta, f, ensure_ascii=False)
        with open(self.working_path, "w", encoding="utf-8") as f:
            json.dump(list(self.working_memory), f, ensure_ascii=False)
        with open(self.scratch_path, "w", encoding="utf-8") as f:
            f.write(self.scratchpad)

    def load_memories(self) -> None:
        if os.path.exists(self.episodic_index_path):
            self.episodic_index = faiss.read_index(self.episodic_index_path)
        if os.path.exists(self.episodic_meta_path):
            with open(self.episodic_meta_path, encoding="utf-8") as f:
                self.episodic_meta = json.load(f)
        if os.path.exists(self.working_path):
            with open(self.working_path, encoding="utf-8") as f:
                self.working_memory = deque(
                    json.load(f), maxlen=self.config.working_memory_size
                )
        if os.path.exists(self.scratch_path):
            with open(self.scratch_path, encoding="utf-8") as f:
                self.scratchpad = f.read()
