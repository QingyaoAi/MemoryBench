from tqdm import tqdm
from typing import Dict, List

from src.agent.light import LightAgent, LightAgentConfig
from src.solver.base import BaseSolver


class LightSolver(BaseSolver):
    """MemoryBench solver wrapping the three-layer LIGHT memory system."""

    AGENT_CLASS = LightAgent

    def __init__(self, config: LightAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "LIGHT"
        self.current_corpus_doc_ids: List[str] = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        return super()._create_or_load_memory(dialogs, can_thread=False)

    # ------------------------------------------------------------------
    # Corpus ingestion (Locomo + DialSim share the same shape).
    # Each session becomes a single episodic chunk; every turn also
    # lands verbatim in working memory; the scratchpad is updated once
    # per session.
    # ------------------------------------------------------------------
    def memory_locomo_conversation(self, conversation, session_cnt: int):
        pbar = tqdm(
            total=session_cnt,
            desc="LIGHT ingesting corpus",
            ascii=True,
            dynamic_ncols=False,
            ncols=80,
        )
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation[f"session_{session_idx}_date_time"]
            session = conversation[f"session_{session_idx}"]

            session_lines = [f"[{session_date_time}]"]
            for turn in session:
                line = f"{turn['speaker']}: {turn['text']}"
                session_lines.append(line)
                self.agent.working_memory.append(line.upper())
            session_text = "\n".join(session_lines)

            doc_id = f"corpus_session_{session_idx}"
            self.agent.add_memory(
                content=session_text,
                doc_id=doc_id,
                metadata={
                    "session_idx": session_idx,
                    "timestamp": session_date_time,
                },
            )
            self.current_corpus_doc_ids.append(doc_id)

            # One scratchpad update per session keeps the LLM cost bounded.
            self.agent._update_scratchpad(session_text)

            session_idx += 1
            pbar.update(1)

    def memory_dialsim_conversation(self, conversation, session_cnt: int):
        # Same shape as Locomo for LIGHT's purposes.
        return self.memory_locomo_conversation(conversation, session_cnt)

    def delete_conversation_memory(self):
        for did in self.current_corpus_doc_ids:
            self.agent.delete_memory(did)
        self.current_corpus_doc_ids = []
