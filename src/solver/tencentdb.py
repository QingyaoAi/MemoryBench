from tqdm import tqdm
from typing import Dict, List

from src.agent.tencentdb import TencentDBAgent, TencentDBAgentConfig
from src.solver.base import BaseSolver


class TencentDBSolver(BaseSolver):
    """MemoryBench solver wrapping the TencentDB Agent Memory system."""

    AGENT_CLASS = TencentDBAgent

    def __init__(self, config: TencentDBAgentConfig, memory_cache_dir: str):
        super().__init__(config, memory_cache_dir)
        self.method_name = "TencentDB"
        self._current_corpus_sessions: List[str] = []

    def create_or_load_memory(self, dialogs: List[Dict]):
        # TencentDB's gateway state is not thread-safe for concurrent seeds.
        return super()._create_or_load_memory(dialogs, can_thread=False)

    # ------------------------------------------------------------------
    # Corpus ingestion for Locomo / DialSim datasets.
    # Each session is seeded as one TencentDB session so the L0→L1
    # pipeline can build per-session context.
    # ------------------------------------------------------------------

    def memory_locomo_conversation(self, conversation: Dict, session_cnt: int):
        pbar = tqdm(
            total=session_cnt,
            desc="TencentDB ingesting corpus",
            ascii=True,
            dynamic_ncols=False,
            ncols=80,
        )
        session_idx = 1
        while f"session_{session_idx}" in conversation:
            session_date_time = conversation.get(f"session_{session_idx}_date_time", "")
            session = conversation[f"session_{session_idx}"]

            messages: List[Dict[str, str]] = []
            for turn in session:
                if turn.get("speaker") and turn.get("text"):
                    role = "user" if turn["speaker"].lower() not in ("ai", "assistant") else "assistant"
                    messages.append({"role": role, "content": turn["text"]})

            if messages:
                session_key = f"corpus_session_{session_idx}"
                sessions = self.agent._messages_to_seed_sessions(messages, session_key)
                if sessions:
                    try:
                        self.agent._client.seed(
                            data={"sessions": sessions},
                            session_key=session_key,
                            auto_fill_timestamps=True,
                        )
                        self._current_corpus_sessions.append(session_key)
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).warning(
                            "TencentDB seed failed for corpus session %d: %s", session_idx, exc
                        )

            session_idx += 1
            pbar.update(1)
        pbar.close()

    def memory_dialsim_conversation(self, conversation: Dict, session_cnt: int):
        return self.memory_locomo_conversation(conversation, session_cnt)

    def delete_conversation_memory(self):
        # TencentDB does not expose a per-session delete endpoint in the
        # gateway API, so we clear our tracking list only.
        self._current_corpus_sessions = []
