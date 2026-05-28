# Tests

## `test_refactor.py`

Integration test for the registry / corpus-format refactor. Loads the
local `TinyDataset/` and exercises the public `memorybench` API plus the
new `src.memory_systems` registry.

### Run (offline, no API calls)

```bash
cd MemoryBench/
python -m unittest tests.test_refactor -v
```

### Optional live LLM ping

The final test case is skipped unless `MEMORYBENCH_LLM_PING=1`. When
enabled it reads credentials from `../API_config.json` (one directory
above the repo) and sends a 16-token "ok" prompt via the Anthropic SDK.

```bash
MEMORYBENCH_LLM_PING=1 python -m unittest tests.test_refactor.TestOptionalLLMPing -v
```

The token is never written to disk, printed, or included in failure
messages. `API_config.json` is `.gitignore`'d defensively.
