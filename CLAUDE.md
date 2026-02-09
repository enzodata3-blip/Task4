# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Context

This is the `model_a` workspace in an A/B testing experiment. It is paired with a `model_b` directory under the same parent (`Task4/`). Session logs, transcripts, and metrics are routed to `../logs/model_a/` automatically by the hooks infrastructure.

## A/B Testing Infrastructure

The `.claude/hooks/` directory contains session lifecycle hooks that run automatically:

- **`capture_session_event.py`** — Fires on `SessionStart` and `SessionEnd`. Captures git metadata (commit hash, branch) at session start and logs session boundary events.
- **`process_transcript.py`** — Fires on `Stop` (incremental) and `SessionEnd` (final). Reads the raw JSONL transcript, deduplicates messages, extracts thinking blocks, aggregates token usage, analyzes tool calls, and generates a session summary with git diff metrics.
- **`claude_code_capture_utils.py`** — Shared utilities: detects `model_a`/`model_b` lane from the working directory path, resolves the experiment root (the parent containing both `model_a/` and `model_b/`), reads `manifest.json` for task/model assignments, and routes log files to `../logs/<model_lane>/`.

The experiment root is detected by walking up the directory tree to find the first ancestor that contains both `model_a/` and `model_b/` subdirectories. A `manifest.json` at the experiment root can specify `task_id` and `assignments` (mapping lane names to model names).

## Log Output

All session data is written as JSONL to `../logs/model_a/session_<session_id>.jsonl`. Each file contains: `session_start`, deduplicated `user`/`assistant`/`assistant_thinking` messages with A/B metadata, and a final `session_summary` with token totals, tool call counts, thinking metrics, and git diff stats.
