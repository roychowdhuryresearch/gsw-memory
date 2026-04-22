from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console

console = Console()


_INPUT_PRICE_PER_M = 0.15
_OUTPUT_PRICE_PER_M = 0.60


class StageLogger:
    def __init__(
        self,
        *,
        stage_name: str,
        output_dir: str | Path,
        verbose: bool = False,
        show_thinking: bool = False,
    ) -> None:
        self.stage_name = str(stage_name or "stage").strip() or "stage"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = bool(verbose)
        self.show_thinking = bool(show_thinking)
        self.progress_path = self.output_dir / f"{self.stage_name}_progress.jsonl"
        self._managed_loggers: list[logging.Logger] = []
        # Token/cost tracking
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.question_input_tokens: int = 0
        self.question_output_tokens: int = 0
        self.questions_completed: int = 0
        self._setup_logging()

    def _setup_logging(self) -> None:
        logger_base = f"query_then_sleep.{self.stage_name}.{self.output_dir.name}"
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        def _reset_logger(name: str, level: int) -> logging.Logger:
            logger = logging.getLogger(name)
            logger.setLevel(level)
            logger.propagate = False
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass
            self._managed_loggers.append(logger)
            return logger

        self.logger = _reset_logger(logger_base, logging.DEBUG)
        execution_handler = logging.FileHandler(self.logs_dir / "execution.log")
        execution_handler.setLevel(logging.DEBUG)
        execution_handler.setFormatter(file_formatter)
        self.logger.addHandler(execution_handler)

        self.tool_logger = _reset_logger(f"{logger_base}.tools", logging.INFO)
        tool_handler = logging.FileHandler(self.logs_dir / "tool_calls.jsonl")
        tool_handler.setFormatter(logging.Formatter("%(message)s"))
        self.tool_logger.addHandler(tool_handler)

        self.error_logger = _reset_logger(f"{logger_base}.errors", logging.ERROR)
        error_handler = logging.FileHandler(self.logs_dir / "errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.error_logger.addHandler(error_handler)

        self.trace_logger = _reset_logger(f"{logger_base}.trace", logging.INFO)
        trace_handler = logging.FileHandler(self.logs_dir / "agent_trace.log")
        trace_handler.setFormatter(logging.Formatter("%(message)s"))
        self.trace_logger.addHandler(trace_handler)

        # Human-readable trace file
        self._readable_trace_path = self.logs_dir / "readable_trace.md"
        self._readable_trace_file = open(self._readable_trace_path, "w")
        self._readable_trace_file.write(f"# {self.stage_name} — Agent Reasoning Trace\n\n")
        self._readable_trace_file.flush()

        console.print(f"  Logs directory: {self.logs_dir}")
        console.print(f"  Trace log: {self.logs_dir / 'agent_trace.log'}")
        console.print(f"  Readable trace: {self._readable_trace_path}")
        self.logger.info("Stage logger initialized: %s", self.stage_name)

    def progress(self, phase: str, status: str, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "stage": self.stage_name,
            "phase": str(phase or "").strip(),
            "status": str(status or "").strip(),
            **data,
        }
        with open(self.progress_path, "a") as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def _track_tokens(self, event_type: str, data: Dict[str, Any]) -> None:
        """Accumulate token counts from model_response events."""
        if event_type == "model_response":
            inp = int(data.get("prompt_tokens", 0) or 0)
            out = int(data.get("completion_tokens", 0) or 0)
            self.total_input_tokens += inp
            self.total_output_tokens += out
            self.question_input_tokens += inp
            self.question_output_tokens += out
        elif event_type == "query_started":
            self.question_input_tokens = 0
            self.question_output_tokens = 0
        elif event_type == "query_finished":
            self.questions_completed += 1

    @property
    def total_cost(self) -> float:
        return (self.total_input_tokens / 1_000_000) * _INPUT_PRICE_PER_M + (self.total_output_tokens / 1_000_000) * _OUTPUT_PRICE_PER_M

    @property
    def question_cost(self) -> float:
        return (self.question_input_tokens / 1_000_000) * _INPUT_PRICE_PER_M + (self.question_output_tokens / 1_000_000) * _OUTPUT_PRICE_PER_M

    def _format_tokens(self, tokens: int) -> str:
        if tokens >= 1_000_000:
            return f"{tokens / 1_000_000:.1f}M"
        if tokens >= 1_000:
            return f"{tokens / 1_000:.1f}K"
        return str(tokens)

    def event(self, event_type: str, data: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "event_type": str(event_type or "event").strip() or "event",
            "timestamp": datetime.now().isoformat(),
            "data": data or {},
        }
        rendered = json.dumps(payload, ensure_ascii=False, default=str)
        self.trace_logger.info(rendered)
        self.tool_logger.info(rendered)
        self._track_tokens(payload["event_type"], payload["data"])

        # Write to human-readable trace file
        readable = _format_readable_trace(payload["event_type"], payload["data"])
        if readable and hasattr(self, "_readable_trace_file") and self._readable_trace_file and not self._readable_trace_file.closed:
            self._readable_trace_file.write(readable + "\n")
            self._readable_trace_file.flush()

        # Always-show events (not gated by verbose)
        always_show = _summarize_event_always(payload["event_type"], payload["data"], logger=self)
        if always_show:
            console.print(always_show)
        elif self.verbose:
            summary = _summarize_event_verbose(payload["event_type"], payload["data"], show_thinking=self.show_thinking)
            if summary:
                console.print(summary)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)
        if self.verbose:
            console.print((message % args) if args else message)

    def warning(self, message: str, *args: Any) -> None:
        self.logger.warning(message, *args)
        if self.verbose:
            console.print((message % args) if args else message)

    def error(self, message: str, *args: Any, exc_info: bool = False) -> None:
        self.logger.error(message, *args, exc_info=exc_info)
        self.error_logger.error(message, *args, exc_info=exc_info)
        console.print((message % args) if args else message)

    def log_exception(self, message: str, exc: Exception) -> None:
        self.error_logger.error("%s: %s", message, exc, exc_info=True)
        self.logger.error("%s: %s", message, exc, exc_info=True)
        console.print(f"{message}: {exc}")

    def close(self) -> None:
        if hasattr(self, "_readable_trace_file") and self._readable_trace_file and not self._readable_trace_file.closed:
            self._readable_trace_file.close()
        for logger in self._managed_loggers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass


def _summarize_event_always(event_type: str, data: Dict[str, Any], *, logger: "StageLogger") -> str:
    """Events that ALWAYS print to console, regardless of verbose flag."""
    if event_type == "query_started":
        q_id = str(data.get("question_id", "") or "")[:8]
        question = str(data.get("question", "") or "")
        return f"\n  [bold cyan]\u2753 {q_id}: {question}[/bold cyan]"

    if event_type == "query_finished":
        q_id = str(data.get("question_id", "") or "")[:8]
        f1 = data.get("f1")
        em = data.get("exact_match")
        answer = str(data.get("answer", "") or "")
        gold = str(data.get("gold_answer", "") or "")
        status = data.get("status", "ok")
        iters = data.get("tool_iterations", 0)
        q_tokens = logger.question_input_tokens + logger.question_output_tokens
        q_cost = logger.question_cost
        bridge_used = data.get("bridge_evidence_used", False)
        bridge_str = " | \U0001f309 bridge" if bridge_used else ""

        if f1 is not None and float(f1) >= 0.8:
            icon = "[green]\u2705[/green]"
        elif f1 is not None and float(f1) > 0:
            icon = "[yellow]\u26a0\ufe0f[/yellow]"
        else:
            icon = "[red]\u274c[/red]"

        f1_str = f"{float(f1):.2f}" if f1 is not None else "?"
        cost_str = f"${q_cost:.3f}" if q_cost > 0 else ""
        tokens_str = logger._format_tokens(q_tokens)
        total_cost_str = f"${logger.total_cost:.2f}"

        return (
            f"  {icon} {q_id}: F1={f1_str} | "
            f"[dim]Answer:[/dim] '{answer}' | [dim]Gold:[/dim] '{gold}' | "
            f"{iters} calls | {tokens_str} ({cost_str}){bridge_str}\n"
            f"    [dim]\U0001f4ca Running: {logger.questions_completed} Q | "
            f"{logger._format_tokens(logger.total_input_tokens + logger.total_output_tokens)} tokens | "
            f"cost: {total_cost_str}[/dim]"
        )

    if event_type == "query_model_error":
        error = str(data.get("error", "") or "")[:200]
        stage = data.get("stage", "?")
        return f"  [red]\u26a0 Model error ({stage}): {error}[/red]"

    return ""


def _summarize_event_verbose(event_type: str, data: Dict[str, Any], *, show_thinking: bool) -> str:
    """Events shown only when verbose=True."""
    if event_type == "query_decomposition":
        subs = data.get("sub_questions", [])
        if subs:
            sub_list = " | ".join(str(s.get("sub_question", s) if isinstance(s, dict) else s)[:60] for s in subs[:5])
            return f"    [dim]\U0001f4cb Sub-Qs: {sub_list}[/dim]"
        return ""

    if event_type == "tool_call":
        tool = data.get("tool_name", data.get("name", "?"))
        args = data.get("arguments", {})
        # Show key argument concisely
        key_arg = ""
        for k in ("query", "entity_name", "doc_id", "start_entity"):
            if k in args:
                key_arg = f"{k}='{args[k]}'"
                break
        if not key_arg and args:
            first_key = next(iter(args))
            key_arg = f"{first_key}='{str(args[first_key])[:30]}'"
        return f"    [dim]\U0001f527 {tool}({key_arg})[/dim]"

    if event_type == "tool_result":
        tool = data.get("tool_name", data.get("name", "?"))
        result = data.get("result", {})
        # Summarize result concisely
        if isinstance(result, dict):
            if "entities" in result:
                count = len(result["entities"]) if isinstance(result["entities"], list) else "?"
                return f"    [dim]\u2713 {tool} \u2192 {count} entities[/dim]"
            if "qa_pairs" in result:
                count = len(result["qa_pairs"]) if isinstance(result["qa_pairs"], list) else "?"
                return f"    [dim]\u2713 {tool} \u2192 {count} QA pairs[/dim]"
            if "results" in result:
                count = len(result["results"]) if isinstance(result["results"], list) else "?"
                return f"    [dim]\u2713 {tool} \u2192 {count} results[/dim]"
            if "error" in result:
                return f"    [dim]\u2717 {tool} \u2192 error: {str(result['error'])[:60]}[/dim]"
        return f"    [dim]\u2713 {tool} \u2192 ok[/dim]"

    if event_type == "model_response":
        if show_thinking:
            reasoning = str(data.get("reasoning_preview", "") or "").strip()
            text = str(data.get("text_preview", "") or "").strip()
            content = reasoning or text
            if content:
                # Show first 200 chars of thinking, indented
                lines = content[:200].split("\n")
                formatted = "\n      ".join(lines)
                return f"    [dim italic]\U0001f4ad {formatted}[/dim italic]"
        return ""

    if event_type in {"analysis_started", "guided_sleep_started"}:
        return f"  [bold]\u25b6 {event_type}[/bold]"

    if event_type in {"analysis_finished", "guided_sleep_finished"}:
        return f"  [bold]\u2713 {event_type}[/bold]"

    return ""


# Keep old function name for backward compatibility
def _summarize_event(event_type: str, data: Dict[str, Any], *, show_thinking: bool) -> str:
    return _summarize_event_verbose(event_type, data, show_thinking=show_thinking)


def _format_readable_trace(event_type: str, data: Dict[str, Any]) -> str:
    """Format event as human-readable markdown for the trace log file."""
    if event_type == "query_started":
        q_id = str(data.get("question_id", ""))[:8]
        question = str(data.get("question", ""))
        doc_ids = data.get("doc_ids", [])
        return f"\n---\n\n## Question: {question}\n\n**ID:** `{q_id}` | **Docs:** {len(doc_ids)}\n"

    if event_type == "query_decomposition":
        subs = data.get("sub_questions", [])
        if not subs:
            return ""
        lines = ["**Sub-questions:**"]
        for i, s in enumerate(subs, 1):
            sq = s.get("sub_question", s) if isinstance(s, dict) else str(s)
            lines.append(f"  {i}. {sq}")
        return "\n".join(lines) + "\n"

    if event_type == "model_response":
        reasoning = str(data.get("reasoning_preview", "") or "").strip()
        text_preview = str(data.get("text_preview", "") or "").strip()
        stage = data.get("stage", "")
        tokens = data.get("total_tokens", 0)
        content = reasoning or text_preview
        if not content:
            return ""
        # Show full reasoning, not truncated
        return f"\n**Agent thinking** ({stage}, {tokens} tokens):\n> {content}\n"

    if event_type == "tool_call":
        tool = data.get("tool_name", data.get("name", "?"))
        args = data.get("arguments", {})
        args_str = json.dumps(args, ensure_ascii=False, default=str) if args else ""
        if len(args_str) > 300:
            args_str = args_str[:300] + "..."
        return f"\n**Tool call:** `{tool}({args_str})`"

    if event_type == "tool_result":
        tool = data.get("tool_name", data.get("name", "?"))
        result = data.get("result", {})
        # Summarize result
        if isinstance(result, dict):
            if "error" in result:
                return f"  → **Error:** {result['error']}"
            if "entities" in result and isinstance(result["entities"], list):
                entities = result["entities"]
                names = [str(e.get("name", e.get("entity_name", "?")))[:30] for e in entities[:5]] if entities else []
                return f"  → **{len(entities)} entities found:** {', '.join(names)}"
            if "qa_pairs" in result and isinstance(result["qa_pairs"], list):
                qas = result["qa_pairs"]
                samples = []
                for qa in qas[:3]:
                    q = str(qa.get("question", ""))[:50]
                    a = str(qa.get("answer", ""))[:30]
                    samples.append(f"Q: {q} → A: {a}")
                return f"  → **{len(qas)} QA pairs:**\n" + "\n".join(f"    - {s}" for s in samples)
            if "results" in result and isinstance(result["results"], list):
                return f"  → **{len(result['results'])} results**"
            # Generic dict summary
            keys = list(result.keys())[:5]
            return f"  → **Result keys:** {keys}"
        if isinstance(result, list):
            return f"  → **{len(result)} items**"
        return f"  → {str(result)[:200]}"

    if event_type == "query_finished":
        q_id = str(data.get("question_id", ""))[:8]
        f1 = data.get("f1")
        answer = str(data.get("answer", ""))
        gold = str(data.get("gold_answer", ""))
        status = data.get("status", "ok")
        iters = data.get("tool_iterations", 0)
        missing = data.get("missing_relations", [])
        bridge_used = data.get("bridge_evidence_used", False)

        f1_str = f"{float(f1):.2f}" if f1 is not None else "?"
        icon = "✅" if f1 is not None and float(f1) >= 0.8 else ("⚠️" if f1 is not None and float(f1) > 0 else "❌")

        lines = [
            f"\n### Result: {icon} F1={f1_str}\n",
            f"- **Answer:** {answer}",
            f"- **Gold:** {gold}",
            f"- **Tool calls:** {iters}",
            f"- **Bridge used:** {'Yes' if bridge_used else 'No'}",
        ]
        if missing:
            lines.append(f"- **Missing relations:** {', '.join(str(m) for m in missing)}")
        return "\n".join(lines) + "\n"

    if event_type == "query_model_error":
        error = str(data.get("error", ""))[:500]
        stage = data.get("stage", "?")
        return f"\n**⚠ Model error ({stage}):** {error}\n"

    return ""


def compact_text(text: str, limit: int = 400) -> str:
    raw = str(text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[:limit] + "..."
