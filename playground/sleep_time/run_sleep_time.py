#!/usr/bin/env python3
"""
Standalone Script for Agentic Sleep-Time GSW Exploration

Features:
- Command-line configuration
- Structured logging (console + file + tool calls + errors)
- Progress checkpointing and resume capability
- JSON output with detailed statistics
- Enhanced error handling and validation enforcement
- Graceful interrupt handling

Usage:
    python run_sleep_time.py --num_docs 100 --num_entities 20
    python run_sleep_time.py --resume_from checkpoints/checkpoint_entity_10.json
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

#import .env
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from playground.simple_entity_search import EntitySearcher
from src.gsw_memory.sleep_time.tools import GSWTools
from src.gsw_memory.sleep_time.agentic_reconciler import AgenticReconciler

console = Console()


class AgentOutputParser:
    """Parse and clean agent's messy text output for display."""

    @staticmethod
    def parse_agent_text(content: str) -> Dict[str, Any]:
        """
        Parse messy agent output like:
        'analysisWe have...assistantcommentary...assistantfinal...'

        Returns:
            Dict with cleaned_text and extracted sections
        """
        if not content:
            return {"cleaned_text": "[No text output]", "raw": ""}

        parts = {
            "analysis": [],
            "commentary": [],
            "final_summary": None,
            "raw": content
        }

        # Extract 'analysis' sections
        analysis_pattern = r'analysis(.*?)(?=assistant|$)'
        for match in re.finditer(analysis_pattern, content, re.DOTALL | re.IGNORECASE):
            text = match.group(1).strip()
            if text and not text.startswith('commentary'):
                parts["analysis"].append(text)

        # Extract 'final' section
        final_pattern = r'final(.*)$'
        final_match = re.search(final_pattern, content, re.DOTALL | re.IGNORECASE)
        if final_match:
            parts["final_summary"] = final_match.group(1).strip()

        # Clean up for display
        cleaned = content
        # Strip <think> tags (closed and unclosed) from display
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL)
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL)
        # Remove 'assistant' tags
        cleaned = re.sub(r'assistant(?:commentary|analysis|final)?', '\n', cleaned, flags=re.IGNORECASE)
        # Remove function call artifacts
        cleaned = re.sub(r'to=functions\.\w+', '', cleaned)
        # Remove json markers that aren't actual json
        cleaned = re.sub(r'json\{', '{\n', cleaned)
        # Clean up multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        # Trim
        cleaned = cleaned.strip()

        parts["cleaned_text"] = cleaned if cleaned else "[No readable text]"

        return parts


class InteractiveDisplay:
    """Handle interactive display of agent execution with Rich formatting."""

    def __init__(self, console: Console, show_thinking: bool = False):
        self.console = console
        self.show_thinking = show_thinking
        self.parser = AgentOutputParser()

    def show_iteration_header(self, entity: str, iteration: int, max_iter: int):
        """Display iteration header."""
        self.console.print(f"\n[bold cyan]{'─'*70}[/bold cyan]")
        self.console.print(f"[bold]Entity:[/bold] {entity}  |  [bold]Iteration:[/bold] {iteration}/{max_iter}")
        self.console.print(f"[bold cyan]{'─'*70}[/bold cyan]\n")

    def show_tool_call(self, tool_name: str, arguments: Dict[str, Any]):
        """Display tool call with formatted JSON."""
        self.console.print(f"[cyan]🛠️  Tool:[/cyan] [bold]{tool_name}[/bold]")

        # Format args as JSON with syntax highlighting
        try:
            args_json = json.dumps(arguments, indent=2)
            syntax = Syntax(args_json, "json", theme="monokai", padding=(0, 2))
            self.console.print(syntax)
        except:
            self.console.print(f"   {arguments}")

    def show_tool_result(self, result: Any, tool_name: str = ""):
        """Display tool result (truncated)."""
        result_str = str(result)
        is_error = isinstance(result, dict) and 'error' in result

        if is_error:
            # Error result
            error_msg = result.get('error', 'Unknown error')
            self.console.print(f"[red]✗ Error:[/red] {error_msg}")
            if 'hint' in result:
                self.console.print(f"   [yellow]💡 Hint:[/yellow] {result['hint']}")
            if 'invalid_docs' in result:
                self.console.print(f"   [yellow]Invalid docs:[/yellow] {result['invalid_docs']}")
            if 'valid_docs_range' in result:
                self.console.print(f"   [dim]Valid range:[/dim] {result['valid_docs_range']}")
        else:
            # Success result (show full result without truncation)
            self.console.print(f"[green]✓ Result:[/green] [dim]{result_str}[/dim]")

        self.console.print()  # Blank line

    def show_planner_decision(self, decision: Dict[str, Any]):
        """Display compact planner decision details in verbose mode."""
        decision_type = str(decision.get("decision_type", "unknown"))
        source = str(decision.get("source", "unknown"))
        reason = str(decision.get("reason", "")).strip()
        fallback_reason = str(decision.get("fallback_reason", "")).strip()
        attempt = str(decision.get("attempt", "")).strip()

        parts = [
            f"decision_type={decision_type}",
            f"source={source}",
        ]
        if attempt:
            parts.append(f"attempt={attempt}")
        if reason:
            parts.append(f"reason={reason}")
        if fallback_reason:
            parts.append(f"fallback_reason={fallback_reason}")

        self.console.print(
            f"[yellow]🧭 Planner:[/yellow] [dim]{' | '.join(parts)}[/dim]"
        )
        self.console.print()

    def show_worker_event(self, event_type: str, payload: Dict[str, Any]):
        """Display compact worker lifecycle events in verbose mode."""
        if event_type == "rlm_worker_call":
            parts = [
                f"call={payload.get('call_index')}",
                f"depth={payload.get('depth')}",
                f"optional={payload.get('include_optional')}",
                f"planner_action={payload.get('planner_action')}",
                f"planner_source={payload.get('planner_source')}",
                f"invoked={payload.get('worker_invoked')}",
            ]
            reason = str(payload.get("planner_reason", "")).strip()
            if reason:
                parts.append(f"reason={reason}")
            self.console.print(f"[cyan]⚙️  Worker Call:[/cyan] [dim]{' | '.join(parts)}[/dim]")
            self.console.print()
            return

        parts = [
            f"call={payload.get('call_index')}",
            f"candidates={payload.get('candidates_count')}",
            f"accepted_delta={payload.get('accepted_delta')}",
            f"rejected_delta={payload.get('rejected_delta')}",
            f"parse_stage={payload.get('parse_stage')}",
            f"need_recursion={payload.get('need_recursion')}",
        ]
        notes = str(payload.get("worker_notes", "")).strip()
        if notes:
            parts.append(f"notes={notes}")
        self.console.print(f"[cyan]🧾 Worker Result:[/cyan] [dim]{' | '.join(parts)}[/dim]")
        self.console.print()

    def show_validation_result(self, result: Dict[str, Any]):
        """Special display for validation results."""
        is_valid = result.get('valid', False)
        confidence = result.get('confidence', 0)
        reasoning = result.get('reasoning', 'N/A')
        stage = result.get('stage', 'tool')
        failure_codes = result.get('failure_codes', [])
        codes_text = ", ".join(failure_codes) if failure_codes else "None"

        if is_valid:
            self.console.print(Panel(
                f"[bold green]✅ VALIDATION PASSED[/bold green]\n"
                f"Stage: {stage}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Reasoning: {reasoning}",
                border_style="green",
                title="Validation"
            ))
        else:
            self.console.print(Panel(
                f"[bold red]❌ VALIDATION FAILED[/bold red]\n"
                f"Stage: {stage}\n"
                f"Confidence: {confidence:.2f}\n"
                f"Failure codes: {codes_text}\n"
                f"Reasoning: {reasoning}",
                border_style="red",
                title="Validation"
            ))

    def show_bridge_created(self, bridge_id: str, question: str, answers):
        """Display bridge creation success."""
        answers_str = ", ".join(answers) if isinstance(answers, list) else str(answers)
        self.console.print(Panel(
            f"[bold magenta]🎉 BRIDGE CREATED[/bold magenta]\n"
            f"ID: {bridge_id}\n"
            f"Q: {question}\n"
            f"A: {answers_str}",
            border_style="magenta",
            title="Bridge"
        ))

    def show_agent_thinking(self, content: str):
        """Display agent's text output (cleaned)."""
        if not content:
            return

        parsed = self.parser.parse_agent_text(content)

        if parsed["cleaned_text"] and parsed["cleaned_text"] != "[No readable text]":
            self.console.print(f"\n[yellow]💭 Agent Thinking:[/yellow]")

            # Limit length based on show_thinking flag
            text = parsed["cleaned_text"]
            if self.show_thinking:
                # Full reasoning mode: show complete reasoning without truncation
                pass
            else:
                # Default mode in verbose: show first 300 chars for context
                if len(text) > 300:
                    text = text[:300] + "\n\n[...use --show-thinking for full reasoning...]"

            self.console.print(Panel(
                text,
                border_style="yellow",
                padding=(1, 2),
                title="Agent Reasoning"
            ))

    def show_progress_summary(self, entities_done: int, total: int,
                             bridges: int, tokens: int,
                             input_tokens: int = 0, output_tokens: int = 0):
        """Show progress summary."""
        self.console.print(f"\n[dim]{'─'*70}[/dim]")
        self.console.print(
            f"[cyan]Progress:[/cyan] {entities_done}/{total} entities | "
            f"{bridges} bridges | {input_tokens:,} in / {output_tokens:,} out tokens"
        )
        self.console.print(f"[dim]{'─'*70}[/dim]\n")


class SleepTimeRunner:
    """Orchestrates sleep-time exploration with checkpointing and logging."""

    def __init__(self, args: argparse.Namespace):
        """
        Initialize runner with configuration.

        Args:
            args: Parsed command-line arguments
        """
        self.args = args
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()

        # Setup output directories
        self.output_dir = Path(args.output_dir) / f"run_{self.run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()

        # State
        self.entity_searcher = None
        self.agent = None
        self.explored_entities = []
        self.all_bridges = []
        self.errors = []
        self.interrupted = False

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info(f"="*80)
        self.logger.info(f"Sleep-Time Exploration Run: {self.run_id}")
        self.logger.info(f"="*80)
        self.logger.info(f"Configuration: {vars(args)}")

    def _setup_logging(self):
        """Setup structured logging to console and files."""
        # Create logs directory
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Main logger
        self.logger = logging.getLogger("sleep_time")
        self.logger.setLevel(logging.DEBUG)

        # File handler - detailed logs
        file_handler = logging.FileHandler(logs_dir / "execution.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Tool call logger - separate JSON log
        self.tool_logger = logging.getLogger("sleep_time.tools")
        self.tool_logger.setLevel(logging.INFO)
        tool_handler = logging.FileHandler(logs_dir / "tool_calls.jsonl")
        tool_handler.setFormatter(logging.Formatter("%(message)s"))
        self.tool_logger.addHandler(tool_handler)

        # Error logger - separate error log
        self.error_logger = logging.getLogger("sleep_time.errors")
        self.error_logger.setLevel(logging.ERROR)
        error_handler = logging.FileHandler(logs_dir / "errors.log")
        error_handler.setFormatter(file_formatter)
        self.error_logger.addHandler(error_handler)

        # Trace logger - human-readable agent execution trace
        self.trace_logger = logging.getLogger("sleep_time.trace")
        self.trace_logger.setLevel(logging.INFO)
        trace_handler = logging.FileHandler(logs_dir / "agent_trace.log")
        trace_handler.setFormatter(logging.Formatter("%(message)s"))  # Plain text, no timestamps
        self.trace_logger.addHandler(trace_handler)

        # Also log to agentic_reconciler module
        agent_logger = logging.getLogger("src.gsw_memory.sleep_time.agentic_reconciler")
        agent_logger.setLevel(logging.DEBUG)
        agent_logger.addHandler(file_handler)

        console.print(f"[green]✓ Logging setup complete[/green]")
        console.print(f"  Logs directory: {logs_dir}")
        console.print(f"  Trace log: {logs_dir / 'agent_trace.log'}")

    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        console.print("\n[yellow]⚠ Interrupt received! Saving checkpoint...[/yellow]")
        self.interrupted = True
        if self.agent:
            self._save_checkpoint(force=True)
        console.print("[green]✓ Checkpoint saved. Exiting...[/green]")
        sys.exit(0)

    def _create_callback_handler(self, display: Optional[InteractiveDisplay] = None) -> Callable:
        """
        Create callback handler that routes agent events to InteractiveDisplay and/or trace file.

        Args:
            display: Optional InteractiveDisplay for console output. If None, only logs to trace file.
        """
        def callback(event_type: str, data: Dict[str, Any]):
            # Route to display if available
            if display:
                if event_type == "iteration_start":
                    display.show_iteration_header(
                        data["entity"],
                        data["iteration"],
                        data["max_iterations"]
                    )
                elif event_type == "tool_call":
                    display.show_tool_call(data["tool"], data["arguments"])
                elif event_type == "tool_result":
                    display.show_tool_result(data["result"], data["tool"])
                elif event_type == "validation":
                    display.show_validation_result(data)
                elif event_type == "bridge_created":
                    display.show_bridge_created(
                        data["bridge_id"],
                        data["question"],
                        data.get("answers", data.get("answer", ""))
                    )
                elif event_type == "agent_thinking":
                    display.show_agent_thinking(data["content"])
                elif event_type == "rlm_edge_summary":
                    edge = data.get("edge", {})
                    display.show_tool_result(
                        {
                            "edge": edge,
                            "accepted": data.get("accepted", 0),
                            "attempted": data.get("attempted", 0),
                            "rejected": data.get("rejected", 0),
                            "edge_tokens": data.get("edge_tokens", 0),
                            "budget_exhausted": data.get("budget_exhausted", False),
                        },
                        tool_name="rlm_edge_summary",
                    )
                elif event_type == "rlm_doc_summary":
                    display.show_tool_result(
                        {
                            "doc_id": data.get("entity"),
                            "bridges_created": data.get("bridges_created", 0),
                            "pending_edges": data.get("pending_edges", 0),
                            "edges": len(data.get("edge_results", [])),
                        },
                        tool_name="rlm_doc_summary",
                    )
                elif event_type == "hybrid_planner_decision":
                    display.show_planner_decision(data)
                elif event_type in {"rlm_worker_call", "rlm_worker_result"}:
                    display.show_worker_event(event_type, data)

            # Always log to trace file with visual separators
            if event_type == "iteration_start":
                self.trace_logger.info(f"\n\n{'='*80}")
                self.trace_logger.info(f"Entity: {data['entity']} | Iteration: {data['iteration']}/{data['max_iterations']}")
                self.trace_logger.info(f"{'='*80}\n")

            elif event_type == "tool_call":
                self.trace_logger.info(f"\n{'-'*40}")
                self.trace_logger.info(f"Tool: {data['tool']}")
                self.trace_logger.info(f"Args: {json.dumps(data['arguments'], indent=2)}")

            elif event_type == "tool_result":
                result_str = str(data["result"])
                self.trace_logger.info(f"Result: {result_str}")
                self.trace_logger.info(f"{'-'*40}")

            elif event_type == "validation":
                self.trace_logger.info(f"\n>>> VALIDATION <<<")
                self.trace_logger.info(f"Valid: {data.get('valid')} | Confidence: {data.get('confidence')}")
                if 'stage' in data:
                    self.trace_logger.info(f"  Stage: {data['stage']}")
                if 'failure_codes' in data:
                    self.trace_logger.info(f"  Failure codes: {data['failure_codes']}")
                if 'invalid_docs' in data:
                    self.trace_logger.warning(f"  Invalid docs: {data['invalid_docs']}")
                    self.trace_logger.info(f"  Hint: {data.get('hint', 'N/A')}")
                if 'reasoning' in data:
                    self.trace_logger.info(f"  Reasoning: {data['reasoning']}")

            elif event_type == "bridge_created":
                self.trace_logger.info(f"\n{'*'*80}")
                self.trace_logger.info(f"✓ BRIDGE CREATED: {data['bridge_id']}")
                self.trace_logger.info(f"  Q: {data['question']}")
                answers = data.get("answers", data.get("answer", ""))
                answers_str = ", ".join(answers) if isinstance(answers, list) else str(answers)
                self.trace_logger.info(f"  A: {answers_str}")
                self.trace_logger.info(f"{'*'*80}")

            elif event_type == "agent_thinking":
                # Always log reasoning to trace file
                if data["content"]:
                    parser = AgentOutputParser()
                    parsed = parser.parse_agent_text(data["content"])
                    if parsed["cleaned_text"] != "[No readable text]":
                        # Log full reasoning to trace file without truncation
                        text = parsed["cleaned_text"]
                        self.trace_logger.info(f"\n💭 Agent Reasoning:")
                        self.trace_logger.info(text)

            elif event_type == "agent_finished":
                self.trace_logger.info(f"\n{'='*80}")
                self.trace_logger.info(f"✓ Agent finished exploration")
                self.trace_logger.info(f"{'='*80}\n")
            elif event_type == "rlm_edge_summary":
                self.trace_logger.info(f"\n[rlm_edge] {json.dumps(data, ensure_ascii=False)}")
            elif event_type == "rlm_doc_summary":
                self.trace_logger.info(f"\n[rlm_doc] {json.dumps(data, ensure_ascii=False)}")
            elif event_type == "hybrid_planner_decision":
                self.trace_logger.info(f"\n[planner] {json.dumps(data, ensure_ascii=False)}")
                self.tool_logger.info(
                    json.dumps(
                        {
                            "event_type": "hybrid_planner_decision",
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        },
                        ensure_ascii=False,
                    )
                )
            elif event_type == "rlm_worker_call":
                self.trace_logger.info(f"\n[worker_call] {json.dumps(data, ensure_ascii=False)}")
                self.tool_logger.info(
                    json.dumps(
                        {
                            "event_type": "rlm_worker_call",
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        },
                        ensure_ascii=False,
                    )
                )
            elif event_type == "rlm_worker_result":
                self.trace_logger.info(f"\n[worker_result] {json.dumps(data, ensure_ascii=False)}")
                self.tool_logger.info(
                    json.dumps(
                        {
                            "event_type": "rlm_worker_result",
                            "timestamp": datetime.now().isoformat(),
                            "data": data,
                        },
                        ensure_ascii=False,
                    )
                )

        return callback

    def load_gsws(self) -> EntitySearcher:
        """Load GSW structures and build indexes."""
        console.print(f"\n[cyan]Loading {self.args.num_docs} GSWs...[/cyan]")
        self.logger.info(f"Loading GSWs from: {self.args.gsw_path}")

        try:
            entity_searcher = EntitySearcher(
                num_documents=self.args.num_docs,
                path_to_gsw_files=self.args.gsw_path,
                cache_dir=self.args.cache_dir,
                rebuild_cache=False,
                verbose=False,  # Suppress EntitySearcher logs
                use_bm25=True,
                use_gpu_for_qa_index=False
            )

            console.print(f"[green]✓ Loaded {len(entity_searcher.gsw_by_doc_id)} GSWs[/green]")
            self.logger.info(f"Successfully loaded {len(entity_searcher.gsw_by_doc_id)} GSWs")

            return entity_searcher

        except Exception as e:
            console.print(f"[red]✗ Failed to load GSWs: {e}[/red]")
            self.logger.error(f"Failed to load GSWs: {e}", exc_info=True)
            raise

    def initialize_agent(self, display: Optional[InteractiveDisplay] = None) -> AgenticReconciler:
        """
        Initialize the agentic reconciler.

        Args:
            display: Optional InteractiveDisplay for console output. If None, only logs to trace file.
        """
        console.print(f"\n[cyan]Initializing agent with model: {self.args.model}[/cyan]")

        # Always create callback handler for trace logging (with or without display)
        output_callback = self._create_callback_handler(display)

        try:
            agent = AgenticReconciler(
                entity_searcher=self.entity_searcher,
                model_name=self.args.model,
                budget={
                    "max_entities": self.args.num_entities,
                    "max_documents": self.args.num_docs,
                    "max_tokens": self.args.max_tokens
                },
                verbose=False,  # Suppress default verbose - we use callback
                output_callback=output_callback,
                reasoning_effort=self.args.reasoning_effort,
                base_url=getattr(self.args, 'base_url', None),
                bridge_verifier_enabled=not self.args.disable_bridge_verifier,
                bridge_verifier_threshold=self.args.bridge_verifier_threshold,
                bridge_verifier_fail_open=self.args.bridge_verifier_fail_open,
                bridge_verifier_model=self.args.bridge_verifier_model,
                pipeline_mode=self.args.pipeline_mode,
                hybrid_scope=self.args.hybrid_scope,
                root_model=self.args.root_model,
                worker_model=self.args.worker_model,
                edge_max_depth=self.args.edge_max_depth,
                edge_max_calls=self.args.edge_max_calls,
                edge_max_tokens=self.args.edge_max_tokens,
                max_optional_docs_per_edge=self.args.max_optional_docs_per_edge,
            )

            console.print(f"[green]✓ Agent initialized[/green]")
            if hasattr(agent, 'provider'):
                if agent.provider == "together":
                    console.print(f"  [dim]Reasoning effort: {self.args.reasoning_effort}[/dim]")
                elif agent.provider == "vllm":
                    console.print(f"  [dim]vllm base URL: {self.args.base_url}[/dim]")
            console.print(
                f"  [dim]Bridge verifier: {'off' if self.args.disable_bridge_verifier else 'on'} "
                f"(model: {self.args.bridge_verifier_model or self.args.model}, "
                f"threshold: {self.args.bridge_verifier_threshold:.2f})[/dim]"
            )
            console.print(
                f"  [dim]Pipeline: {self.args.pipeline_mode} "
                f"(root={self.args.root_model or self.args.model}, "
                f"worker={self.args.worker_model or self.args.model})[/dim]"
            )
            self.logger.info(
                f"Agent initialized with model: {self.args.model} "
                f"(provider: {getattr(agent, 'provider', 'unknown')}, "
                f"reasoning_effort: {self.args.reasoning_effort}, "
                f"bridge_verifier_enabled: {not self.args.disable_bridge_verifier}, "
                f"bridge_verifier_model: {self.args.bridge_verifier_model or self.args.model}, "
                f"bridge_verifier_threshold: {self.args.bridge_verifier_threshold}, "
                f"bridge_verifier_fail_open: {self.args.bridge_verifier_fail_open}, "
                f"pipeline_mode: {self.args.pipeline_mode}, "
                f"hybrid_scope: {self.args.hybrid_scope}, "
                f"root_model: {self.args.root_model or self.args.model}, "
                f"worker_model: {self.args.worker_model or self.args.model}, "
                f"edge_max_depth: {self.args.edge_max_depth}, "
                f"edge_max_calls: {self.args.edge_max_calls}, "
                f"edge_max_tokens: {self.args.edge_max_tokens}, "
                f"max_optional_docs_per_edge: {self.args.max_optional_docs_per_edge})"
            )

            return agent

        except Exception as e:
            console.print(f"[red]✗ Failed to initialize agent: {e}[/red]")
            self.logger.error(f"Failed to initialize agent: {e}", exc_info=True)
            raise

    def _save_checkpoint(self, entity_count: Optional[int] = None, force: bool = False):
        """
        Save checkpoint to resume later.

        Args:
            entity_count: Number of entities explored (for filename)
            force: Force save even if not at checkpoint interval
        """
        if not force and entity_count and entity_count % self.args.checkpoint_interval != 0:
            return

        checkpoint_file = self.checkpoint_dir / f"checkpoint_entity_{len(self.explored_entities)}.json"

        checkpoint_data = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "config": vars(self.args),
            "progress": {
                "entities_explored": len(self.explored_entities),
                "explored_entity_list": self.explored_entities,
                "bridges_created": len(self.all_bridges),
                "tokens_used": self.agent.tokens_used if self.agent else 0,
                "input_tokens": self.agent.input_tokens if self.agent else 0,
                "output_tokens": self.agent.output_tokens if self.agent else 0,
                "rlm_metrics": self.agent.rlm_metrics if self.agent else {},
                "duration_seconds": time.time() - self.start_time
            },
            "bridges": self.all_bridges,
            "errors": self.errors
        }

        # Atomic write: write to temp file, then rename
        temp_file = checkpoint_file.with_suffix(".tmp")
        with open(temp_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        temp_file.rename(checkpoint_file)

        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        if not force:
            console.print(f"  [dim]→ Checkpoint saved ({len(self.explored_entities)} entities)[/dim]")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load checkpoint to resume exploration.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data
        """
        console.print(f"\n[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")

        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.explored_entities = checkpoint["progress"]["explored_entity_list"]
            self.all_bridges = checkpoint["bridges"]
            self.errors = checkpoint.get("errors", [])

            console.print(f"[green]✓ Checkpoint loaded[/green]")
            console.print(f"  Entities explored: {len(self.explored_entities)}")
            console.print(f"  Bridges created: {len(self.all_bridges)}")
            self.logger.info(f"Resumed from checkpoint: {len(self.explored_entities)} entities explored")

            return checkpoint

        except Exception as e:
            console.print(f"[red]✗ Failed to load checkpoint: {e}[/red]")
            self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            raise

    def explore_entities(self):
        """Main exploration loop."""
        # Check if seed entities are provided — if not, let agent discover autonomously
        if not getattr(self.args, 'seed_entities_file', None):
            return self._explore_corpus_autonomous()

        # Seed file provided — use entity-by-entity exploration
        tools = GSWTools(self.entity_searcher)

        with open(self.args.seed_entities_file) as f:
            seed_names = json.load(f)
        entities_to_explore = [{"name": name} for name in seed_names]
        console.print(f"[cyan]Loaded {len(entities_to_explore)} seed entities from {self.args.seed_entities_file}[/cyan]")

        # Filter out already explored entities
        entities_to_explore = [
            e for e in entities_to_explore
            if e["name"] not in self.explored_entities
        ]

        if not entities_to_explore:
            console.print("[yellow]No more entities to explore[/yellow]")
            return

        console.print(f"\n[bold cyan]Starting exploration of {len(entities_to_explore)} entities[/bold cyan]")

        # Suppress curator progress bars if in verbose mode
        if self.args.verbose:
            os.environ["CURATOR_DISABLE_PROGRESS"] = "1"

        # Choose display mode based on --verbose flag
        if self.args.verbose:
            # INTERACTIVE MODE: Show detailed output with callbacks
            display = InteractiveDisplay(console, show_thinking=self.args.show_thinking)

            # Reinitialize agent with callback
            self.agent = self.initialize_agent(display)

            for i, entity_info in enumerate(entities_to_explore, 1):
                if self.interrupted:
                    break

                entity_name = entity_info["name"]

                console.print(f"\n[bold magenta]═══ Entity {i}/{len(entities_to_explore)}: {entity_name} ═══[/bold magenta]")

                self.logger.info(f"Starting exploration of entity {i}/{len(entities_to_explore)}: {entity_name}")

                try:
                    # Explore with interactive display (callbacks handle output)
                    result = self.agent.explore_entity(entity_name, max_iterations=self.args.max_iterations)

                    # Track results
                    self.explored_entities.append(entity_name)
                    new_bridges = self.agent.get_all_bridges()
                    bridges_before = len(self.all_bridges)
                    self.all_bridges = new_bridges
                    bridges_created = len(new_bridges) - bridges_before

                    self.logger.info(
                        f"Completed {entity_name}: {result['iterations']} iterations, "
                        f"{result['tool_calls']} tool calls, {bridges_created} bridges"
                    )

                    # Save checkpoint
                    self._save_checkpoint(entity_count=len(self.explored_entities))

                    # Show progress summary
                    display.show_progress_summary(
                        i, len(entities_to_explore),
                        len(self.all_bridges), self.agent.tokens_used,
                        self.agent.input_tokens, self.agent.output_tokens
                    )

                except Exception as e:
                    self.logger.error(f"Failed to explore {entity_name}: {e}", exc_info=True)
                    self.error_logger.error(f"Entity: {entity_name} | Error: {e}\n{traceback.format_exc()}")
                    self.errors.append({
                        "entity": entity_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

        else:
            # PROGRESS BAR MODE: Existing code
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:

                main_task = progress.add_task(
                    "[cyan]Exploring entities...",
                    total=len(entities_to_explore)
                )

                for i, entity_info in enumerate(entities_to_explore, 1):
                    if self.interrupted:
                        break

                    entity_name = entity_info["name"]
                    progress.update(main_task, description=f"[cyan]Exploring: {entity_name}")

                    self.logger.info(f"Starting exploration of entity {i}/{len(entities_to_explore)}: {entity_name}")

                    try:
                        # Explore entity with timeout
                        result = self._explore_entity_with_timeout(entity_name)

                        # Track results
                        self.explored_entities.append(entity_name)
                        new_bridges = self.agent.get_all_bridges()
                        bridges_before = len(self.all_bridges)
                        self.all_bridges = new_bridges
                        bridges_created = len(new_bridges) - bridges_before

                        self.logger.info(
                            f"Completed {entity_name}: "
                            f"{result['iterations']} iterations, "
                            f"{result['tool_calls']} tool calls, "
                            f"{bridges_created} bridges"
                        )

                        # Save checkpoint
                        self._save_checkpoint(entity_count=len(self.explored_entities))

                        # Update progress
                        progress.update(
                            main_task,
                            advance=1,
                            description=f"[cyan]{len(self.explored_entities)}/{len(entities_to_explore)} entities | "
                                        f"{len(self.all_bridges)} bridges | "
                                        f"{self.agent.input_tokens:,} in / {self.agent.output_tokens:,} out"
                        )

                    except Exception as e:
                        self.logger.error(f"Failed to explore {entity_name}: {e}", exc_info=True)
                        self.error_logger.error(f"Entity: {entity_name} | Error: {e}\n{traceback.format_exc()}")
                        self.errors.append({
                            "entity": entity_name,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })

                        # Continue to next entity
                        progress.update(main_task, advance=1)
                        continue

        console.print(f"\n[green]✓ Exploration complete[/green]")
        self.logger.info(f"Exploration complete: {len(self.explored_entities)} entities explored")

    def _explore_corpus_autonomous(self):
        """Explore documents one by one, feeding each doc_id to the agent."""
        if self.args.pipeline_mode in {"rlm", "hybrid"}:
            mode_label = "deterministic RLM scheduler" if self.args.pipeline_mode == "rlm" else f"hybrid scheduler ({self.args.hybrid_scope})"
            console.print(f"\n[bold cyan]Exploring corpus with {mode_label}[/bold cyan]")
            if self.args.verbose:
                display = InteractiveDisplay(console, show_thinking=self.args.show_thinking)
                self.agent = self.initialize_agent(display)
            result = self.agent.explore_entity(doc_id=None, max_iterations=self.args.max_iterations)
            self.all_bridges = self.agent.get_all_bridges()
            self.explored_entities = list(self.agent.tools.explored_documents)
            self.logger.info(
                "Corpus exploration complete: docs=%s bridges=%s mode=%s scope=%s",
                result.get("documents_explored", len(self.explored_entities)),
                len(self.all_bridges),
                result.get("mode"),
                result.get("hybrid_scope", self.args.hybrid_scope),
            )
            self._save_checkpoint(entity_count=len(self.explored_entities))
            console.print(f"\n[green]✓ Exploration complete[/green]")
            return

        doc_ids = sorted(self.entity_searcher.gsw_by_doc_id.keys())
        console.print(f"\n[bold cyan]Exploring {len(doc_ids)} documents one by one[/bold cyan]")

        # Suppress curator progress bars in verbose mode
        if self.args.verbose:
            os.environ["CURATOR_DISABLE_PROGRESS"] = "1"

        if self.args.verbose:
            display = InteractiveDisplay(console, show_thinking=self.args.show_thinking)
            self.agent = self.initialize_agent(display)

        for i, doc_id in enumerate(doc_ids, 1):
            if self.interrupted:
                break

            console.print(f"\n[bold magenta]═══ Document {i}/{len(doc_ids)}: {doc_id} ═══[/bold magenta]")
            self.logger.info(f"Starting exploration of document {i}/{len(doc_ids)}: {doc_id}")

            try:
                result = self.agent.explore_entity(doc_id=doc_id, max_iterations=self.args.max_iterations)

                self.all_bridges = self.agent.get_all_bridges()
                self.explored_entities = list(self.agent.tools.explored_documents)

                self.logger.info(
                    f"Completed {doc_id}: {result['iterations']} iterations, "
                    f"{result['tool_calls']} tool calls, {len(self.all_bridges)} total bridges"
                )

                self._save_checkpoint(entity_count=len(self.explored_entities))

            except Exception as e:
                self.logger.error(f"Failed to explore document {doc_id}: {e}", exc_info=True)
                self.error_logger.error(f"Document: {doc_id} | Error: {e}\n{traceback.format_exc()}")
                self.errors.append({
                    "entity": doc_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                continue

        console.print(f"\n[green]✓ Exploration complete[/green]")
        self.logger.info(f"Exploration complete: {len(self.all_bridges)} bridges created across {len(doc_ids)} documents")

    def _explore_entity_with_timeout(self, entity_name: str, timeout: int = 300) -> Dict[str, Any]:
        """
        Explore entity with timeout protection.

        Args:
            entity_name: Entity to explore
            timeout: Max time in seconds (default: 5 minutes)

        Returns:
            Exploration result dict
        """
        # Note: Simple implementation - for production, use multiprocessing with timeout
        result = self.agent.explore_entity(entity_name, max_iterations=self.args.max_iterations)
        return result

    def generate_report(self) -> Dict[str, Any]:
        """Generate final report with statistics."""
        duration = time.time() - self.start_time

        # Calculate statistics
        valid_bridges = [b for b in self.all_bridges if b.get("validated", True)]
        invalid_bridges = [b for b in self.all_bridges if not b.get("validated", True)]

        confidences = [b.get("confidence", 0) for b in self.all_bridges if "confidence" in b]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        hop_counts = {}
        for bridge in self.all_bridges:
            hop = bridge.get("hop_count", len(bridge.get("source_docs", [])))
            hop_counts[hop] = hop_counts.get(hop, 0) + 1

        report = {
            "metadata": {
                "run_id": self.run_id,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": duration,
                "model": self.args.model,
                "num_docs_loaded": len(self.entity_searcher.gsw_by_doc_id),
                "config": vars(self.args)
            },
            "summary": {
                "entities_explored": len(self.explored_entities),
                "total_bridges": len(self.all_bridges),
                "valid_bridges": len(valid_bridges),
                "invalid_bridges": len(invalid_bridges),
                "tokens_used": self.agent.tokens_used if self.agent else 0,
                "input_tokens": self.agent.input_tokens if self.agent else 0,
                "output_tokens": self.agent.output_tokens if self.agent else 0,
                "rlm_metrics": self.agent.rlm_metrics if self.agent else {},
                "avg_confidence": round(avg_confidence, 3),
                "avg_bridges_per_entity": round(len(self.all_bridges) / len(self.explored_entities), 2) if self.explored_entities else 0,
                "hop_distribution": hop_counts,
                "total_errors": len(self.errors)
            },
            "bridges": self.all_bridges,
            "explored_entities": self.explored_entities,
            "errors": self.errors
        }

        return report

    def save_results(self, report: Dict[str, Any]):
        """Save results in multiple formats."""
        # Main JSON output
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        console.print(f"\n[green]✓ Results saved to: {results_file}[/green]")
        self.logger.info(f"Results saved to: {results_file}")

        # Bridges CSV (for easy analysis)
        bridges_csv = self.output_dir / "bridges.csv"
        if self.all_bridges:
            import csv
            with open(bridges_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "bridge_id", "question", "answers", "reverse_question", "reverse_answers",
                    "source_docs", "confidence", "hop_count", "entities_involved"
                ])
                writer.writeheader()
                for bridge in self.all_bridges:
                    writer.writerow({
                        "bridge_id": bridge.get("bridge_id", ""),
                        "question": bridge.get("question", ""),
                        "answers": ";".join(bridge.get("answers", [])),
                        "reverse_question": bridge.get("reverse_question", ""),
                        "reverse_answers": ";".join(bridge.get("reverse_answers", [])),
                        "source_docs": ";".join(bridge.get("source_docs", [])),
                        "confidence": bridge.get("confidence", 0),
                        "hop_count": bridge.get("hop_count", 0),
                        "entities_involved": ";".join(bridge.get("entities_involved", []))
                    })
            console.print(f"  → Bridges CSV: {bridges_csv}")

        # Summary text file
        summary_file = self.output_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Sleep-Time Exploration Summary\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Run ID: {report['metadata']['run_id']}\n")
            f.write(f"Duration: {report['metadata']['duration_seconds']:.1f}s\n")
            f.write(f"Model: {report['metadata']['model']}\n\n")
            f.write(f"Entities Explored: {report['summary']['entities_explored']}\n")
            f.write(f"Total Bridges: {report['summary']['total_bridges']}\n")
            f.write(f"  Valid: {report['summary']['valid_bridges']}\n")
            f.write(f"  Invalid: {report['summary']['invalid_bridges']}\n")
            f.write(f"Avg Confidence: {report['summary']['avg_confidence']:.3f}\n")
            f.write(f"Input Tokens: {report['summary']['input_tokens']:,}\n")
            f.write(f"Output Tokens: {report['summary']['output_tokens']:,}\n")
            f.write(f"Errors: {report['summary']['total_errors']}\n\n")
            rlm_metrics = report['summary'].get('rlm_metrics', {})
            if rlm_metrics:
                f.write("RLM Metrics:\n")
                for key in [
                    "root_calls",
                    "worker_calls",
                    "verifier_calls",
                    "edges_explored",
                    "edges_with_bridges",
                    "recursive_invocations",
                    "edges_skipped_no_path",
                    "planner_actions_overridden",
                    "calls_blocked_no_progress",
                    "alias_resolution_hits",
                    "docs_attempted",
                    "docs_completed",
                ]:
                    if key in rlm_metrics:
                        f.write(f"  {key}: {rlm_metrics[key]}\n")
                f.write("\n")
            f.write(f"Hop Distribution:\n")
            for hop, count in sorted(report['summary']['hop_distribution'].items()):
                f.write(f"  {hop}-hop: {count}\n")

        console.print(f"  → Summary: {summary_file}")

    def print_summary(self, report: Dict[str, Any]):
        """Print summary to console."""
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            f"[bold cyan]Sleep-Time Exploration Complete[/bold cyan]\n\n"
            f"Run ID: {report['metadata']['run_id']}\n"
            f"Duration: {report['metadata']['duration_seconds']:.1f}s\n"
            f"Model: {report['metadata']['model']}",
            border_style="cyan",
            box=box.DOUBLE
        ))

        # Statistics table
        table = Table(title="Exploration Statistics", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        table.add_row("Entities Explored", str(report['summary']['entities_explored']))
        table.add_row("Total Bridges", str(report['summary']['total_bridges']))
        table.add_row("  ✓ Valid", str(report['summary']['valid_bridges']), style="green")
        table.add_row("  ✗ Invalid", str(report['summary']['invalid_bridges']), style="red")
        table.add_row("Avg Confidence", f"{report['summary']['avg_confidence']:.3f}")
        table.add_row("Avg Bridges/Entity", f"{report['summary']['avg_bridges_per_entity']:.2f}")
        table.add_row("Input Tokens", f"{report['summary']['input_tokens']:,}")
        table.add_row("Output Tokens", f"{report['summary']['output_tokens']:,}")
        rlm_metrics = report['summary'].get('rlm_metrics', {})
        if rlm_metrics:
            table.add_row("RLM Edge Count", str(rlm_metrics.get("edges_explored", 0)))
            table.add_row("RLM Recursions", str(rlm_metrics.get("recursive_invocations", 0)))
            table.add_row("RLM Worker Calls", str(rlm_metrics.get("worker_calls", 0)))
            table.add_row("RLM No-Path Skips", str(rlm_metrics.get("edges_skipped_no_path", 0)))
            table.add_row("RLM Action Overrides", str(rlm_metrics.get("planner_actions_overridden", 0)))
        table.add_row("Errors", str(report['summary']['total_errors']), style="yellow" if report['summary']['total_errors'] > 0 else "green")

        console.print(table)

        # Hop distribution
        if report['summary']['hop_distribution']:
            console.print("\n[bold]Hop Distribution:[/bold]")
            for hop, count in sorted(report['summary']['hop_distribution'].items()):
                console.print(f"  {hop}-hop: {count} bridges")

    def run(self):
        """Main execution flow."""
        try:
            # Load GSWs or checkpoint
            if self.args.resume_from:
                checkpoint = self.load_checkpoint(self.args.resume_from)
                # Load config from checkpoint
                self.entity_searcher = self.load_gsws()
                self.agent = self.initialize_agent()
                # Restore token counts
                self.agent.tokens_used = checkpoint["progress"]["tokens_used"]
                self.agent.input_tokens = checkpoint["progress"].get("input_tokens", 0)
                self.agent.output_tokens = checkpoint["progress"].get("output_tokens", 0)
                saved_rlm = checkpoint["progress"].get("rlm_metrics")
                if isinstance(saved_rlm, dict):
                    self.agent.rlm_metrics.update(saved_rlm)
            else:
                self.entity_searcher = self.load_gsws()
                self.agent = self.initialize_agent()

            # Run exploration
            self.explore_entities()

            # Generate and save report
            report = self.generate_report()
            self.save_results(report)
            self.print_summary(report)

        except KeyboardInterrupt:
            console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
            self.logger.warning("Interrupted by user")
        except Exception as e:
            console.print(f"\n[red]✗ Fatal error: {e}[/red]")
            self.logger.error(f"Fatal error: {e}", exc_info=True)
            self.error_logger.error(f"Fatal error: {e}\n{traceback.format_exc()}")
            raise
        finally:
            # Final checkpoint on exit
            if self.agent and self.explored_entities:
                self._save_checkpoint(force=True)


def main():
    """Parse arguments and run exploration."""
    parser = argparse.ArgumentParser(
        description="Agentic Sleep-Time GSW Exploration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data configuration
    parser.add_argument("--num_docs", type=int, default=20,
                        help="Number of GSW documents to load")
    parser.add_argument("--gsw_path", type=str,
                        default="/mnt/SSD1/shreyas/SM_GSW/2wiki/networks",
                        help="Path to GSW files directory")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/SSD1/shreyas/SM_GSW/2wiki/.gsw_cache",
                        help="Path to cache directory")

    # Exploration configuration
    parser.add_argument("--num_entities", type=int, default=2,
                        help="Number of entities to explore (legacy mode). In RLM mode, corpus breadth is controlled by --num_docs.")
    parser.add_argument("--min_docs", type=int, default=2,
                        help="Minimum documents an entity must appear in")
    parser.add_argument("--filter_generic", action="store_true",
                        help="Filter out generic entities (nationalities, years)")
    parser.add_argument("--seed_entities_file", type=str, default=None,
                        help="JSON file with seed entity names (list of strings). Overrides browse_entities.")

    # Model configuration
    parser.add_argument("--model", type=str, default="Qwen3-30B-A3B-Thinking-2507",
                        help="Model name (OpenAI: gpt-4o, gpt-4o-mini | Together AI: Qwen/Qwen3-235B-A22B-Thinking-2507, openai/gpt-oss-120b, meta-llama/... | vllm: any name when --base_url is set)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Base URL for OpenAI-compatible API server (e.g. http://127.0.0.1:6379/v1 for a local vllm instance). When set, the OpenAI client is used regardless of model name.")
    parser.add_argument("--max_tokens", type=int, default=500_000,
                        help="Maximum token budget")
    parser.add_argument("--max_iterations", type=int, default=30,
                        help="Maximum tool call iterations per entity")
    parser.add_argument("--reasoning_effort", type=str, default="medium",
                        choices=["low", "medium", "high"],
                        help="Reasoning effort for Together AI models (low/medium/high) - higher = better reasoning but slower. Default: medium for balance.")
    parser.add_argument("--disable_bridge_verifier", action="store_true",
                        help="Disable bridge verifier subagent gate for create_bridge_qa.")
    parser.add_argument("--bridge_verifier_threshold", type=float, default=0.75,
                        help="Minimum verifier score (0-1) needed to keep a created bridge.")
    parser.add_argument("--bridge_verifier_fail_open", action="store_true",
                        help="If set, bridge verifier API/parse errors won't block bridge creation.")
    parser.add_argument("--bridge_verifier_model", type=str, default=None,
                        help="Optional model for verifier subagent. Defaults to --model.")
    parser.add_argument("--pipeline_mode", type=str, default="legacy", choices=["legacy", "rlm", "hybrid"],
                        help="Exploration pipeline mode: legacy, deterministic RLM scheduler, or hybrid self-controller.")
    parser.add_argument("--hybrid_scope", type=str, default="doc_edge",
                        choices=["edge", "doc_edge", "corpus_doc_edge"],
                        help="Hybrid autonomy scope: edge-only, doc+edge, or corpus+doc+edge.")
    parser.add_argument("--root_model", type=str, default=None,
                        help="Optional root-stage model for RLM mode. Defaults to --model.")
    parser.add_argument("--worker_model", type=str, default=None,
                        help="Optional worker-stage model for RLM mode. Defaults to --model.")
    parser.add_argument("--edge_max_depth", type=int, default=1,
                        help="Maximum recursion depth per edge in RLM mode.")
    parser.add_argument("--edge_max_calls", type=int, default=2,
                        help="Maximum worker invocations per edge in RLM mode.")
    parser.add_argument("--edge_max_tokens", type=int, default=3000,
                        help="Per-edge token budget in RLM mode.")
    parser.add_argument("--max_optional_docs_per_edge", type=int, default=2,
                        help="Maximum optional fuzzy docs included per edge packet in RLM mode.")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="logs/sleep_time",
                        help="Output directory for results and logs")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Save checkpoint every N entities")

    # Interactive display options
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed tool calls and results in real-time")
    parser.add_argument("--show-thinking", action="store_true",
                        help="Also show agent's reasoning/analysis text (requires --verbose)")

    # Resume capability
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint file")

    args = parser.parse_args()

    # Print configuration
    scope_line = (
        f"Documents: {args.num_docs}\n"
        if args.pipeline_mode in {"rlm", "hybrid"}
        else f"Documents: {args.num_docs}\nEntities: {args.num_entities}\n"
    )

    console.print(Panel.fit(
        f"[bold cyan]Agentic Sleep-Time Exploration[/bold cyan]\n\n"
        f"{scope_line}"
        f"Model: {args.model}\n"
        f"Pipeline: {args.pipeline_mode}\n"
        f"Hybrid scope: {args.hybrid_scope}\n"
        f"Bridge verifier: {'off' if args.disable_bridge_verifier else 'on'} "
        f"(threshold={args.bridge_verifier_threshold:.2f})\n"
        f"Output: {args.output_dir}",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # Run exploration
    runner = SleepTimeRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
