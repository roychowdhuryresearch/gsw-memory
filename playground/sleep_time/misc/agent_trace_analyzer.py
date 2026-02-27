"""
Agent Trace Analyzer - Interactive Streamlit Dashboard

Visualize and analyze agent traces from sleep-time exploration runs.
Helps debug agent behavior, track bridge creation, and compare runs.

Usage:
    streamlit run playground/agent_trace_analyzer.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Agent Trace Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
LOGS_DIR = Path("logs/sleep_time")
REPETITION_PATTERNS = [
    "Wait,", "Actually,", "Let me think", "But wait",
    "Okay,", "So maybe", "However,", "Although,"
]

# ============================================================================
# PARSER FUNCTIONS
# ============================================================================

def find_available_runs() -> List[str]:
    """Find all available run directories."""
    if not LOGS_DIR.exists():
        return []

    runs = []
    for run_dir in sorted(LOGS_DIR.iterdir(), reverse=True):
        if run_dir.is_dir() and run_dir.name.startswith("run_"):
            trace_file = run_dir / "logs" / "agent_trace.log"
            if trace_file.exists():
                runs.append(run_dir.name)
    return runs


def parse_results_json(run_dir: Path) -> Optional[Dict]:
    """Load and parse results.json."""
    results_file = run_dir / "results.json"
    if not results_file.exists():
        return None

    with open(results_file, 'r') as f:
        return json.load(f)


def parse_agent_trace(trace_file: Path) -> List[Dict]:
    """
    Parse agent_trace.log into structured iterations.

    Returns:
        List of iteration dicts with entity, iteration_num, reasoning, tool_calls, bridges
    """
    try:
        with open(trace_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        st.error(f"Failed to read trace file: {e}")
        return []

    if not content.strip():
        st.warning("Trace file is empty")
        return []

    iterations = []

    # Split by iteration headers
    iteration_pattern = r'={80}\nEntity: (.*?) \| Iteration: (\d+)/(\d+)\n={80}'
    splits = re.split(iteration_pattern, content)

    # Check if any iterations were found
    if len(splits) <= 1:
        st.warning(f"""
        ⚠️ No iteration headers found in trace file.

        File size: {len(content)} characters
        Expected pattern: `Entity: <name> | Iteration: <num>/<max>`

        First 200 characters:
        ```
        {content[:200]}
        ```
        """)
        return []

    # Process each iteration
    for i in range(1, len(splits), 4):
        if i+3 > len(splits):
            break

        entity = splits[i].strip()
        iteration_num = int(splits[i+1])
        max_iterations = int(splits[i+2])
        iteration_content = splits[i+3]

        # Extract reasoning
        reasoning_match = re.search(r'💭 Agent Reasoning:(.*?)(?:----------------------------------------|\n\n={80}|$)',
                                   iteration_content, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        # Extract tool calls
        tool_calls = []
        tool_pattern = r'Tool: (.*?)\nArgs: (.*?)\nResult: (.*?)(?:----------------------------------------|\n\n|$)'
        for match in re.finditer(tool_pattern, iteration_content, re.DOTALL):
            tool_name = match.group(1).strip()
            args_str = match.group(2).strip()
            result_str = match.group(3).strip()

            # Try to parse JSON
            try:
                args = json.loads(args_str) if args_str else {}
            except:
                args = args_str

            try:
                # Handle both single results and list results
                result = json.loads(result_str) if result_str else {}
            except:
                result = result_str

            tool_calls.append({
                'name': tool_name,
                'args': args,
                'result': result
            })

        # Extract bridge creations
        bridges = []
        bridge_pattern = r'\*+ ✓ BRIDGE CREATED: (.*?)\n  Q: (.*?)\n  A: (.*?)\n\*+'
        for match in re.finditer(bridge_pattern, iteration_content):
            bridge_id = match.group(1).strip()
            question = match.group(2).strip()
            answer = match.group(3).strip()
            bridges.append({
                'id': bridge_id,
                'question': question,
                'answer': answer
            })

        iterations.append({
            'entity': entity,
            'iteration_num': iteration_num,
            'max_iterations': max_iterations,
            'reasoning': reasoning,
            'reasoning_lines': len(reasoning.split('\n')) if reasoning else 0,
            'tool_calls': tool_calls,
            'bridges': bridges,
            'num_tools': len(tool_calls),
            'num_bridges': len(bridges)
        })

    return iterations


def detect_batch_calls(iterations: List[Dict]) -> Dict:
    """Analyze batch mode usage in create_bridge_qa calls."""
    total_bridge_calls = 0
    batch_calls = 0
    total_bridges_created = 0
    bridges_per_call = []

    for iteration in iterations:
        for tool_call in iteration['tool_calls']:
            if tool_call['name'] == 'create_bridge_qa':
                total_bridge_calls += 1

                # Check if result is a list (batch mode) or dict (single mode)
                result = tool_call['result']
                if isinstance(result, list):
                    # Batch mode
                    batch_calls += 1
                    num_bridges = len(result)
                    bridges_per_call.append(num_bridges)
                    total_bridges_created += num_bridges
                elif isinstance(result, dict):
                    # Single mode
                    bridges_per_call.append(1)
                    total_bridges_created += 1

    batch_adoption = (batch_calls / total_bridge_calls * 100) if total_bridge_calls > 0 else 0

    return {
        'total_calls': total_bridge_calls,
        'batch_calls': batch_calls,
        'single_calls': total_bridge_calls - batch_calls,
        'batch_adoption_pct': batch_adoption,
        'total_bridges': total_bridges_created,
        'bridges_per_call': bridges_per_call,
        'avg_bridges_per_call': sum(bridges_per_call) / len(bridges_per_call) if bridges_per_call else 0
    }


def calculate_metrics(iterations: List[Dict], results: Optional[Dict]) -> Dict:
    """Calculate comprehensive metrics."""
    total_tool_calls = sum(it['num_tools'] for it in iterations)
    total_bridges = sum(it['num_bridges'] for it in iterations)

    reasoning_lengths = [it['reasoning_lines'] for it in iterations if it['reasoning_lines'] > 0]
    avg_reasoning_lines = sum(reasoning_lengths) / len(reasoning_lengths) if reasoning_lengths else 0

    # Tool usage
    tool_counts = Counter()
    for iteration in iterations:
        for tool_call in iteration['tool_calls']:
            tool_counts[tool_call['name']] += 1

    # Bridge efficiency
    bridge_efficiency = (total_bridges / total_tool_calls * 100) if total_tool_calls > 0 else 0

    # Batch mode analysis
    batch_stats = detect_batch_calls(iterations)

    metrics = {
        'total_iterations': len(iterations),
        'total_tool_calls': total_tool_calls,
        'total_bridges': total_bridges,
        'bridge_efficiency': bridge_efficiency,
        'avg_reasoning_lines': avg_reasoning_lines,
        'max_reasoning_lines': max(reasoning_lengths) if reasoning_lengths else 0,
        'tool_counts': tool_counts,
        'batch_stats': batch_stats,
        'entities': list(set(it['entity'] for it in iterations))
    }

    # Add results.json data if available
    if results:
        metrics.update({
            'run_id': results['metadata'].get('run_id'),
            'model': results['metadata'].get('model'),
            'duration_minutes': results['metadata'].get('duration_seconds', 0) / 60,
            'tokens_used': results['summary'].get('tokens_used', 0),
            'avg_confidence': results['summary'].get('avg_confidence', 0)
        })

    return metrics


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_bridge_efficiency_timeline(iterations: List[Dict]) -> go.Figure:
    """Plot cumulative bridges vs tool calls over time."""
    cumulative_bridges = []
    cumulative_tools = []
    iteration_nums = []

    bridges = 0
    tools = 0

    for it in iterations:
        bridges += it['num_bridges']
        tools += it['num_tools']
        cumulative_bridges.append(bridges)
        cumulative_tools.append(tools)
        iteration_nums.append(it['iteration_num'])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iteration_nums,
        y=cumulative_bridges,
        mode='lines+markers',
        name='Cumulative Bridges',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=iteration_nums,
        y=cumulative_tools,
        mode='lines+markers',
        name='Cumulative Tool Calls',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Bridge Creation Timeline',
        xaxis_title='Iteration',
        yaxis_title='Count',
        hovermode='x unified'
    )

    return fig


def plot_tool_usage_chart(tool_counts: Counter) -> go.Figure:
    """Bar chart of tool call frequencies."""
    tools = list(tool_counts.keys())
    counts = list(tool_counts.values())

    fig = px.bar(
        x=tools,
        y=counts,
        labels={'x': 'Tool Name', 'y': 'Number of Calls'},
        title='Tool Usage Distribution'
    )

    fig.update_layout(xaxis_tickangle=-45)

    return fig


def plot_reasoning_length_distribution(iterations: List[Dict]) -> go.Figure:
    """Histogram of reasoning block lengths."""
    lengths = [it['reasoning_lines'] for it in iterations if it['reasoning_lines'] > 0]

    fig = px.histogram(
        x=lengths,
        nbins=20,
        labels={'x': 'Reasoning Length (lines)', 'y': 'Frequency'},
        title='Reasoning Length Distribution'
    )

    # Add vertical line at 50 (threshold for "long reasoning")
    fig.add_vline(x=50, line_dash="dash", line_color="red",
                  annotation_text="Long Reasoning Threshold")

    return fig


def plot_batch_mode_adoption(batch_stats: Dict) -> go.Figure:
    """Pie chart showing batch vs single bridge creation."""
    labels = ['Batch Mode', 'Single Mode']
    values = [batch_stats['batch_calls'], batch_stats['single_calls']]
    colors = ['#2ecc71', '#3498db']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors),
        textinfo='label+percent+value',
        hole=0.3
    )])

    fig.update_layout(
        title=f'Bridge Creation Mode Distribution<br><sub>Batch Adoption: {batch_stats["batch_adoption_pct"]:.1f}%</sub>'
    )

    return fig


# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_metrics_cards(metrics: Dict):
    """Display key metrics in card layout."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Bridges",
            metrics['total_bridges'],
            delta=f"{metrics['bridge_efficiency']:.1f}% efficiency"
        )

    with col2:
        st.metric(
            "Batch Adoption",
            f"{metrics['batch_stats']['batch_adoption_pct']:.1f}%",
            delta=f"{metrics['batch_stats']['batch_calls']} batch calls"
        )

    with col3:
        st.metric(
            "Avg Reasoning",
            f"{metrics['avg_reasoning_lines']:.0f} lines",
            delta=f"Max: {metrics['max_reasoning_lines']}"
        )

    with col4:
        st.metric(
            "Tool Calls",
            metrics['total_tool_calls'],
            delta=f"{metrics['total_iterations']} iterations"
        )


def render_iteration(iteration: Dict, show_full_reasoning: bool = False):
    """Render a single iteration with collapsible sections."""
    # Add prominent bridge indicator at start of title
    if iteration['num_bridges'] > 0:
        iteration_title = f"🌉 [{iteration['num_bridges']} BRIDGE{'S' if iteration['num_bridges'] > 1 else ''}] Iteration {iteration['iteration_num']}: {iteration['entity']}"
    else:
        iteration_title = f"Iteration {iteration['iteration_num']}: {iteration['entity']}"

    # Add warning emoji for long reasoning
    if iteration['reasoning_lines'] > 50:
        iteration_title += " ⚠️"

    with st.expander(iteration_title):
        # Detect batch mode
        is_batch_mode = False
        for tool_call in iteration.get('tool_calls', []):
            if tool_call['name'] == 'create_bridge_qa':
                result = tool_call.get('result', {})
                if isinstance(result, list):
                    is_batch_mode = True
                    break

        # Metadata row
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.caption(f"🔧 Tools: {iteration['num_tools']}")
        with meta_col2:
            st.caption(f"💭 Reasoning: {iteration['reasoning_lines']} lines")
        with meta_col3:
            if iteration['num_bridges'] > 0:
                batch_indicator = " 🔄 Batch" if is_batch_mode else ""
                st.success(f"🌉 Bridges: **{iteration['num_bridges']}**{batch_indicator}")
            else:
                st.caption(f"🌉 Bridges: {iteration['num_bridges']}")

        st.divider()

        # Reasoning block
        if iteration['reasoning']:
            st.markdown("**💭 Agent Reasoning:**")

            if iteration['reasoning_lines'] > 50:
                st.warning(f"⚠️ Long reasoning detected ({iteration['reasoning_lines']} lines)")

            if show_full_reasoning or iteration['reasoning_lines'] <= 30:
                st.code(iteration['reasoning'], language='text')
            else:
                # Show first 20 lines with expand option
                lines = iteration['reasoning'].split('\n')
                preview = '\n'.join(lines[:20])

                with st.container():
                    st.code(preview + '\n... [truncated]', language='text')
                    if st.button(f"Show full reasoning", key=f"reasoning_{iteration['entity']}_{iteration['iteration_num']}"):
                        st.code(iteration['reasoning'], language='text')

        st.divider()

        # Tool calls
        if iteration['tool_calls']:
            st.markdown("**🔧 Tool Calls:**")

            for i, tool_call in enumerate(iteration['tool_calls']):
                st.markdown(f"**{i+1}. {tool_call['name']}**")

                # Arguments
                with st.expander("Arguments", expanded=False):
                    st.json(tool_call['args'])

                # Result
                with st.expander("Result", expanded=False):
                    result = tool_call['result']

                    # Check if batch mode
                    if isinstance(result, list) and tool_call['name'] == 'create_bridge_qa':
                        st.info(f"🔄 Batch Mode: {len(result)} bridges created")

                    st.json(result)

        # Bridges created
        if iteration['bridges']:
            st.divider()
            st.markdown("**✅ Bridges Created:**")

            for bridge in iteration['bridges']:
                st.success(f"**{bridge['id']}**")
                st.markdown(f"- **Q:** {bridge['question']}")
                st.markdown(f"- **A:** {bridge['answer']}")


def render_bridge_table(results: Dict):
    """Render bridges as interactive table."""
    if not results or 'bridges' not in results:
        st.warning("No bridge data available")
        return

    bridges = results['bridges']

    # Create DataFrame
    df = pd.DataFrame([
        {
            'Question': b['question'],
            'Answer': b['answer'],
            'Source Docs': ', '.join(b['source_docs']),
            'Confidence': b.get('confidence', 0),
            'Hop Count': b.get('hop_count', len(b['source_docs']))
        }
        for b in bridges
    ])

    # Display with custom column config
    st.dataframe(
        df,
        column_config={
            "Confidence": st.column_config.ProgressColumn(
                "Confidence",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
        },
        use_container_width=True,
        height=400
    )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🔍 Agent Trace Analyzer")
    st.markdown("*Visualize and analyze agent traces from sleep-time exploration*")

    # Sidebar
    st.sidebar.header("Configuration")

    # Find available runs
    available_runs = find_available_runs()

    if not available_runs:
        st.error(f"No runs found in {LOGS_DIR}")
        st.info("Run `python playground/run_sleep_time.py` to generate traces")
        return

    # Run selection
    selected_run = st.sidebar.selectbox(
        "Select Run",
        available_runs,
        format_func=lambda x: x.replace("run_", "")
    )

    run_dir = LOGS_DIR / selected_run
    trace_file = run_dir / "logs" / "agent_trace.log"

    # Load data
    with st.spinner("Loading trace data..."):
        iterations = parse_agent_trace(trace_file)
        results = parse_results_json(run_dir)
        metrics = calculate_metrics(iterations, results)

    # Early validation - check if we have data to display
    if not iterations:
        st.error("❌ No iterations found in trace file")
        st.info("""
        **Possible reasons:**
        - Trace file is empty
        - Trace format doesn't match expected pattern
        - Run might have failed or been interrupted

        **Expected format:**
        ```
        ================================================================================
        Entity: <name> | Iteration: <num>/<max>
        ================================================================================

        💭 Agent Reasoning:
        <reasoning text>

        Tool: <tool_name>
        Args: {...}
        Result: {...}
        ```
        """)
        return

    # Filters
    st.sidebar.header("Filters")

    # Handle empty entities
    if metrics['entities']:
        entity_filter = st.sidebar.multiselect(
            "Entities",
            metrics['entities'],
            default=metrics['entities']
        )
    else:
        st.sidebar.warning("⚠️ No entities found")
        entity_filter = []

    # Safe slider ranges
    max_iteration = max(it['iteration_num'] for it in iterations)
    max_reasoning = max(metrics['max_reasoning_lines'], 1)  # Ensure at least 1

    iteration_range = st.sidebar.slider(
        "Iteration Range",
        1,
        max_iteration,
        (1, max_iteration)
    )

    min_reasoning_lines = st.sidebar.slider(
        "Min Reasoning Lines",
        0,
        max_reasoning,
        0
    )

    show_full_reasoning = st.sidebar.checkbox("Show Full Reasoning", value=False)

    # Filter iterations
    filtered_iterations = [
        it for it in iterations
        if it['entity'] in entity_filter
        and iteration_range[0] <= it['iteration_num'] <= iteration_range[1]
        and it['reasoning_lines'] >= min_reasoning_lines
    ]

    # Quick stats in sidebar
    st.sidebar.header("Quick Stats")
    st.sidebar.metric("Total Bridges", metrics['total_bridges'])
    st.sidebar.metric("Efficiency", f"{metrics['bridge_efficiency']:.1f}%")
    st.sidebar.metric("Batch Adoption", f"{metrics['batch_stats']['batch_adoption_pct']:.1f}%")

    if results:
        st.sidebar.metric("Duration", f"{metrics.get('duration_minutes', 0):.1f} min")
        st.sidebar.metric("Tokens Used", f"{metrics.get('tokens_used', 0):,}")

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview",
        "📜 Trace Viewer",
        "🌉 Bridges",
        "🔧 Tools",
        "💭 Reasoning",
        "🔄 Batch Mode"
    ])

    # ========== OVERVIEW TAB ==========
    with tab1:
        st.header("Overview")

        # Metrics cards
        render_metrics_cards(metrics)

        st.divider()

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(
                plot_bridge_efficiency_timeline(iterations),
                use_container_width=True,
                key="overview_bridge_timeline"
            )

        with col2:
            st.plotly_chart(
                plot_tool_usage_chart(metrics['tool_counts']),
                use_container_width=True,
                key="overview_tool_usage"
            )

        # Run metadata
        if results:
            st.header("Run Metadata")
            meta_col1, meta_col2, meta_col3 = st.columns(3)

            with meta_col1:
                model = results.get('metadata', {}).get('model', 'Unknown')
                st.metric("Model", model.split('/')[-1])
            with meta_col2:
                max_iter = results.get('config', {}).get('max_iterations', 'N/A')
                st.metric("Max Iterations", max_iter)
            with meta_col3:
                reasoning_effort = results.get('config', {}).get('reasoning_effort', 'N/A')
                st.metric("Reasoning Effort", reasoning_effort)

    # ========== TRACE VIEWER TAB ==========
    with tab2:
        st.header("Trace Viewer")

        # Filter controls
        col1, col2 = st.columns([3, 1])
        with col1:
            search_query = st.text_input("🔍 Search in reasoning", "")
        with col2:
            show_only_bridges = st.checkbox("🌉 Only bridge iterations", value=False)

        # Count bridge-creating iterations
        bridge_iterations = [it for it in filtered_iterations if it['num_bridges'] > 0]
        total_bridge_count = sum(it['num_bridges'] for it in bridge_iterations)

        # Info banner with bridge summary
        if bridge_iterations:
            bridge_iter_nums = ', '.join(str(it['iteration_num']) for it in bridge_iterations[:10])
            if len(bridge_iterations) > 10:
                bridge_iter_nums += f", ... (+{len(bridge_iterations) - 10} more)"

            st.success(f"🌉 **{len(bridge_iterations)} iterations created {total_bridge_count} bridges**: Iterations {bridge_iter_nums}")

        st.info(f"Showing {len(filtered_iterations)} iterations (filtered from {len(iterations)})")

        # Apply filters
        display_iterations = filtered_iterations
        if show_only_bridges:
            display_iterations = [it for it in display_iterations if it['num_bridges'] > 0]

        for iteration in display_iterations:
            # Search filter
            if search_query and search_query.lower() not in iteration['reasoning'].lower():
                continue

            render_iteration(iteration, show_full_reasoning)

    # ========== BRIDGES TAB ==========
    with tab3:
        st.header("Bridge Analysis")

        if results:
            render_bridge_table(results)

            st.divider()

            # Bridge stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Bridges", results['summary']['total_bridges'])
            with col2:
                st.metric("Avg Confidence", f"{results['summary']['avg_confidence']:.2f}")
            with col3:
                hop_dist = results['summary'].get('hop_distribution', {})
                st.metric("Hop Distribution", ', '.join(f"{k}-hop: {v}" for k, v in hop_dist.items()))
        else:
            st.warning("No results.json found")

    # ========== TOOLS TAB ==========
    with tab4:
        st.header("Tool Analytics")

        # Tool frequency
        st.plotly_chart(
            plot_tool_usage_chart(metrics['tool_counts']),
            use_container_width=True,
            key="tools_tool_usage"
        )

        # Tool details
        st.subheader("Tool Usage Details")
        tool_df = pd.DataFrame([
            {'Tool': tool, 'Calls': count}
            for tool, count in metrics['tool_counts'].most_common()
        ])
        st.dataframe(tool_df, use_container_width=True)

    # ========== REASONING TAB ==========
    with tab5:
        st.header("Reasoning Analysis")

        # Distribution
        st.plotly_chart(
            plot_reasoning_length_distribution(iterations),
            use_container_width=True,
            key="reasoning_length_dist"
        )

        # Pattern detection
        st.subheader("Repetitive Patterns")
        all_reasoning = ' '.join(it['reasoning'] for it in iterations)

        pattern_counts = {}
        for pattern in REPETITION_PATTERNS:
            count = all_reasoning.count(pattern)
            if count > 0:
                pattern_counts[pattern] = count

        if pattern_counts:
            pattern_df = pd.DataFrame([
                {'Pattern': pattern, 'Count': count}
                for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
            ])
            st.dataframe(pattern_df, use_container_width=True)

        # Longest reasoning
        st.subheader("Top 5 Longest Reasoning Blocks")
        longest = sorted(iterations, key=lambda x: x['reasoning_lines'], reverse=True)[:5]

        for i, it in enumerate(longest):
            with st.expander(f"#{i+1}: Iteration {it['iteration_num']} ({it['reasoning_lines']} lines)"):
                st.code(it['reasoning'], language='text')

    # ========== BATCH MODE TAB ==========
    with tab6:
        st.header("Batch Mode Analysis")

        batch_stats = metrics['batch_stats']

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Bridge Calls", batch_stats['total_calls'])
        with col2:
            st.metric("Batch Calls", batch_stats['batch_calls'],
                     delta=f"{batch_stats['batch_adoption_pct']:.1f}%")
        with col3:
            st.metric("Avg Bridges/Call", f"{batch_stats['avg_bridges_per_call']:.2f}")

        # Pie chart
        st.plotly_chart(
            plot_batch_mode_adoption(batch_stats),
            use_container_width=True,
            key="batch_adoption_pie"
        )

        # Batch size distribution
        if batch_stats['bridges_per_call']:
            st.subheader("Batch Size Distribution")
            batch_sizes = Counter(batch_stats['bridges_per_call'])

            fig = px.bar(
                x=list(batch_sizes.keys()),
                y=list(batch_sizes.values()),
                labels={'x': 'Bridges per Call', 'y': 'Frequency'},
                title='Batch Size Distribution'
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_size_dist")


if __name__ == "__main__":
    main()
