#!/usr/bin/env python3
"""
Streamlit app for inspecting GSW generation outputs and creating platinum datasets.

This app allows you to:
- Inspect raw text, thinking traces, and GSW outputs side-by-side
- Mark high-quality samples for platinum dataset creation
- Export curated samples for training/evaluation

Usage:
    streamlit run playground/gsw_creation_local/gsw_inspector_streamlit.py
"""

import json
import os
import time
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from openai import OpenAI


# ==============================================================================
# Configuration
# ==============================================================================

# Default file paths
DEFAULT_INPUT_FILE = "pred_gsws_train_thinking_traces.json"
STATE_FILE = ".gsw_inspector_state.json"

# OpenAI API Key
OPENAI_API_KEY = "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"

# Quality status options
STATUS_PLATINUM = "platinum"
STATUS_REVIEW = "review"
STATUS_REJECT = "reject"
STATUS_SKIP = "skip"

STATUS_OPTIONS = {
    STATUS_PLATINUM: {"label": "✅ Platinum", "color": "green"},
    STATUS_REVIEW: {"label": "⚠️ Review", "color": "orange"},
    STATUS_REJECT: {"label": "❌ Reject", "color": "red"},
    STATUS_SKIP: {"label": "⏸️ Skip", "color": "gray"}
}

# LLM-as-a-Judge Prompt for Quality Assessment
LLM_AS_A_JUDGE_PROMPT = """You are an expert evaluator of Generative Semantic Workspace (GSW) structures.

Your task is to assess the quality of a predicted GSW extraction from raw text.

Evaluation Criteria:

1. ENTITY COMPLETENESS (0-1 score)
   - Are all important entities from the text extracted?
   - Are entities atomic (not bundled)?
   - Are dates, locations, and other answer-bearing values captured as entities?
   - Missing critical entities significantly lowers this score

2. RELATIONSHIP ACCURACY (0-1 score)
   - Are verb phrases/relationships correctly identified?
   - Do questions properly capture the relationships stated in the text?
   - Are questions bidirectional (both A→B and B→A direction)?
   - Do the questions make semantic sense?

3. FORMAT COMPLIANCE (0-1 score)
   - No pronouns in questions? (e.g., avoid "Who did he marry?")
   - Answers are entity IDs only?
   - Exactly two questions per verb phrase?
   - Complete content captured (no missing "that" clauses, conditions, etc.)?
   - Entity IDs referenced in answers actually exist in entity_nodes?

4. HALLUCINATION CHECK (0-1 score, where 1 = no hallucinations)
   - Are all entities and relationships actually present in the input text?
   - No fabricated dates, names, relationships, or information?
   - Information correctly reflects what's stated in the source?

5. OVERALL QUALITY (0-100 score)
   - Weighted combination:
     * 30% entity completeness
     * 30% relationship accuracy
     * 20% format compliance
     * 20% no hallucinations

Provide your assessment in JSON format with this structure:
{
  "overall_score": <integer 0-100>,
  "entity_completeness": <float 0-1>,
  "relationship_accuracy": <float 0-1>,
  "format_compliance": <float 0-1>,
  "hallucination_score": <float 0-1>,
  "strengths": ["list of specific things done well"],
  "issues": ["list of specific problems or errors"],
  "missing_entities": ["entities from the text that should have been extracted"],
  "hallucinated_info": ["any fabricated information not in the source"],
  "recommendation": "<one of: platinum|review|reject>"
}

Recommendation guidelines:
- "platinum": overall_score >= 85, no critical issues
- "review": overall_score 60-84, has some issues but salvageable
- "reject": overall_score < 60, major problems or many hallucinations
"""


# ==============================================================================
# Data Loading
# ==============================================================================

@st.cache_data
def load_thinking_traces(file_path: str) -> Dict[str, Any]:
    """Load thinking traces from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def load_state(state_path: str) -> Dict[str, Any]:
    """Load saved review state if it exists."""
    if Path(state_path).exists():
        with open(state_path, 'r') as f:
            return json.load(f)
    return {}


def save_state(state_path: str, state_data: Dict[str, Any]):
    """Save review state to file."""
    with open(state_path, 'w') as f:
        json.dump(state_data, f, indent=2)


# ==============================================================================
# Session State Initialization
# ==============================================================================

def initialize_session_state(data: Dict[str, Any], state_path: str):
    """Initialize Streamlit session state."""
    if 'initialized' not in st.session_state:
        # Load saved state
        saved_state = load_state(state_path)

        # Initialize current index
        st.session_state.current_index = saved_state.get('current_index', 0)

        # Initialize review statuses (dict: index -> status)
        st.session_state.review_statuses = saved_state.get('review_statuses', {})

        # Initialize notes (dict: index -> note)
        st.session_state.notes = saved_state.get('notes', {})

        # Initialize AI assessments cache (dict: index -> assessment)
        st.session_state.ai_assessments = saved_state.get('ai_assessments', {})

        # Total number of samples
        st.session_state.total_samples = len(data['entries'])

        # Filter mode
        st.session_state.filter_mode = saved_state.get('filter_mode', 'all')

        # Mark as initialized
        st.session_state.initialized = True


def persist_state(state_path: str):
    """Save current session state to file."""
    state_data = {
        'current_index': st.session_state.current_index,
        'review_statuses': st.session_state.review_statuses,
        'notes': st.session_state.notes,
        'ai_assessments': st.session_state.ai_assessments,
        'filter_mode': st.session_state.filter_mode,
        'last_saved': datetime.now().isoformat()
    }
    save_state(state_path, state_data)


# ==============================================================================
# Navigation Functions
# ==============================================================================

def get_filtered_indices() -> List[int]:
    """Get list of indices based on current filter mode."""
    all_indices = list(range(st.session_state.total_samples))

    if st.session_state.filter_mode == 'all':
        return all_indices
    elif st.session_state.filter_mode == 'platinum':
        return [i for i in all_indices if st.session_state.review_statuses.get(str(i)) == STATUS_PLATINUM]
    elif st.session_state.filter_mode == 'unmarked':
        return [i for i in all_indices if str(i) not in st.session_state.review_statuses]
    elif st.session_state.filter_mode == 'marked':
        return [i for i in all_indices if str(i) in st.session_state.review_statuses]
    return all_indices


def navigate_to(index: int, state_path: str):
    """Navigate to a specific index."""
    filtered_indices = get_filtered_indices()
    if filtered_indices:
        # Ensure index is within bounds
        index = max(0, min(index, len(filtered_indices) - 1))
        st.session_state.current_index = filtered_indices[index]
        persist_state(state_path)


# ==============================================================================
# Pretty Print GSW Rendering
# ==============================================================================

def render_gsw_pretty(gsw: Dict[str, Any], token_usage: Dict[str, Any]):
    """Render GSW in human-readable pretty format."""
    entity_nodes = gsw.get('entity_nodes', [])
    verb_nodes = gsw.get('verb_phrase_nodes', [])

    # Calculate total questions
    total_questions = sum(len(v.get('questions', [])) for v in verb_nodes)

    # Summary stats
    st.markdown(f"""
**📊 GSW Summary**

`Entities: {len(entity_nodes)}  |  Relationships: {len(verb_nodes)}  |  Questions: {total_questions}`
""")

    st.markdown("---")

    # Entities Section
    st.markdown(f"#### 📌 ENTITIES ({len(entity_nodes)})")

    if entity_nodes:
        entity_text = ""
        for entity in entity_nodes:
            entity_id = entity.get('id', 'unknown')
            entity_name = entity.get('name', 'Unknown')

            # Get roles and states
            roles_info = []
            for role in entity.get('roles', []):
                role_name = role.get('role', 'unknown')
                states = role.get('states', [])
                roles_info.append((role_name, states))

            # Format entity
            entity_text += f"\n📌 **{entity_name}** `({entity_id})`\n"
            for role_name, states in roles_info:
                entity_text += f"   - **Role:** {role_name}\n"
                if states:
                    entity_text += f"   - **States:** {', '.join(states)}\n"
            entity_text += "\n"

        st.markdown(entity_text)
    else:
        st.info("No entities found")

    st.markdown("---")

    # Relationships Section
    st.markdown(f"#### 🔗 RELATIONSHIPS ({len(verb_nodes)})")

    if verb_nodes:
        for verb in verb_nodes:
            verb_id = verb.get('id', 'unknown')
            verb_phrase = verb.get('phrase', 'Unknown phrase')
            questions = verb.get('questions', [])

            # Create expander for each relationship
            with st.expander(f"🔗 **{verb_phrase}** `({verb_id})` — {len(questions)} questions"):
                if questions:
                    st.markdown("**Questions:**")
                    for q in questions:
                        q_text = q.get('text', 'No text')
                        q_answers = q.get('answers', [])
                        answer_str = ', '.join(q_answers) if isinstance(q_answers, list) else str(q_answers)
                        st.markdown(f"  • {q_text} → `[{answer_str}]`")
                else:
                    st.info("No questions for this relationship")
    else:
        st.info("No relationships found")

    # Token usage if available
    if token_usage:
        st.markdown("---")
        with st.expander("📊 Token Usage Statistics"):
            st.json(token_usage)

    # Raw JSON fallback
    with st.expander("🔍 Show Full GSW JSON (for debugging)"):
        st.json(gsw)


# ==============================================================================
# AI Quality Assessment Function
# ==============================================================================

def assess_gsw_quality_with_openai(
    raw_text: str,
    gsw: Dict[str, Any],
    model: str = "gpt-4o",
    api_key: str = None
) -> Optional[Dict[str, Any]]:
    """
    Assess GSW quality using OpenAI LLM-as-a-Judge.

    Args:
        raw_text: Original input text
        gsw: Predicted GSW structure
        model: OpenAI model to use
        api_key: OpenAI API key

    Returns:
        Dict with assessment results (scores, findings, etc.) or None on error
    """
    if not api_key:
        return None

    try:
        client = OpenAI(api_key=api_key)

        # Format GSW as JSON string
        gsw_json = json.dumps(gsw, indent=2)

        # Create user prompt
        user_content = f"""<input_text>
{raw_text}
</input_text>

<predicted_gsw_json>
{gsw_json}
</predicted_gsw_json>

Assess the quality of this predicted GSW structure. Evaluate entity extraction, relationships, format compliance, and check for hallucinations. Provide detailed scores and findings."""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_AS_A_JUDGE_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # Request JSON output
        )

        # Parse response
        assessment_text = response.choices[0].message.content
        assessment = json.loads(assessment_text)

        return assessment

    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return None


# ==============================================================================
# Batch Assessment Function
# ==============================================================================

def batch_assess_samples(
    data: Dict[str, Any],
    state_path: str,
    api_key: str,
    max_samples: int = None,
    rate_limit_rpm: int = 10
) -> None:
    """
    Batch assess all samples that don't have cached assessments.
    Runs with rate limiting and updates session state progressively.

    Args:
        data: Full dataset with entries
        state_path: Path to save state
        api_key: OpenAI API key
        max_samples: Optional limit on number of samples to assess
        rate_limit_rpm: Rate limit in requests per minute (default: 10)
    """
    # Find samples without assessments
    unassessed_indices = []
    for i in range(len(data['entries'])):
        if str(i) not in st.session_state.ai_assessments:
            unassessed_indices.append(i)

    if max_samples:
        unassessed_indices = unassessed_indices[:max_samples]

    if not unassessed_indices:
        return

    # Create progress tracking
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Estimate cost (approximate $0.01-0.02 per assessment with GPT-4o)
    estimated_cost = len(unassessed_indices) * 0.015
    st.sidebar.info(f"📊 Assessing {len(unassessed_indices)} samples\n💰 Estimated cost: ${estimated_cost:.2f}")

    # Rate limiting: seconds between requests
    delay = 60 / rate_limit_rpm

    successful = 0
    failed = 0

    for idx, sample_idx in enumerate(unassessed_indices):
        # Check for stop signal
        if st.session_state.get('stop_assessment', False):
            status_text.warning(f"⏸️ Assessment stopped by user ({successful} completed, {failed} failed)")
            break

        sample = data['entries'][sample_idx]
        status_text.text(f"Assessing sample {sample_idx} ({idx+1}/{len(unassessed_indices)})...")

        # Assess this sample
        try:
            assessment = assess_gsw_quality_with_openai(
                raw_text=sample.get('raw_text', ''),
                gsw=sample.get('gsw', {}),
                model="gpt-4o",
                api_key=api_key
            )

            if assessment:
                st.session_state.ai_assessments[str(sample_idx)] = assessment

                # Auto-apply AI recommendation ONLY if not manually reviewed yet
                if str(sample_idx) not in st.session_state.review_statuses:
                    ai_recommendation = assessment.get('recommendation', 'review')
                    # Map AI recommendation to status
                    if ai_recommendation == 'platinum':
                        st.session_state.review_statuses[str(sample_idx)] = STATUS_PLATINUM
                    elif ai_recommendation == 'reject':
                        st.session_state.review_statuses[str(sample_idx)] = STATUS_REJECT
                    else:  # 'review' or unknown
                        st.session_state.review_statuses[str(sample_idx)] = STATUS_REVIEW

                persist_state(state_path)
                successful += 1
            else:
                failed += 1
        except Exception as e:
            status_text.error(f"Error assessing sample {sample_idx}: {e}")
            failed += 1

        # Update progress
        progress_bar.progress((idx + 1) / len(unassessed_indices))

        # Rate limiting (sleep between requests)
        if idx < len(unassessed_indices) - 1:
            time.sleep(delay)

    # Final status
    progress_bar.empty()
    status_text.success(f"✅ Assessment complete! {successful} successful, {failed} failed")
    st.session_state.pop('stop_assessment', None)


def backfill_ai_recommendations(state_path: str) -> int:
    """
    Apply AI recommendations to all assessed but unmarked samples.

    This is useful for applying statuses to samples that were assessed
    before the auto-apply feature was implemented.

    Args:
        state_path: Path to save state

    Returns:
        Number of samples that had status applied
    """
    applied_count = 0

    for sample_idx_str, assessment in st.session_state.ai_assessments.items():
        # Only apply if not already manually reviewed
        if sample_idx_str not in st.session_state.review_statuses:
            ai_recommendation = assessment.get('recommendation', 'review')

            # Map AI recommendation to status
            if ai_recommendation == 'platinum':
                st.session_state.review_statuses[sample_idx_str] = STATUS_PLATINUM
            elif ai_recommendation == 'reject':
                st.session_state.review_statuses[sample_idx_str] = STATUS_REJECT
            else:  # 'review' or unknown
                st.session_state.review_statuses[sample_idx_str] = STATUS_REVIEW

            applied_count += 1

    if applied_count > 0:
        persist_state(state_path)

    return applied_count


# ==============================================================================
# Statistics Functions
# ==============================================================================

def get_statistics() -> Dict[str, int]:
    """Calculate statistics about review progress."""
    stats = {
        'total': st.session_state.total_samples,
        'platinum': 0,
        'review': 0,
        'reject': 0,
        'skip': 0,
        'unmarked': 0
    }

    for i in range(st.session_state.total_samples):
        status = st.session_state.review_statuses.get(str(i))
        if status:
            stats[status] += 1
        else:
            stats['unmarked'] += 1

    return stats


# ==============================================================================
# Export Functions
# ==============================================================================

def export_platinum_dataset(data: Dict[str, Any], export_format: str = 'full') -> Dict[str, Any]:
    """Export platinum samples to dataset format."""
    platinum_samples = []

    for i in range(st.session_state.total_samples):
        if st.session_state.review_statuses.get(str(i)) == STATUS_PLATINUM:
            entry = data['entries'][i].copy()

            # Add review metadata
            entry['platinum_status'] = STATUS_PLATINUM
            entry['quality_notes'] = st.session_state.notes.get(str(i), '')
            entry['marked_at'] = datetime.now().isoformat()

            # Add AI assessment if available
            ai_assessment = st.session_state.ai_assessments.get(str(i))
            if ai_assessment:
                entry['ai_assessment'] = ai_assessment

            # Format based on export option
            if export_format == 'gsw_only':
                entry = {
                    'index': entry['index'],
                    'gsw': entry['gsw']
                }
            elif export_format == 'compact':
                # Remove verbose fields
                entry.pop('token_usage', None)

            platinum_samples.append(entry)

    # Create output dataset
    stats = get_statistics()

    # Count how many platinum samples have AI assessments
    ai_assessed_count = sum(
        1 for sample in platinum_samples
        if 'ai_assessment' in sample
    )

    output = {
        'metadata': {
            'created_from': DEFAULT_INPUT_FILE,
            'total_samples': st.session_state.total_samples,
            'platinum_count': stats['platinum'],
            'ai_assessed_count': ai_assessed_count,
            'created_at': datetime.now().isoformat(),
            'export_format': export_format,
            'source_cache_dir': data.get('cache_dir', 'unknown')
        },
        'samples': platinum_samples
    }

    return output


# ==============================================================================
# Main UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title="GSW Inspector",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 GSW Inspector & Platinum Dataset Creator")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # File selection
        input_file = st.text_input(
            "Input file",
            value=DEFAULT_INPUT_FILE,
            help="Path to thinking traces JSON file"
        )

        state_path = Path(input_file).parent / STATE_FILE

        # Load data
        try:
            data = load_thinking_traces(input_file)
            initialize_session_state(data, str(state_path))
        except FileNotFoundError:
            st.error(f"File not found: {input_file}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading file: {e}")
            st.stop()

        st.success(f"✓ Loaded {len(data['entries'])} samples")

        # Statistics
        st.header("Statistics")
        stats = get_statistics()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total", stats['total'])
            st.metric("Platinum", stats['platinum'], delta=None)
            st.metric("Review", stats['review'], delta=None)
        with col2:
            st.metric("Unmarked", stats['unmarked'], delta=None)
            st.metric("Rejected", stats['reject'], delta=None)
            st.metric("Skipped", stats['skip'], delta=None)

        # Progress bar
        reviewed = stats['total'] - stats['unmarked']
        progress = reviewed / stats['total'] if stats['total'] > 0 else 0
        st.progress(progress)
        st.caption(f"Progress: {reviewed}/{stats['total']} ({progress*100:.1f}%)")

        # Filter mode
        st.header("Filter")
        filter_mode = st.selectbox(
            "Show samples",
            options=['all', 'unmarked', 'marked', 'platinum'],
            format_func=lambda x: {
                'all': 'All samples',
                'unmarked': 'Unmarked only',
                'marked': 'Marked only',
                'platinum': 'Platinum only'
            }[x],
            index=['all', 'unmarked', 'marked', 'platinum'].index(st.session_state.filter_mode)
        )

        if filter_mode != st.session_state.filter_mode:
            st.session_state.filter_mode = filter_mode
            navigate_to(0, str(state_path))
            st.rerun()

        # AI Auto-Assessment Section
        st.header("AI Auto-Assessment")

        if not OPENAI_API_KEY:
            st.warning("⚠️ Set OPENAI_API_KEY to enable")
        else:
            # Auto-backfill existing assessments on first load
            if 'backfill_done' not in st.session_state:
                assessed_count = len(st.session_state.ai_assessments)
                marked_count = len(st.session_state.review_statuses)

                # If there are assessed samples without marks, backfill them
                if assessed_count > marked_count:
                    applied = backfill_ai_recommendations(str(state_path))
                    if applied > 0:
                        st.success(f"✅ Auto-applied {applied} AI recommendations from previous assessments")
                st.session_state.backfill_done = True

            # Count unassessed samples
            unassessed_count = sum(
                1 for i in range(st.session_state.total_samples)
                if str(i) not in st.session_state.ai_assessments
            )

            assessed_count = st.session_state.total_samples - unassessed_count

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Assessed", assessed_count)
            with col2:
                st.metric("Unassessed", unassessed_count)

            # Batch assessment controls
            col_assess, col_stop = st.columns([2, 1])
            with col_assess:
                if st.button("🚀 Assess All", disabled=unassessed_count == 0, use_container_width=True):
                    batch_assess_samples(data, str(state_path), OPENAI_API_KEY)
                    st.rerun()

            with col_stop:
                if st.button("⏸️ Stop", use_container_width=True):
                    st.session_state.stop_assessment = True

            # Auto-start option
            auto_assess = st.checkbox("Auto-assess on startup", value=False,
                                     help="Automatically assess all unassessed samples when app starts")

            if auto_assess:
                st.session_state.auto_assess_on_startup = True
                # Trigger assessment if unassessed samples exist and not already done
                if unassessed_count > 0 and not st.session_state.get('auto_assess_done', False):
                    st.info("🔄 Starting auto-assessment...")
                    batch_assess_samples(data, str(state_path), OPENAI_API_KEY)
                    st.session_state.auto_assess_done = True
                    st.rerun()
            else:
                st.session_state.auto_assess_on_startup = False
                st.session_state.auto_assess_done = False

            # Backfill button for existing assessments
            st.markdown("---")
            if st.button("🔄 Apply AI Recommendations to Existing Assessments",
                        help="Apply AI recommendations to all assessed but unmarked samples",
                        use_container_width=True):
                applied = backfill_ai_recommendations(str(state_path))
                if applied > 0:
                    st.success(f"✅ Applied AI recommendations to {applied} samples!")
                    st.rerun()
                else:
                    st.info("ℹ️ No unmarked assessed samples found")

        # Export section
        st.header("Export")
        export_format = st.selectbox(
            "Export format",
            options=['full', 'compact', 'gsw_only'],
            format_func=lambda x: {
                'full': 'Full (with thinking traces)',
                'compact': 'Compact (no token usage)',
                'gsw_only': 'GSW only'
            }[x]
        )

        if st.button("📥 Export Platinum Dataset", type="primary"):
            if stats['platinum'] == 0:
                st.warning("No platinum samples marked yet!")
            else:
                output = export_platinum_dataset(data, export_format)
                output_json = json.dumps(output, indent=2)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"gsw_platinum_dataset_{timestamp}.json"

                st.download_button(
                    label=f"Download {filename}",
                    data=output_json,
                    file_name=filename,
                    mime="application/json"
                )
                st.success(f"✓ Ready to download {stats['platinum']} platinum samples")

    # Main content area
    filtered_indices = get_filtered_indices()

    if not filtered_indices:
        st.warning(f"No samples found with filter: {st.session_state.filter_mode}")
        return

    # Find current position in filtered list
    try:
        current_position = filtered_indices.index(st.session_state.current_index)
    except ValueError:
        current_position = 0
        st.session_state.current_index = filtered_indices[0]

    # Navigation controls
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])

    with nav_col1:
        if st.button("⏮️ First", disabled=current_position == 0):
            navigate_to(0, str(state_path))
            st.rerun()

    with nav_col2:
        if st.button("◀️ Previous", disabled=current_position == 0):
            navigate_to(current_position - 1, str(state_path))
            st.rerun()

    with nav_col3:
        st.markdown(f"**Sample {current_position + 1} of {len(filtered_indices)}** (Index: {st.session_state.current_index})")

    with nav_col4:
        if st.button("Next ▶️", disabled=current_position >= len(filtered_indices) - 1):
            navigate_to(current_position + 1, str(state_path))
            st.rerun()

    with nav_col5:
        if st.button("Last ⏭️", disabled=current_position >= len(filtered_indices) - 1):
            navigate_to(len(filtered_indices) - 1, str(state_path))
            st.rerun()

    # Jump to index
    jump_index = st.number_input(
        "Jump to index",
        min_value=0,
        max_value=st.session_state.total_samples - 1,
        value=st.session_state.current_index,
        step=1
    )
    if jump_index != st.session_state.current_index:
        st.session_state.current_index = jump_index
        persist_state(str(state_path))
        st.rerun()

    st.divider()

    # Get current sample
    current_sample = data['entries'][st.session_state.current_index]
    current_status = st.session_state.review_statuses.get(str(st.session_state.current_index))

    # Status indicator
    if current_status:
        status_info = STATUS_OPTIONS[current_status]
        st.info(f"**Status:** {status_info['label']}")

    # Quality assessment panel
    st.subheader("Quality Assessment")

    status_cols = st.columns(4)
    for idx, (status_key, status_info) in enumerate(STATUS_OPTIONS.items()):
        with status_cols[idx]:
            if st.button(
                status_info['label'],
                key=f"btn_{status_key}",
                type="primary" if current_status == status_key else "secondary",
                use_container_width=True
            ):
                st.session_state.review_statuses[str(st.session_state.current_index)] = status_key
                persist_state(str(state_path))
                st.rerun()

    # Notes field
    current_note = st.session_state.notes.get(str(st.session_state.current_index), '')
    note = st.text_area(
        "Quality notes",
        value=current_note,
        height=100,
        placeholder="Add notes about this sample's quality..."
    )

    if note != current_note:
        st.session_state.notes[str(st.session_state.current_index)] = note
        persist_state(str(state_path))

    # AI Quality Assessment Section
    st.markdown("---")
    st.markdown("#### 🤖 AI Quality Assessment")

    # Check if OpenAI API key is available
    if not OPENAI_API_KEY:
        st.warning("⚠️ OpenAI API key not found. Set `OPENAI_API_KEY` environment variable to use AI assessment.")
    else:
        # Check if assessment exists in cache
        current_idx_str = str(st.session_state.current_index)
        cached_assessment = st.session_state.ai_assessments.get(current_idx_str)

        if cached_assessment:
            # Assessment exists - show results directly with clear button
            col_title, col_clear = st.columns([5, 1])
            with col_title:
                st.markdown("##### Assessment Results")
            with col_clear:
                if st.button("🔄", help="Clear and re-assess", use_container_width=True):
                    # Remove from cache to re-run
                    del st.session_state.ai_assessments[current_idx_str]
                    # Also clear the status so it can be re-set by new AI assessment
                    if current_idx_str in st.session_state.review_statuses:
                        del st.session_state.review_statuses[current_idx_str]
                    persist_state(str(state_path))
                    st.rerun()
        else:
            # No assessment yet - show button to assess
            col_assess, col_info = st.columns([2, 2])
            with col_assess:
                if st.button("📊 Assess This Sample", type="secondary", use_container_width=True):
                    # Call OpenAI API with spinner
                    with st.spinner("Assessing GSW quality with GPT-4o..."):
                        raw_text = current_sample.get('raw_text', '')
                        gsw = current_sample.get('gsw', {})

                        assessment = assess_gsw_quality_with_openai(
                            raw_text=raw_text,
                            gsw=gsw,
                            model="gpt-4o",
                            api_key=OPENAI_API_KEY
                        )

                        if assessment:
                            # Cache the assessment
                            st.session_state.ai_assessments[current_idx_str] = assessment

                            # Auto-apply AI recommendation if not manually reviewed yet
                            if current_idx_str not in st.session_state.review_statuses:
                                ai_recommendation = assessment.get('recommendation', 'review')
                                if ai_recommendation == 'platinum':
                                    st.session_state.review_statuses[current_idx_str] = STATUS_PLATINUM
                                elif ai_recommendation == 'reject':
                                    st.session_state.review_statuses[current_idx_str] = STATUS_REJECT
                                else:
                                    st.session_state.review_statuses[current_idx_str] = STATUS_REVIEW

                            persist_state(str(state_path))
                            st.rerun()

            with col_info:
                st.caption("💡 No AI assessment yet. Click to assess or use 'Assess All' in sidebar.")

        # Display assessment results if available
        if cached_assessment:
            # Overall score as prominent metric
            overall_score = cached_assessment.get('overall_score', 0)

            # Color-code based on score
            if overall_score >= 85:
                score_color = "green"
                score_emoji = "🟢"
            elif overall_score >= 60:
                score_color = "orange"
                score_emoji = "🟡"
            else:
                score_color = "red"
                score_emoji = "🔴"

            st.markdown(f"### {score_emoji} Overall Score: {overall_score}/100")

            # Sub-scores in 4 columns
            st.markdown("**Detailed Scores:**")
            score_cols = st.columns(4)

            with score_cols[0]:
                entity_score = cached_assessment.get('entity_completeness', 0)
                st.metric("Entities", f"{entity_score:.2f}/1.0")

            with score_cols[1]:
                relationship_score = cached_assessment.get('relationship_accuracy', 0)
                st.metric("Relationships", f"{relationship_score:.2f}/1.0")

            with score_cols[2]:
                format_score = cached_assessment.get('format_compliance', 0)
                st.metric("Format", f"{format_score:.2f}/1.0")

            with score_cols[3]:
                hallucination_score = cached_assessment.get('hallucination_score', 0)
                st.metric("No Hallucinations", f"{hallucination_score:.2f}/1.0")

            # Strengths
            strengths = cached_assessment.get('strengths', [])
            if strengths:
                st.markdown("**✅ Strengths:**")
                for strength in strengths:
                    st.markdown(f"  - {strength}")

            # Issues
            issues = cached_assessment.get('issues', [])
            if issues:
                st.markdown("**⚠️ Issues:**")
                for issue in issues:
                    st.markdown(f"  - {issue}")

            # Missing entities
            missing_entities = cached_assessment.get('missing_entities', [])
            if missing_entities:
                st.markdown("**❌ Missing Entities:**")
                for entity in missing_entities:
                    st.markdown(f"  - {entity}")

            # Hallucinated info
            hallucinated = cached_assessment.get('hallucinated_info', [])
            if hallucinated:
                st.markdown("**🚨 Hallucinated Information:**")
                for info in hallucinated:
                    st.markdown(f"  - {info}")

            # AI Recommendation
            recommendation = cached_assessment.get('recommendation', 'unknown')
            rec_emoji = {
                'platinum': '✅',
                'review': '⚠️',
                'reject': '❌'
            }.get(recommendation, '❓')

            st.markdown(f"**{rec_emoji} AI Recommendation:** `{recommendation.upper()}`")

            # Expandable full details
            with st.expander("🔍 Show Full Assessment JSON"):
                st.json(cached_assessment)

    st.divider()

    # Row-by-row vertical layout for inspection
    st.subheader("Sample Inspection")

    # Section 1: Raw Input Text (Full Width)
    st.markdown("### 📄 Raw Input Text")
    raw_text = current_sample.get('raw_text', '')
    if raw_text:
        st.text_area(
            "",
            value=raw_text,
            height=300,
            disabled=True,
            label_visibility="collapsed"
        )
        st.caption(f"Characters: {len(raw_text)} | Words: {len(raw_text.split())}")
    else:
        st.warning("No raw text available")

    st.divider()

    # Section 2: Thinking Trace (Collapsible)
    st.markdown("### 🧠 Thinking Trace")
    thinking_trace = current_sample.get('thinking_trace', '')
    if thinking_trace:
        with st.expander("Show thinking trace", expanded=False):
            st.text_area(
                "",
                value=thinking_trace,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )
            st.caption(f"Characters: {len(thinking_trace)} | Words: {len(thinking_trace.split())}")
    else:
        st.info("No thinking trace available")

    st.divider()

    # Section 3: GSW Pretty Print (Full Width)
    st.markdown("### 🔗 GSW Structure")
    gsw = current_sample.get('gsw', {})
    if gsw:
        token_usage = current_sample.get('token_usage', {})
        render_gsw_pretty(gsw, token_usage)
    else:
        st.warning("No GSW output available")


if __name__ == "__main__":
    main()
