import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from typing import Dict, List, Any
import numpy as np
import re
import string
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="Agent Trace Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    /* Dark mode base styles */
    .stApp {
        background-color: #0e1117 !important;
        color: #fafafa !important;
    }
    
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #64b5f6;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #64b5f6;
        border: 1px solid #333;
        color: #fafafa;
    }
    .success-metric {
        border-left-color: #4caf50;
        background-color: rgba(76, 175, 80, 0.1);
    }
    .warning-metric {
        border-left-color: #ff9800;
        background-color: rgba(255, 152, 0, 0.1);
    }
    .error-metric {
        border-left-color: #f44336;
        background-color: rgba(244, 67, 54, 0.1);
    }
    
    /* Tool calls */
    .tool-call {
        background-color: #1e1e1e;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        border-left: 3px solid #64b5f6;
        border: 1px solid #333;
        color: #fafafa;
    }
    
    /* Comment sections */
    .comment-section {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #444;
        color: #fafafa;
        margin: 0.5rem 0;
    }
    
    /* Question header */
    .question-header {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #64b5f6;
        border: 1px solid #333;
        margin: 1rem 0;
        color: #fafafa;
    }
    
    /* Status indicators */
    .status-correct {
        text-align: center;
        padding: 10px;
        background-color: rgba(76, 175, 80, 0.2);
        border-radius: 5px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #fafafa;
    }
    .status-incorrect {
        text-align: center;
        padding: 10px;
        background-color: rgba(244, 67, 54, 0.2);
        border-radius: 5px;
        border: 1px solid rgba(244, 67, 54, 0.3);
        color: #fafafa;
    }
    
    /* Streamlit component overrides */
    .stTextInput > div > div > input {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a4a !important;
    }
    .stSelectbox > div > div > select {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a4a !important;
    }
    .stTextArea > div > div > textarea {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a4a !important;
    }
    .stButton > button {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-color: #4a4a4a !important;
    }
    .stButton > button:hover {
        background-color: #333 !important;
        border-color: #666 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #0e1117 !important;
    }
    
    /* Chart backgrounds */
    .js-plotly-plot {
        background-color: #1e1e1e !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e1e1e !important;
        color: #fafafa !important;
        border-color: #333 !important;
    }
    
    /* Success messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.2) !important;
        color: #4caf50 !important;
        border-color: rgba(76, 175, 80, 0.3) !important;
    }
    
    /* Info messages */
    .stInfo {
        background-color: rgba(33, 150, 243, 0.2) !important;
        color: #2196f3 !important;
        border-color: rgba(33, 150, 243, 0.3) !important;
    }
    
    /* Answer comparison styling */
    .answer-comparison {
        margin: 1rem 0;
    }
    
    .answer-box {
        min-height: 100px;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .answer-header {
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and cache the JSON data"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []

def load_comments() -> Dict[int, List[Dict[str, Any]]]:
    """Load saved comments from file"""
    comments_file = "agent_trace_comments.json"
    if os.path.exists(comments_file):
        try:
            with open(comments_file, 'r') as f:
                comments = json.load(f)
                # Validate comment structure
                validated_comments = {}
                for question_id, comment_list in comments.items():
                    if isinstance(comment_list, list):
                        # Convert question_id to int if it's a string
                        try:
                            qid = int(question_id) if isinstance(question_id, str) else question_id
                            # Validate each comment is a dictionary with required fields
                            validated_comment_list = []
                            for comment in comment_list:
                                if isinstance(comment, dict) and 'text' in comment:
                                    validated_comment_list.append(comment)
                            validated_comments[qid] = validated_comment_list
                        except ValueError:
                            # Skip invalid question IDs
                            continue
                return validated_comments
        except Exception as e:
            st.warning(f"⚠️ Could not load comments: {str(e)}")
            return {}
    return {}

@st.cache_data
def load_musique_data() -> Dict[str, Any]:
    """Load musique_50_q.json data"""
    musique_file = "musique_50_q.json"
    if os.path.exists(musique_file):
        try:
            with open(musique_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load musique data: {str(e)}")
            return []
    return []

@st.cache_data
def load_hipporag_data() -> Dict[str, Any]:
    """Load HippoRAG results data"""
    hipporag_file = "logs/musique_50_q_result_dict_hipporagv2.json"
    if os.path.exists(hipporag_file):
        try:
            with open(hipporag_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load HippoRAG data: {str(e)}")
            return {}
    return {}

def calculate_hipporag_metrics(hipporag_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate metrics from HippoRAG data"""
    if not hipporag_data:
        return {}
    
    total_questions = len(hipporag_data)
    exact_matches = 0
    f1_scores = []
    
    for question_id, result in hipporag_data.items():
        if isinstance(result, dict):
            # Check for exact match
            predicted = result.get('hipporag_2_predicted_answer', '').strip()
            gold_answers = result.get('gold_answers', [])
            if isinstance(gold_answers, list) and predicted in gold_answers:
                exact_matches += 1
            
            # Calculate F1 score
            if gold_answers:
                f1_score = calculate_f1_score(gold_answers, predicted)
                f1_scores.append(f1_score)
    
    exact_match_rate = (exact_matches / total_questions) * 100 if total_questions > 0 else 0
    avg_f1_score = np.mean(f1_scores) * 100 if f1_scores else 0
    
    return {
        'total_questions': total_questions,
        'exact_match_rate': exact_match_rate,
        'avg_f1_score': avg_f1_score,
        'exact_matches': exact_matches
    }

def get_supporting_documents(question_text: str, support_indices: List[int], musique_data: List[Dict]) -> List[Dict]:
    """Get supporting documents for a question based on support indices"""
    supporting_docs = []
    
    # Find the question in musique data by matching question text
    question_data = None
    
    # Try to find exact or similar question text match
    for item in musique_data:
        if 'question' in item:
            musique_question = item['question'].strip()
            agent_question = question_text.strip()
            
            # Try exact match first
            if musique_question == agent_question:
                question_data = item
                break
            
            # Try partial match (if exact doesn't work)
            if musique_question in agent_question or agent_question in musique_question:
                question_data = item
                break
    
    # If no match found, try to find by question similarity
    if not question_data:
        # Use a simple similarity check (first few words)
        agent_words = question_text.split()[:5]  # First 5 words
        for item in musique_data:
            if 'question' in item:
                musique_words = item['question'].split()[:5]
                if agent_words == musique_words:
                    question_data = item
                    break
    
    if question_data and 'paragraphs' in question_data:
        paragraphs = question_data['paragraphs']
        for idx in support_indices:
            if 0 <= idx < len(paragraphs):
                supporting_docs.append(paragraphs[idx])
    
    return supporting_docs, question_data

def normalize_answer(answer: str) -> str:
    """
    Normalize answer string using HippoRAG's methodology.
    
    Applies the following transformations:
    1. Convert to lowercase
    2. Remove punctuation characters
    3. Remove articles "a", "an", "the"
    4. Normalize whitespace (collapse multiple spaces)
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def calculate_f1_score(gold_answers: List[str], predicted_answer: str) -> float:
    """Calculate F1 score using HippoRAG methodology"""
    def compute_f1(gold: str, predicted: str) -> float:
        gold_tokens = normalize_answer(gold).split()
        predicted_tokens = normalize_answer(predicted).split()
        common = Counter(predicted_tokens) & Counter(gold_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            return 0.0

        precision = 1.0 * num_same / len(predicted_tokens) if predicted_tokens else 0.0
        recall = 1.0 * num_same / len(gold_tokens) if gold_tokens else 0.0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)

    f1_scores = [compute_f1(gold, predicted_answer) for gold in gold_answers]
    return float(np.max(f1_scores))

def save_comments(comments: Dict[int, List[Dict[str, Any]]]):
    """Save comments to file"""
    try:
        # Validate comments structure before saving
        validated_comments = {}
        for question_id, comment_list in comments.items():
            if isinstance(comment_list, list):
                validated_comment_list = []
                for comment in comment_list:
                    if isinstance(comment, dict) and 'text' in comment:
                        validated_comment_list.append(comment)
                if validated_comment_list:
                    validated_comments[question_id] = validated_comment_list
        
        # Create backup of existing file if it exists
        backup_file = "agent_trace_comments_backup.json"
        if os.path.exists("agent_trace_comments.json"):
            try:
                import shutil
                shutil.copy2("agent_trace_comments.json", backup_file)
            except Exception as e:
                st.warning(f"⚠️ Could not create backup: {str(e)}")
        
        with open("agent_trace_comments.json", 'w') as f:
            json.dump(validated_comments, f, indent=2)
    except Exception as e:
        st.error(f"❌ Failed to save comments: {str(e)}")
        raise

def calculate_metrics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate various metrics from the data"""
    if not data:
        return {}
    
    total_questions = len(data)
    question_types = {}
    exact_matches = 0
    f1_scores = []
    tool_calls_distribution = {}
    reasoning_lengths = []
    
    for entry in data:
        # Question type distribution
        q_type = entry.get('question_type', 'Unknown')
        question_types[q_type] = question_types.get(q_type, 0) + 1
        
        # Exact match check
        predicted = entry.get('predicted_answer', '').strip()
        gold_answers = entry.get('gold_answers', [])
        if predicted in gold_answers:
            exact_matches += 1
        
        # F1 score calculation using HippoRAG methodology
        if gold_answers:
            f1_score = calculate_f1_score(gold_answers, predicted)
            f1_scores.append(f1_score)
        
        # Tool calls analysis
        tool_calls = entry.get('tool_calls', [])
        num_tool_calls = len(tool_calls)
        tool_calls_distribution[num_tool_calls] = tool_calls_distribution.get(num_tool_calls, 0) + 1
        
        # Reasoning length
        reasoning = entry.get('reasoning', '')
        reasoning_lengths.append(len(reasoning.split()))
    
    exact_match_rate = (exact_matches / total_questions) * 100 if total_questions > 0 else 0
    avg_f1_score = np.mean(f1_scores) * 100 if f1_scores else 0
    
    return {
        'total_questions': total_questions,
        'exact_match_rate': exact_match_rate,
        'avg_f1_score': avg_f1_score,
        'question_types': question_types,
        'tool_calls_distribution': tool_calls_distribution,
        'avg_reasoning_length': np.mean(reasoning_lengths) if reasoning_lengths else 0,
        'exact_matches': exact_matches
    }

def render_question_details(entry: Dict[str, Any], comments: Dict[int, List[Dict[str, Any]]], question_id: int, selected_question_idx: int, musique_data: List[Dict] = None, hipporag_data: Dict[str, Any] = None):
    """Render detailed view of a single question"""
    
    # Question information with better layout
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"**Question:** {entry.get('question', 'N/A')}")
        st.markdown(f"**Type:** {entry.get('question_type', 'N/A')}")
    
    with col2:
        # Exact match indicator with F1 score
        predicted = entry.get('predicted_answer', '').strip()
        gold_answers = entry.get('gold_answers', [])
        is_exact_match = predicted in gold_answers
        
        # Calculate F1 for this question
        f1_score = calculate_f1_score(gold_answers, predicted) if gold_answers else 0.0
        
        status_emoji = "✅" if is_exact_match else "❌"
        status_text = "Exact Match" if is_exact_match else "Partial Match"
        status_class = "status-correct" if is_exact_match else "status-incorrect"
        st.markdown(f"""
        <div class="{status_class}">
            <h3>{status_emoji} {status_text}</h3>
            <p>F1: {f1_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.metric("Tool Calls", len(entry.get('tool_calls', [])))
        st.metric("Supporting Docs", len(entry.get('supporting_doc_ids', [])))
    
    # Answers comparison
    st.markdown("### 📝 Answer Analysis")
    
    # Get HippoRAG answer for this question
    hipporag_answer = "N/A"
    if hipporag_data:
        for qid, result in hipporag_data.items():
            if isinstance(result, dict) and qid == str(question_id):
                hipporag_answer = result.get('hipporag_2_predicted_answer', 'N/A')
                break
    
    # Add some spacing for better visual separation
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 Agent Answer:**")
        agent_answer = entry.get('predicted_answer', 'N/A')
        if agent_answer and agent_answer.strip():
            st.info(agent_answer)
        else:
            st.warning("No answer provided")
    
    with col2:
        st.markdown("**🦛 HippoRAG Answer:**")
        if hipporag_answer and hipporag_answer.strip() and hipporag_answer != 'N/A':
            st.info(hipporag_answer)
        else:
            st.warning("No HippoRAG answer available")
    
    with col3:
        st.markdown("**🏆 Gold Answers:**")
        if gold_answers:
            for i, gold_answer in enumerate(gold_answers):
                st.success(f"{i+1}. {gold_answer}")
        else:
            st.warning("No gold answers available")
    
    # Add spacing after the comparison
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reasoning
    st.markdown("### 🤔 Agent Reasoning")
    reasoning = entry.get('reasoning', 'No reasoning provided')
    st.text_area("Reasoning Process", reasoning, height=120, disabled=True, key=f"reasoning_{question_id}")
    
    # Question decomposition (if available)
    decomposition = entry.get('question_decomposition_gold', [])
    if decomposition:
        st.markdown("### 🧩 Question Decomposition")
        for i, step in enumerate(decomposition):
            with st.expander(f"Step {i+1}: {step.get('question', 'N/A')}", expanded=i==0):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Answer:** {step.get('answer', 'N/A')}")
                with col2:
                    st.markdown(f"**Support Index:** {step.get('paragraph_support_idx', 'N/A')}")
        
        # Show supporting documents if musique data is available
        if musique_data:
            st.markdown("### 📚 Supporting Documents")
            
            # Collect all support indices from decomposition
            support_indices = []
            for step in decomposition:
                support_idx = step.get('paragraph_support_idx')
                if support_idx is not None and support_idx not in support_indices:
                    support_indices.append(support_idx)
            
            if support_indices:
                # Try to find the question in musique data by matching question text
                question_text = entry.get('question', '')
                supporting_docs, matched_question = get_supporting_documents(question_text, support_indices, musique_data)
                
                if supporting_docs:
                    st.success(f"✅ Found {len(supporting_docs)} supporting documents")
                    for i, doc in enumerate(supporting_docs):
                        with st.expander(f"📄 Document {i+1}: {doc.get('title', 'Untitled')} (Index: {doc.get('idx', 'N/A')})", expanded=i==0):
                            st.markdown(f"**Title:** {doc.get('title', 'Untitled')}")
                            st.markdown(f"**Index:** {doc.get('idx', 'N/A')}")
                            st.markdown(f"**Supporting:** {'✅ Yes' if doc.get('is_supporting', False) else '❌ No'}")
                            st.markdown("**Content:**")
                            st.text_area(
                                "Document Text",
                                doc.get('paragraph_text', 'No content available'),
                                height=200,
                                disabled=True,
                                key=f"doc_{question_id}_{i}"
                            )
                else:
                    st.warning(f"⚠️ No supporting documents found for question: {question_text[:100]}...")
                    with st.expander("🔍 Debug: Question Matching"):
                        st.write("**Agent Question:**")
                        st.write(question_text)
                        st.write("**Available Musique Questions (first 5):**")
                        for i, item in enumerate(musique_data[:5]):
                            st.write(f"{i+1}. {item.get('question', 'N/A')}")
                        if matched_question:
                            st.write("**Matched Question:**")
                            st.write(matched_question.get('question', 'N/A'))
                        else:
                            st.write("**No question match found**")
            else:
                st.info("📝 No support indices found in question decomposition")
    
    # Tool calls with better organization
    tool_calls = entry.get('tool_calls', [])
    if tool_calls:
        st.markdown("### 🔧 Tool Calls")
        for i, tool_call in enumerate(tool_calls):
            tool_name = tool_call.get('tool', 'Unknown')
            with st.expander(f"🔧 Tool Call {i+1}: {tool_name}", expanded=i==0):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Arguments:**")
                    st.json(tool_call.get('arguments', {}))
                with col2:
                    st.markdown("**Result:**")
                    st.json(tool_call.get('result', []))
    else:
        st.info("No tool calls recorded")
    
    # Comments section with improved styling
    question_comments = comments.get(question_id, [])
    comment_count = len(question_comments)
    
    st.markdown(f"### 💬 Comments & Notes ({comment_count})")
    
    # Debug info (can be removed later)
    with st.expander("🔧 Debug: Comment Status"):
        st.write(f"**Question ID:** {question_id}")
        st.write(f"**Question ID Type:** {type(question_id)}")
        st.write(f"**Comments for this question:** {len(question_comments)}")
        st.write(f"**All comment keys:** {list(comments.keys())}")
        st.write(f"**Total comments in file:** {sum(len(comments.get(qid, [])) for qid in comments)}")
    
    # Add new comment with better key management
    comment_key = f"comment_input_{question_id}_{selected_question_idx}"
    new_comment = st.text_area(
        f"Add your comment or analysis for Question {question_id}", 
        key=comment_key,
        placeholder="Enter your thoughts, observations, or analysis here...",
        help="Your comments will be saved and can be exported later"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        add_button_key = f"add_comment_btn_{question_id}_{selected_question_idx}"
        if st.button("💬 Add Comment", key=add_button_key, disabled=not new_comment.strip()):
            if new_comment.strip():
                # Ensure comments dict exists for this question
                if question_id not in comments:
                    comments[question_id] = []
                
                # Add the new comment
                new_comment_data = {
                    'text': new_comment.strip(),
                    'timestamp': datetime.now().isoformat(),
                    'user': 'User'
                }
                comments[question_id].append(new_comment_data)
                
                # Save to file
                try:
                    save_comments(comments)
                    st.success(f"✅ Comment added successfully! (Total: {len(comments[question_id])})")
                    # Set flag to indicate comments were updated
                    st.session_state.comments_updated = True
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to save comment: {str(e)}")
    
    # Display existing comments with better formatting
    if question_comments:
        st.markdown("**Previous Comments:**")
        for i, comment in enumerate(question_comments, 1):
            try:
                # Validate comment structure
                if not isinstance(comment, dict):
                    st.warning(f"⚠️ Invalid comment structure at index {i}")
                    continue
                
                timestamp = comment.get('timestamp', 'Unknown')
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        pass
                
                st.markdown(f"""
                <div class="comment-section">
                    <strong>👤 {comment.get('user', 'User')}</strong> - <em>{timestamp}</em> (Comment #{i})<br>
                    {comment.get('text', '')}
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error displaying comment {i}: {str(e)}")
    else:
        st.info("📝 No comments yet. Be the first to add one!")
    
    # Comment management section
    if question_comments:
        st.markdown("### 🛠️ Comment Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear All Comments", key=f"clear_comments_{question_id}"):
                if question_id in comments:
                    del comments[question_id]
                    save_comments(comments)
                    st.success("✅ All comments cleared!")
                    st.session_state.comments_updated = True
                    st.rerun()
        
        with col2:
            if st.button("🔄 Reload Comments", key=f"reload_comments_{question_id}"):
                st.session_state.comments_updated = True
                st.rerun()

def render_hipporag_comparison(entry: Dict[str, Any], hipporag_data: Dict[str, Any], question_id: int):
    """Render HippoRAG comparison for a question"""
    if not hipporag_data:
        return
    
    # Find corresponding HippoRAG result
    hipporag_result = None
    for qid, result in hipporag_data.items():
        if isinstance(result, dict) and result.get('question_id') == question_id:
            hipporag_result = result
            break
    
    if not hipporag_result:
        return
    
    st.markdown("### 🦛 HippoRAG Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**HippoRAG Answer:**")
        # breakpoint()
        hipporag_answer = hipporag_result.get('hipporag_2_predicted_answer', 'N/A')
        st.info(hipporag_answer)
    
    with col2:
        st.markdown("**Agent Answer:**")
        agent_answer = entry.get('predicted_answer', 'N/A')
        st.info(agent_answer)
    
    # Performance comparison
    gold_answers = entry.get('gold_answers', [])
    
    # Calculate scores for both
    hipporag_exact = hipporag_answer.strip() in gold_answers
    agent_exact = agent_answer.strip() in gold_answers
    
    hipporag_f1 = calculate_f1_score(gold_answers, hipporag_answer) if gold_answers else 0.0
    agent_f1 = calculate_f1_score(gold_answers, agent_answer) if gold_answers else 0.0
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Exact Match:**")
        hipporag_emoji = "✅" if hipporag_exact else "❌"
        agent_emoji = "✅" if agent_exact else "❌"
        st.markdown(f"HippoRAG: {hipporag_emoji}")
        st.markdown(f"Agent: {agent_emoji}")
    
    with col2:
        st.markdown("**F1 Score:**")
        st.markdown(f"HippoRAG: {hipporag_f1:.3f}")
        st.markdown(f"Agent: {agent_f1:.3f}")
    
    with col3:
        st.markdown("**Winner:**")
        if hipporag_exact and not agent_exact:
            st.success("🦛 HippoRAG")
        elif agent_exact and not hipporag_exact:
            st.success("🤖 Agent")
        elif hipporag_f1 > agent_f1:
            st.success("🦛 HippoRAG")
        elif agent_f1 > hipporag_f1:
            st.success("🤖 Agent")
        else:
            st.info("🤝 Tie")

def main():
    st.markdown('<h1 class="main-header">🤖 Agent Trace Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar for file selection
    st.sidebar.header("Data Selection")
    
    # Find available log files
    log_dir = "logs"
    available_files = []
    if os.path.exists(log_dir):
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.endswith('.json') and 'agentic_multi_file_results' in file:
                    full_path = os.path.join(root, file)
                    available_files.append(full_path)
    
    if not available_files:
        st.error("No agent trace result files found in the logs directory!")
        return
    
    selected_file = st.sidebar.selectbox(
        "Select Result File",
        available_files,
        format_func=lambda x: os.path.basename(x)
    )
    
    # Load data
    data = load_data(selected_file)
    
    # Initialize session state for comments if not exists
    if 'comments_updated' not in st.session_state:
        st.session_state.comments_updated = False
    
    # Load comments (reload if updated)
    comments = load_comments()
    musique_data = load_musique_data()
    hipporag_data = load_hipporag_data()
    
    # Reset the comments updated flag after loading
    st.session_state.comments_updated = False
    
    if not data:
        st.error("No data loaded!")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(data)
    hipporag_metrics = calculate_hipporag_metrics(hipporag_data)
    
    # Main dashboard
    st.header("📊 Overview Dashboard")
    
    # Agent vs HippoRAG comparison
    if hipporag_metrics:
        st.subheader("🤖 Agent vs 🦛 HippoRAG Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", metrics['total_questions'])
        
        with col2:
            agent_em_class = "success-metric" if metrics['exact_match_rate'] >= 80 else "warning-metric" if metrics['exact_match_rate'] >= 60 else "error-metric"
            hipporag_em_class = "success-metric" if hipporag_metrics['exact_match_rate'] >= 80 else "warning-metric" if hipporag_metrics['exact_match_rate'] >= 60 else "error-metric"
            
            st.markdown(f"""
            <div class="metric-card {agent_em_class}">
                <h3>Agent Exact Match</h3>
                <h2>{metrics['exact_match_rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card {hipporag_em_class}">
                <h3>HippoRAG Exact Match</h3>
                <h2>{hipporag_metrics['exact_match_rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            agent_f1_class = "success-metric" if metrics['avg_f1_score'] >= 80 else "warning-metric" if metrics['avg_f1_score'] >= 60 else "error-metric"
            hipporag_f1_class = "success-metric" if hipporag_metrics['avg_f1_score'] >= 80 else "warning-metric" if hipporag_metrics['avg_f1_score'] >= 60 else "error-metric"
            
            st.markdown(f"""
            <div class="metric-card {agent_f1_class}">
                <h3>Agent F1 Score</h3>
                <h2>{metrics['avg_f1_score']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card {hipporag_f1_class}">
                <h3>HippoRAG F1 Score</h3>
                <h2>{hipporag_metrics['avg_f1_score']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # Calculate improvement
            em_improvement = metrics['exact_match_rate'] - hipporag_metrics['exact_match_rate']
            f1_improvement = metrics['avg_f1_score'] - hipporag_metrics['avg_f1_score']
            
            improvement_class = "success-metric" if em_improvement > 0 else "error-metric" if em_improvement < 0 else "warning-metric"
            st.markdown(f"""
            <div class="metric-card {improvement_class}">
                <h3>EM Improvement</h3>
                <h2>{em_improvement:+.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            improvement_class = "success-metric" if f1_improvement > 0 else "error-metric" if f1_improvement < 0 else "warning-metric"
            st.markdown(f"""
            <div class="metric-card {improvement_class}">
                <h3>F1 Improvement</h3>
                <h2>{f1_improvement:+.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Original metrics display when no HippoRAG data
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Questions", metrics['total_questions'])
        
        with col2:
            exact_match_class = "success-metric" if metrics['exact_match_rate'] >= 80 else "warning-metric" if metrics['exact_match_rate'] >= 60 else "error-metric"
            st.markdown(f"""
            <div class="metric-card {exact_match_class}">
                <h3>Exact Match</h3>
                <h2>{metrics['exact_match_rate']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            f1_class = "success-metric" if metrics['avg_f1_score'] >= 80 else "warning-metric" if metrics['avg_f1_score'] >= 60 else "error-metric"
            st.markdown(f"""
            <div class="metric-card {f1_class}">
                <h3>F1 Score</h3>
                <h2>{metrics['avg_f1_score']:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.metric("Avg Reasoning Length", f"{metrics['avg_reasoning_length']:.1f} words")
    
    # Charts
    st.header("📈 Detailed Analysis")
    
    if hipporag_metrics:
        # Performance comparison chart
        st.subheader("🤖 Agent vs 🦛 HippoRAG Performance Comparison")
        
        comparison_data = {
            'Metric': ['Exact Match', 'F1 Score'],
            'Agent': [metrics['exact_match_rate'], metrics['avg_f1_score']],
            'HippoRAG': [hipporag_metrics['exact_match_rate'], hipporag_metrics['avg_f1_score']]
        }
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Agent',
            x=comparison_data['Metric'],
            y=comparison_data['Agent'],
            marker_color='#64b5f6'
        ))
        
        fig.add_trace(go.Bar(
            name='HippoRAG',
            x=comparison_data['Metric'],
            y=comparison_data['HippoRAG'],
            marker_color='#4caf50'
        ))
        
        fig.update_layout(
            title="Performance Comparison",
            barmode='group',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#fafafa'),
            title_font_color='#fafafa',
            yaxis_title="Score (%)",
            yaxis=dict(gridcolor='#333', zerolinecolor='#333')
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Question type distribution
        if metrics['question_types']:
            fig = px.pie(
                values=list(metrics['question_types'].values()),
                names=list(metrics['question_types'].keys()),
                title="Question Type Distribution"
            )
            # Apply dark mode theme
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                title_font_color='#fafafa'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tool calls distribution
        if metrics['tool_calls_distribution']:
            fig = px.bar(
                x=list(metrics['tool_calls_distribution'].keys()),
                y=list(metrics['tool_calls_distribution'].values()),
                title="Tool Calls Distribution",
                labels={'x': 'Number of Tool Calls', 'y': 'Frequency'}
            )
            # Apply dark mode theme
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#fafafa'),
                title_font_color='#fafafa'
            )
            fig.update_xaxes(gridcolor='#333', zerolinecolor='#333')
            fig.update_yaxes(gridcolor='#333', zerolinecolor='#333')
            st.plotly_chart(fig, use_container_width=True)
    
    # Question filtering and search
    st.header("🔍 Question Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by question type
        all_types = list(set(entry.get('question_type', 'Unknown') for entry in data))
        selected_type = st.selectbox("Filter by Question Type", ["All"] + all_types)
    
    with col2:
        # Filter by correctness
        correctness_filter = st.selectbox("Filter by Correctness", ["All", "Correct", "Incorrect"])
    
    with col3:
        # Search by question content
        search_term = st.text_input("Search in questions", "")
    
    # Apply filters
    filtered_data = data
    if selected_type != "All":
        filtered_data = [entry for entry in filtered_data if entry.get('question_type') == selected_type]
    
    if correctness_filter != "All":
        is_correct = correctness_filter == "Correct"
        filtered_data = [
            entry for entry in filtered_data 
            if (entry.get('predicted_answer', '').strip() in entry.get('gold_answers', [])) == is_correct
        ]
    
    if search_term:
        filtered_data = [
            entry for entry in filtered_data 
            if search_term.lower() in entry.get('question', '').lower()
        ]
    
    st.markdown(f"**Showing {len(filtered_data)} of {len(data)} questions**")
    
    # Question navigation
    if filtered_data:
        # Navigation controls
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        
        with col1:
            # Question selection dropdown with better formatting
            question_options = []
            for i, entry in enumerate(filtered_data):
                question_id = entry.get('question_id', i)
                question_text = entry.get('question', 'N/A')[:60] + "..." if len(entry.get('question', '')) > 60 else entry.get('question', 'N/A')
                question_type = entry.get('question_type', 'Unknown')
                question_options.append(f"Q{question_id} ({question_type}): {question_text}")
            
            selected_question_idx = st.selectbox(
                "📋 Select Question",
                range(len(filtered_data)),
                format_func=lambda x: question_options[x] if x < len(question_options) else f"Question {x}"
            )
        
        with col2:
            if st.button("⬅️ Previous", disabled=selected_question_idx == 0, help="Go to previous question"):
                selected_question_idx = max(0, selected_question_idx - 1)
                st.rerun()
        
        with col3:
            if st.button("Next ➡️", disabled=selected_question_idx == len(filtered_data) - 1, help="Go to next question"):
                selected_question_idx = min(len(filtered_data) - 1, selected_question_idx + 1)
                st.rerun()
        
        with col4:
            # Quick jump to specific question ID
            all_question_ids = [entry.get('question_id', i) for i, entry in enumerate(filtered_data)]
            jump_to_id = st.selectbox("🔍 Jump to ID", ["Select ID"] + all_question_ids, help="Quickly jump to a specific question ID")
            if jump_to_id != "Select ID":
                try:
                    target_idx = all_question_ids.index(jump_to_id)
                    if target_idx != selected_question_idx:
                        selected_question_idx = target_idx
                        st.rerun()
                except ValueError:
                    pass
        
        # Display selected question
        if 0 <= selected_question_idx < len(filtered_data):
            entry = filtered_data[selected_question_idx]
            question_id = entry.get('question_id', selected_question_idx)
            
            # Show question info at top with better styling
            predicted = entry.get('predicted_answer', '').strip()
            gold_answers = entry.get('gold_answers', [])
            is_exact_match = predicted in gold_answers
            
            # Calculate F1 for this specific question
            f1_score = calculate_f1_score(gold_answers, predicted) if gold_answers else 0.0
            
            st.markdown(f"""
            <div class="question-header">
                <h3>📊 Question {selected_question_idx + 1} of {len(filtered_data)} (ID: {question_id})</h3>
                <p><strong>Type:</strong> {entry.get('question_type', 'N/A')} | <strong>Exact Match:</strong> {'✅ Yes' if is_exact_match else '❌ No'} | <strong>F1:</strong> {f1_score:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display the selected question details
            render_question_details(entry, comments, question_id, selected_question_idx, musique_data, hipporag_data)
            render_hipporag_comparison(entry, hipporag_data, question_id)
    else:
        st.info("No questions match the current filters.")
    
    # Export functionality
    st.header("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export filtered results
        if st.button("📊 Export Results", help="Export filtered questions and answers"):
            try:
                # Create a more detailed export with question context
                export_data = []
                for entry in filtered_data:
                    question_id = entry.get('question_id', 'Unknown')
                    export_item = {
                        'question_id': question_id,
                        'question': entry.get('question', 'N/A'),
                        'question_type': entry.get('question_type', 'N/A'),
                        'predicted_answer': entry.get('predicted_answer', 'N/A'),
                        'gold_answers': '; '.join(entry.get('gold_answers', [])),
                        'exact_match': entry.get('predicted_answer', '').strip() in entry.get('gold_answers', []),
                        'f1_score': calculate_f1_score(entry.get('gold_answers', []), entry.get('predicted_answer', '')) if entry.get('gold_answers') else 0.0,
                        'num_tool_calls': len(entry.get('tool_calls', [])),
                        'reasoning_length': len(entry.get('reasoning', '').split())
                    }
                    export_data.append(export_item)
                
                df = pd.DataFrame(export_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"agent_trace_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_results"
                )
                st.success(f"✅ Exported {len(export_data)} questions")
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
    
    with col2:
        # Export comments with question context
        if st.button("💬 Export Comments", help="Export all comments with question context"):
            try:
                comments_df = []
                for question_id, question_comments in comments.items():
                    # Find the question data for context
                    question_data = None
                    for entry in data:
                        if entry.get('question_id') == question_id:
                            question_data = entry
                            break
                    
                    for comment in question_comments:
                        comment_row = {
                            'question_id': question_id,
                            'question': question_data.get('question', 'N/A') if question_data else 'N/A',
                            'question_type': question_data.get('question_type', 'N/A') if question_data else 'N/A',
                            'comment': comment.get('text', '').strip(),
                            'timestamp': comment.get('timestamp', ''),
                            'user': comment.get('user', 'User')
                        }
                        
                        # Format timestamp
                        if comment_row['timestamp']:
                            try:
                                dt = datetime.fromisoformat(comment_row['timestamp'])
                                comment_row['formatted_timestamp'] = dt.strftime("%Y-%m-%d %H:%M:%S")
                            except:
                                comment_row['formatted_timestamp'] = comment_row['timestamp']
                        else:
                            comment_row['formatted_timestamp'] = 'Unknown'
                        
                        comments_df.append(comment_row)
                
                if comments_df:
                    df = pd.DataFrame(comments_df)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comments CSV",
                        data=csv,
                        file_name=f"agent_trace_comments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_comments"
                    )
                    st.success(f"✅ Exported {len(comments_df)} comments")
                else:
                    st.info("📝 No comments to export")
            except Exception as e:
                st.error(f"❌ Comment export failed: {str(e)}")
    
    with col3:
        # Export JSON backup
        if st.button("💾 Export JSON Backup", help="Export all data as JSON backup"):
            try:
                backup_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'total_questions': len(data),
                    'filtered_questions': len(filtered_data),
                    'total_comments': sum(len(comments.get(qid, [])) for qid in comments),
                    'questions': filtered_data,
                    'comments': comments
                }
                
                json_data = json.dumps(backup_data, indent=2)
                st.download_button(
                    label="📥 Download JSON Backup",
                    data=json_data,
                    file_name=f"agent_trace_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="download_backup"
                )
                st.success("✅ JSON backup ready")
            except Exception as e:
                st.error(f"❌ Backup export failed: {str(e)}")
    
    # HippoRAG comparison export
    if hipporag_metrics:
        st.header("🦛 HippoRAG Comparison Export")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export comparison results
            if st.button("📊 Export Comparison", help="Export Agent vs HippoRAG comparison"):
                try:
                    comparison_data = []
                    for entry in filtered_data:
                        question_id = entry.get('question_id', 'Unknown')
                        
                        # Find corresponding HippoRAG result
                        hipporag_result = None
                        for qid, result in hipporag_data.items():
                            if isinstance(result, dict) and result.get('question_id') == question_id:
                                hipporag_result = result
                                break
                        comparison_item = {
                            'question_id': question_id,
                            'question': entry.get('question', 'N/A'),
                            'question_type': entry.get('question_type', 'N/A'),
                            'agent_answer': entry.get('predicted_answer', 'N/A'),
                            'hipporag_answer': hipporag_result.get('hipporag_2_predicted_answer', 'N/A') if hipporag_result else 'N/A',
                            'gold_answers': '; '.join(entry.get('gold_answers', [])),
                            'agent_exact_match': entry.get('predicted_answer', '').strip() in entry.get('gold_answers', []),
                            'hipporag_exact_match': hipporag_result.get('hipporag_2_predicted_answer', '').strip() in entry.get('gold_answers', []) if hipporag_result else False,
                            'agent_f1_score': calculate_f1_score(entry.get('gold_answers', []), entry.get('predicted_answer', '')) if entry.get('gold_answers') else 0.0,
                            'hipporag_f1_score': calculate_f1_score(entry.get('gold_answers', []), hipporag_result.get('hipporag_2_predicted_answer', '')) if hipporag_result and entry.get('gold_answers') else 0.0,
                            'winner': 'Agent' if (entry.get('predicted_answer', '').strip() in entry.get('gold_answers', []) and not (hipporag_result.get('hipporag_2_predicted_answer', '').strip() in entry.get('gold_answers', []) if hipporag_result else False)) else 'HippoRAG' if hipporag_result and (hipporag_result.get('hipporag_2_predicted_answer', '').strip() in entry.get('gold_answers', []) and not (entry.get('predicted_answer', '').strip() in entry.get('gold_answers', []))) else 'Tie'
                        }
                        comparison_data.append(comparison_item)
                    
                    df = pd.DataFrame(comparison_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Comparison CSV",
                        data=csv,
                        file_name=f"agent_vs_hipporag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_comparison"
                    )
                    st.success(f"✅ Exported {len(comparison_data)} comparison results")
                except Exception as e:
                    st.error(f"❌ Comparison export failed: {str(e)}")
        
        with col2:
            # Export summary metrics
            if st.button("📈 Export Summary", help="Export summary metrics comparison"):
                try:
                    summary_data = {
                        'export_timestamp': datetime.now().isoformat(),
                        'agent_metrics': metrics,
                        'hipporag_metrics': hipporag_metrics,
                        'improvement': {
                            'exact_match_improvement': metrics['exact_match_rate'] - hipporag_metrics['exact_match_rate'],
                            'f1_score_improvement': metrics['avg_f1_score'] - hipporag_metrics['avg_f1_score']
                        }
                    }
                    
                    json_data = json.dumps(summary_data, indent=2)
                    st.download_button(
                        label="📥 Download Summary JSON",
                        data=json_data,
                        file_name=f"agent_vs_hipporag_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key="download_summary"
                    )
                    st.success("✅ Summary export ready")
                except Exception as e:
                    st.error(f"❌ Summary export failed: {str(e)}")

if __name__ == "__main__":
    main() 