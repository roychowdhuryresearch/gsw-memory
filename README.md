# 🤖 Agent Trace Analyzer

A comprehensive Streamlit application for analyzing and tracking agent trace results from different types of questions. This tool provides detailed insights into agent performance, tool usage patterns, and allows for collaborative commenting on individual questions.

## Features

### 📊 Dashboard Overview
- **Real-time Metrics**: Total questions, accuracy percentage, correct answers count, and average reasoning length
- **Visual Analytics**: Interactive charts showing question type distribution and tool calls patterns
- **Performance Tracking**: Color-coded accuracy indicators (green for ≥80%, yellow for ≥60%, red for <60%)

### 🔍 Detailed Analysis
- **Question Filtering**: Filter by question type (2hop, 3hop1, 4hop1, 4hop2, etc.)
- **Correctness Filtering**: View only correct or incorrect answers
- **Search Functionality**: Search through question content
- **Pagination**: Navigate through large datasets efficiently

### 📝 Interactive Features
- **Detailed Question View**: Expand each question to see:
  - Original question and predicted answer
  - Gold standard answers
  - Correctness status
  - Agent reasoning
  - Tool calls with full JSON details
  - Question decomposition steps
- **Comment System**: Add and view comments for each question
- **Export Capabilities**: Download filtered results and comments as CSV

### 🎨 Modern UI
- **Responsive Design**: Works on desktop and mobile
- **Custom Styling**: Professional appearance with color-coded metrics
- **Intuitive Navigation**: Sidebar for file selection and main content area

## Installation

1. **Clone or download the files** to your local machine

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure your data structure**:
   - Place your agent trace result JSON files in a `logs/` directory
   - Files should be named with `agentic_multi_file_results` in the filename
   - The app will automatically detect and list available files

## Usage

1. **Start the application**:
   ```bash
   streamlit run agent_trace_analyzer.py
   ```

2. **Select your data file**:
   - Use the sidebar to choose from available result files
   - The app will automatically load and analyze the data

3. **Explore the dashboard**:
   - View overview metrics at the top
   - Examine charts for patterns
   - Use filters to focus on specific question types or correctness

4. **Analyze individual questions**:
   - Click on question expanders to see detailed information
   - Review agent reasoning and tool calls
   - Add comments for collaboration or notes

5. **Export results**:
   - Download filtered results as CSV
   - Export comments separately

## Data Format

The app expects JSON files with the following structure:

```json
[
  {
    "question_id": 0,
    "question": "Your question text here",
    "question_type": "2hop",
    "predicted_answer": "Agent's answer",
    "gold_answers": ["Correct answer 1", "Correct answer 2"],
    "supporting_doc_ids": ["doc1", "doc2"],
    "question_decomposition_gold": [
      {
        "id": 13548,
        "question": "Sub-question 1",
        "answer": "Sub-answer 1",
        "paragraph_support_idx": 1
      }
    ],
    "reasoning": "Agent's reasoning process",
    "tool_calls": [
      {
        "tool": "tool_name",
        "arguments": {...},
        "result": [...]
      }
    ],
    "num_tool_calls": 3,
    "approach": "method_used"
  }
]
```

## File Structure

```
your-project/
├── agent_trace_analyzer.py    # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── logs/                     # Your data directory
│   └── agentic_2wiki_20250729_135333/
│       └── agentic_multi_file_results.json
└── agent_trace_comments.json # Auto-generated comments file
```

## Customization

### Adding New Metrics
To add new analysis metrics, modify the `calculate_metrics()` function in the main script.

### Styling Changes
Update the CSS in the `st.markdown()` section at the top of the script to customize colors and layout.

### Additional Filters
Add new filter options by extending the filtering logic in the main function.

## Troubleshooting

### Common Issues

1. **No files found**: Ensure your JSON files are in the `logs/` directory and contain `agentic_multi_file_results` in the filename

2. **Import errors**: Make sure all dependencies are installed:
   ```bash
   pip install streamlit pandas plotly numpy
   ```

3. **Large file loading**: The app uses caching to handle large files efficiently. If you encounter memory issues, consider splitting your data into smaller files.

4. **Comments not saving**: Ensure the app has write permissions in the directory where it's running.

## Contributing

Feel free to extend this application with additional features such as:
- User authentication for comments
- More advanced analytics and visualizations
- Integration with external databases
- Real-time collaboration features
- Advanced export formats (Excel, PDF reports)

## License

This project is open source and available under the MIT License.