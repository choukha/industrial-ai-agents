```
industrial-ai-agents/
├── itisa/
│   ├── app.py                     # Main Gradio UI and agent logic
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_processing_tool.py  # Tool for loading and describing data
│   │   ├── plotting_tool.py         # Tools for various Plotly visualizations
│   │   └── analysis_tool.py         # Tools for anomaly detection, trend/seasonality
│   ├── results/                   # Directory to save plots and analysis results
│   │   └── .gitkeep                 # Placeholder to keep the directory in git
│   ├── data/                      # Directory for uploaded CSVs and intermediate data
│   │   └── .gitkeep                 # Placeholder to keep the directory in git
│   ├── templates/                 # Optional: For HTML templates if needed later
│   │   └── .gitkeep
│   ├── static/                    # Optional: For CSS/JS if UI is customized later
│   │   └── .gitkeep
│   └── .env                       # For environment variables (e.g., OLLAMA_API_BASE)
|   |___ streaming_utils.py        # For streaming agent respone to UI
└── idoca/                         # Your other agent folder
    └── ...
```