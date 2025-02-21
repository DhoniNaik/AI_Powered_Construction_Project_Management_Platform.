├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── data/
│   ├── labor_force_training.csv
│   ├── resource_training.csv
│   ├── risk_training.csv
│   ├── work_assignments_training.csv
│   ├── project_timeline_training.csv
│   └── supply_chain_training.csv
├── pages/
│   ├── project_input.py
│   ├── risk_analysis.py
│   ├── safety_monitoring.py
│   ├── supply_chain.py
│   └── task_management.py
├── utils/
│   ├── ml_models.py
│   └── data_generator.py
├── .gitignore
├── README.md
└── main.py
```

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/construction-project-management.git
cd construction-project-management
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add:
```
OPENAI_API_KEY=your_openai_api_key
```

4. Run the application:
```bash
streamlit run main.py