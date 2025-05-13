# Industrial AI Agents
Practical examples of Industrial AI Agents using Open Source tools


# Setup Instructions
# Navigate to your project root directory
cd industrial-ai-agents

# Create a virtual environment
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt


# Install the package in development mode
pip install -e .

# to run idoca
cd agents 
python -m idoca.main