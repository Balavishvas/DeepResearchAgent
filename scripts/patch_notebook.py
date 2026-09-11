import json

path = r'd:\DeepResearchAgent\DeepResearchAgent\notebooks\agent2.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 0 - disable LangSmith + load env
setup_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'setup-env-001',
    'metadata': {},
    'outputs': [],
    'source': [
        'import os\n',
        '# Disable LangSmith tracing to stop 403 errors\n',
        'os.environ["LANGCHAIN_TRACING_V2"] = "false"\n',
        'os.environ["LANGCHAIN_API_KEY"] = ""\n',
        'from dotenv import load_dotenv\n',
        'load_dotenv()\n',
        'print("Environment loaded successfully")'
    ]
}

# Cell 1 - model initialization with confirmed working model
model_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'model-init-001',
    'metadata': {},
    'outputs': [],
    'source': [
        'import os\n',
        'import getpass\n',
        'from langchain_google_genai import ChatGoogleGenerativeAI\n',
        '\n',
        'if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):\n',
        '    os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter your Google Gemini API Key: ")\n',
        '\n',
        'compress_model = ChatGoogleGenerativeAI(\n',
        '    model="gemini-3.5-flash-lite",\n',
        '    temperature=1.0,\n',
        '    max_tokens=None,\n',
        '    timeout=None,\n',
        ')\n',
        'summarization_model = compress_model\n',
        'print("Model ready:", compress_model.model)'
    ]
}

# Keep only non-empty existing cells
existing = [c for c in nb['cells'] if ''.join(c.get('source', [])).strip()]
nb['cells'] = [setup_cell, model_cell] + existing

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully with 2 new cells:")
print("  Cell 0: Environment setup + LangSmith disabled")
print("  Cell 1: Model init with gemini-3.5-flash-lite")
