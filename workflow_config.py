# workflow_config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# --- VECTOR DATABASES ---
DB_BSDD = r"C:\Users\Tjorven\OneDrive\Documenten\Schoolwerk\universiteit\2de master b.ir.arch\Masterproef\VSC\Masterproef\bsDD\bsdd_vector_db"
DB_EXPRESS = r"C:\Users\Tjorven\OneDrive\Documenten\Schoolwerk\universiteit\2de master b.ir.arch\Masterproef\VSC\Masterproef\EXPRESS\ifc_vector_db"

# --- API ---
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")
if not OPENROUTER_KEY:
    raise EnvironmentError(
        "OPENROUTER_KEY is not set. Add it to your .env file."
    )
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# --- MODELLEN ---
MODEL_GENERATOR = "anthropic/claude-sonnet-4.6"
MODEL_JUDGE = "openai/gpt-4o"

# --- RAG PARAMETERS ---
RAG_K_BSDD = 10           # final chunks selected per query (bSDD)
RAG_K_EXPRESS = 15        # final chunks selected per query (EXPRESS — higher: schema defs are verbose)
RAG_FETCH_K_BSDD = 30     # MMR candidate pool per query (bSDD)
RAG_FETCH_K_EXPRESS = 45  # MMR candidate pool per query (EXPRESS)
RAG_LAMBDA_BSDD = 0.7     # MMR relevance/diversity balance for bSDD (0=diversity, 1=similarity)
RAG_LAMBDA_EXPRESS = 0.7  # MMR relevance/diversity balance for EXPRESS

# --- FEEDBACK LOOPS ---
MAX_SYNTAX_ITERATIONS = 3
MAX_SEMANTIC_ITERATIONS = 5

# --- JUDGE SELF-CONSISTENCY ---
JUDGE_RUNS = 1        # number of independent judge runs per evaluation
JUDGE_TEMPERATURE = 0.0  # temperature for judge runs (0 = deterministic)

# --- OUTPUT ---
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)