# main_workflow.py
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agentic_workflow.graph import run_compliance_check
from workflow_config import OPENROUTER_KEY

if __name__ == "__main__":
    run_compliance_check(
        building_rule="Every door in the building must have a minimum height of 2.1 meters.",
        ifc_file_path="ifc_models/rac_basic_sample_project.ifc",
        api_key=OPENROUTER_KEY,
    )