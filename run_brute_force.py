import importlib
import src.brute_force.brute_force as bf
from src.utils.enums import BruteForceMode, Models, PromptTypes
import datetime

importlib.reload(bf)

# SET THIS VARIABLES FOR YOUR EXPERIMENT
MODEL = Models.GEMMA_2
PROMPT_TYPE = PromptTypes.SELECTING_PROMPT
RUN_ID = f"FINAL_{PROMPT_TYPE}_{MODEL}_{datetime.date.today()}"
MODE = BruteForceMode.SELECTING
RESTART_SCRIPT_PATH = "control_scripts/docker/restart_ollama_with_gpu.sh"

brute_force = bf.BruteForce(
    config_path="./src/config/config.json",
    pairs_path="er_datasets/D2/pairs.json",
    restart_script_path=RESTART_SCRIPT_PATH,
    d1_records_path="er_datasets/D2/d1_records.json",
    d2_records_path="er_datasets/D2/d2_records.json",
    ground_truth_path="er_datasets/D2/gtclean.csv",
    verbose=True,
)


if MODE == BruteForceMode.MATCHING:
    brute_force.run(
        mode=BruteForceMode.MATCHING,
        model=MODEL,
        temperature=0.0,
        run_id=RUN_ID,
        prompt_type=PROMPT_TYPE,
    )

elif MODE == BruteForceMode.COMPARING:
    brute_force.run(
        mode=BruteForceMode.COMPARING,
        model=MODEL,
        temperature=0.0,
        run_id=RUN_ID,
        prompt_type=PROMPT_TYPE,
    )

elif MODE == BruteForceMode.SELECTING:
    brute_force.run(
        mode=BruteForceMode.SELECTING,
        model=MODEL,
        temperature=0.0,
        run_id=RUN_ID,
        prompt_type=PROMPT_TYPE,
    )

else:
    raise NotImplementedError(f"BruteForce {MODE} is not valid!")
