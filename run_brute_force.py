import importlib
import src.brute_force.brute_force as bf
from src.utils.enums import BruteForceMode, Models, PromptTypes
import datetime

importlib.reload(bf)

# SET THESE VARIABLES FOR YOUR EXPERIMENT
MODEL = "ALL"
# MODEL = Models.GEMMA4
PROMPT_TYPE = PromptTypes.MATCHING_PROMPT_ONLY
MAX_INPUT_TOKENS = 8192
THINK = False
MODE = BruteForceMode.MATCHING
DATASET = "D5"
USE_TQDM = False
RESTART_SCRIPT_PATH = "control_scripts/docker/restart_ollama_with_gpu.sh"

# "ALL" runs all the available models
if MODEL == "ALL":
    models = [
        Models.GEMMA4,
        # Replaced with Gemma4
        # Models.GEMMA_2,
        # No longer needed
        # Models.FALCON_3,
        Models.PHI_4,
        # Replaced with Qwen 3.5
        # Models.QWEN_2_5,
        Models.QWEN_3_5_9B,
        # Last new addition GLM 4.6v Flash
        Models.GLM_4_6V_FLASH_10B,
    ]
else:
    models = [MODEL]

for model in models:
    print(f"RUNNING BRUTE FORCE FOR {model}")

    run_id = f"{MAX_INPUT_TOKENS}_max_{PROMPT_TYPE}_{model}_{datetime.date.today()}"
    dataset_path = f"er_datasets/{DATASET}"
    log_path = f"/home/papadakis_korovesis/llms-for-labeling-in-ER/logs/brute_force__{model}__{datetime.date.today().strftime('%d_%m_%Y')}__.txt"

    brute_force = bf.BruteForce(
        config_path="./src/config/config.json",
        pairs_path=f"{dataset_path}/pairs.json",
        restart_script_path=RESTART_SCRIPT_PATH,
        d1_records_path=f"{dataset_path}/d1_records.json",
        d2_records_path=f"{dataset_path}/d2_records.json",
        ground_truth_path=f"{dataset_path}/gtclean.csv",
        verbose=True,
        use_tqdm=USE_TQDM,
    )

    brute_force.restart_ollama_server()

    brute_force.run(
        mode=MODE,
        model=model,
        temperature=0.0,
        think=THINK,
        run_id=run_id,
        prompt_type=PROMPT_TYPE,
        max_input_tokens=MAX_INPUT_TOKENS,
        log_path=log_path,
    )

    # Delete the runner to release its LLM and database resources before the next model.
    del brute_force

    print("--------------\n--------------\n--------------\n")
