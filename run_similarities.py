from src.similarities.similarities import SimilarityCalculator
from src.utils.enums import Embedding_Models, SimilarityMetric

MODE = "D4"  # Choose between D2, D3, D4, D10

#################
# D2 EMBEDDINGS #
#################

if MODE == "D2":
    DB_PATH = "er_datasets/D2/results.duckdb"
    PAIRS_PATH = "er_datasets/D2/pairs.json"
    GROUND_TRUTH_PATH = "er_datasets/D2/gtclean.csv"

    # Calculate embeddings time

    similarity_calculator = SimilarityCalculator(
        db_path=DB_PATH,
        pairs_path=PAIRS_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        verbose=True,
        to_csv=True,
        normalize=False,
        csv_separator="|",
    )

    # ALL MINI LM V12
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_L12_V2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.MINI_LM_L12_V2,
    )

    # ALL MINI LM V6
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_V6
    )

    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.MINI_LM_V6
    )

    # GTE QWEN 7B
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.GTE_QWEN2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.GTE_QWEN2
    )

    # EMBER V1
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.EMBER_V1
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.EMBER_V1
    )

    # SFR Embedding Mistral
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )

    # BGE M3
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_M3
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_M3
    )

    # BGE EN ICL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_EN_ICL
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_EN_ICL
    )

    # STELLA EN
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.STELLA_EN
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.STELLA_EN
    )

    # USING TASK
    ############

    # SFR Embedding Mistral
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
        use_task=True,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
        use_task=True,
    )

    # EMBER V1
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.EMBER_V1,
        use_task=True,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.EMBER_V1,
        use_task=True,
    )

    # STELLA EN
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.STELLA_EN,
        use_task=True,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.STELLA_EN,
        use_task=True,
    )

    # E5-MISTRAL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.E5_MISTRAL_7B,
        use_task=True,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.E5_MISTRAL_7B,
        use_task=True,
    )

#################
# D3 EMBEDDINGS #
#################

elif MODE == "D3":
    DB_PATH = "er_datasets/D3/results.duckdb"
    PAIRS_PATH = "er_datasets/D3/pairs.json"
    GROUND_TRUTH_PATH = "er_datasets/D3/gtclean.csv"

    # Calculate embeddings time

    similarity_calculator = SimilarityCalculator(
        db_path=DB_PATH,
        pairs_path=PAIRS_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        verbose=True,
        to_csv=True,
        normalize=False,
        csv_separator="#",
    )

    # ALL MINI LM V12
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_L12_V2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.MINI_LM_L12_V2,
    )

    # ALL MINI LM V6
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_V6
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.MINI_LM_V6
    )

    # GTE QWEN 7B
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.GTE_QWEN2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.GTE_QWEN2
    )

    # EMBER V1
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.EMBER_V1
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.EMBER_V1
    )

    # SFR Embedding Mistral
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )

    # BGE M3
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_M3
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_M3
    )

    # BGE EN ICL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_EN_ICL
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_EN_ICL
    )

    # STELLA EN
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.STELLA_EN
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.STELLA_EN
    )

    # E5 MISTRAL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.E5_MISTRAL_7B
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.E5_MISTRAL_7B,
    )


#################
# D4 EMBEDDINGS #
#################

elif MODE == "D4":
    DB_PATH = "er_datasets/D4/results.duckdb"
    PAIRS_PATH = "er_datasets/D4/pairs.json"
    GROUND_TRUTH_PATH = "er_datasets/D4/gtclean.csv"

    # Calculate embeddings time

    similarity_calculator = SimilarityCalculator(
        db_path=DB_PATH,
        pairs_path=PAIRS_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        verbose=True,
        to_csv=True,
        normalize=False,
        csv_separator="%",
    )

    # ALL MINI LM V12
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_L12_V2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.MINI_LM_L12_V2,
    )

    # ALL MINI LM V6
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_V6
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.MINI_LM_V6
    )

    # GTE QWEN 7B
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.GTE_QWEN2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.GTE_QWEN2
    )

    # EMBER V1
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.EMBER_V1
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.EMBER_V1
    )

    # SFR Embedding Mistral
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )

    # BGE M3
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_M3
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_M3
    )

    # BGE EN ICL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_EN_ICL
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_EN_ICL
    )

    # STELLA EN
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.STELLA_EN
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.STELLA_EN
    )

    # TASK

    # E5 MISTRAL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.E5_MISTRAL_7B,
        use_task=True,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.E5_MISTRAL_7B,
        use_task=True,
    )

##################
# D10 EMBEDDINGS #
##################

elif MODE == "D10":
    DB_PATH = "er_datasets/D10/results.duckdb"
    PAIRS_PATH = "er_datasets/D10/pairs.json"
    GROUND_TRUTH_PATH = "er_datasets/D10/gtclean.csv"

    # Calculate embeddings time

    similarity_calculator = SimilarityCalculator(
        db_path=DB_PATH,
        pairs_path=PAIRS_PATH,
        ground_truth_path=GROUND_TRUTH_PATH,
        verbose=True,
        to_csv=True,
        normalize=False,
        csv_separator="|",
    )

    # ALL MINI LM V12
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_L12_V2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.MINI_LM_L12_V2,
    )

    # ALL MINI LM V6
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.MINI_LM_V6
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.MINI_LM_V6
    )

    # GTE QWEN 7B
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.GTE_QWEN2
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.GTE_QWEN2
    )

    # EMBER V1
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.EMBER_V1
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.EMBER_V1
    )

    # SFR Embedding Mistral
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN,
        embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    )

    # BGE M3
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_M3
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_M3
    )

    # BGE EN ICL
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_EN_ICL
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_EN_ICL
    )

    # STELLA EN
    # Cosine
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.STELLA_EN
    )
    # Euclidian
    similarity_calculator.calculate_similarities(
        metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.STELLA_EN
    )

    # TODO: Enable this after run is complete
    
    # E5 MISTRAL
    # Cosine
    # similarity_calculator.calculate_similarities(
    #     metric=SimilarityMetric.COSINE,
    #     embedding_model=Embedding_Models.E5_MISTRAL_7B,
    # )
    # # Euclidian
    # similarity_calculator.calculate_similarities(
    #     metric=SimilarityMetric.EUCLIDIAN,
    #     embedding_model=Embedding_Models.E5_MISTRAL_7B,
    # )
