from src.similarities.similarities import SimilarityCalculator
from src.utils.enums import Embedding_Models, SimilarityMetric

DB_PATH = "er_datasets/D2/results.duckdb"
SIMILARITY_METRIC = SimilarityMetric.EUCLIDIAN
EMBEDDING_MODEL = Embedding_Models.EMBER_V1
PAIRS_PATH = "er_datasets/D2/pairs.json"
GROUND_TRUTH_PATH = "er_datasets/D2/gtclean.csv"

# TODO: Calculate recall, prec
# IN top 3 for example we could have 1 true positive and 2 false negatives

# Calculate embeddings time

similarity_calculator = SimilarityCalculator(
    db_path=DB_PATH,
    pairs_path=PAIRS_PATH,
    ground_truth_path=GROUND_TRUTH_PATH,
    verbose=True,
    to_csv=True,
    normalize=False,
)

################
# WITHOUT TASK #
################

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.MINI_LM_L12_V2,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.MINI_LM_L12_V2,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.EMBER_V1
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.EMBER_V1
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.MINI_LM_V6,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.MINI_LM_V6,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.GTE_QWEN2,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.GTE_QWEN2,
    use_task=False,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.STELLA_EN
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.STELLA_EN
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_EN_ICL
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_EN_ICL
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.BGE_M3,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.BGE_M3,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.JASPER_TOKEN_COMPRESSION,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.JASPER_TOKEN_COMPRESSION,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.OCTEN_EMBEDDING_4B,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.OCTEN_EMBEDDING_4B,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.NEMOTRON_3_EMBED_1B,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.NEMOTRON_3_EMBED_1B,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.JINA_EMBEDDINGS_V5_OMNI_SMALL,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.JINA_EMBEDDINGS_V5_OMNI_SMALL,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.QWEN_3_5_EMBEDDING_4B,
    use_embeddings_optimal=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.QWEN_3_5_EMBEDDING_4B,
    use_embeddings_optimal=True,
)

#############
# WITH TASK #
#############


similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.E5_MISTRAL_7B,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.E5_MISTRAL_7B,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.SFR_EMBEDDING_MISTRAL,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.EMBER_V1,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.EMBER_V1,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.PHI_3,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.PHI_3,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.STELLA_EN,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.STELLA_EN,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.COSINE,
    embedding_model=Embedding_Models.QWEN_2_5_7B,
    use_task=True,
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN,
    embedding_model=Embedding_Models.QWEN_2_5_7B,
    use_task=True,
)
