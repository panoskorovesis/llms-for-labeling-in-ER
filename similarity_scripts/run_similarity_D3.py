from src.similarities.similarities import SimilarityCalculator
from src.utils.enums import Embedding_Models, SimilarityMetric

DB_PATH = "er_datasets/D3/results.duckdb"
PAIRS_PATH = "er_datasets/D3/pairs.json"
GROUND_TRUTH_PATH = "er_datasets/D3/gtclean.csv"

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
    csv_separator="#",
)

################
# WITHOUT TASK #
################

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
    metric=SimilarityMetric.COSINE, embedding_model=Embedding_Models.BGE_M3
)

similarity_calculator.calculate_similarities(
    metric=SimilarityMetric.EUCLIDIAN, embedding_model=Embedding_Models.BGE_M3
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