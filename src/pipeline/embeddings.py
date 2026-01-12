from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
import numpy as np
from sentence_transformers import SentenceTransformer
import os

# Initialize the sentence transformer model
MODEL_NAME = "all-MiniLM-L6-v2"
model = None

def load_model():
    """Load the sentence transformer model."""
    global model
    if model is None:
        model = SentenceTransformer(MODEL_NAME)
    return model

def get_embeddings_udf():
    """Create a UDF for generating embeddings."""
    model = load_model()
    
    def _get_embeddings(text):
        if not text or not text.strip():
            return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2
        
        # Convert text to embedding
        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    
    return F.udf(_get_embeddings, ArrayType(FloatType()))

def generate_embeddings_spark(spark: SparkSession, df: DataFrame, text_col: str = "message") -> DataFrame:
    """
    Generate embeddings for text data using a pre-trained sentence transformer.
    
    Args:
        spark: Active SparkSession
        df: Input DataFrame containing text data
        text_col: Name of the column containing text to embed
        
    Returns:
        DataFrame with an additional 'embedding' column
    """
    print("Generating embeddings...")
    
    # Cache the DataFrame to avoid recomputation
    df.cache()
    
    # Get the embeddings UDF
    get_embeddings = get_embeddings_udf()
    
    # Generate embeddings
    df_with_embeddings = df.withColumn("embedding", get_embeddings(F.col(text_col)))
    
    # Show sample embeddings
    print("Sample embeddings generated:")
    df_with_embeddings.select("userid", text_col, "embedding").show(3, truncate=50)
    
    # Persist the result
    df_with_embeddings.persist()
    
    return df_with_embeddings

def build_faiss_index(embeddings: np.ndarray):
    """
    Build a FAISS index for efficient similarity search.
    
    Args:
        embeddings: Numpy array of embeddings
        
    Returns:
        FAISS index
    """
    try:
        import faiss  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "faiss is not installed. Install faiss-cpu to use build_faiss_index()."
        ) from e

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_embeddings(df: DataFrame, output_path: str):
    """
    Save embeddings to disk.
    
    Args:
        df: DataFrame containing embeddings
        output_path: Path to save the embeddings
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.write.parquet(output_path, mode="overwrite")
    print(f"Embeddings saved to {output_path}")

def load_embeddings(spark: SparkSession, input_path: str) -> DataFrame:
    """
    Load embeddings from disk.
    
    Args:
        spark: Active SparkSession
        input_path: Path to the saved embeddings
        
    Returns:
        DataFrame with embeddings
    """
    return spark.read.parquet(input_path)

if __name__ == "__main__":
    # For local testing
    from pyspark.sql import SparkSession
    from ingestion import ingest_data
    
    spark = SparkSession.builder \
        .appName("EmbeddingGeneration") \
        .getOrCreate()
    
    try:
        # Test the embedding generation
        df = ingest_data(spark)
        df_with_embeddings = generate_embeddings_spark(spark, df)
        
        # Show the schema and sample data
        df_with_embeddings.printSchema()
        df_with_embeddings.show(2, truncate=50)
        
        # Convert to pandas for FAISS testing
        pdf = df_with_embeddings.toPandas()
        embeddings = np.array(pdf['embedding'].tolist())
        
        # Build and test FAISS index
        index = build_faiss_index(embeddings)
        print(f"FAISS index built with {index.ntotal} vectors")
        
    finally:
        spark.stop()
