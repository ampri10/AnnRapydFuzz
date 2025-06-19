# AnnRapydFuzz
Efficient string matcher

📦 fuzzy-match-ann

Efficient fuzzy string matching using Approximate Nearest Neighbors (ANN) and RapidFuzz, optimized for large datasets with Polars and chunking.



🔍 Use Case

This package helps detect similar text values in large datasets (e.g., company names, product titles) efficiently — without comparing all pairs.



⚙️ Features



🧼 Smart text cleaning (removes suffixes, punctuation, whitespace)




🔢 Optional alphabetical chunking to speed up processing




🚀 Fast ANN candidate generation using TfidfVectorizer + NearestNeighbors




🧠 Accurate scoring with RapidFuzz




🗃️ Export results as CSV


📥 Installation

bash
CopyEdit
pip install polars pandas rapidfuzz scikit-learn


📘 Example Usage

from fuzzy_match_ann import run_fuzzy_matching

# Run without chunks
results = run_fuzzy_matching(
    df, text_col='merchant_name', id_col='merchant_id',
    threshold=85, n_neighbors=50, use_chunks=False
)

# Run with chunks (split alphabet into 4 parts)
results = run_fuzzy_matching(
    df, text_col='merchant_name', id_col='merchant_id',
    threshold=85, n_neighbors=50,
    use_chunks=True, chunk_num=1, total_chunks=4
)


📤 Output Format

Each match contains:




id_left, id_right




name_left, name_right




fuzzy_score




chunk




original_name_left, original_name_right





🧠 How It Works



Clean text: normalize case, remove suffixes, strip punctuation.




ANN (Approx. Nearest Neighbors): use TF-IDF + NearestNeighbors to get candidate pairs.




Scoring: refine matches using fuzz.ratio from RapidFuzz.




Chunking (optional): split dataset alphabetically to process faster.
