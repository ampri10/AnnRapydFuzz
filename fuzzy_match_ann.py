import polars as pl
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from datetime import datetime
import re
import string
import math

# ---------- Clean names ----------
def clean_column(df, column_name):
    df_copy = df.copy()
    df_copy[f"{column_name}_cleaned"] = (
        df_copy[column_name]
        .str.lower()
        .str.replace(r'\b(ehf|inc|corp|corporation|llc|ltd|limited|co|company)\b\.?', '', regex=True)
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df_copy

def print_chunk_info_from_data(df: pd.DataFrame, cleaned_col: str, total_chunks: int):
    df_pl = pl.DataFrame(df)
    chunk_ranges = get_dynamic_chunk_ranges(total_chunks)

    print("\n\U0001F4CA Chunk Distribution Based on Data:")
    print("=" * 50)

    total = 0
    for chunk_id, (start_letter, end_letter) in chunk_ranges.items():
        filtered = df_pl.filter(
            pl.col(cleaned_col).is_not_null() &
            (pl.col(cleaned_col).str.len_chars() > 0) &
            (pl.col(cleaned_col).str.slice(0, 1).str.to_lowercase() >= start_letter) &
            (pl.col(cleaned_col).str.slice(0, 1).str.to_lowercase() <= end_letter)
        )
        count = len(filtered)
        total += count
        print(f"Chunk {chunk_id}: {start_letter.upper()}–{end_letter.upper()} → {count:,} rows")

    print("=" * 50)
    print(f"Total matched rows: {total:,}")
    print(f"Total rows in cleaned DataFrame: {len(df):,}")
    print(f"Unmatched rows (likely null/empty): {len(df) - total:,}")


def get_dynamic_chunk_ranges(num_chunks):
    alphabet = list(string.ascii_lowercase)
    chunk_size = math.ceil(len(alphabet) / num_chunks)
    chunks = {}

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(alphabet)) - 1
        chunks[i + 1] = (alphabet[start], alphabet[end])

    return chunks

def filter_chunk_data(df_pl, chunk_num, name_col, total_chunks):
    chunk_ranges = get_dynamic_chunk_ranges(total_chunks)
    start_letter, end_letter = chunk_ranges[chunk_num]

    return df_pl.filter(
        pl.col(name_col).is_not_null() &
        (pl.col(name_col).str.len_chars() > 0) &
        (pl.col(name_col).str.slice(0, 1).str.to_lowercase() >= start_letter) &
        (pl.col(name_col).str.slice(0, 1).str.to_lowercase() <= end_letter)
    )

# ---------- ANN logic ----------
def get_ann_candidates(texts, n_neighbors=15):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4))
    tfidf = vec.fit_transform(texts)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric='cosine', algorithm='auto')
    nn.fit(tfidf)
    distances, indices = nn.kneighbors(tfidf)

    pairs = set()
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:
            pair = tuple(sorted((i, j)))
            pairs.add(pair)
    return list(pairs)

# ---------- Match pairs ----------
def fuzzy_match_candidates(chunk_df, name_col, original_col, id_col, chunk_num, threshold, n_neighbors):
    print("\U0001F50D Using ANN to generate candidate pairs...")
    candidate_pairs = get_ann_candidates(chunk_df[name_col], n_neighbors=n_neighbors)
    print(f"→ Total candidate pairs: {len(candidate_pairs):,}")

    results = []
    for idx1, idx2 in candidate_pairs:
        row1 = chunk_df.iloc[idx1]
        row2 = chunk_df.iloc[idx2]

        if row1[id_col] != row2[id_col]:
            score = fuzz.ratio(row1[name_col], row2[name_col])
            if score >= threshold:
                results.append({
                    f'{id_col}_left': row1[id_col],
                    f'{id_col}_right': row2[id_col],
                    'name_left': row1[name_col],
                    'name_right': row2[name_col],
                    'fuzzy_score': score,
                    'chunk': chunk_num,
                    'original_name_left': row1.get(original_col, ''),
                    'original_name_right': row2.get(original_col, '')
                })
    return results

# ---------- Export Results ----------
def export_results(results_df, chunk_num, threshold, text_col,n_neighbors):
    if len(results_df) > 0:
        filename = f"fuzzy_matches_{text_col}_chunk_{chunk_num}_threshold_{threshold}_n_neighbors_{n_neighbors}.csv"
        results_df.write_csv(filename)
        print(f"\U0001F4BE Exported to {filename} — {len(results_df):,} matches")
    else:
        print("🟡 No matches to export.")

# ---------- Runner ----------
def run_fuzzy_matching(
    df,
    text_col,
    id_col,
    threshold=80,
    n_neighbors=15,
    use_chunks=True,
    chunk_num=1,
    total_chunks=1
):
    if text_col not in df.columns or id_col not in df.columns:
        raise ValueError("Column name not found in dataframe")

    print(f"\n🧼 Cleaning text column: '{text_col}'")
    df_cleaned = clean_column(df, text_col)
    cleaned_col = f"{text_col}_cleaned"

    if use_chunks:
        print_chunk_info_from_data(df_cleaned, cleaned_col, total_chunks)

    df_pl = pl.DataFrame(df_cleaned)

    if use_chunks:
        print(f"🔠 Filtering chunk {chunk_num} of {total_chunks}...")
        chunk_df = filter_chunk_data(df_pl, chunk_num, cleaned_col, total_chunks).to_pandas()
        print(f"→ Chunk {chunk_num} contains {len(chunk_df):,} rows")
    else:
        print("📦 Using full dataset (no chunks)")
        chunk_df = df_cleaned

    if len(chunk_df) == 0:
        print("⚠️ No data in selection")
        return pl.DataFrame()

    print(f"→ Running fuzzy matching on {len(chunk_df):,} rows")
    results = fuzzy_match_candidates(
        chunk_df=chunk_df,
        name_col=cleaned_col,
        original_col=text_col,
        id_col=id_col,
        chunk_num=chunk_num,
        threshold=threshold,
        n_neighbors=n_neighbors
    )

    results_df = pl.DataFrame(results)
    print(f"✅ Done — Matches above threshold: {len(results_df)}")

    export_results(results_df, chunk_num, threshold, text_col,n_neighbors)

    return results_df
