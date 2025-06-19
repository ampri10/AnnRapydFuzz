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
