# README 

# DSND Recommendation Project

This notebook walks through building a full recommendation engine using IBM Watson Studio data. You’ll see three flavors of recommendations—popularity (rank‑based), collaborative filtering (user–user and SVD), and content‑based (TF‑IDF + K‑Means clustering). By the end, you’ll have functions that recommend articles for brand‑new users, users with sparse history, and users with rich interaction logs.

## Getting Started

To grab a copy of the project and run it locally, just clone this repository and open the Jupyter notebook in your preferred environment:

```bash
git clone https://github.com/MatthewJay412/dsnd-recommendation-project.git
cd dsnd-recommendation-project/starter
jupyter notebook Recommendations_with_IBM.ipynb
```

### Dependencies

This project uses common Python data‑science libraries. You’ll need:

Python 3 (tested on 3.7+)
pandas
numpy
scikit‑learn
matplotlib
Jupyter Notebook

If you’re missing anything, install via pip:
```
pip install pandas numpy scikit-learn matplotlib jupyter
```

### Installation

Clone the repo

```
git clone https://github.com/MatthewJay412/dsnd-recommendation-project.git
cd dsnd-recommendation-project/starter
```

Create a virtual environment (optional but recommended)

```
python3 -m venv venv
source venv/bin/activate      # On Windows, use: venv\Scripts\activate
```

Install required packages

```
pip install pandas numpy scikit-learn matplotlib jupyter
```

Launch the notebook

```
jupyter notebook Recommendations_with_IBM.ipynb
```

That’s it. You should see the notebook open in your browser, ready for exploration.

## Testing

There are “Test your functions here” cells sprinkled throughout the notebook. To verify everything works:

Run Restart & Run All from the Kernel menu.
Each “Test…” cell will print a success message (e.g., “Nice job!” or “Great job!”).
If you see any AssertionError, go back to the function above it and tweak your logic until it passes.
When all assertions pass, your implementation is correct.

### Break Down Tests

**Rank‑Based Tests (Section 2)**

  get_article_names, get_user_articles, and get_ranked_article_unique_counts each have asserts to check expected outputs.

**Collaborative Filtering Tests (Section 3)**

  get_top_sorted_users is tested by looking up “most similar” user IDs for sample users (1, 2, 131).

  new_user_recs cell asserts that the top 10 popular articles match Udacity’s ground truth set.

  **Content‑Based Tests (Section 4)**

  After building TF‑IDF, LSA, and K‑Means, the get_similar_articles and make_content_recs functions are tested for a sample article.

  **SVD‐Based Tests (Section 5)**

  get_svd_similar_article_ids is run for article ID 4, and the assert verifies that the returned 10 recommendations match the expected set.

If all of these test cells print their success messages, then your code is behaving exactly as required.

## Project Instructions

**Exploratory Data Analysis (Section 1)**
Load the user–article interaction and article metadata. Then compute basic stats like how many unique users we have, how many unique articles, and plot out user activity and article popularity to see the big picture.

**Rank‑Based Recommendations (Section 2)**
Implement get_article_names(article_ids, df) so you can map a list of IDs to their actual titles. Next, write get_user_articles(user_id, user_item) to fetch a user’s read article IDs and titles. Then build get_ranked_article_unique_counts(article_ids, user_item) to tally up how many unique users saw each article and sort them from most to least popular. And finally, create new_user_recs by grabbing the top 10 most popular articles—perfect for anyone who has no history yet.

**Collaborative Filtering (Section 3)**
Compute cosine similarity between users using get_top_sorted_users(user_id, user_item). Use that to craft user_user_recs_part2(user_id, m, user_item, df), which recommends up to m new articles based on what similar users have clicked—prioritizing the most popular items among their peers. Then run SVD on the user–item matrix with 200 latent features:

```
import numpy as np
U, S, vt = np.linalg.svd(user_item.values, full_matrices=False)
```

And implement get_svd_similar_article_ids(article_id, vt, user_item, include_similarity=False) to find the top 10 articles whose 200‑dimensional latent vectors are most alike to the given article.

**Content‑Based Recommendations (Section 4)**
First, dedupe down to df_unique_articles so it only has article_id and title. Next, vectorize those titles with TF‑IDF (using max_features=200, min_df=5, max_df=0.75) and then run LSA (n_components=50) to squash it into meaningful themes. After that, cluster the LSA output with K‑Means—pick your k from an elbow plot (for example, 76). Then assign each article a title_cluster in df. Next, implement get_similar_articles(article_id, df) so it returns all other IDs in the same cluster. And finally, write make_content_recs(article_id, n, df) to rank those cluster neighbors by how many unique readers they’ve had and return the top n.

**SVD Article Recommendations (Section 5)**
Use that previously calculated vt (200 latent features) to find the top 10 “nearest neighbor” articles to any given ID via cosine similarity.

## Built With

Python 3 
Jupyter Notebook 
pandas 
NumPy 
scikit‑learn 
    TfidfVectorizer, TruncatedSVD, Normalizer, KMeans (content‑based pipeline).
    cosine_similarity (both user‑user and article‑article).
matplotlib – Plotting distribution charts and elbow curves.

## License

[License](LICENSE.txt)
