import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.preprocessing import clean_data


def run_loyalty_model(df, knobs, lambda_parameter, multipliers, tier_mix):
    cust = clean_data(df.copy())

    # Make sure the numeric values are indeed numeric
    for c in ["purchase_amount_usd", "previous_purchases", "frequency_of_purchases",
              "subscription_status", "discount_applied", "review_rating", "age"]:
        if c in cust.columns:
            cust[c] = pd.to_numeric(cust[c], errors="coerce")

    # Since the data is synthetic, we only have one row per consumer, so we assume the last purchase is the average amount that they usually spend
    cust["freq_per_year"] = cust["frequency_of_purchases"].fillna(1.0)
    cust["total_spend_est"] = (cust["purchase_amount_usd"].fillna(0) * cust["freq_per_year"])
    cust["avg_purchase"] = cust["purchase_amount_usd"].fillna(0)

    # Fill in the missing values for the engagement block
    cust["sub_rate"] = cust["subscription_status"].fillna(0)
    cust["discount_rate"] = cust["discount_applied"].fillna(0)
    cust["avg_rating"] = cust["review_rating"].fillna(0)

    # Now we bild the blocks with scores

    # Spend/Value block is estimated by annual spend and average purchase
    s_spend = np.nanmean(np.vstack([
        norm_quantile_0_100(nz(cust["total_spend_est"])),
        norm_quantile_0_100(nz(cust["avg_purchase"])),
    ]), axis=0)

    # For engagement, we include sub + promos + rating (also add a cap so promo-chasers do not dominate)
    s_engage = np.nanmean(np.vstack([
        norm_quantile_0_100(nz(cust["sub_rate"])),
        norm_quantile_0_100(nz(cust["discount_rate"])),
        norm_quantile_0_100(nz(cust["avg_rating"])),
    ]), axis=0)

    # Assemble block scores
    block_scores = pd.DataFrame({
        "customer_id": cust["customer_id"],
        "s_spend": s_spend,
        "s_engage": s_engage,
    })

    block_cols = sorted([c for c in block_scores.columns if c.startswith("s_")])

    # Clustering
    X = block_scores[block_cols].values
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    best_labels = km.fit_predict(StandardScaler().fit_transform(X))
    block_scores["cluster"] = best_labels

    # Compute the unsupervised contribution, scale it and assigns a score based on the cluster to each consumer
    row_eq = block_scores[['s_spend', 's_engage']].mean(axis=1)
    cluster_base = row_eq.groupby(block_scores["cluster"]).mean()
    cluster_ranks = cluster_base.rank(method='dense').astype(int) - 1
    unsup = block_scores["cluster"].map(cluster_ranks)
    unsup_scaled = (unsup / unsup.max()) * 100 if unsup.max() > 0 else 0
    block_scores["unsup_signal"] = unsup_scaled

    # Calculate the average scores in each cluster
    cluster_profiles = block_scores.groupby('cluster')[['s_spend', 's_engage']].mean().round(1)

    # Identify each archetype based on the cluster
    low_score_cluster = cluster_profiles.sum(axis=1).idxmin()
    high_score_cluster = cluster_profiles.sum(axis=1).idxmax()

    # Identify the remaining two clusters
    remaining_clusters = [c for c in cluster_profiles.index if c not in [low_score_cluster, high_score_cluster]]

    # Sort the remaining clusters
    if cluster_profiles.loc[remaining_clusters[0], 's_spend'] > cluster_profiles.loc[remaining_clusters[1], 's_spend']:
        transactional_cluster = remaining_clusters[0]
        advocate_cluster = remaining_clusters[1]
    else:
        transactional_cluster = remaining_clusters[1]
        advocate_cluster = remaining_clusters[0]

    # Create the mapping from cluster number to archetype
    archetype_map = {
        high_score_cluster: 'Brand Champions',
        transactional_cluster: 'Transactional Spenders',
        advocate_cluster: 'Brand Advocates',
        low_score_cluster: 'Passive Customers'
    }

    # Add the 'archetype' column to the DataFrame
    block_scores['archetype'] = block_scores['cluster'].map(archetype_map)

    # Normalise the knobs
    knobs = {k: v for k, v in knobs.items() if f"s_{k}" in block_scores.columns}
    tot = sum(knobs.values()) or 1.0
    knobs = {k: v / tot for k, v in knobs.items()}

    block_scores["U_seed"] = U_from_knobs(block_scores, knobs)

    # Create the multiplier and alter the boosts/penalties
    archetype_multipliers = multipliers

    # Create a column with the multiplier for each customer
    block_scores['multiplier'] = block_scores['archetype'].map(archetype_multipliers)

    # Set our confidence parameter and create the tierer
    LAMBDA = lambda_parameter
    base_score = LAMBDA * block_scores["U_seed"].values + (1 - LAMBDA) * block_scores["unsup_signal"].values

    # Apply the strategic multiplier
    score = base_score * block_scores['multiplier'].values
    block_scores["score"] = np.clip(score, 0, 100)

    # Add the tier distribution (has to add to 1)
    TARGET_MIX = tier_mix

    thr = thresholds_from_mix(block_scores["score"].values, TARGET_MIX)

    block_scores["tier"] = [to_tier(s, thr) for s in block_scores["score"].values]

    # Final output
    final_df = clean_data(df.copy())
    tier_map = block_scores.set_index("customer_id")["tier"]
    score_map = block_scores.set_index("customer_id")["score"]
    archetype = block_scores.set_index("customer_id")["archetype"]

    final_df["tier"] = final_df["customer_id"].map(tier_map)
    final_df["score"] = final_df["customer_id"].map(score_map).round(1)
    final_df["archetype"] = final_df["customer_id"].map(archetype)
    final_df['score'] = final_df['score'].round(1)

    return final_df, block_scores


# Define a normalisation function
def norm_quantile_0_100(s, invert=False):
    x = pd.to_numeric(s, errors="coerce").to_numpy().reshape(-1, 1)
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(s)),
        output_distribution='uniform',
        random_state=42
    )
    y = (qt.fit_transform(x).ravel() * 100)
    return (100 - y) if invert else y


# Create a function that fills in missing values
def nz(s):
    return pd.to_numeric(s, errors="coerce").fillna(s.median())


# Define the function U(x)
def U_from_knobs(df, weights):
    u = np.zeros(len(df))
    for k, w in weights.items():
        u += w * df[f"s_{k}"].values
    return u


# Define the function that gives us tier cutoffs
def thresholds_from_mix(scores, mix):
    cum = 0
    thr = {}
    order = ["Platinum", "Gold", "Silver", "Regular"]
    for lvl in order[:-1]:
        thr[lvl] = np.quantile(scores, 1 - (cum + mix[lvl]))
        cum += mix[lvl]
    thr["Regular"] = -np.inf
    return thr


# Function that assigns the tiers
def to_tier(s, thr):
    if s >= thr["Platinum"]:
        return "Platinum"
    if s >= thr["Gold"]:
        return "Gold"
    if s >= thr["Silver"]:
        return "Silver"
    return "Regular"
