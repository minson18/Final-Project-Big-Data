import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans


def load_data_and_labels(data_path, submission_path):
    df = pd.read_csv(data_path)
    sub = pd.read_csv(submission_path)
    if "id" in df.columns:
        df = df.sort_values("id").reset_index(drop=True)
    if "id" in sub.columns:
        sub = sub.sort_values("id").reset_index(drop=True)
    X = df.drop(columns=["id"], errors="ignore").values
    labels = sub["label"].values
    return X, labels


def plot_pair_scatter(X, labels, save_path="scatter_pairs.png"):
    combos = list(combinations(range(X.shape[1]), 2))
    cols = 3
    rows = int(np.ceil(len(combos) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()
    for ax, (i, j) in zip(axes, combos):
        ax.scatter(X[:, i], X[:, j], c=labels, s=5, alpha=0.6)
        ax.set_xlabel(f"Feature {i+1}")
        ax.set_ylabel(f"Feature {j+1}")
        ax.set_title(f"Feature {i+1} vs {j+1}")
    for ax in axes[len(combos) :]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved pairwise scatter plots to {save_path}")


def plot_pca(X, labels, save_path="pca_scatter.png"):
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)
    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, s=5, alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Embedding")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved PCA scatter to {save_path}")


def plot_tsne(X, labels, save_path="tsne_scatter.png"):
    X_log = np.log1p(X)
    tsne = TSNE(n_components=2, random_state=42, init="pca")
    proj = tsne.fit_transform(X_log)
    plt.figure(figsize=(8, 6))
    plt.scatter(proj[:, 0], proj[:, 1], c=labels, s=5, alpha=0.6)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("t-SNE Embedding")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t-SNE scatter to {save_path}")


def silhouette_hyperparam_curve(
    X_scaled, n_clusters, max_inits=10, save_path="silhouette_curve.png"
):
    scores = []
    inits = list(range(1, max_inits + 1))
    sample_idx = None
    N = X_scaled.shape[0]
    if N > 10000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, 10000, replace=False)
    for n in inits:
        km = MiniBatchKMeans(
            n_clusters=n_clusters, n_init=n, batch_size=1000, random_state=42
        )
        labels_km = km.fit_predict(X_scaled)
        if sample_idx is not None:
            score_val = silhouette_score(X_scaled[sample_idx], labels_km[sample_idx])
        else:
            score_val = silhouette_score(X_scaled, labels_km)
        scores.append(score_val)
    plt.figure(figsize=(8, 6))
    plt.plot(inits, scores, marker="o")
    plt.xlabel("n_init")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette vs n_init")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved silhouette hyperparameter curve to {save_path}")


def stability_heatmap(X_scaled, n_clusters, runs=10, save_path="stability_heatmap.png"):
    import seaborn as sns

    all_labels = []
    N = X_scaled.shape[0]
    sample_idx = None
    if N > 10000:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(N, 10000, replace=False)
    for i in range(runs):
        km = MiniBatchKMeans(
            n_clusters=n_clusters, n_init=1, batch_size=1000, random_state=i
        )
        lab = km.fit_predict(X_scaled)
        all_labels.append(lab)
    ari = np.zeros((runs, runs))
    for i in range(runs):
        for j in range(runs):
            if sample_idx is not None:
                ari[i, j] = adjusted_rand_score(
                    all_labels[i][sample_idx], all_labels[j][sample_idx]
                )
            else:
                ari[i, j] = adjusted_rand_score(all_labels[i], all_labels[j])
    plt.figure(figsize=(8, 6))
    sns.heatmap(ari, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Stability ARI Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved stability heatmap to {save_path}")


def cluster_profile_table(X, labels, save_path="cluster_profiles.csv"):
    df = pd.DataFrame(np.log1p(X))
    df["label"] = labels
    profile = df.groupby("label").mean()
    profile.to_csv(save_path)
    print(f"Saved cluster profile table to {save_path}")


if __name__ == "__main__":
    # Analyze public results
    X_pub, labels_pub = load_data_and_labels("public_data.csv", "public_submission.csv")
    X_log = np.log1p(X_pub)
    X_scaled = StandardScaler().fit_transform(X_log)
    plot_pair_scatter(X_pub, labels_pub)
    plot_pca(X_pub, labels_pub)
    plot_tsne(X_pub, labels_pub)
    silhouette_hyperparam_curve(X_scaled, len(np.unique(labels_pub)))
    stability_heatmap(X_scaled, len(np.unique(labels_pub)))
    cluster_profile_table(X_pub, labels_pub)
