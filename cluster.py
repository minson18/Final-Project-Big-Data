import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.special import gammaln
from grader import score


def poisson_logpmf(X, lambdas):
    eps = 1e-9
    X = np.clip(X, 0, None)
    lambdas = np.clip(lambdas, eps, None)
    log_probs = []
    for lam in lambdas:
        log_p = X * np.log(lam) - lam - gammaln(X + 1)
        log_probs.append(np.sum(log_p, axis=1))
    return np.stack(log_probs, axis=1)


def fit_poisson_mixture(X, k, max_iter=50, tol=1e-4, init_labels=None, seed=42):
    np.random.seed(seed)
    N, d = X.shape
    if init_labels is None:
        init_labels = np.random.randint(0, k, size=N)
    lambdas = np.zeros((k, d))
    pis = np.zeros(k)
    for j in range(k):
        mask = init_labels == j
        count = np.sum(mask)
        if count > 0:
            lambdas[j] = X[mask].mean(axis=0) + 1e-3
            pis[j] = count / N
        else:
            lambdas[j] = np.mean(X, axis=0) + 1e-3
            pis[j] = 1.0 / k
    prev_ll = None
    for _ in range(max_iter):
        log_probs = poisson_logpmf(X, lambdas) + np.log(pis + 1e-12)
        log_sum = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        resp = np.exp(log_probs - log_sum)
        Nk = resp.sum(axis=0)
        pis = Nk / N
        lambdas = (resp.T @ X) / (Nk[:, None] + 1e-12) + 1e-3
        ll = np.sum(log_sum)
        if prev_ll is not None and abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    labels = resp.argmax(axis=1)
    return labels


def cluster_file(path, n_dims, silhouette_samples=10000):
    k = 4 * n_dims - 1
    print("Processing {} with k={}...".format(path, k))
    df = pd.read_csv(path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    X = df.values
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)
    km = MiniBatchKMeans(n_clusters=k, n_init=100, batch_size=1000, random_state=42)
    km_labels = km.fit_predict(X_scaled)
    sample_idx = None
    if X_scaled.shape[0] > silhouette_samples:
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(X_scaled.shape[0], silhouette_samples, replace=False)
    if sample_idx is not None:
        sil_before = silhouette_score(X_scaled[sample_idx], km_labels[sample_idx])
    else:
        sil_before = silhouette_score(X_scaled, km_labels)
    labels = fit_poisson_mixture(X, k, init_labels=km_labels)
    if sample_idx is not None:
        sil_after = silhouette_score(X_scaled[sample_idx], labels[sample_idx])
    else:
        sil_after = silhouette_score(X_scaled, labels)
    return labels, sil_before, sil_after


if __name__ == "__main__":
    public_labels, sil_b, sil_a = cluster_file("public_data.csv", 4)
    pd.DataFrame({"id": range(len(public_labels)), "label": public_labels}).to_csv(
        "public_submission.csv", index=False
    )
    print("Public silhouette before EM: {:.4f}".format(sil_b))
    print("Public silhouette after EM: {:.4f}".format(sil_a))
    print("Public Score: {:.4f}".format(score(public_labels.tolist())))

    private_labels, sil_pb, sil_pa = cluster_file("private_data.csv", 6)
    pd.DataFrame({"id": range(len(private_labels)), "label": private_labels}).to_csv(
        "private_submission.csv", index=False
    )
    print("Private silhouette before EM: {:.4f}".format(sil_pb))
    print("Private silhouette after EM: {:.4f}".format(sil_pa))
