import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
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
    return np.stack(log_probs, axis=1)  # (N, k)


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
    for i in range(max_iter):
        log_probs = poisson_logpmf(X, lambdas) + np.log(pis + 1e-12)
        log_sum = np.logaddexp.reduce(log_probs, axis=1, keepdims=True)
        resp = np.exp(log_probs - log_sum)
        Nk = resp.sum(axis=0)
        pis = Nk / N
        lambdas = (resp.T @ X) / (Nk[:, None] + 1e-12) + 1e-3
        ll = np.sum(log_sum)
        if prev_ll is not None and np.abs(ll - prev_ll) < tol:
            break
        prev_ll = ll
    labels = resp.argmax(axis=1)
    return labels, lambdas, pis


def cluster_file(path, n_dims):
    k = 4 * n_dims - 1
    print(f"Processing {path} with k={k} clusters...")
    df = pd.read_csv(path).drop(columns=["id"])
    X = df.values
    # Preprocessing for KMeans
    X_log = np.log1p(X)
    X_scaled = StandardScaler().fit_transform(X_log)
    km = MiniBatchKMeans(n_clusters=k, n_init=100, batch_size=1000, random_state=42)
    km_labels = km.fit_predict(X_scaled)
    # Poisson Mixture refinement in original count space
    labels, _, _ = fit_poisson_mixture(
        X, k, max_iter=50, tol=1e-4, init_labels=km_labels
    )
    return labels


if __name__ == "__main__":
    # Public dataset
    public_labels = cluster_file("public_data.csv", 4)
    pd.DataFrame({"id": range(len(public_labels)), "label": public_labels}).to_csv(
        "public_submission.csv", index=False
    )
    print(f"Public Score: {score(public_labels.tolist()):.4f}")

    # # Private dataset (uncomment if needed)
    # private_labels = cluster_file("private_data.csv", 6)
    # pd.DataFrame({"id": range(len(private_labels)), "label": private_labels}).to_csv(
    #     "private_submission.csv", index=False
    # )
