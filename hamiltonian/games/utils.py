from scipy.stats import ortho_group
import torch


def make_spd_matrix(n_samples, dim, mu=0.0, L=1.0, num_zeros=0):
    A = torch.rand(n_samples, dim, dim)
    X = torch.bmm(A.transpose(-1, -2), A)
    U, s, V = torch.svd(X)
    diag = torch.zeros(n_samples, dim)
    random_eig = torch.rand(n_samples - num_zeros, dim)
    random_eig = (random_eig - random_eig.min()) / (random_eig.max() - random_eig.min())
    diag[: n_samples - num_zeros] = mu + random_eig * (L - mu)
    X = torch.bmm(torch.bmm(U, torch.diag_embed(diag)), V.transpose(-1, -2))
    return X


def make_sym_matrix(n_samples, dim, mu=0.0, L=1.0, num_zeros=0):
    A = torch.rand(n_samples, dim, dim)
    Q, _ = torch.qr(A)
    eigs = torch.zeros(n_samples, dim)
    random_eig = torch.rand(n_samples, dim - num_zeros)
    if random_eig.numel() > 1:
        max_eig = random_eig.max()  # 1, keepdim=True)[0]
        min_eig = random_eig.min()  # 1, keepdim=True)[0]
        random_eig = (random_eig - min_eig) / (max_eig - min_eig)
    else:
        random_eig[:] = 1
    eigs[:, : dim - num_zeros] = mu + random_eig * (L - mu)
    X = torch.bmm(torch.bmm(Q, torch.diag_embed(eigs)), Q.transpose(-1, -2))
    return X


def make_random_matrix(n_samples, dim, mu=0.0, L=1.0, normal=False):
    A = torch.rand(n_samples, dim, dim)
    U, S, V = torch.svd(A)
    S = torch.rand(n_samples, dim)
    if S.numel() > 1:
        S_min = S.min()  # 1, keepdim=True)[0]
        S_max = S.max()  # 1, keepdim=True)[0]
        S = (S - S_min) / (S_max - S_min)
    else:
        S[:] = 1
    S = mu + S * (L - mu)

    if normal:
        V = U.transpose(-2, -1)

    X = torch.bmm(torch.bmm(U, torch.diag_embed(S)), V)
    return X


def make_commutative_matrix(n_samples, dim, mu=(0, 0, 0), L=(1, 1, 1)):
    mu = torch.tensor(mu).view(3, 1, 1)
    L = torch.tensor(L).view(3, 1, 1)

    U = torch.tensor(ortho_group.rvs(dim, size=n_samples), dtype=torch.float32).view(
        n_samples, dim, dim
    )

    s = torch.randn(3, n_samples, dim, dtype=torch.float32)
    s_min, s_max = s.min(-1, keepdim=True)[0], s.max(-1, keepdim=True)[0]
    s = (s - s_min) / (s_max - s_min)
    s = s.view(3, n_samples, dim)
    s = mu + s * (L - mu)

    mask = 2 * (torch.randint_like(s[2], 2) - 0.5)
    s[2] = mask * s[2]
    s = s.float()

    A = torch.bmm(torch.bmm(U, torch.diag_embed(s[0])), U.transpose(-2, -1))
    C = torch.bmm(torch.bmm(U, torch.diag_embed(s[1])), U.transpose(-2, -1))
    B = torch.bmm(torch.bmm(U, torch.diag_embed(s[2])), U.transpose(-2, -1))

    return A, C, B
