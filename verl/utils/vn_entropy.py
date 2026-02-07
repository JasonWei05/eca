import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VNEntropyCalculator:
    """
    Computes Von Neumann entropy using low-rank eigenvalue optimization.

    Von Neumann entropy: S = -Tr(ρ log ρ) where ρ is the density matrix.

    For a probability distribution over embeddings, ρ = Σ_i p_i |e_i⟩⟨e_i|.
    We use the low-rank structure: eigenvalues of ρ equal eigenvalues of the
    Gram matrix G_ij = sqrt(p_i) sqrt(p_j) ⟨e_i|e_j⟩.

    Optimizations applied:
    1. Top-k token selection
    2. Duality trick for Gram matrix (use smaller of K vs d)
    3. PCA projection to reduce embedding dimension
    """

    device: torch.device
    pca_dim: int
    top_k: int
    embedding_mean: torch.Tensor
    pca_matrix: torch.Tensor | None
    embeddings_proj: torch.Tensor

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        pca_dim: int = 64,
        top_k: int = 64,
    ):
        """
        Args:
            embedding_matrix: (vocab_size, hidden_dim) token embeddings
            pca_dim: Dimension to project embeddings to (default 64)
            top_k: Number of top tokens to consider (default 64)
        """
        logger.info(f"[VNEntropyCalculator] __init__: START (pca_dim={pca_dim}, top_k={top_k})")
        self.device = embedding_matrix.device
        self.pca_dim = pca_dim

        # Validate top_k doesn't exceed vocab size
        vocab_size = embedding_matrix.shape[0]
        self.top_k = min(top_k, vocab_size)
        logger.info(f"[VNEntropyCalculator] __init__: vocab_size={vocab_size}, top_k={self.top_k}")

        embeddings_float = embedding_matrix.float()

        # Store mean for centering (required for correct PCA projection)
        self.embedding_mean = embeddings_float.mean(dim=0)
        centered = embeddings_float - self.embedding_mean

        if pca_dim < embeddings_float.shape[1]:
            # Compute PCA on centered embeddings
            logger.info(f"[VNEntropyCalculator] __init__: computing PCA ({embeddings_float.shape[1]} -> {pca_dim})")
            self.pca_matrix = self._compute_pca(centered, pca_dim)
            logger.info("[VNEntropyCalculator] __init__: PCA computed, projecting embeddings")
            # Project centered embeddings and normalize
            self.embeddings_proj = F.normalize(centered @ self.pca_matrix, dim=-1)
        else:
            self.pca_matrix = None
            self.embeddings_proj = F.normalize(centered, dim=-1)
        logger.info(f"[VNEntropyCalculator] __init__: DONE, embeddings_proj shape {self.embeddings_proj.shape}")

    def _compute_pca(self, centered: torch.Tensor, n_components: int) -> torch.Tensor:
        """Compute PCA projection matrix using randomized SVD.

        Args:
            centered: Already centered embeddings (vocab_size, hidden_dim)
            n_components: Number of principal components

        Returns:
            Projection matrix (hidden_dim, n_components)

        Raises:
            RuntimeError: If both SVD methods fail.
        """
        try:
            U, S, Vh = torch.svd_lowrank(centered.T, q=n_components)
        except RuntimeError as e:
            # svd_lowrank can fail for degenerate/ill-conditioned matrices
            logger.warning(f"svd_lowrank failed ({e}), falling back to full SVD")
            U, S, Vh = torch.linalg.svd(centered.T, full_matrices=False)
            U = U[:, :n_components]

        return U  # (hidden_dim, pca_dim)

    def update_pca(self, new_embedding_matrix: torch.Tensor) -> None:
        """Recompute PCA and projected embeddings.

        Args:
            new_embedding_matrix: Updated embedding matrix (vocab_size, hidden_dim)
        """
        logger.info(f"[VNEntropyCalculator] update_pca: START, shape {new_embedding_matrix.shape}")
        self.device = new_embedding_matrix.device
        embeddings_float = new_embedding_matrix.float()

        # Update mean and center
        self.embedding_mean = embeddings_float.mean(dim=0)
        centered = embeddings_float - self.embedding_mean

        if self.pca_dim < embeddings_float.shape[1]:
            logger.info(f"[VNEntropyCalculator] update_pca: PCA {embeddings_float.shape[1]} -> {self.pca_dim}")
            self.pca_matrix = self._compute_pca(centered, self.pca_dim)
            logger.info("[VNEntropyCalculator] update_pca: projecting embeddings")
            self.embeddings_proj = F.normalize(centered @ self.pca_matrix, dim=-1)
        else:
            self.pca_matrix = None
            self.embeddings_proj = F.normalize(centered, dim=-1)
        logger.info("[VNEntropyCalculator] update_pca: DONE")

    def compute_vn_entropy(
        self,
        logits: torch.Tensor,
        chunk_size: int = 2048,
    ) -> torch.Tensor:
        """
        Compute Von Neumann entropy for a batch of logits using top-k tokens.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            chunk_size: Process this many positions at a time for memory efficiency

        Returns:
            vn_entropy: (batch_size, seq_len)
        """
        logger.info(f"[VNEntropyCalculator] compute_vn_entropy: START, logits {logits.shape}, chunk_size={chunk_size}")
        # Ensure embeddings are on same device as logits
        if self.embeddings_proj.device != logits.device:
            logger.info(f"[VNEntropyCalculator] compute_vn_entropy: moving embeddings to {logits.device}")
            self.embeddings_proj = self.embeddings_proj.to(logits.device)
            self.embedding_mean = self.embedding_mean.to(logits.device)

        K = self.top_k
        d = self.embeddings_proj.shape[1]

        B, L, V = logits.shape
        total_positions = B * L

        logits = logits.detach()
        vn_entropy = torch.zeros(B, L, device=logits.device, dtype=torch.float32)

        # Process one batch item at a time to avoid OOM from full reshape
        num_chunks_per_batch = (L + chunk_size - 1) // chunk_size
        total_chunks = B * num_chunks_per_batch
        logger.info(f"[VNEntropyCalculator] compute_vn_entropy: {total_positions} positions, {total_chunks} chunks")
        print(f"[VNEntropyCalculator] compute_vn_entropy: {total_positions} positions, {total_chunks} chunks")
        global_chunk_idx = 0

        for batch_idx in range(B):
            batch_logits = logits[batch_idx]  # (L, V)

            for chunk_idx, start in enumerate(range(0, L, chunk_size)):
                end = min(start + chunk_size, L)
                chunk_len = end - start
                
                chunk_logits = batch_logits[start:end]  # (chunk_len, V)
                top_logits, top_indices = torch.topk(chunk_logits, K, dim=-1)

                # Softmax over top-k
                probs = F.softmax(top_logits.float(), dim=-1)

                # Gather embeddings: (chunk_len, K, d)
                selected_emb = self.embeddings_proj[top_indices.reshape(-1)].reshape(chunk_len, K, d)

                # Weighted embeddings: w_i = sqrt(p_i) * e_i
                sqrt_probs = torch.sqrt(probs).unsqueeze(-1)  # (chunk_len, K, 1)
                w = selected_emb * sqrt_probs  # (chunk_len, K, d)

                # Duality trick: use smaller Gram matrix
                if d < K:
                    gram = w.transpose(-1, -2) @ w  # (chunk_len, d, d)
                    gram_size = d
                else:
                    gram = w @ w.transpose(-1, -2)  # (chunk_len, K, K)
                    gram_size = K

                # Add small regularization for numerical stability
                eye = torch.eye(gram_size, device=gram.device, dtype=gram.dtype)
                gram = gram + 1e-6 * eye

                # Compute eigenvalues and VN entropy
                eigs = torch.linalg.eigvalsh(gram)

                eigs = torch.clamp(eigs, min=1e-10)
                chunk_entropy = -torch.sum(eigs * torch.log(eigs), dim=-1)

                vn_entropy[batch_idx, start:end] = chunk_entropy

                global_chunk_idx += 1

        logger.info("[VNEntropyCalculator] compute_vn_entropy: DONE")
        print("[VNEntropyCalculator] compute_vn_entropy: DONE")
        return vn_entropy
