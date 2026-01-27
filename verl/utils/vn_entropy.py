import torch
import torch.nn.functional as F


class VNEntropyCalculator:
    """
    Computes Von Neumann entropy using low-rank eigenvalue optimization.
    
    Optimizations applied:
    1. Top-p/Top-k token selection
    2. Low-rank eigenvalue computation (exploit rank-k structure)
    3. PCA projection (optional)
    """

    def __init__(self, embedding_matrix: torch.Tensor, pca_dim: int = 512, use_pca: bool = True):
        self.device = embedding_matrix.device
        self.embeddings = F.normalize(embedding_matrix.float(), dim=-1)
        self.pca_dim = pca_dim
        self.use_pca = use_pca
        
        if self.use_pca and self.pca_dim < self.embeddings.shape[1]:
            self.pca_matrix = self._compute_pca(embedding_matrix.float(), pca_dim)
            self.embeddings_proj = self.embeddings @ self.pca_matrix
        else:
            self.pca_matrix = None
            self.embeddings_proj = self.embeddings

    def _compute_pca(self, embeddings: torch.Tensor, n_components: int) -> torch.Tensor:
        """Compute PCA projection matrix using randomized SVD."""
        mean = embeddings.mean(dim=0, keepdim=True)
        centered = embeddings - mean
        try:
            # randomized SVD is faster
            U, S, Vh = torch.svd_lowrank(centered.T, q=n_components)
            return U
        except Exception:
            # Fallback
            U, S, Vh = torch.linalg.svd(centered.T, full_matrices=False)
            return U[:, :n_components]

    def update_embeddings(self, new_embedding_matrix: torch.Tensor):
        """Update embedding matrix (call after actor training step if embeddings change, or just update PCA)."""
        self.device = new_embedding_matrix.device
        self.embeddings = F.normalize(new_embedding_matrix.float(), dim=-1)
        if self.use_pca and self.pca_matrix is not None:
            self.embeddings_proj = self.embeddings @ self.pca_matrix
        else:
            self.embeddings_proj = self.embeddings

    def update_pca(self, new_embedding_matrix: torch.Tensor):
        """Recompute PCA projection (call every N steps)."""
        self.device = new_embedding_matrix.device
        if self.use_pca and self.pca_dim < new_embedding_matrix.shape[1]:
            self.pca_matrix = self._compute_pca(new_embedding_matrix.float(), self.pca_dim)
            self.embeddings = F.normalize(new_embedding_matrix.float(), dim=-1)
            self.embeddings_proj = self.embeddings @ self.pca_matrix
        else:
            self.embeddings = F.normalize(new_embedding_matrix.float(), dim=-1)
            self.embeddings_proj = self.embeddings

    def compute_vn_entropy(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """
        Compute VN entropy for a batch of logits.

        Args:
            logits: (batch_size, seq_len, vocab_size)
            top_p: float in [0, 1], e.g. 0.99 means top 99% of probability mass

        Returns:
            vn_entropy: (batch_size, seq_len)
        """
        probs = F.softmax(logits.float(), dim=-1)
        
        # We need to handle batch dimensions. 
        # For efficiency, we can process flattened batch if needed, but let's try to keep dims.
        # However, top-p filtering is tricky with variable cutoff per position.
        # The design doc suggests top-k (fixed k) is easier for tensor ops.
        # But user asked for top-p.
        
        # To make it vectorized with top-p, we usually take top-k that covers p for most, 
        # or just use a fixed large k (e.g. 50-100) as approximation.
        # The user said "top-p".
        
        # Let's implement per-position logic if we can't vectorize easily, 
        # OR use a fixed large K (e.g. 100) and then mask out those beyond top-p?
        # But masked embeddings are tricky for low-rank.
        
        # Actually, the original implementation in entropy_comparison.py processes ONE logit vector at a time?
        # "vn_calculator.compute_vn_entropy(lg, top_p) for lg in logits_list"
        # Yes, it iterates. That's slow for training.
        
        # We should vectorize.
        # Strategy: Take top-K (e.g. 100). 
        # If top-p cutoff is smaller than K, zero out the rest.
        # Renormalize.
        
        K = 500  # Upper bound for top-p filtering
        
        top_probs, top_indices = torch.topk(probs, K, dim=-1)
        # top_probs: (B, L, K)
        
        cumsum = torch.cumsum(top_probs, dim=-1)
        mask = cumsum <= top_p
        # Include the first token that exceeds threshold
        # (cumsum shift right, fill 0, < top_p)
        mask = torch.cat([torch.ones_like(mask[..., :1]), mask[..., :-1]], dim=-1)
        
        # Zero out probs outside mask
        top_probs = top_probs * mask
        top_probs = top_probs / (top_probs.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Get embeddings
        # self.embeddings_proj: (V, D)
        # top_indices: (B, L, K)
        # We need to gather: (B, L, K, D)
        B, L, K_dim = top_indices.shape
        flat_indices = top_indices.view(-1)
        selected_embeddings = self.embeddings_proj[flat_indices].view(B, L, K_dim, -1)
        
        # Low-rank calc
        sqrt_probs = torch.sqrt(top_probs).unsqueeze(-1) # (B, L, K, 1)
        weighted_embeddings = selected_embeddings * sqrt_probs # (B, L, K, D)
        
        # Gram matrix: (B, L, K, D) @ (B, L, D, K) -> (B, L, K, K)
        gram_matrix = torch.matmul(weighted_embeddings, weighted_embeddings.transpose(-1, -2))
        
        # Eigenvalues
        # batch eigvalsh
        # Reshape to (B*L, K, K)
        gram_flat = gram_matrix.view(-1, K_dim, K_dim)
        eigenvalues = torch.linalg.eigvalsh(gram_flat)
        eigenvalues = torch.clamp(eigenvalues, min=1e-10)
        
        vn_entropy = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)
        vn_entropy = vn_entropy.view(B, L)
        
        return vn_entropy
