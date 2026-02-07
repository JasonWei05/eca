"""
Token-level entropy heatmap visualization.
Renders each token as a colored box (blue=low, red=high entropy).
Produces heatmaps for Shannon + 3 VN entropy configs on first 2 AIME responses.
"""

import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
from safetensors import safe_open

MODEL_PATH = "/home/tiger/jason/eca/Qwen3-4b-Instruct"
CACHE_FILE = "/home/tiger/jason/eca/aime_outputs.json"
TOP_K = 64
PCA_DIMS = [32, 64, 128]
NUM_PROBLEMS = 2


def compute_pca_matrix(embeddings, pca_dim):
    centered = embeddings - embeddings.mean(dim=0)
    _, _, V = torch.pca_lowrank(centered, q=pca_dim)
    return V


def compute_vn_entropy(logits_seq, emb_normalized, pca_matrices, top_k, pca_dim):
    V_pca = pca_matrices[pca_dim]
    proj_emb = F.normalize(emb_normalized @ V_pca, dim=-1)
    top_logits, top_indices = torch.topk(logits_seq, top_k, dim=-1)
    probs = F.softmax(top_logits.float(), dim=-1)
    sel_emb = proj_emb[top_indices]
    w = torch.sqrt(probs).unsqueeze(-1) * sel_emb
    K, d = w.shape[-2], w.shape[-1]
    if d < K:
        gram = w.transpose(-1, -2) @ w
    else:
        gram = w @ w.transpose(-1, -2)
    gram = gram + 1e-6 * torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    eigs = torch.clamp(torch.linalg.eigvalsh(gram), min=1e-10)
    return -torch.sum(eigs * torch.log(eigs), dim=-1)


def render_heatmap(tokens, values, title, output_path, logits_seq=None, tokenizer=None, cmap_name="coolwarm"):
    """Render a token heatmap where each token is a colored box with text."""
    # Layout params
    fig_width = 20
    x_pad = 0.15
    y_pad = 0.08
    box_height = 0.45
    font_size = 7
    char_width = 0.12  # approx width per character

    # Determine high-entropy threshold (top 20%)
    threshold = np.percentile(values, 80)

    # Build display strings: for high-entropy tokens, show top-5 with probabilities
    display_tokens = []
    for i, tok in enumerate(tokens):
        if values[i] >= threshold and logits_seq is not None and tokenizer is not None:
            top5 = torch.topk(logits_seq[i], 5, dim=-1)
            top5_probs = F.softmax(logits_seq[i].float(), dim=-1)[top5.indices].tolist()
            parts = []
            for tid, p in zip(top5.indices.tolist(), top5_probs):
                alt = tokenizer.decode([tid]).strip()
                if not alt:
                    alt = "·"
                alt = alt.replace("$", "\\$").replace("\n", "\\n")
                parts.append(f"{alt}:{p:.2f}")
            display_tokens.append(f"{tok} ({', '.join(parts)})")
        else:
            display_tokens.append(tok)

    # Compute box widths based on display token text length
    box_widths = []
    for dtok in display_tokens:
        w = max(len(dtok) * char_width, 0.4)
        box_widths.append(w)

    # Layout tokens into rows
    rows = []
    current_row = []
    current_x = x_pad
    max_x = fig_width - x_pad
    for i, (dtok, bw) in enumerate(zip(display_tokens, box_widths)):
        if current_x + bw > max_x and current_row:
            rows.append(current_row)
            current_row = []
            current_x = x_pad
        current_row.append((i, dtok, current_x, bw))
        current_x += bw + 0.08
    if current_row:
        rows.append(current_row)

    n_rows = len(rows)
    fig_height = max(3, n_rows * (box_height + y_pad) + 1.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, fig_width)
    ax.set_ylim(0, fig_height)
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=10)

    vmin, vmax = np.min(values), np.max(values)
    if vmin == vmax:
        vmax = vmin + 1
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    for row_idx, row in enumerate(rows):
        y = fig_height - 1.0 - row_idx * (box_height + y_pad)
        for (tok_idx, dtok, x, bw) in row:
            color = cmap(norm(values[tok_idx]))
            rect = FancyBboxPatch(
                (x, y - box_height / 2), bw, box_height,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=(0.8, 0.8, 0.8, 0.5), linewidth=0.3
            )
            ax.add_patch(rect)
            # Choose text color for readability
            lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            text_color = "white" if lum < 0.5 else "black"
            # Escape $ to prevent matplotlib mathtext parsing
            dtok = dtok.replace("$", "\\$")
            ax.text(
                x + bw / 2, y, dtok,
                ha="center", va="center", fontsize=font_size,
                color=text_color, fontfamily="monospace",
                clip_on=True
            )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Token Entropy", fontsize=10)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(CACHE_FILE) as f:
        cache_data = json.load(f)[:NUM_PROBLEMS]
    print(f"Processing {len(cache_data)} problems")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Embeddings for VN entropy
    with safe_open(f"{MODEL_PATH}/model-00001-of-00003.safetensors", framework="pt") as f:
        raw_embeddings = f.get_tensor("model.embed_tokens.weight").float().cuda()
    centered = raw_embeddings - raw_embeddings.mean(dim=0)
    emb_normalized = F.normalize(centered, dim=-1)

    pca_matrices = {}
    for pd in PCA_DIMS:
        pca_matrices[pd] = compute_pca_matrix(raw_embeddings, pd)

    for prob_idx, item in enumerate(cache_data):
        full_text = item["prompt"] + item["generated_text"]
        input_ids = tokenizer(full_text, return_tensors="pt")["input_ids"].to(model.device)
        prompt_ids = tokenizer(item["prompt"], return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]
        gen_len = input_ids.shape[1] - prompt_len

        print(f"\nProblem {prob_idx+1}: {gen_len} generated tokens")

        # Get token strings for the generated portion
        gen_token_ids = input_ids[0, prompt_len:]
        token_strs = []
        for tid in gen_token_ids:
            t = tokenizer.decode([tid.item()])
            # Clean up for display
            t = t.replace("\n", "\\n")
            if not t.strip():
                t = "·"
            token_strs.append(t)

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
        logits_seq = outputs.logits[0, prompt_len - 1:-1].float()

        # Shannon entropy
        probs_full = F.softmax(logits_seq, dim=-1)
        shannon = -torch.sum(probs_full * torch.log(probs_full + 1e-10), dim=-1)
        shannon_np = shannon.cpu().numpy()

        render_heatmap(
            token_strs, shannon_np,
            f"Problem {prob_idx+1} — Shannon Entropy",
            f"/home/tiger/jason/eca/heatmap_p{prob_idx+1}_shannon.png",
            logits_seq=logits_seq, tokenizer=tokenizer
        )

        # VN entropy for each PCA dim
        for pd in PCA_DIMS:
            chunk_size = 2048
            vn_chunks = []
            for start in range(0, gen_len, chunk_size):
                end = min(start + chunk_size, gen_len)
                vn = compute_vn_entropy(logits_seq[start:end], emb_normalized, pca_matrices, TOP_K, pd)
                vn_chunks.append(vn.cpu().numpy())
            vn_np = np.concatenate(vn_chunks)

            render_heatmap(
                token_strs, vn_np,
                f"Problem {prob_idx+1} — VN Entropy (k={TOP_K}, pca={pd})",
                f"/home/tiger/jason/eca/heatmap_p{prob_idx+1}_vn_k{TOP_K}_pca{pd}.png",
                logits_seq=logits_seq, tokenizer=tokenizer
            )


if __name__ == "__main__":
    main()
