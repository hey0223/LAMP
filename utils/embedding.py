import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Model / device setup
# ============================================================

MODEL_PATH = os.environ.get(
    "MINICPM_EMBED_MODEL_PATH",
    "./MiniCPM-Embedding",
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False,
    local_files_only=True,
)

model = AutoModel.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=MODEL_DTYPE,
    local_files_only=True,
).to(DEVICE)
model.eval()

HIDDEN_SIZE = getattr(model.config, "hidden_size", 2304)


# ============================================================
# Core embedding utilities
# ============================================================


def _minicpm_sentence_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, D]
    counts = mask.sum(dim=1).clamp(min=1.0)                          # [B, 1]
    return summed / counts


def _format_query_text(text: str, instruction: str = "") -> str:
    text = "" if text is None else str(text).strip()
    instruction = "" if instruction is None else str(instruction).strip()
    if instruction:
        return f"Instruction: {instruction}\nQuery: {text}"
    return f"Query: {text}"


def _format_document_text(text: str) -> str:
    return "" if text is None else str(text).strip()



def _encode_texts(texts: List[str]) -> torch.Tensor:
    if len(texts) == 0:
        return torch.empty((0, HIDDEN_SIZE), device=DEVICE, dtype=torch.float32)

    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    with torch.no_grad():
        if DEVICE.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(**tokens)
                pooled = _minicpm_sentence_pool(outputs.last_hidden_state, tokens["attention_mask"])
        else:
            outputs = model(**tokens)
            pooled = _minicpm_sentence_pool(outputs.last_hidden_state, tokens["attention_mask"])

    pooled = pooled.float()
    return pooled



def get_query_embedding(text: str, instruction: str = "") -> torch.Tensor:
    formatted = _format_query_text(text, instruction=instruction)
    return _encode_texts([formatted]).squeeze(0)



def batch_get_query_embeddings(texts: List[str], instruction: str = "") -> torch.Tensor:
    formatted = [_format_query_text(t, instruction=instruction) for t in texts]
    return _encode_texts(formatted)



def get_document_embedding(text: str) -> torch.Tensor:
    formatted = _format_document_text(text)
    return _encode_texts([formatted]).squeeze(0)



def batch_get_document_embeddings(texts: List[str]) -> torch.Tensor:
    formatted = [_format_document_text(t) for t in texts]
    return _encode_texts(formatted)


# ============================================================
# 32D compressed embedding path restored for statement selection
# ============================================================

embedding_projection_hidden_256 = nn.Linear(HIDDEN_SIZE, 256, dtype=MODEL_DTYPE).to(DEVICE)
embedding_compressor_256_32 = nn.Linear(256, 32, dtype=MODEL_DTYPE).to(DEVICE)
embedding_compressor_32_5 = nn.Linear(32, 5, dtype=MODEL_DTYPE).to(DEVICE)

for module in [
    embedding_projection_hidden_256,
    embedding_compressor_256_32,
    embedding_compressor_32_5,
]:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False



def get_embedding(text: str) -> torch.Tensor:
    """
    Restored original selector behavior:
      text -> high-dimensional MiniCPM embedding -> 256D -> 32D
    Returns shape [32].
    """
    doc_emb = get_document_embedding(text).to(device=DEVICE, dtype=MODEL_DTYPE).unsqueeze(0)  # [1, H]
    with torch.no_grad():
        emb_256 = embedding_projection_hidden_256(doc_emb)   # [1, 256]
        emb_32 = embedding_compressor_256_32(emb_256)        # [1, 32]
    return emb_32.squeeze(0)



def batch_get_embeddings(texts: List[str]) -> torch.Tensor:
    """
    Batch version of the restored 32D selector embedding path.
    Returns shape [B, 32].
    """
    if len(texts) == 0:
        return torch.empty((0, 32), device=DEVICE, dtype=MODEL_DTYPE)

    doc_embs = batch_get_document_embeddings(texts).to(device=DEVICE, dtype=MODEL_DTYPE)  # [B, H]
    with torch.no_grad():
        emb_256 = embedding_projection_hidden_256(doc_embs)  # [B, 256]
        emb_32 = embedding_compressor_256_32(emb_256)        # [B, 32]
    return emb_32


def Household_embed(evaluation: str, obs_text: str) -> torch.Tensor:
    """
    Original household embedding path restored:
      evaluation + obs_text -> 32D embedding -> layer norm -> 5D projection
    Returns shape [5].
    """
    combined_text = f"\n{evaluation}\n\n=== OBSERVATION REASONING ===\n{obs_text}\n"
    household_embed = get_embedding(combined_text)
    household_embed = F.layer_norm(household_embed.unsqueeze(0), (32,)).squeeze(0)
    with torch.no_grad():
        household_embed_5d = embedding_compressor_32_5(household_embed.unsqueeze(0)).squeeze(0)
    return household_embed_5d


# ============================================================
# Restored original numeric-feature statement selector
# ============================================================


def encode_player_vector(
    player_id: int,
    economic_status: int,
    personal_labor_productivity: float,
    personal_wealth: float,
    step: int,
    wealth_guesses: list,
    trust_levels: list,
) -> torch.Tensor:
    """
    Generate a numeric observation vector of shape (56,):
      - player ID one-hot: 10
      - economic status one-hot: 3
      - labor productivity: 1
      - wealth: 1
      - step: 1
      - wealth guesses one-hot: 10 x 3 = 30
      - trust levels: 10
    Total = 56
    """
    device = DEVICE
    dtype = MODEL_DTYPE

    clamped_player_id = int(max(0, min(int(player_id), 9)))
    id_one_hot = F.one_hot(torch.tensor(clamped_player_id, device=device), num_classes=10).to(dtype)

    clamped_econ = int(max(0, min(int(economic_status), 2)))
    econ_one_hot = F.one_hot(torch.tensor(clamped_econ, device=device), num_classes=3).to(dtype)

    step_norm = torch.tensor([float(step) / 100.0], dtype=dtype, device=device)
    labor_norm = torch.tensor(
        [float(personal_labor_productivity) / (float(personal_labor_productivity) + 1.0)],
        dtype=dtype,
        device=device,
    )
    wealth_norm = torch.tensor(
        [float(personal_wealth) / (float(personal_wealth) + 1e5)],
        dtype=dtype,
        device=device,
    )

    wealth_guesses = list(wealth_guesses)[:10] + [1] * max(0, 10 - len(list(wealth_guesses)))
    wg_tensor = torch.tensor(wealth_guesses[:10], dtype=torch.long, device=device)
    wg_tensor = torch.clamp(wg_tensor, 0, 2)
    wealth_guesses_one_hot = F.one_hot(wg_tensor, num_classes=3).to(dtype).flatten()

    trust_levels = list(trust_levels)[:10] + [5.0] * max(0, 10 - len(list(trust_levels)))
    trust_levels_norm = torch.tensor(trust_levels[:10], dtype=dtype, device=device) / 10.0

    player_vector = torch.cat(
        [
            id_one_hot,
            econ_one_hot,
            labor_norm,
            wealth_norm,
            step_norm,
            wealth_guesses_one_hot,
            trust_levels_norm,
        ],
        dim=0,
    )
    return player_vector


class PlayerEncoder(nn.Module):
    def __init__(self, input_dim=56, hidden_dim=32, output_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=MODEL_DTYPE)
        self.fc2 = nn.Linear(hidden_dim, output_dim, dtype=MODEL_DTYPE)
        self.layer_norm = nn.LayerNorm(output_dim, dtype=MODEL_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return self.layer_norm(out)


class ResidualSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dtype=MODEL_DTYPE,
        ).to(DEVICE)
        self.layer_norm = nn.LayerNorm(embed_dim, dtype=MODEL_DTYPE).to(DEVICE)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_out, attention_weights = self.self_attention(x, x, x)
        out = self.layer_norm(x + attention_out)
        return out, attention_weights


class CriticHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=MODEL_DTYPE)
        self.fc2 = nn.Linear(hidden_dim, 1, dtype=MODEL_DTYPE)
        self.layer_norm = nn.LayerNorm(hidden_dim, dtype=MODEL_DTYPE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)
        return self.fc2(x)


class SelfAttentionPolicy(nn.Module):
    """
    Restored original selector logic:
    numeric household features + observation text embedding + candidate statement embeddings
    -> self-attention -> state embedding -> candidate distribution.
    """
    def __init__(self, player_dim=56, embed_dim=32, num_heads=4):
        super().__init__()
        self.player_encoder = PlayerEncoder(
            input_dim=player_dim,
            hidden_dim=32,
            output_dim=embed_dim,
        )
        self.self_attention = ResidualSelfAttention(embed_dim, num_heads)
        self.critic_head = CriticHead(input_dim=embed_dim, hidden_dim=max(1, embed_dim // 2))
        self.embed_dim = embed_dim

    def forward(
        self,
        player_vec: torch.Tensor,         # [56]
        obs_embedding: torch.Tensor,      # [32]
        action_embeddings: torch.Tensor,  # [K, 32]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        player_embed = self.player_encoder(player_vec).unsqueeze(0)   # [1, 32]
        obs_embed = obs_embedding.unsqueeze(0)                        # [1, 32]

        all_inputs = torch.cat([player_embed, obs_embed, action_embeddings], dim=0).unsqueeze(0)  # [1, 2+K, 32]
        attended, attention_weights = self.self_attention(all_inputs)  # attended: [1, 2+K, 32]
        attended = attended.squeeze(0)                                 # [2+K, 32]

        state_embed = torch.mean(attended[:2], dim=0)                  # [32]
        value = self.critic_head(state_embed)                          # [1]

        action_vecs = attended[2:]                                     # [K, 32]
        action_logits = torch.matmul(state_embed, action_vecs.T)       # [K]
        action_probs = F.softmax(action_logits.float(), dim=-1)        # [K]

        attn_matrix = attention_weights.squeeze(0).float()             # [2+K, 2+K]
        return action_probs, action_logits.float(), attn_matrix


_POLICY_MODEL_CACHE = None



def run_policy_inference(
    player_id: int,
    economic_status: int,
    personal_labor_productivity: float,
    personal_wealth: float,
    step: int,
    wealth_guesses: list,
    trust_levels: list,
    obs_text: str,
    candidate_action_texts: list,
    player_dim: int = 56,
    embed_dim: int = 32,
    num_heads: int = 4,
    temperature: float = 1.0,
):
    """
    Hybrid compatibility wrapper.

    Internally uses the original selector structure, but returns the 29-version
    triple expected by utils/dialogue.py:
        probs, scores, attention_weights
    """
    del temperature  # kept only for backward compatibility with 29-version call sites

    global _POLICY_MODEL_CACHE

    try:
        if candidate_action_texts is None or len(candidate_action_texts) == 0:
            empty = torch.empty(0, dtype=torch.float32, device=DEVICE)
            empty_attn = torch.empty((0, 0), dtype=torch.float32, device=DEVICE)
            return empty, empty, empty_attn

        if _POLICY_MODEL_CACHE is None:
            _POLICY_MODEL_CACHE = SelfAttentionPolicy(
                player_dim=player_dim,
                embed_dim=embed_dim,
                num_heads=num_heads,
            ).to(DEVICE)
            _POLICY_MODEL_CACHE.eval()

        policy_model = _POLICY_MODEL_CACHE

        player_vec = encode_player_vector(
            player_id=player_id,
            economic_status=economic_status,
            personal_labor_productivity=personal_labor_productivity,
            personal_wealth=personal_wealth,
            step=step,
            wealth_guesses=wealth_guesses,
            trust_levels=trust_levels,
        ).to(device=DEVICE, dtype=MODEL_DTYPE)

        obs_embedding = get_embedding(obs_text).to(device=DEVICE, dtype=MODEL_DTYPE)
        action_embeddings = batch_get_embeddings(candidate_action_texts).to(device=DEVICE, dtype=MODEL_DTYPE)

        action_probs, action_logits, attention_weights = policy_model(
            player_vec,
            obs_embedding,
            action_embeddings,
        )
        return action_probs, action_logits, attention_weights

    except Exception as e:
        print(f"Error in policy inference: {str(e)}")
        raise
