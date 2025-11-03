import torch
from src.retriever.training_utils import info_nce_loss, encode, get_embeddings, load_tokenizer, load_encoder

def test_info_nce_loss():
    q = torch.randn(4, 768)
    p = torch.randn(4, 768)
    n = torch.randn(4, 768)

    loss = info_nce_loss(q, p, n)
    assert loss.item() > 0
    assert torch.isfinite(loss)

def test_encode_and_embed():
    tok = load_tokenizer("intfloat/multilingual-e5-base")
    model = load_encoder("intfloat/multilingual-e5-base", "cpu")

    batch = ["Привет", "Мир"]
    tokens = encode(tok, batch, 16, "cpu")
    emb = get_embeddings(model, tokens)

    assert emb.shape[0] == 2
    assert emb.shape[1] > 100
