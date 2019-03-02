import torch.nn as nn
import torch.nn.functional as F
import copy

from model_build import attention, decoder, embeddings, encoder, feed_forward, positional_encoding


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = attention.MultiHeadedAttention(h, d_model)
    ff = feed_forward.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = positional_encoding.PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        encoder.Encoder(encoder.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder.Decoder(decoder.DecoderLayer(d_model, c(attn), c(attn),
                                             c(ff), dropout), N),
        nn.Sequential(embeddings.Embeddings(d_model, src_vocab + 1), c(position)),
        nn.Sequential(embeddings.Embeddings(d_model, tgt_vocab + 1), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


