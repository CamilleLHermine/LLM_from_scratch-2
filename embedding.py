import torch
import torch.nn.functional as F
import numpy as np
import random
import manual_tokenizer as mt
import data_sampling as ds

def embed_inputs_v1(inputs, vocab_size=50257, output_dim=256):

    batch_size, seq_len = inputs.shape

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)

    pos_embedding_layer = torch.nn.Embedding(seq_len, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(seq_len))

    input_embeddings = token_embeddings + pos_embeddings

    return input_embeddings