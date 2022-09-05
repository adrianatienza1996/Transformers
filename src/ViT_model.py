import torch
import torch.nn as nn

import math
from einops.layers.torch import Rearrange

from einops import repeat


# The iniput size will be (224, 224)
# We will separate the input in (16, 16) patch size

class Patch_Embeddings(nn.Module):
    def __init__(self, s1=16, s2=16, model_dim=512, in_channels=3, conv_projection=True):
        super(Patch_Embeddings, self).__init__()
        # According papers, projector can be both Linear Layer or Conv Layer
        self.conv_projection = conv_projection

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, model_dim, kernel_size=(s1, s2), stride=(s1, s2)),
            Rearrange('b d p1 p2 -> b (p1 p2) d')) if self.conv_projection == True \
        \
        else nn.Sequential(
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=s1, s2=s2),
            nn.Linear(s1 * s2 * in_channels, model_dim))

        self.class_token = nn.Parameter(torch.randn(1, 1, model_dim))

    def forward(self, x):

        h = self.net(x)
        
        # Cloning the class embedding token for every object in the batch (p = # patches, d = model dimension)
        cls_token = repeat(self.class_token, '() p d -> b p d', b=x.shape[0])
        h = torch.cat([cls_token, h], dim=1)
        return h


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim=512, max_seq_len=5000, do_prob=0.1):

        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        self.dropout = nn.Dropout(do_prob)
        self.model_dim = model_dim

        position_id = torch.arange(0, max_seq_len).unsqueeze(1)
        frequencies = torch.pow(10000., -torch.arange(0, model_dim, 2, dtype=torch.float) / model_dim)
        positional_encodings = torch.zeros(max_seq_len, model_dim)
        positional_encodings[:, 0::2] = torch.sin(position_id * frequencies)  
        positional_encodings[:, 1::2] = torch.cos(position_id * frequencies)  
        self.register_buffer('positional_encodings', positional_encodings)

        

    def forward(self, tokens):
        embeddings = self.embedding(tokens) * math.sqrt(self.model_dim)
        tmp_encoding = self.positional_encodings[:embeddings.shape[1]]
        return self.dropout(embeddings + tmp_encoding)



class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, input_size=(224, 224), patch_size=(16, 16), model_dim=512):
        super(LearnablePositionalEmbedding, self).__init__()
        self.positional_encodings = nn.Parameter(torch.randn((input_size[0] // patch_size[0]) * (input_size[1] // patch_size[1]) + 1, model_dim))

    def forward(self, x):
        return x + self.positional_encodings