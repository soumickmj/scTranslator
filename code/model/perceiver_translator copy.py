import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig

class PerceiverIOTranslator(nn.Module):
    def __init__(
        self,
        input_emb_dim=128,
        output_emb_dim=128,
        num_input_tokens=20000,
        num_output_tokens=1000,
        perceiver_model_name='deepmind/language-perceiver'
    ):
        super().__init__()
        config = PerceiverConfig.from_pretrained(perceiver_model_name)
        self.input_proj = nn.Linear(input_emb_dim, config.d_model)
        self.perceiver = PerceiverModel.from_pretrained(perceiver_model_name, config=config)

        self.d_latents = config.d_latents
        self.num_output_tokens = num_output_tokens

        # Learnable queries: [num_output_tokens, d_latents]
        self.output_queries = nn.Parameter(torch.randn(num_output_tokens, self.d_latents))

        # Multihead cross-attention: queries attend to latents
        # Queries: [batch, num_output_tokens, d_latents]
        # Keys, Values: [batch, num_latents, d_latents]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.d_latents,
            num_heads=8,        # 8 is usually sufficient; you can tune this
            batch_first=True
        )

        # Output projection to final output embedding dim (128)
        self.output_proj = nn.Linear(self.d_latents, output_emb_dim)

    def forward(self, seq_in):
        batch_size = seq_in.size(0)
        seq_in_proj = self.input_proj(seq_in)  # [batch, 20000, 768]

        # 1. Get Perceiver latents: [batch, num_latents, d_latents]
        perceiver_out = self.perceiver(inputs=seq_in_proj).last_hidden_state  # [batch, 256, 1280]

        # 2. Expand learned output queries to batch: [batch, 1000, 1280]
        queries = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # 3. Cross-attention: output tokens attend to latents
        # Q: queries [batch, num_output_tokens, d_latents]
        # K,V: latents [batch, num_latents, d_latents]
        out, _ = self.cross_attn(
            query=queries,       # [batch, 1000, 1280]
            key=perceiver_out,   # [batch, 256, 1280]
            value=perceiver_out  # [batch, 256, 1280]
        )  # out: [batch, 1000, 1280]

        # 4. Project to target output dim
        seq_out = self.output_proj(out)  # [batch, 1000, 128]
        return seq_out

# Example usage
batch_size = 2
seq_in = torch.randn(batch_size, 20000, 128)
translator = PerceiverIOTranslator()
seq_out = translator(seq_in)
print(seq_out.shape)   # [2, 1000, 128]
