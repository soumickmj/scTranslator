import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig

from performer_enc_dec import extract_and_set_enc_dec_kwargs

class PerceiverIOTranslator(nn.Module):
    def __init__(
        self,
        input_emb_dim=128,
        output_emb_dim=128,
        num_input_tokens=20000,
        num_output_tokens=1000,
        n_cross_attn_heads=8,
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
            num_heads=n_cross_attn_heads,        # 8 is usually sufficient; you can tune this
            batch_first=True
        )

        # Output projection to final output embedding dim (128)
        self.output_proj = nn.Linear(self.d_latents, output_emb_dim)

    def forward(self, seq_in):
        batch_size = seq_in.size(0)
        seq_in_proj = self.input_proj(seq_in.transpose(1,2).contiguous())  # [batch, 20000, 768] (.transpose(1,2).contiguous() is required to make it compatible with the existing Enc-Dec code)

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
        return seq_out.transpose(1,2).contiguous() #(.transpose(1,2).contiguous() is required to make it compatible with the existing Enc-Dec code)

def custom_model_forward(self, seq_in, seq_inID, seq_outID, **kwargs):
    #seq_in: RNA values following the shape [batch_size x number of origin RNA genes]
    #seq_inID: RNA gene IDs following the shape [batch_size x number of origin RNA genes]
    #seq_outID: protein IDs following the shape [batch_size x number of origin proteins]
    #number of origin RNA genes and number of origin proteins here are set to 20k and 1k, respectively
    #kwargs: is a dict, containing enc_mask and dec_mask, telling us which tokens to ignore (zero-padded tokens)
    enc_kwargs, dec_kwargs, kwargs = extract_and_set_enc_dec_kwargs(kwargs)
    encodings = self.enc(seq_in, seq_inID, return_encodings = True, **enc_kwargs)# batch_size, input_seq_lenth, dim
    seq_out = self.translator(encodings.transpose(1,2).contiguous()).transpose(1,2).contiguous() # batch_size, out_seq_lenth, dim 
    return encodings, self.dec(seq_out, seq_outID, **dec_kwargs), seq_out
    
def perceiver4translator(model, perceiver, use_IKD=False):
    model.translator = perceiver
    if use_IKD:
        model.forward = custom_model_forward
    return model