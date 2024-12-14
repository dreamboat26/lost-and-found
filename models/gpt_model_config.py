from models.helper import Config

# class GPTConfig:
#     vocab_size = 40000
#     emb_dim = 512
#     max_seq_len = 512
#     num_heads = 8
#     drop_prob = 0.1
#     ff_multiplier = 4
#     num_blocks = 12


class GPTConfig(Config):
    """
    Model Config for GPT model.
    """

    vocab_size = 50257
    emb_dim = 128
    num_heads = 4
    drop_prob = 0.1
    ff_multiplier = 1
    num_blocks = 2


if __name__ == "__main__":
    g = GPTConfig()
    g.print_config()
