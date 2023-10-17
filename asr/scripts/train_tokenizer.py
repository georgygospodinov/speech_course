import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/texts.txt",
    model_prefix='data/tokenizer/bpe_1024_bos_eos',
    vocab_size=1024,
    model_type="bpe",
    user_defined_symbols=["<s>","</s>"]
)
