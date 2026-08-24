from tokenizers import Tokenizer, pre_tokenizers, processors, decoders, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

trainer = trainers.BpeTrainer(
    vocab_size=8192,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

files = ["tinystories_corpus.txt"]
tokenizer.train(files, trainer)

tokenizer.save("tinystories_tokenizer.json")
print("Tokenizer BPE for Tinystories created successfully.")
