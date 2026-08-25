# Trains a ByteLevel BPE tokenizer on the corpus and saves it to JSON.
# Byte-level = every byte is a valid token start, so there are no unknown chars.
from tokenizers import Tokenizer, pre_tokenizers, processors, decoders, trainers
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()

special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]

trainer = trainers.BpeTrainer(
    vocab_size=16384,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
)

files = ["fineweb_corpus.txt"]
tokenizer.train(files, trainer)

tokenizer.save("fineweb_tokenizer.json")
print("Tokenizer BPE for fineweb created successfully.")
