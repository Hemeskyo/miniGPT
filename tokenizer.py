from tokenizers import Tokenizer


class BPETokenizerWrapper:
    def __init__(self, json_path="tinystories_tokenizer.json"):
        self.tokenizer = Tokenizer.from_file(json_path)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
