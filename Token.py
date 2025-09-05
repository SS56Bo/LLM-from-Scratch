# Feeling cute, might learn embeddings later on

#[text]
training_set = "The quick brown fox jumps over the lazy dog!"

tokens = training_set.encode("utf-8")
tokens = list(map(int, tokens))
print(tokens)

class Tokenizer:
    def __init__(self):
        self.token = []

    def convert_to_utf(self, data: str)->list:
        self.token = data.encode("utf-8")
        self.token = list(map(int, tokens))
        return self.token
