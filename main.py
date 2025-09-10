from Pipeline import *
from LLM import *

txt = Text()
token = Tokenizer()

raw_text = txt.extract_text("Rough.pdf")
raw_tokens = token.tokenize_strings(raw_text)

print(f"Token length: {len(raw_tokens)}")
result_data = GPTDataLoader.create_dataloader(text=raw_tokens, batch_size=8, max_length=4, stride=4 ,shuffle=False)

data_iter = iter(result_data)
inputs, targets = next(data_iter)
embedding_model = EmbedText(50257, 256)

print(f"Embedding model size: {embedding_model.convert_tokens_to_embeddings(input_text=inputs).shape}")