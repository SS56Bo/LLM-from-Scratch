from Pipeline import *
from LLM import *

txt = Text()
token = Tokenizer()

raw_text = txt.extract_text("Rough.pdf")
raw_tokens = token.tokenize_strings(raw_text)

print(f"Token length: {len(raw_tokens)}")
result_data = GPTDataLoader.create_dataloader(text=raw_tokens, batch_size=8, max_length=4, stride=4 ,shuffle=False)
result_iter = iter(result_data)
first_batch = next(result_iter)
second_batch = next(result_iter)
print(first_batch)
print(second_batch)