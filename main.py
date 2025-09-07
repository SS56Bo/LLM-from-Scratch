from Pipeline import *

txt = Text()
token = Tokenizer()
raw_text = txt.extract_text("Rough.pdf")
raw_tokens = token.tokenize_strings(raw_text)

print(f"Token length: {len(raw_tokens)}")
print(raw_tokens.type)