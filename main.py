from Pipeline import Text, Tokenizer

txt = Text()
token = Tokenizer()
raw_text = txt.extract_text("Wow.pdf")
raw_tokens = token.tokenize_strings(raw_text)

token.sliding_window(raw_tokens, 50)