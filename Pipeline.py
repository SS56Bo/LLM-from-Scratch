import tiktoken

tokenizer = tiktoken.encoding_for_model("gpt-2")
test = tokenizer.encode("The quick brown fox jumps over the lazy dog! Another one")
print("-------")
print(test)
print("-------")
print(tokenizer.decode(test))