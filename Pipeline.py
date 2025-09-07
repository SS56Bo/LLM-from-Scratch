# felt cute, might write a small embedding model later
import tiktoken
import fitz
import re
from tqdm import tqdm

class Tokenizer:
    def __init__(self):
        self.tokenizer = tiktoken.encoding_for_model("gpt-2")
        
    def tokenize_strings(self, text:str):
        test = self.tokenizer.encode(text)
        return test
    
    def sliding_window(self, tokens:list[int], con_window=10):
        self.context_window = con_window
        for i in range(0, self.context_window+1):
            context = tokens[:i]
            target = tokens[i]
            print(f"{self.tokenizer.decode(context)} ---> {self.tokenizer.decode([target])}")
    
class Text:
    def __init__(self):
        self.text = ""
    
    def text_format(self, input: str):
        clean = input.replace("\n", " ").strip()
        clean = re.sub(r"[();`]", "", clean)
        return clean

    def extract_text(self, filename: str):
        file_text = fitz.open(filename)
        for i in tqdm(range(len(file_text))):
            pages = file_text[i]
            got_text = self.text_format(pages.get_text())
            self.text += got_text
        return self.text
