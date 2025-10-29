
import urllib.request
import re
import tiktoken

'''1. importing the data'''

url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")
file_path = "the-verdict.txt"
urllib.request.urlretrieve(url,file_path)

with open("the-verdict.txt","r", encoding="utf-8") as f:
    raw_text=f.read()

print('number of characters: ', len(raw_text))
print(raw_text[:99])

''' 2. Tokenizing'''

class ManualTokenizer:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}

    def encode(self, text):
        split_text = re.split(r'([--|,.:;?_!"()\|\s])', raw_text)
        split_text = [item.strip() for item in split_text if item.strip()]
        split_text = [item if item in self.str_to_int else "<|unknown|>" for item in split_text]
        ids=[self.str_to_int[s] for s in split_text]
        return ids
    
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.:;!?"()\'])',r'\1', text)
        return text


''' 3. Building vocabulary and converting tokens into IDs'''

vocabulary = sorted(list(set(raw_text)))
vocabulary.extend(["<|endoftext|>", "<|unknown|>"])

vocab_size = len(vocabulary)
print(vocab_size)

vocab = {token:integer for integer, token in enumerate(vocabulary)}

tokenizer=ManualTokenizer(vocab)
ids=tokenizer.encode(raw_text)

''' alternative tokenizer '''

tokenizerBPE = tiktoken.get_encoding("gpt2")






