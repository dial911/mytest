from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from dotenv import load_dotenv

load_dotenv()
model = AutoModelForCausalLM.from_pretrained("./qwen0.5b")
tokenizer = AutoTokenizer.from_pretrained("./qwen0.5b")

def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r') as file:
        content = file.read()

    return [chunk for chunk in content.split("\n\n")]


if __name__ == "__main__":
    chunks = split_into_chunks("doc.md")
    print(chunks)
