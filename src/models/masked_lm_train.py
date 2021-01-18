from tokenizers.models import BPE

tokenizer = BPE(
    "./models/EsperBERTo-small/vocab.json",
    "./models/EsperBERTo-small/merges.txt",
)
