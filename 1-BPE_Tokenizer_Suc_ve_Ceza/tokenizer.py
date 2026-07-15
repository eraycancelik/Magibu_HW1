from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=15000, special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)
tokenizer.train(files=["suc_ve_ceza.txt"], trainer=trainer)  # ya da train_from_iterator

output = tokenizer.encode("merhaba dünya")
print(output.ids, output.tokens)

tokenizer.save("tokenizer.json")
