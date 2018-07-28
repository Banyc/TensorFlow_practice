import codecs
import sys

RAW_DATA = "./PTB/data/ptb.train.txt"
VOCAB = "./PTB/data/ptb.vocab"
OUTPUT_DATA = "./PTB/data/ptb.train"

with codecs.open(VOCAB, 'r', 'utf-8') as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

f_in = codecs.open(RAW_DATA, 'r', 'utf-8')
f_out = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in f_in:
    words = line.strip().split() + ["<eos>"]
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    f_out.write(out_line)
f_in.close()
f_out.close()
