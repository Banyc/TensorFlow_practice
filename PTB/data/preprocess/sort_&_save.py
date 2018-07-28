import codecs
import collections
from operator import itemgetter

RAW_DATA = "./PTB/data/ptb.train.txt"
VOCAB_OUTPUT = "./PTB/data/ptb.vocab"

counter = collections.Counter()
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():
            counter[word] += 1

sorted_word_to_cnt = sorted(
    counter.items(),
    key=itemgetter(1),
    reverse=True,
)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# char of the signal of the end of a sentence. "End of sentence"
sorted_words = ["<eos>"] + sorted_words

with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
