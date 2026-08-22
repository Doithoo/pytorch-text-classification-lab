from text_classifier.data.tokenizer import SimpleWordTokenizer

texts = ["GPU training is useful.", "NLP uses Unicode text: 中文。"]
tokenizer = SimpleWordTokenizer.build(texts, vocab_size=32, min_frequency=1, max_length=16)
for text in texts:
    print(text)
    print(tokenizer.tokenize(text))
    print(tokenizer.encode(text))
print(tokenizer.metadata())
