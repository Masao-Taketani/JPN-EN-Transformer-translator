# Japanese-English Transformer Translator

## Dataset
[Tanaka Corpus](http://www.edrdg.org/wiki/index.php/Tanaka_Corpus#Downloads)

## Tokenizer
- [SentencePiece](https://github.com/google/sentencepiece)(Used for both Japanese and English.)
- Set word size 8000 for Japanese and English respectively.

## Train
```
python train.py
```

## Translate
```
python translate.py [Japanese Sentence] (e.g. python translate.py お元気ですか。)
```
Or you can execute translation using Jupyter Notebook(`notebooks/test.ipynb`).

## 
## Reference
- Papers
  - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)
- Codes
  - [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)
