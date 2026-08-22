# AG News Dataset Note

[中文](ag-news.zh-CN.md) | [Documentation index](../README.md)

AG News is a news-topic classification dataset. This project uses the CSV mirror in `mhjabreel/CharCnn_Keras`, containing 120,000 training-source rows and 7,600 test rows balanced across four classes. The project draws 10% of the training source per class for validation.

Original paper citation:

```bibtex
@inproceedings{zhang2015character,
  title={Character-level Convolutional Networks for Text Classification},
  author={Zhang, Xiang and Zhao, Junbo and LeCun, Yann},
  booktitle={Advances in Neural Information Processing Systems},
  year={2015}
}
```

The data originates from AG's corpus of news articles. Terms for the mirror and source data are not the same as this project's MIT source-code license. Before using, publishing, or redistributing raw text, derived vocabulary, or error samples, verify upstream terms, applicable law, and organizational policy.

The repository does not commit raw train/test CSV, checkpoints, full tokenizer files, or per-example errors. Recorded runs retain aggregate evidence; if organization policy permits, detailed error text and tokenizer can be obtained from Kaggle output. Before publishing derived vocabulary or source text, verify upstream terms and privacy.

Fixed URLs and SHA-256 detect upstream byte changes. A checksum proves byte identity; it does not grant usage rights.
