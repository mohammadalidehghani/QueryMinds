# Bug Report

No critical format errors or parsing issues were detected during the processing and validation of `.conllu` files.
All files were successfully loaded and processed without interruption to the pipeline.

## Minor observed irregularities
- Mathematical expressions and LaTeX-style symbols (e.g. `∈`, `η`, `{}`, `→`, `≤`) are often tokenized as multiple separate tokens.
- Some files have long sentences due to unrecognized punctuation boundaries.
- Several technical terms (e.g. model names, abbreviations, algorithm names) are incorrectly marked as \texttt{NOUN} instead of \texttt{PROPN}.
- Several files with complex formulas have seen an increase in the number of tokens without any structural errors.

## Impact Analysis
These phenomena are considered minor tokenization artifacts that do not affect the overall functionality of the corpus.
The errors are systematic in nature, related to symbols and terms from scientific fields, and can be corrected in future phases through additional segmentation rules and customized lexical resources.

## Conclusion
The corpus is valid, stable, and ready for Milestone 2.
All identified irregularities are minor and do not affect the integrity or usability of the data.
It is recommended to introduce a customized tokenizer for mathematical expressions and recognize domain-specific names to further increase the accuracy of annotations.