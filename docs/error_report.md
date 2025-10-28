# Error Report

No critical format or parsing errors were detected across the processed `.conllu` files.

### Minor observations:
- Mathematical expressions and LaTeX-style symbols (e.g., `∈`, `η`, `{}`) were split into multiple tokens.
- Some files contain long single sentences due to unrecognized punctuation boundaries.
- A few technical terms were misclassified as NOUN instead of PROPN (proper noun).

These are minor tokenization artifacts, not structural errors.  
Overall dataset is valid and ready for Milestone 2

