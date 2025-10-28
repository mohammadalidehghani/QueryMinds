# Quality Control Notes

During manual inspection of 20 randomly selected `.conllu` files, all files were found to be valid and correctly formatted.

- Each file contains proper CoNLL-U structure (10 columns, `# text` and `# sent_id` metadata).
- Lemmatization and POS tagging are accurate.
- Sentence segmentation is generally consistent, though some long mathematical sentences remain unbroken.
- No empty or corrupted `.conllu` files were found.

**Conclusion:** CoNLL-U passes quality validation with minor tokenization artifacts typical for scientific text.
