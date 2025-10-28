# Quality control notes

During a manual inspection of twenty randomly selected `.conllu` files, all files were found to be correctly formatted and valid.
The check included structural correctness, annotation accuracy (lemma, POS tags), and consistency of sentence and token segmentation.

## Structural check
- Each file contains a valid CoNLL-U structure with all 10 required columns (`ID`, `FORM`, `LEMMA`, `UPOS`, `XPOS`, `FEATS`, `HEAD`, `DEPREL`, `DEPS`, `MISC`).
- The `# text` and `# sent_id` metadata are present and correctly matched to the corresponding sentences.
- No empty, incomplete, or corrupted files were found.
- The order of tokens and sentences is consistent, with no skipped or duplicates.

## Annotation accuracy
- Lemmatization and POS marking are mostly accurate and in line with the expected forms in scientific texts.
- A few minor differences were observed for domain-specific terms (especially technical terms and foreign words).
- For some mathematical expressions, long symbolic expressions were noted to remain unparsed, which is common for scientific texts.
- Syntactic links (HEAD and DEPREL) are not included in the detailed analysis at this stage, but are formally structured correctly.

## Segmentation consistency
- Sentence segmentation is generally consistent, with minor deviations in documents containing complex formulas and LaTeX-style expressions.
- Long paragraphs with technical explanations are sometimes treated as one sentence, which causes a higher number of tokens in individual segments.
- No cases of incorrectly joined sentences or omitted punctuation boundaries were found.

## Conclusion
Overall, all reviewed `.conllu` files pass the quality check with minimal tokenization artifacts, which is common in scientific and technical texts.
The corpus structure is stable, the files are consistent and ready for the next processing phase (Milestone 2).
Additional optimizations may include improved segmentation and expanded dictionaries for domain-specific terms.