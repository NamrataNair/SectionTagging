# Contents

topic-annotate: This is a modified version of ClarityNLP SectionTagger, used to generate clinical 
sentences and section header labels in fast-text format.

scripts: This contains python script used for the the text classification model.

Notes on the embeddings:
=======================

1) The bert-base-clinical-cased embedding used in the python script can be obtained from ClinicalBERT using the following commands:

!wget -O pretrained_bert_tf.tar.gz https://www.dropbox.com/s/8armk04fu16algz/pretrained_bert_tf.tar.gz\?dl\=1
!tar -xzf pretrained_bert_tf.tar.gz
!tar -xzf pretrained_bert_tf/bert_pretrain_output_all_notes_150000.tar.gz
!mv bert_pretrain_output_all_notes_150000 bert-base-clinical-cased
!mv ./bert-base-clinical-cased/bert_config.json ./bert-base-clinical-cased/config.json

2) The cui2vec_embed_vectors.bin used in the python script can be obtained by emailing the authors.

3) The forward-lm.pt, backward-lm.pt are fine-tuned versions of pubmed-forward and pubmed-backward learning models
 and can also be obtained upon request.
 
 

