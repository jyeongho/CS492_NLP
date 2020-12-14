# CS492_NLP
CS492_NLP : QuestionAnswering task for korsquad dataset

## File discription
### model_electra.py
---
Modified class of Electra models for retrospective reader paper [arXiv:2001.09694](https://arxiv.org/pdf/2001.09694.pdf). For external front verfication, we used 'ElectraForSequenceClassification' class. For internel front verfication, we used 'ElectraForQuestionAnswering' class.Model classes are from huggingface[1]. And, referring to retrospective reader code[2], a part of the class was modified.
<br/>[1] : https://github.com/huggingface/transformers/blob/master/src/transformers/models/electra/modeling_electra.py
<br/>[2] : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/transformers/modeling_albert.py

### open_squad.py
---
To applying retrospective paper, we modified some functions and added class.<br/>
1) Add 'is_impossible', 'pq_end_pos' into SquadFeature.
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad.py#L96
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad.py#L279
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad.py#L729
- Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/transformers/data/processors/squad.py

2) Change criterion for choosing examples.
<br/> We modify this part to choose examples using word-based similarity
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad.py#L559

3) Create new SquadResult_with_score class to contain ext score.
<br/> This class is used in 'run_two_model.py' and created for [compute_predictions_logits_with_score](https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L712) function in 'open_squad_metrics.py' .
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad.py#L813

### open_squad_metrics.py
---
1) Add functions for fixing error in get_final_text function.
<br/> When 'get_final_text' function does change pred_text using orig_text, there are some error cases related with de-tokenization. For example, [UNK] or '##' in pred_text, unicode '\xa0' instead of ' ', unicode '\u200e' instead of '', unknown signs etc. To avoid thie cases, we add some functions.
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L282
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L289
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L329

2) Create function for prediction to use ext score and diff score.
<br/> To make prediction using ext score and diff score, we add new function. If ext_score + diff_score is less than threshold, make prediction using QA model's output, otherwise, make null prediction.
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L712

### run_cls.py
---
This file is for training Sketchy Reading Module in retrospective paper. Using ElectraForSequenceClassification model, this file does training by classification task. After training, the output of ElectraForSequenceClassification model will be used to calculating ext score.
<br/> Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_cls.py

### run_squad.py
---
This file is for training intensive Reading Module in retrospective paper. Using ElectraForQuestionAnswering model, this file does training by QA task. After training, the output of ElectraForQuestionAnswering model will be used to calculating diff score and predicting answer.
<br/> Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_squad_seq_sc.py

### run_save_model.py
---

### run_two_model.py
---

### run_nsml.sh
---

### setup.py
---


## The path of pre-trained model in NSML
