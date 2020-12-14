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



### run_cls.py



### run_squad.py



### run_save_model.py


### run_two_model.py


### run_nsml.sh


### setup.py

## The path of pre-trained model in NSML
