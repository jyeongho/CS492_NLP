# CS492_NLP
CS492_NLP : QuestionAnswering task for korsquad dataset

## File discription
### model_electra.py
---
Modified class of Electra models for retrospective reader paper [arXiv:2001.09694](https://arxiv.org/pdf/2001.09694.pdf). For external front verfication, we used 'ElectraForSequenceClassification' class. For internel front verfication, we used 'ElectraForQuestionAnswering' class. Model classes are from huggingface[1]. And, referring to retrospective reader code[2], a part of the class was modified.
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
<br/> When 'get_final_text' function try to change pred_text using orig_text, there are some error cases related with de-tokenization. For example, [UNK] or '##' in pred_text, unicode '\xa0' instead of ' ', unicode '\u200e' instead of '', unknown signs etc. To avoid thie cases, we add some functions.
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L282
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L289
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L329

2) Create function for making prediction using ext score and diff score.
<br/> To make prediction using ext score and diff score, we add new function. If ext_score + diff_score is less than threshold, make prediction using QA model's output, otherwise, make null prediction.
- https://github.com/jyeongho/CS492_NLP/blob/main/open_squad_metrics.py#L712

### run_cls.py
---
This file is for training Sketchy Reading Module in retrospective paper. Using ElectraForSequenceClassification model, this file does training for classification task. After training, the output of ElectraForSequenceClassification model will be used to calculating ext score.
<br/> Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_cls.py

### run_squad.py
---
This file is for training intensive Reading Module in retrospective paper. Using ElectraForQuestionAnswering model, this file does training for QA task. After training, the output of ElectraForQuestionAnswering model will be used to calculating diff score and predicting answer.
<br/> Reference : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/examples/run_squad_seq_sc.py

### run_save_model.py
---
Thie file is for saving models in separate sessions into one session. Because Sketchy Reading Module and Intensive Reading Module are saved in different sessions, we need to save these model in one session for convenience.

### run_two_model.py
---
This file is for making predictions using Classification model and QuestionAnswering model. Use ElectraForSequenceClassification model as sketchy reading module. Use ElectraForQuestionAnswering model as intensive reading module. Since this file is only for making prediction, there is no train function.

### run_nsml.sh
---
In this file, there are some commmands and arguments for executing files. 
1) Command for training intensive reading module
- https://github.com/jyeongho/CS492_NLP/blob/main/run_nsml.sh#L3

2) Command for training sketchy reading module
- https://github.com/jyeongho/CS492_NLP/blob/main/run_nsml.sh#L6

3) Command for saving two models into one session
<br/> Before running this command, you should enter session name of intensive reading module in [here](https://github.com/jyeongho/CS492_NLP/blob/main/run_save_model.py#L164), and enter session name of sketchy reading module in [here](https://github.com/jyeongho/CS492_NLP/blob/main/run_save_model.py#L157).
- https://github.com/jyeongho/CS492_NLP/blob/main/run_nsml.sh#L9

4) Command for making prediction using two models
<br/> Before running this command, you should enter session name, which created by command 3, in [here](https://github.com/jyeongho/CS492_NLP/blob/main/run_two_model.py#L547).
- https://github.com/jyeongho/CS492_NLP/blob/main/run_nsml.sh#L12

### setup.py
---
Packages for our project

## The path of pre-trained model in NSML
---
When we saved models, we forgot to change model name in arguments. The model name is bert, but the actual model is KoELECTRA. Sorry for the confusion.
1. Training KoELECTRA during 10 epochs. The only difference from baseline is using KoELECTRA model instead of BERT.
- kaist006/korquad-open-ldbd3/58/bert_best

2. Training intensive reading module and sketchy reading module based on KoELECTRA during 5 epochs. Same as baseline, training dataset is created by choosing three paragraphs.
- Intensive reading module : kaist006/korquad-open-ldbd3/78/bert_best
- Sketchy reading module : kaist006/korquad-open-ldbd3/81/electra_best

3. Training intensive reading module and sketchy reading module based on KoELECTRA during 5 epochs. Training dataset is created by choosing **five paragraphs**.
- Intensive reading module : kaist006/korquad-open-ldbd3/83/bert_best
- Sketchy reading module : kaist006/korquad-open-ldbd3/84/electra_best

4. Training intensive reading module and sketchy reading module based on KoELECTRA during 5 epochs. Training dataset is created by choosing **random paragraphs**.
- Intensive reading module : kaist006/korquad-open-ldbd3/92/bert_best
- Sketchy reading module : kaist006/korquad-open-ldbd3/93/electra_best

5. Training intensive reading module and sketchy reading module based on KoELECTRA during 5 epochs. Training dataset is created by choosing **paragraphs using relevance**.
- Intensive reading module : kaist006/korquad-open-ldbd3/103/bert_best
- Sketchy reading module : kaist006/korquad-open-ldbd3/105/electra_best

6. Training intensive reading module and sketchy reading module based on KoELECTRA during 5 epochs. Training dataset is created by choosing **paragraphs using word-based similarity**.
- Intensive reading module : kaist006/korquad-open-ldbd3/120/bert_best
- Sketchy reading module : kaist006/korquad-open-ldbd3/121/electra_best

7. Session that contains two models. This sessions are created by executing 'run_save_model.py' file.
- For models in session 78, 81 : kaist006/korquad-open-ldbd3/117/saved
- For models in session 83, 84 : kaist006/korquad-open-ldbd3/158/saved
- For models in session 92, 93 : kaist006/korquad-open-ldbd3/159/saved
- For models in session 103, 105 : kaist006/korquad-open-ldbd3/160/saved
- For models in session 120, 121 : kaist006/korquad-open-ldbd3/180/saved
  
8. Session for making predictions using two models. This sessions are created by executing 'run_two_model.py' file and used for submitting.
- For models in session 117 : kaist006/korquad-open-ldbd3/156/best_model
- For models in session 158 : kaist006/korquad-open-ldbd3/162/best_model
- For models in session 159 : kaist006/korquad-open-ldbd3/163/best_model
- For models in session 160 : kaist006/korquad-open-ldbd3/164/best_model
- For models in session 180 : kaist006/korquad-open-ldbd3/181/best_model

9. After changing 'get_final_text' function in open_squad_metrics.py, we created new sessions for re-submitting models.
- For models in session 160 : kaist006/korquad-open-ldbd3/245/best_model
- For models in session 180 : kaist006/korquad-open-ldbd3/242/best_model
