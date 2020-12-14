# CS492_NLP
CS492_NLP : QuestionAnswering task for korsquad dataset

## File discription
### model_electra.py
> Modified class of Electra models for retrospective reader paper [arXiv:2001.09694](https://arxiv.org/pdf/2001.09694.pdf)
For external front verfication, we used 'ElectraForSequenceClassification' class.
For internel front verfication, we used 'ElectraForQuestionAnswering' model.
Model classes are from huggingface. Reference code : https://github.com/huggingface/transformers/blob/master/src/transformers/models/electra/modeling_electra.py
And, referring to retrospective reader code, a part of the class was modified. Reference code : https://github.com/cooelf/AwesomeMRC/blob/master/transformer-mrc/transformers/modeling_albert.py



### open_squad.py


### open_squad_metrics.py



### run_cls.py



### run_squad.py



### run_save_model.py


### run_two_model.py


### run_nsml.sh


### setup.py

## The path of pre-trained model in NSML
