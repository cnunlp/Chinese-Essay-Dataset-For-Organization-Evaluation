# Chinese Essay Dataset For Organization Evaluation
A Chinese argumentative student essay dataset for Organization Evaluation and Sentence Function Identification.

## Introduction

We built a **Chinese argumentative student essay dataset** for Organization Evaluation and Sentence Function Identification.

## Task Definition

 - **Organization Evaluation(OE)**. Evaluating the organization of argumentative student essays.
 - **Sentence Function Identification(SFI)**. Identifying the function of sentence in argumentative student essays.
 - **Paragragh Function Identification(PFI)**. Identifying the function of paragragh in argumentative student essays.

## Dataset
### Discourse Elements

The concept of discourse elements is borrowed from [Burstein et al., 2003] and indicates the functions of sentences. In this paper, we refer discourse elements as both sentence functions and paragraph functions.

**Sentence Functions.** We mainly follow the definition and taxonomy proposed by [Burstein et al., 2003] except that we divide support into evidence and elaboration to give more details. The sentence functions include:

　　**Introduction** is to introduce the background or attract readers’ attention before making claims.

　　**Thesis** expresses the central claim of the writer with respect to the essays topic.

　　**Main Idea** asserts foundational ideas or aspects that are related to the thesis.

　　**Evidence indicates** examples and other types of evidence that are used to support the main ideas and thesis.

　　**Elaboration further** explains the main ideas or evidence, but contains no evidence.

　　**Conclusion summarizes** the full essay and echos or extends the central claim.
  
**Paragraph Functions.** The function of a paragraph is determined according to the functions of its sentences. We consider the following paragraph functions:

　　**IntroductionPara** contains introduction sentences but does not have thesis or main idea sentences.

　　**ThesisPara** contains at least a thesis sentence.

　　**IdeaPara** contains at least a main idea sentence but does not have a thesis sentence.

　　**SupportPara** contains evidence or elaboration sentences but does not contain thesis, main idea or conclusion sentences.

　　**ConclusionPara** contains conclusion sentences but does not have thesis sentences.
  
### Organization Grades
We represent organization quality with three grades.

　　**Bad** The essay is poorly structured. It is incomplete or misses key discourse elements.

　　**Medium** The essay is well structured and complete, but could be further improved.

　　**Great** The essay is fairly well structured and the organization is very clear and logical.

### Organization Grades
Basic Statistics

| Basic Statistics | Number |
| :----------------------- | :------: |
|\#Essays | 1,220 |
|Avg. \#paragraph per essay | 8 |
|Avg. \#sentences per essay | 28 |
|Avg. \#words per sentence | 21 |
|Sentence Functions|
|Introduction |3,125|
|Thesis |1,061|
|Main Idea |4,948|
|Evidence |6,569|
|Elaboration |13,351|
|Conclusion |3,379|
|Paragraph Functions||
|IntroductionPara |893|
|ThesisPara |864|
|IdeaPara |3,379|
|SupportPara |2,788|
|ConclusionPara |1,796|
|Organization Grades|
|Great |245|
|Medium |670|
|Bad |305|

### Dataset File

    ./data/all_data.json It includes all Argumentative Student Essays, and essays are formated by json. One line is one essay.
    ./data/cv_folds.txt It includes the IDs of the essays and was split into 5 sets for cross validation.
    
The description of keys of json data in all_data.json, as follows:
~~~
{
 "title": "The title of the essay, which had been split to word", 
 "score": "The Organization Grades of the essay", 
 "sents": "All sentences of the essay, which had been split to word",
 "labels": "The labels of every sentences", 
 "gid": "The Global position of every sentences", 
 "lid": "The Local position of every sentences", 
 "pid": "The Paragraph position of every sentences", 
 "paras": "The labels of every paragraph"
}
~~~

An example code of loading dataset as follows:

~~~Python
import utils

in_file = './data/all_data.json'
embed_filename = './embd/tecent_embeddings.txt' # The embeddings from https://ai.tencent.com/ailab/nlp/en/embedding.html
title = True
max_len = 40
en_documents, en_labels, features, scores, vec_size, grids, en_paralabels = utils.getSamplesAndFeatures(in_file, embed_filename, title=title)
pad_documents, pad_labels = utils.sentence_padding(en_documents, en_labels, max_len, vec_size)

folds_file = './data/cv_folds.txt'
folds = utils.loadFolds(folds_file)

fold_k = 0 # The number of validation set, which do not for training

train_fold = []
for i in range(len(folds)):
    if i != fold_k:
        train_fold += folds[i]
train_docs = [pad_documents[i] for i in train_fold]
train_labels = [pad_labels[i] for i in train_fold]
train_features = [features[i] for i in train_fold]
train_scores = [scores[i] for i in train_fold]
train_grids = [grids[i] for i in train_fold]
train_paras = [en_paralabels[i] for i in train_fold]

~~~
## Reference
The dataset is released with this paper:

    @inproceedings{ijcai2020-536,
         title     = {Hierarchical Multi-task Learning for Organization Evaluation of Argumentative Student Essays},
         author    = {Song, Wei and Song, Ziyao and Liu, Lizhen and Fu, Ruiji},
         booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
                      Artificial Intelligence, {IJCAI-20}},
         publisher = {International Joint Conferences on Artificial Intelligence Organization},             
         editor    = {Christian Bessiere}	
         pages     = {3875--3881},
         year      = {2020},
         month     = {7},
         note      = {Main track}
         doi       = {10.24963/ijcai.2020/536},
         url       = {https://doi.org/10.24963/ijcai.2020/536},
    }


The references:

[1] Jill Burstein, Daniel Marcu, and Kevin Knight. 2003. Finding the write stuff: Automatic identification of discourse structure in student essays. IEEE Intelligent Systems, 18(1):32–39.
