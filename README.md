# Chinese-Argumentative-Student-Essays-Dataset
A Chinese argumentative student essay dataset for Organization Evaluation and Sentence Function Identification.

## Introduction

We proposed a **Chinese argumentative student essay dataset** for Organization Evaluation and Sentence Function Identification.

## Task Definition

 - **Organization Evaluation(OE)**. Evaluating the organization of argumentative student essays.
 - **Sentence Function Identification(SFI)**. Identificating the function of sentence in argumentative student essays.

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


The references of explaining simile concepts and some other references:

[1] Jill Burstein, Daniel Marcu, and Kevin Knight. 2003. Finding the write stuff: Automatic identification of discourse structure in student essays. IEEE Intelligent Systems, 18(1):32–39.
