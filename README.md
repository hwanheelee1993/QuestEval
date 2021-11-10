# QuestEval - forked for evaluating FEC
![GitHub](https://img.shields.io/github/license/ThomasScialom/QuestEval)
![release](https://img.shields.io/github/v/release/ThomasScialom/QuestEval)
 
QuestEval is a **NLG metric** to assess if two different inputs contain the same information. The metric, based on Question Generation and Answering can deal with **multimodal** and **multilingual** inputs. 
It is the result from an (on-going) international collaboration, and so far it tackles various tasks:


## 1/ Installing QuestEval
```
$ conda create --name questeval python=3.9
$ conda activate questeval
```
**WARNING**: You need to install, before any package, correct version of [pytorch](https://pytorch.org/get-started/locally/#start-locally) linked to your cuda version.
```
(questeval) $ conda install pytorch cudatoolkit=10.1 -c pytorch
```

```
(questeval) $ conda install pip
(questeval) $ pip install -e .
```

## 2/ Using QuestEval 

The default `task` is `text2text` and the default `language` is `en`. It allows to measure the content similarity between any two English texts. This means that **QuestEval can be used to evaluate any NLG task where references are available**. Alternatively, we can compare the hyothesis to the source as detailed below.  
For tasks specificities, see below. 

Here is an example. Note that the code can take time since it requires generating and answering a set of questions. However, if you let the parameter `use_cache` to its default value at `True`, running the same example again will be very fast this time.

```
from questeval.questeval_metric import QuestEval
questeval = QuestEval(no_cuda=True)

source_1 = "Since 2000, the recipient of the Kate Greenaway medal has also been presented with the Colin Mears award to the value of 35000."
prediction_1 = "Since 2000, the winner of the Kate Greenaway medal has also been given to the Colin Mears award of the Kate Greenaway medal."
references_1 = [
    "Since 2000, the recipient of the Kate Greenaway Medal will also receive the Colin Mears Awad which worth 5000 pounds",
    "Since 2000, the recipient of the Kate Greenaway Medal has also been given the Colin Mears Award."
]

source_2 = "He is also a member of another Jungiery boyband 183 Club."
prediction_2 = "He also has another Jungiery Boyband 183 club."
references_2 = [
    "He's also a member of another Jungiery boyband, 183 Club.", 
    "He belonged to the Jungiery boyband 183 Club."
]

score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2],
    list_references=[references_1, references_2]
)

print(score)
```
Output:
```
{'corpus_score': 0.6115364039438322, 
'ex_level_scores': [0.5698804143875364, 0.6531923935001279]}
```

In the output, you have access to the `corpus_score` which corresponds to the average of each example score stored in `ex_level_scores`. Note that the score is always between 0 and 1.


### Reference-less mode


```
score = questeval.corpus_questeval(
    hypothesis=[prediction_1, prediction_2], 
    sources=[source_1, source_2]
)

print(score)
```
Output:
```
{'corpus_score': 0.5538727587335324, 
'ex_level_scores': [0.5382940950847808, 0.569451422382284]}
```

