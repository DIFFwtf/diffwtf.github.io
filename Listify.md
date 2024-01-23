---
layout: page
title: Listify
permalink: /Listify/
---


# Listification is HYBRID intelligence

Annotated lists help form the basis of algorithms behind semiotic and semantic graphs ... it's really a matter of growing or extending noospheres with organic, "almost mycorrhizal" living conceptual pathways.

It begins with a very simple lists ... which becomes checkist or perhaps a priorizable outline driving an open issue deck ... along the way, we add annoations and make new connections ... continuously editing and revising ... to eventually think in visual graphs and then do that with intelligence infrastructure in code, such as ConnectedPapers ... with hopefully standard data API format such as maybe a simplistic CSV, slightly more valuable JSON or a graphQL data API ... so that others can re-use the connections.

The manual EXERCISE of continually curating annotated reading lists is really about the routine discipline of little tasks, eg daily reading, that advance the personal discipline as well as mental/social fitness ... without the regular exercise of different knowledge development *muscles* atrophy occurs, especially in areas of knowledge where communities evolve and grow or die, the language and terminology changes, where different minds become more or less engaged.

# [Using CheckLists: Behavioral testing of Textual Entailment models](https://blog.allenai.org/using-checklists-with-allennlp-behavioral-testing-of-textual-entailment-models-1a0aa43cdb28)

Reproducibility is a foundational component of scientific progress, in any field. Reproducibility is the entire point of using a checklist filled out at paper submission time is to descriptively remind authors of relevant information to report [for those who might look at the paper with much different perspective], while preserving the freedom for authors to do so however they see fit while systematically annotating the author's thought process.

CheckLists enable practitioners to construct and run behavioral tests (i.e., black box tests that only evaluate the input and output to test a system) on models in a task-agnostic way. It allows for the construction of suites of tests defined by inputs and expected outputs, which are designed to check for robustness across several general linguistic capabilities; eg. Vocabulary, Logic, Negation, etc.

NLP systems are complex ... but the world is watching a lot of different developments in the field, looking for advances that will lead to broadly adopted applications. As the impact of NLP grows, so too do the consequences of reproducibility

NLP is not the first field to evaluate reproducibility; some have even described a “reproducibility crisis” in science. One tool designed to improve reproducibility is a 

Qualitatively evaluating a model’s failures using human evaluation can be expensive, and is often task-dependent. 

In this blog post, we describe how to use CheckList with AllenNLP to catch common issues with new models. As an example, we do an analysis of [textual entailment](https://en.wikipedia.org/wiki/Textual_entailment) models present in the [AllenNLP](https://allenai.org/allennlp/software/allennlp-library) demo [AI2 Tango](https://allenai.org/allennlp/software/ai2-tango). Tango’s built-in mechanism for storing and retrieving results makes sure that researchers can stay flexible when pursuing another idea. No work is duplicated, past results can be found easily, and the way a result was obtained is stored along with the result itself.



## Background
A **CheckList suite** can be defined as a matrix of Capabilities and Test Typesfor a given task. The Capabilities can include (but are not limited to):

Vocabulary + Parts of Speech (POS): Important words/word types for the task. For example, a sentiment analysis model should be able to recognize that words like “good”, “awesome” etc. indicate positive sentiment, while words like “despise”, “bad” indicate negative sentiment.
Taxonomy: Synonyms/antonyms, superclasses and subclasses etc. For example, a model should be able to recognize that “rabbits” are “animals”.
Robustness: To typos, irrelevant changes, etc.
Named Entity Recognition (NER): Appropriately understanding named entities given the task. For instance, in a sentiment analysis task, changing “Jane loved the movie” to “John loved the movie” should have no effect on the predicted sentiment. However, in a question-answering task with the question “Who loved the movie?”, the names are relevant.
Temporal: Understanding the order of events. For example, in a question-answering task with the context “Jane used to be a teacher, but she is now an artist”, a model should answer the question “Is Jane a teacher?” with “No”.
Negation: For instance, changing “The movie was good” to “The movie was not good” should change the sentiment.
Coreference: Understanding which entities are referred to by “his / her”, “former / latter”, etc. For instance, “Jane and John are friends. The former is a teacher.” entails that “Jane is a teacher”.
Semantic Role Labeling (SRL): Understanding roles such as agents and objects.
Logic: Ability to handle symmetry, consistency, and conjunctions. For example, “A is better than B” contradicts “B is better than A”.
Fairness: This can mean different things in different contexts. Typically, we want models to not discriminate against protected groups. For example, given the context “A is a man. B is a woman.”, the sentences “A is a doctor” and “B is a doctor” should have equal probabilities.
Ideally, each Capability should be tested with the following Test Types:

Minimum Functionality Test (MFT): It checks if the predicted output matches the expected output. For example, for a sentiment analysis task, a simple MFT can check if the model always predicts a positive sentiment for very positive words.
Invariance Test (INV): It checks if the predicted output is invariant to some change in the input. For example, for a sentiment analysis task, an INV test can check if the predicted sentiment stays consistent if simple typos are added to the input sentence.
Directional Expectation Test (DIR): It checks if the predicted output changes in some specific way in response to the change in input. For example, for a sentiment analysis task, a DIR test can check if adding a reducer (eg. "good" -> "somewhat good") causes the prediction's positive confidence score to decrease (or at least not increase).
Running CheckList suites with AllenNLP
AllenNLP offers an integration with the CheckList library and offers test suites for multiple tasks. These can be used as starting points for users developing their own suites, and can be updated.

from allennlp.confidence_checks.task_checklists import TextualEntailmentSuite
suite = TextualEntailmentSuite()
suite.describe()
This suite contains 6 tests across 5 capabilities.
Capability: "Vocabulary" (1 tests)
* Name: "A is more COMP than B" entails "B is more antonym(COMP) than A" (100 test cases)
Eg. A is more active than B implies that B is more passive than A
Capability: "NER" (1 tests)
* Name: "A is COMP than B" gives no information about "A is COMP than C" (100 test cases)
Eg. "A is better than B" gives no information about "A is better than C"
Capability: "Temporal" (2 tests)
* Name: "A works as P" gives no information about "A used to work as P" (200 test cases)
Eg. "A is a writer" gives no information about "A was a writer"
* Name: Before != After (99 test cases)
Eg. "A was a writer before they were a journalist" contradicts "A was a writer after they were a journalist"
Capability: "Negation" (1 tests)
* Name: "A is COMP than B" contradicts "A is not COMP than B" (100 test cases)
Eg. A is better than B contradicts A is not better than C
Capability: "Logic" (1 tests)
* Name: "A is COMP than B" contradicts "B is COMP than A" (97 test cases)
Eg. "A is better than B" contradicts "B is better than A"
The above description shows the number of tests for each Capability being tested in the suite. Each test contains multiple test cases. For example,

suite.suite.tests['"A is COMP than B" contradicts "A is not COMP than B"'].to_raw_examples()
["('Patrick is worse than Martha', 'Patrick is not worse than Martha')",
 "('Fiona is bigger than Henry', 'Fiona is not bigger than Henry')",
 "('Amanda is colder than Jack', 'Amanda is not colder than Jack')",
 "('Johnny is colder than Stephanie', 'Johnny is not colder than Stephanie')",
 "('Nick is larger than Stephen', 'Nick is not larger than Stephen')",
.
.
.
An existing suite can also be loaded:

from checklist.test_suite import TestSuite
test_suite = TestSuite.from_file(TestSuite.from_file(cached_path("https://storage.googleapis.com/allennlp-public-models/checklist/textual_entailment_suite.pkl"))
suite = TextualEntailmentSuite(suite=test_suite, probs_key="probs", add_default_tests=False)
A test suite can be run by providing it with a Predictor object.

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz", predictor_name="textual_entailment")
suite.run(predictor, max_examples=15)
The above loads a predictor object for the Decomposable Attention model, and runs 15 test cases for each test.

Once a suite has been run, its results can be summarized as follows:

suite.summary(capabilities=["Vocabulary"])  # Filtering for a specific capability.
Capability: "Vocabulary" (1 tests)
* Name: "A is more COMP than B" entails "B is more antonym(COMP) than A" (99 test cases)
Eg. A is more active than B implies that B is more passive than A
Test cases:      99
Test cases run:  15
Fails (rate):    3 (20.0%)
Example fails:
Premise: Anna is more powerless than Ryan
Hypothesis: Ryan is more powerful than Anna
Original: Entails
Prediction: Entails (0.1), Contradicts (0.8), Neutral (0.1)
----
Premise: Chris is more hopeful than Colin
Hypothesis: Colin is more hopeless than Chris
Original: Entails
Prediction: Entails (0.4), Contradicts (0.1), Neutral (0.5)
----
Premise: Martin is more powerless than Jack
Hypothesis: Jack is more powerful than Martin
Original: Entails
Prediction: Entails (0.4), Contradicts (0.2), Neutral (0.4)
----
The above shows a brief summary of how the model did on tests for a particular capability (“Vocabulary").

The results (and predictions) for tests can also be saved to a file for in-depth perusal.

suite.summary(file='/path/to/output/file.txt')
Analyzing textual entailment models
Textual Entailment is the task of predicting whether, for a pair of sentences, the facts in the first sentence necessarily imply the facts in the second. For instance,

Premise: Two women are wandering along the shore drinking iced tea.
Hypothesis: Two women are sitting on a blanket near some rocks talking about politics.
Label: Contradiction ("sitting" contradicts "wandering").
The AllenNLP Demo showcases different models trained for textual entailment. Users can play with these models by providing their own inputs on the website. In the rest of this post, we use our checklist suite for analyzing the models’ behavior for different capabilities, and highlight some interesting observations. For a complete summary of the suite’s outputs, see here.

We compare the following models:

ELMo-based Decomposable Attention (Decomposable Attention)
RoBERTa finetuned on SNLI (RoBERTa-SNLI)
RoBERTa finetuned on MNLI (Roberta-MNLI)
Robustness
We start with a basic test: Adding typos should not change the predicted label.

We observe that the models fail in interesting ways. Here is a sample of failing examples for the Decomposable Attention model:

Premise: A pregnant lady singing on stage while holding a flag behind her.
Hypothesis: A woman is sick in **bed**.
Prediction: Entails (0.0), Contradicts (1.0), Neutral (0.0)
Premise: A pregnant lady singing on stage while holding a flag behind her.
Hypothesis: A woman is sick in **be.d**
Prediction: Entails (0.0), Contradicts (0.5), Neutral (0.5)
Premise: A man in a **black** shirt is playing golf outside.
Hypothesis: The man wearing the black shirt plays a game of golf.
Prediction: Entails (1.0), Contradicts (0.0), Neutral (0.0)
Premise: A man in a **blcak** shirt is playing golf outside.
Hypothesis: The man wearing the black shirt plays a game of golf.
Prediction: Entails (0.5), Contradicts (0.0), Neutral (0.5)
As can be seen, typos such as “black” → “blcak” cause the model to change its prediction.


Table 1: Failure rates for Robustness tests
Negation
Negation is fairly easy for humans to understand, but surprisingly difficult for simpler NLP models. For instance, the Decomposable Attention model fails on the following cases:

Premise: Andrew is smarter than Sophie
Hypothesis: Andrew is not smarter than Sophie
Original: **Contradicts**
Prediction: **Entails** (0.6), Contradicts (0.3), Neutral (0.1)
----
Premise: Evelyn is smaller than Louise
Hypothesis: Evelyn is not smaller than Louise
Original: **Contradicts**
Prediction: **Entails** (0.9), Contradicts (0.0), Neutral (0.1)
----
It appears that the model is simply mapping the repetition of words in the premise and the hypothesisand assumes entailment if there is sufficient word overlap.

Both the RoBERTa-based models (SNLI and MNLI) do well on this test, and have a 0% failure rate.

Logic
In deploying NLP models, it is important to make sure that the predictions follow the rules of logic. We run the following tests:

A entails A (When premise equals hypothesis, the prediction should be “entails”.)
“A is <comparative> than B” contradicts “B is <comparative> than A”
While the RoBERTa-based models do very well on these simple checks, the Decomposable Attention has a 99% failure rate on the second test! It even fails on 2 test cases for the first test:

Premise: Nobody has a suit
Hypothesis: Nobody has a suit
Original: Entails
Prediction: Entails (0.1), Contradicts (0.9), Neutral (0.0)
----
Premise: Nobody has a shirt on
Hypothesis: Nobody has a shirt on
Original: Entails
Prediction: Entails (0.1), Contradicts (0.9), Neutral (0.0)
This seems to contradict the idea that word overlap is causing the model to predict entailment in all cases, making them interesting outliers.

Fairness
We test whether male and female names are more/less associated with certain professions.

Premise: All people in the park are financiers. Gary is in the park.
Hypothesis: He is a financier.
Prediction: Entails (0.2), Contradicts (0.0), Neutral (0.8)
Premise: All people in the park are financiers. Elizabeth is in the park.
Hypothesis: She is a financier.
Prediction: Entails (0.1), Contradicts (0.0), Neutral (0.9)
We see above that changing “He” → “She” reduces the confidence in entailment. Ideally, since we do not know whether He/She refer to the person in the premise, the confidence scores should not change.

For 10 stereotypically male professions and 10 stereotypically female professions (taken from Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (Bolukbasi et al)), we check whether changing between female and male causes a reduction in the confidence scores for entailment.


Table 2: Failure rates for Fairness tests
Note that all 3 models have higher failure rates when associating stereotypically male professions with women, as compared to associating stereotypically female professions with men. This is a common problem in NLP and it is a key research area to determine how to remove these biases from models. Identifying the problem is the first step though, and CheckLists are here to help!

Summary
AllenNLP is a library that will help you build state of the art models, and with CheckLists you can quickly identify errors in your model to make sure it is accurate and fair!

Below are the overall test results for each of the models in for the textual entailment task.


Table 3: Failure rates for Decomposable Attention Model

Table 4: Failure rates for RoBERTa-SNLI model

Table 5: Failure rates for RoBERTa-MNLI model
References
Ribeiro, M.T., Wu, T., Guestrin, C., & Singh, S. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL.

Bolukbasi, T., Chang, K., Zou, J.Y., Saligrama, V., & Kalai, A.T. (2016). Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NIPS.



## Curated Art Collections

* Landsculpting

* Photography

* Museums and Galleries
## Curated Nanoenterprise Collections

We also think in terms of other lists for Ventures

* Quantum Cloud Kernel Reading List

* Computational Biology Reading List

* Asynchronous Workflow Reading List

* Computational Linguistics Reading List

## Curated Reading Collections

* Classics

* Newsreaders

* Pre-Print Archives

* Threads / Comment Poetry

## Curated Workout Collections

Our process of revising and edit Workout Lists is that we really must do these workouts.

* Tabatas

* 5S Housekeeping

* Hikes

