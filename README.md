# Modeling Dialogue Acts with Content Word Filtering and Speaker Preferences [(BibTeX)(http://www.aclweb.org/anthology/D/D17/D17-1231.bib)]

We present an unsupervised model of dialogue act sequences in conversation. By modeling topical themes as transitioning more slowly than dialogue acts in conversation, our model de-emphasizes content-related words in order to focus on conversational function words that signal dialogue acts. We also incorporate speaker tendencies to use some acts more than others as an additional predictor of dialogue act prevalence beyond temporal dependencies. According to the evaluation presented on two dissimilar corpora, the CNET forum and NPS Chat corpus, the effectiveness of each modeling assumption is found to vary depending on characteristics of the data. De-emphasizing content-related words yields improvement on the CNET corpus, while utilizing speaker tendencies is advantageous on the NPS corpus. The components of our model complement one another to achieve robust performance on both corpora and outperform state-of-the-art baseline models. 


## Source Code

 * Model: <https://github.com/yohanjo/YUtils/tree/master/src/topicmodel/csm>.
 * Data processing: will be uploaded soon.


## BibTeX
```
@InProceedings{liu-EtAl:2017:EMNLP20176,
  author    = {Liu, Yang  and  Han, Kun  and  Tan, Zhao  and  Lei, Yun},
  title     = {Using Context Information for Dialog Act Classification in DNN Framework},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {2170--2178},
  abstract  = {Previous work on dialog act (DA) classification has investigated different
	methods, such as hidden Markov models, maximum entropy, conditional random
	fields, graphical models, and support vector machines.
	A few recent studies explored using deep learning neural networks for DA
	classification, however, it is not clear yet what is the best method for using
	dialog context or DA sequential information, and how much gain it brings. This
	paper proposes several ways of using context information for DA classification,
	all in the deep learning framework. The baseline system classifies each
	utterance using the convolutional neural networks (CNN). Our proposed methods
	include using hierarchical models (recurrent neural networks (RNN) or CNN) for
	DA sequence tagging where the bottom layer takes the sentence CNN
	representation as input, concatenating predictions from the previous utterances
	with the CNN vector for classification, and performing sequence decoding based
	on the predictions from the sentence CNN model. 
	We conduct thorough experiments and comparisons on the Switchboard corpus,
	demonstrate that incorporating context information significantly improves DA
	classification, and show that we achieve new state-of-the-art performance for
	this task.},
  url       = {https://www.aclweb.org/anthology/D17-1231}
}
```
