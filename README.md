# Modeling Dialogue Acts with Content Word Filtering and Speaker Preferences ([BibTeX](http://www.aclweb.org/anthology/D/D17/D17-1232.bib))

We present an unsupervised model of dialogue act sequences in conversation. By modeling topical themes as transitioning more slowly than dialogue acts in conversation, our model de-emphasizes content-related words in order to focus on conversational function words that signal dialogue acts. We also incorporate speaker tendencies to use some acts more than others as an additional predictor of dialogue act prevalence beyond temporal dependencies. According to the evaluation presented on two dissimilar corpora, the CNET forum and NPS Chat corpus, the effectiveness of each modeling assumption is found to vary depending on characteristics of the data. De-emphasizing content-related words yields improvement on the CNET corpus, while utilizing speaker tendencies is advantageous on the NPS corpus. The components of our model complement one another to achieve robust performance on both corpora and outperform state-of-the-art baseline models. 


## Source Code

 * Model: <https://github.com/yohanjo/YUtils/tree/master/src/topicmodel/csm>.
 * Data processing: will be uploaded soon.


## BibTeX
```
@InProceedings{jo-EtAl:2017:EMNLP2017,
  author    = {Jo, Yohan  and  Yoder, Michael  and  Jang, Hyeju  and  Rose, Carolyn},
  title     = {Modeling Dialogue Acts with Content Word Filtering and Speaker Preferences},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {2179--2189},
  url       = {https://www.aclweb.org/anthology/D17-1232}
}
```
