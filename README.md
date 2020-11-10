# sentiment-review-summary

Code for our COLING-2020 paper ["Making the Best Use of Review Summary for Sentiment Analysis"](https://ringos.github.io/files/coling2020_sentiment.pdf)

### Before running the code
- Our experiments were conducted under `pytorch==1.0.1` and `cudatoolkit==9.0`, with `python==3.6`. 
- Some other required packages: tensorboardX, pickle, nltk, numpy. 
- SNAP Amazon Review dataset can be obtained from [this url](http://snap.stanford.edu/data/web-Amazon.html). (We cannot provide the data in our repo due to copyright issue. )
- The paths of the dataset and the embedding file (we use GloVe in our experiments) in the config file need to be filled in before running the code. 

### Run the code
`python main.py --config config_toy`
- /param/config_toy.py is just an example config file. You may create your own config file. 

### Cite
If you find our code useful, please consider citing our paper: 
```
@inproceedings{yang-etal-2020-making,
    author    = {Sen Yang and
                 Leyang Cui and
                 Jun Xie and
                 Yue Zhang},
    title     = {Making the Best Use of Review Summary for Sentiment Analysis},
    booktitle = {Proceedings of the 28th International Conference on Computational
                 Linguistics},
    year      = {2020}
}
```
