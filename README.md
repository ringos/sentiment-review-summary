# sentiment-review-summary

Code for our COLING2020 paper "Making the Best Use of Review Summary for Sentiment Analysis"

### Before you can run the code
- Our experiments were conducted under `pytorch==1.0.1` and `cudatoolkit==9.0`, with `python==3.6`. 
- Some other required packages: tensorboardX, pickle, nltk, numpy. 
- SNAP Amazon Review dataset can be obtained from [this url](http://snap.stanford.edu/data/web-Amazon.html). (We cannot provide the data in our repo due to copyright issue. )
- The dataset path in the config file needs to be correctly assigned to run the code. 

### Run the code
`python main.py --config config_toy`
- /param/config_toy.py is just an example config file. You may create your own config file. 
