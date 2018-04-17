Please make sure that the folloing libraries are installed

import numpy as np
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize,RegexpTokenizer
from nltk.util import ngrams
import random
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
import sys

I found that stopwords was not installed on the data.cs.purdue but in the piazza comments it was posted that we can install it.

follwing are the instructions to run the file:
1. run it as:
python nn_per.py --test_data (path to real_test_file.txt) --test_label (path to real_test_label.txt).
2. Make sure sentences.txt and labels.txt are present in the same folder as the nn_per.py as it is needed to train the network.
3. It will output 5 lines: 
	1. perceptron results _ _ _
	2. MLP results _ _ _
	3. Real Testing on unseen data
	4. perceptron results _ _ _
	5. MLP results _ _ _
	6. Time taken = __ 
4. A screenshot of the working algorithm is included in the report.