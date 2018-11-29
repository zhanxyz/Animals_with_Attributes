------------------------------------------------------------------
Animals with Attributes Dataset, v1.0, May 22th 2009
------------------------------------------------------------------
Python code for Direct Attribute Prediction (DAP) model
(C) May 22, Christoph Lampert, <chl@tuebingen.mpg.de>
------------------------------------------------------------------

0) install the SHOGUN machine learning toolbox: http://www.shogun-toolbox.org/

1) update paths in Python files to your local setup

2) run preprocess.py to convert ASCII features into Python Pickle format
   (written to ./feat directory)

3) execute attributes.sh to train attribute classifiers on train classes
   (results are written to the ./DAP directory) 
   Note: This requires quite some RAM and a lot of CPU time.
   
4) execute DAP_eval.py to predict the test classes and evaluate
