from module_neural_net_v3 import *
from simple_no_text import *
from simple_text import *
from tcn import *
from tcn_functional import *
from tcn_rnn import *
from tcn_gru import *
from tcn_lstm import *

os.chdir(r'C:\Users\vinee\OneDrive - Massachusetts Institute of Technology\MIT\Fall 2020\6.867\Project\Final Data by Morgan')

ticker = 'ALL_TICKERS' 
data = 'numerical'
nn_type = tcn_lstm
sequential = True
onehot = True
run_network(ticker,data,nn_type,sequential,onehot)  
