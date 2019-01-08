import sys
from tet import RnnTet
from learner import MetricLearner
from utils import read_values_labels_files
import warnings
warnings.filterwarnings("ignore")
import time
from functions import CrossEntropy, MeanSquaredError, EMDHinge
from metric import TetMetric


# Read the TET string from the file
#file = open('tet-quali.verbose', 'r')
file = open('../tets/tet-dami.compact', 'r')
tet_txt = file.read()
file.close()

# Initialize the TET
tet = RnnTet()
tet.parse_tet_str(tet_txt, parser='c')
print("#### TET #####")
#tet.print_tet()
print(tet)

# Read the TET values and the labels from the files. values is a list of 
# TETValues converted by the function .arrayfy(), which returns a list-of-lists
# representation of the TETValue.
train_values, train_labels = read_values_labels_files("../data/regression/reduced/dataset.train", "../data/regression/reduced/hindex.train")

test_values, test_labels = read_values_labels_files("../data/regression/reduced/dataset.test", "../data/regression/reduced/hindex.test")

metric = TetMetric()

# Initialize the learner with the TET, the values and the labels
learner = MetricLearner(tet, EMDHinge(0.1), metric, train_values, train_labels)


print("LEARNING...")
start_time = time.time()
# Start the learning procedure. par contains the optimized parameters. 
# For now, only batch gradient descent is implemented.
par = learner.learn(num_iters = 50, optimizer = 'adam', step_size = 0.5)
exe_time = time.time() - start_time
print("+++++++ Learning took {} minutes +++++++".format(int(exe_timei/60)))


print("\nOPTIMIZED PARAMETERS: ", par)
