import sys
from tet import RnnTet
from learner import Learner
from utils import read_values_labels_files
import warnings
warnings.filterwarnings("ignore")
import time
from functions import CrossEntropy, MeanSquaredError

# Read the TET string from the file
#file = open('tet-quali.verbose', 'r')
file = open('../tets/tet.verbose', 'r')
tet_txt = file.read()
file.close()

# Initialize the TET
tet = RnnTet()
tet.parse_tet_str(tet_txt, parser='v')
print("#### TET #####")
#tet.print_tet()
print(tet)

# Read the TET values and the labels from the files. values is a list of 
# TETValues converted by the function .arrayfy(), which returns a list-of-lists
# representation of the TETValue.
train_values, train_labels = read_values_labels_files("../data/reduced/dataset.red.train", "../data/reduced/labels.red.train")

test_values, test_labels = read_values_labels_files("../data/reduced/dataset.red.test", "../data/reduced/labels.red.test")

# Initialize the learner with the TET, the values and the labels
learner = Learner(tet, MeanSquaredError(), train_values, train_labels)

print("LEARNING...")
start_time = time.time()
# Start the learning procedure. par contains the optimized parameters. 
# For now, only batch gradient descent is implemented.
par = learner.learn(optimizer="adam", num_iters=10, step_size=0.1)
exe_time = time.time() - start_time
print("+++++++ Learning took {} seconds +++++++".format(exe_time))


print("\nOPTIMIZED PARAMETERS: ", par)


