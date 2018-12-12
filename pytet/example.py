from tet import RnnTet
from value import TetValue
from autograd import grad
from autograd.misc.flatten import flatten
from autograd.misc.optimizers import adam, sgd
import autograd.numpy as np
from learner import Learner
from utils import read_values_labels_files
import warnings
warnings.filterwarnings("ignore")

# Read the TET string from the file
#file = open('tet-quali.verbose', 'r')
file = open('../tets/tet-quali.verbose', 'r')
tet_txt = file.read()
file.close()

# Initialize the TET
tet = RnnTet()
tet.parse_tet_str(tet_txt, parser='v')
print("#### TET #####")
#tet.print_tet()
print(tet)

########### Uncomment this part for a more detailed example ###############
value = TetValue()
#value.parse_value_str("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])")
value.parse_value_str("(T,[(T,[T:3]):2,(T,[T:2]):1],[T:3])")

print("VALUE: ", value)

npv = value.arrayfy()
print("ARRAY VALUE: ", npv)

params = tet.get_params()
print("PARAMETER: ", params)

evaluation_values = TetValue()

r = tet.forward_value(params, npv, evaluation_values)
print("RESULT: ",r)
print("EVALUATION TREE: ",evaluation_values)
#evaluation_grad = grad(loss, argnum=0)
#gr = evaluation_grad(params, npv, tet, evaluation_values)
#print("GRADIENT: ", gr)

############################################################################



#values, labels = read_values_labels_files("../data/reduced/dataset.red.train", "../data/reduced/labels.red.train")
#
#learner = Learner(tet, values, labels)
#
#par = learner.predict(optimizer="adam", num_iters=5, step_size=0.1)
#
#print(par)
#

