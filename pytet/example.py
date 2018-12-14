from tet import RnnTet, MultiPathTree
from value import TetValue
from autograd import grad
from metric import TetMetric
import time
import warnings
warnings.filterwarnings("ignore")

def loss(par, value, tet, evaluations=None):
    return ((tet.forward_value(par, value, evaluations) - 1)**2)/2

# Read the TET string from the file
file = open('../tets/tet.verbose', 'r')
#file = open('../tets/tet-quali.verbose', 'r')
tet_txt = file.read()
file.close()

# Initialize the TET
tet = RnnTet()
tet.parse_tet_str(tet_txt, parser='v')
print("#### TET #####")
#tet.print_tet()
print(tet)

########### Uncomment this part for a more detailed example ###############
print("TET VALUE 1")
value_1 = TetValue()
value_1.parse_value_str("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])")
#value_1.parse_value_str("(T,[(T,[T:3]):2,(T,[T:2]):1],[T:3])")

print("VALUE: ", value_1)

npv_1 = value_1.arrayfy()
print("ARRAY VALUE: ", npv_1)

params = tet.get_params()
print("PARAMETER: ", params)

evaluation_values = TetValue()

r = tet.forward_value(params, npv_1, evaluation_values)
print("RESULT: ",r)
print("EVALUATION TREE: ",evaluation_values)

mpt_1 = MultiPathTree()
mpt_1.instantiate_tree(tet)
#print(mpt)

#mpt_1.extract_value_path((evaluation_values,1))
#print(mpt_1)
############################################################################

print("TET VALUE 2")
value_2 = TetValue()
value_2.parse_value_str("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])")
#value_2.parse_value_str("(T,[(T,[T:1]):2,(T,[ ]):4,(T,[T:2]):1],[T:1])")

print("VALUE: ", value_2)

npv_2 = value_2.arrayfy()
print("ARRAY VALUE: ", npv_2)

params = tet.get_params()
print("PARAMETER: ", params)

evaluation_values_2 = TetValue()

r = tet.forward_value(params, npv_2, evaluation_values_2)
print("RESULT: ", r)
print("EVALUATION TREE: ", evaluation_values_2)

mpt_2 = MultiPathTree()
mpt_2.instantiate_tree(tet)
#print(mpt)

#mpt_2.extract_value_path((evaluation_values_2,1))
#print(mpt_2)

metric = TetMetric()
#r = metric.mpt_distance(mpt_1, mpt_2)
#print(r)

#metric_grad = grad(metric.mpt_distance)
#print(metric_grad(mpt_1, mpt_2))

r = metric.emd(params, tet, npv_1, npv_2)
print("EMD: ", r)
metric_grad = grad(metric.emd, argnum=0)

start_time = time.time()

r = metric_grad(params, tet, npv_1, npv_2)
print("GRAD", r)
print("++++++++ Computation took {} sec".format(time.time()-start_time))