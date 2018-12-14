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
#file = open('../tets/tet.verbose', 'r')
file = open('../tets/tet-quali.verbose', 'r')
tet_txt = file.read()
file.close()

# Initialize the TET
tet = RnnTet()
tet.parse_tet_str(tet_txt, parser='v')
print("#### TET #####")
#tet.print_tet()
print(tet)

params = tet.get_params()
print("PARAMETERS: ", params, "\n")

print("TET VALUE 1")
value_1 = TetValue()
#value_1.parse_value_str("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])")
value_1.parse_value_str("(T,[(T,[T:3]):2,(T,[T:2]):1],[T:3])")

print("VALUE: ", value_1)

npv_1 = value_1.arrayfy()
#print("ARRAY VALUE: ", npv_1)

mpt_1 = MultiPathTree()
mpt_1.instantiate_tree(tet)


print("\nTET VALUE 2")
value_2 = TetValue()
#value_2.parse_value_str("(T,[(T,[T:8]):4,(T,[T:9]):2,(T,[T:10]):2])")
value_2.parse_value_str("(T,[(T,[T:1]):2,(T,[ ]):4,(T,[T:2]):1],[T:1])")

print("VALUE: ", value_2)

npv_2 = value_2.arrayfy()
#print("ARRAY VALUE: ", npv_2)

params = tet.get_params()

mpt_2 = MultiPathTree()
mpt_2.instantiate_tree(tet)

metric = TetMetric()

r = metric.emd(params, tet, npv_1, npv_2)
print("\nEMD: ", r)
metric_grad = grad(metric.emd, argnum=0)

start_time = time.time()

r = metric_grad(params, tet, npv_1, npv_2)
print("GRADIENT: ", r)
print("\n++++++++ Computation took {} sec +++++++++".format(time.time()-start_time))