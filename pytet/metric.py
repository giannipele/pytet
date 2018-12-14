import autograd.numpy as np
from scipy.stats import wasserstein_distance

class TetMetric():
    def mpt_distance(self, mpt_1, mpt_2):
        if mpt_1.is_leaf and mpt_2.is_leaf:
            return  self._mpn_distance(mpt_1.value_path, mpt_2.value_path)
        else:
            distance = 0
            for child_1, child_2 in zip(mpt_1.children, mpt_2.children):
                distance = distance + (self.mpt_distance(child_1, child_2))

            node_distance = self._mpn_distance(mpt_1.value_path, mpt_2.value_path)
            return node_distance + distance 

    def _mpn_distance(self, vp_1, vp_2):
        assert len(vp_1) > 0
        assert len(vp_2) > 0
        assert len(vp_1[0][0]) == len(vp_2[0][0])
        distance = 0
        for i in range(len(vp_1[0][0])):
            v_1 = self._calculate_marginals(vp_1, i)
#            print(v_1)
            v_2 = self._calculate_marginals(vp_2, i)
#            print(v_2)
            #distance = distance + wasserstein_distance(v_1, v_2)
            distance = distance + self.cdf(v_1, v_2)
        return distance
    
    def _calculate_marginals(self, value_path, i):
        values = []
        for v in value_path:
            p = [v[0][i]]*v[1]
            values = values + p
        return np.array(values)

    def cdf(self, t_1,t_2):
        r = 0
        t = np.sort(np.concatenate((t_1, t_2), axis=0))
        #print("t = ", t)
        for i in range(len(t)-1): 
            r += np.abs(self.f(t[i], t_1) - self.f(t[i], t_2)) * (t[i+1]-t[i])
            #print(i,f(t[i], t_1) - f(t[i], t_2), t[i+1]-t[i] )
        return r

    def f(self, thresh, values):
        n = 0
        for v in values:
            if v <= thresh:
                n += 1
        return n/len(values)
