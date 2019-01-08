from .tet import MultiPathTree
import autograd.numpy as np

class TetMetric:
    def _mpt_emd(self, mpt_1, mpt_2):
        if mpt_1.is_leaf and mpt_2.is_leaf:
            #print("Leaf Node distance: ", self._mpn_emd(mpt_1.value_path, mpt_2.value_path))
            return self._mpn_emd(mpt_1.value_path, mpt_2.value_path)
        else:
            distance = 0
            for child_1, child_2 in zip(mpt_1.children, mpt_2.children):
                distance = distance + (1/len(mpt_1.children))*(self._mpt_emd(child_1, child_2))

            node_distance = self._mpn_emd(mpt_1.value_path, mpt_2.value_path)
            #print("Node distance: ", node_distance)
            return node_distance + distance 

    def _mpn_emd(self, vp_1, vp_2):
        """
        """
        assert len(vp_1) > 0
        assert len(vp_2) > 0
        assert len(vp_1[0][0]) == len(vp_2[0][0])
        distance = 0
        for i in range(len(vp_1[0][0])):
            v_1 = self._calculate_marginals(vp_1, i)
            v_2 = self._calculate_marginals(vp_2, i)
            cdf = self._cdf(v_1, v_2)
            #print(cdf)
            #print("CDF: ", cdf)
            distance = distance + cdf
            #distance = distance + wasserstein_distance(np.asarray(v_1), np.asarray(v_2))
        return distance

    def _calculate_marginals(self, value_path, dim):
        values = []
        for v in value_path:
            p = [v[0][dim]]*v[1]
            values = values + p
        return np.array(values)

    def _cdf(self, t_1,t_2):
        r = 0
        t = np.sort(np.concatenate((t_1, t_2), axis=0))
        for i in range(len(t)-1): 
            r += np.abs(self._F(t[i], t_1) - self._F(t[i], t_2)) * (t[i+1]-t[i])
        return r

    def _F(self, thresh, values):
        n = 0
        for v in values:
            if v <= thresh:
                n += 1
        return n/len(values)

    def emd(self, params, tet, value_1, value_2):
        mpt_1, mpt_2 = MultiPathTree(), MultiPathTree()
        mpt_1.instantiate_tree(tet)
        mpt_2.instantiate_tree(tet)
        mpt_1.compute_value_path(params, value_1, tet)
        #print(mpt_1)
        mpt_2.compute_value_path(params, value_2, tet)
        #print(mpt_2)
        return self._mpt_emd(mpt_1, mpt_2)


