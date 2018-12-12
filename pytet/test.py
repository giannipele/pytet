import unittest
from tetutils import TetValue, TetMultiset, next_token, RnnTet

class TestTETReading(unittest.TestCase):
    
    def test_parsing(self):
        value = TetValue()
        self.assertEqual(value.parse_value("(T,[(T,[T:4]):3,(T,[T:2]):1],[(T,[]):1,(T,[T:8]):6])", 0), 52)

    def test_nodes(self):
        value = TetValue("(T,[(T,[T:4]):3,(T,[T:2]):1],[(T,[]):1,(T,[T:8]):6])")
        self.assertEqual(value.count_nodes(), 74)
    
    def test_strings(self):
        self.assertEqual(next_token("123(123(123)))", ')', parenthesis=True), ("123(123(123))",13))
        self.assertEqual(next_token("123(123(123)))", ')', 4, parenthesis=True), ("123(123)",12))
        #self.assertEqual(next_token('123', ')'), ('123',2))
        node = RnnTet()
        self.assertEqual(node.parse_tet_str(
            "{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}}",0), 40)
        self.assertEqual(node.parse_tet_str(
           "{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}{CHILD (paper0) {NODE {FUNCTION (logistic,-75,10)}{TYPE (author_paper(author0,paper0))}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}}}}", 0), 199)
        self.assertEqual(node.parse_tet_str(
           "{NODE {FUNCTION (logistic,-75,10)}{TYPE ()}{CHILD (paper0) {NODE {FUNCTION (logistic,-75,10)}{TYPE (author_paper(author0,paper0))}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}{CHILD (paper1) {NODE {FUNCTION (identity)}{TYPE (paper_paper(paper1,paper0))}}}}}}}", 0), 274)


if __name__=='__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTETReading)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()
