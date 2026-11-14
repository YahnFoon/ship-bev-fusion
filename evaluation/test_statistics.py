import itertools
import unittest
import numpy as np
from scipy.stats import rankdata, wilcoxon
from evaluation.paired_statistics import exact_wilcoxon, holm, paired_bootstrap, match_conditions

class StatisticsTests(unittest.TestCase):
    def test_no_ties_matches_scipy(self):
        for n in (3, 9, 25):
            d = np.arange(1, n+1, dtype=float)
            d[::3] *= -1
            got = exact_wilcoxon(d)
            expected = wilcoxon(d, alternative='two-sided', method='exact')
            self.assertAlmostEqual(got['pvalue'], expected.pvalue)
            self.assertAlmostEqual(got['statistic'], expected.statistic)

    def test_ties_and_zeros_bruteforce(self):
        d = np.array([0., 1., -1., 2., 2., -4.])
        ranks = rankdata(np.abs(d[d != 0]))
        observed = ranks[d[d != 0] > 0].sum()
        distribution = np.array([sum(r*s for r,s in zip(ranks, signs))
                                 for signs in itertools.product([0,1], repeat=len(ranks))])
        expected = min(1., 2*min(np.mean(distribution <= observed), np.mean(distribution >= observed)))
        self.assertEqual(exact_wilcoxon(d)['pvalue'], expected)
        self.assertEqual(exact_wilcoxon(np.zeros(9))['pvalue'], 1.)

    def test_integer_counts_do_not_overflow(self):
        self.assertEqual(exact_wilcoxon(np.arange(1.,94.))['pvalue'], 2/2**93)

    def test_holm(self):
        np.testing.assert_allclose(holm([.04,.001,.03]), [.06,.003,.06])

    def test_bootstrap_and_pair_direction(self):
        seeds = [1,2,3]
        a = {s:{'a':{'m':s},'b':{'m':s+2}} for s in seeds}
        b = {s:{'a':{'m':s+1},'b':{'m':s+4}} for s in seeds}
        paired, excluded = match_conditions(a,b,seeds,'m')
        self.assertEqual([r['difference'] for r in paired], [-1.,-2.])
        self.assertEqual(excluded, [])
        self.assertEqual(paired_bootstrap([-1.,-2.], 123), (-2.,-1.))
        a[3]['a']['m'] = float('nan')
        paired, excluded = match_conditions(a,b,seeds,'m')
        self.assertEqual(len(paired), 1)
        self.assertEqual(excluded[0]['ours_nonfinite_seeds'], [3])
        del b[1]['a']
        with self.assertRaises(ValueError): match_conditions(a,b,seeds,'m')

if __name__ == '__main__': unittest.main()
