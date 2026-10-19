import unittest
import numpy as np
import pandas as pd
from thesis_eval.analysis.magnitude import summarize, fit_outcomes


class MagnitudeTests(unittest.TestCase):
    def test_decomposition_and_prompt_bootstrap(self):
        frame=pd.DataFrame({'prompt_id':['a','a','b','b'],'strongreject_score':[0.,.25,.75,1.]})
        result=summarize(frame,draws=1000)
        self.assertAlmostEqual(result['mean_score'],result['asr']*result['positive_mean'])
        self.assertEqual(result['n_prompts'],2)
        self.assertEqual(result,summarize(frame,draws=1000))
        self.assertEqual(result['mean_ci_low'],.125)
        self.assertEqual(result['mean_ci_high'],.875)

    def test_invalid_scores_rejected(self):
        with self.assertRaises(ValueError):
            summarize(pd.DataFrame({'prompt_id':['a'],'strongreject_score':[1.1]}))

    def test_same_binary_outcome_different_magnitude(self):
        rng=np.random.default_rng(17); rows=[]
        for prompt in range(80):
            for model in ['a','b']:
                for language in ['a','b','c']:
                    x=rng.normal()
                    score=0. if rng.random()<.4 else (.25 if x<0 else .875)
                    rows.append(dict(prompt_id=str(prompt),model=model,attack_language=language,x=x,strongreject_score=score))
        effects,diagnostics,_=fit_outcomes(pd.DataFrame(rows),('x',),'synthetic')
        e={r['outcome']:r for r in effects}
        self.assertEqual(e['binary']['n'],e['score']['n'])
        self.assertLess(abs(e['binary']['estimate']),.3)
        self.assertGreater(e['score']['estimate'],.5)
        self.assertGreater(e['positive_score']['estimate'],1.)
        self.assertTrue(all(d['valid'] for d in diagnostics))
        self.assertTrue(all(e[k]['average_derivative']>0 for k in ['score','positive_score']))

if __name__=='__main__': unittest.main()
