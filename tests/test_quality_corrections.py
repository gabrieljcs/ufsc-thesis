import unittest
from unittest.mock import patch

from thesis_eval.benchmarks.belebele import canonical_question_id, compute_if_cons
from thesis_eval.analysis.reporting import collinearity_diagnostics, tokenizer_diagnostics
from thesis_eval.evaluation.strongreject import apply_strongreject_result, _evaluate_strongreject


class QualityCorrectionTests(unittest.TestCase):
    def test_reordered_parallel_questions_have_identity_matched_cons(self):
        def row(language, position, question, answer):
            return dict(model='m', language=language, item_id=f'belebele_{position:04d}',
                        canonical_item_id=canonical_question_id(dict(link='https://example.test/p',question_number=question)),
                        predicted_choice=answer, gold=answer)
        rows=[row('eng',1,1,'A'),row('eng',2,2,'B'),row('rus',1,2,'B'),row('rus',2,1,'A')]
        result=compute_if_cons(rows,{'m':'eng'})
        self.assertTrue(all(r['cons_score']==1 for r in result))
        self.assertEqual(result[1]['n_cons_comparable'],2)
        self.assertLess(result[1]['cons_ci_low'],1)

    def test_legacy_positional_and_duplicate_keys_are_rejected(self):
        row=dict(model='m',language='eng',item_id='belebele_0001',prediction='A',gold='A')
        with self.assertRaisesRegex(ValueError,'migrated'):
            compute_if_cons([row],{'m':'eng'})
        row['canonical_item_id']='question-1'
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            compute_if_cons([row,row],{'m':'eng'})

    def test_unparseable_aligned_answer_does_not_become_disagreement(self):
        rows=[dict(model='m',language='eng',item_id='q',prediction='',gold='A'),
              dict(model='m',language='rus',item_id='q',prediction='A',gold='A')]
        result=compute_if_cons(rows,{'m':'eng'})
        self.assertTrue(all(r['cons_score'] is None and r['n_cons_comparable']==0 for r in result))

    def test_full_design_detects_model_confounding_missed_by_pairwise_vif(self):
        rows=[]
        for group in range(3):
            for j in range(12):
                rows.append(dict(model=f'm{group}',model_alignment_pole='weak',excluded=False,
                                 unsafe_binary=j%2,distance=j/12,spec_score=group*10+(j%3)*.01))
        values={r['diagnostic']:r['value'] for r in collinearity_diagnostics(rows)}
        self.assertGreater(values['vif_spec'],5)
        self.assertTrue(values['single_predictor_rerun_triggered'])

    def test_unknown_context_risk_stays_missing(self):
        row=dict(model='m',attack_language='eng',token_inflation=1,tokens_per_char=.2,truncation_risk=None)
        result=tokenizer_diagnostics([row])[0]
        self.assertIsNone(result['truncation_risk_rate'])
        self.assertEqual(result['n_context_usage'],0)

    def test_primary_rubric_retains_raw_judge_and_identity(self):
        raw='1.b 0\n2.b 5\n3.b 5'
        with patch('thesis_eval.evaluation.strongreject._call_openai_chat_completions',return_value=raw):
            result=_evaluate_strongreject('synthetic request','synthetic response','openai_rubric',judge_model='test-judge')
        saved=apply_strongreject_result({},result)
        self.assertEqual(saved['raw_judge_response'],raw)
        self.assertEqual(saved['judge_model'],'test-judge')
        self.assertTrue(saved['rubric_sha256'].startswith('sha256:'))
        self.assertEqual(saved['unsafe_binary'],1)

    def test_nonpositive_covariance_is_not_reported_as_converged(self):
        from types import SimpleNamespace
        import numpy as np
        import pandas as pd
        from thesis_eval.analysis.glmm import _fit_laplace_map, MAIN_PREDICTORS
        result=SimpleNamespace(optim_retvals=SimpleNamespace(success=True,jac=np.zeros(2),fun=1.0,message='success'),
                               cov_params=lambda:np.diag([1.0,-0.1]))
        with patch('statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM.from_formula') as factory:
            factory.return_value.fit_map.return_value=result
            with self.assertRaisesRegex(RuntimeError,'convergence diagnostics'):
                _fit_laplace_map(pd.DataFrame(),MAIN_PREDICTORS,fit='invalid-covariance')


if __name__=='__main__':
    unittest.main()
