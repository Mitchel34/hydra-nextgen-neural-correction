import math
from dataclasses import replace
import unittest
from hydra_correctors.evaluation import (
    HOUR as H, ForecastKey as K, Partition as P, Observation as O,
    UpstreamLineage as L, StackedPrediction as S, issue_available_value,
    fit_unique_hour_scale, validate_disjoint_stack, score_paired)


class ClockTests(unittest.TestCase):
    def test_individual_lead_and_availability_boundaries(self):
        p=P('a','fit',0,101*H,label_available_by=101*H)
        self.assertTrue(p.accepts(K('g',99*H,1),101*H))
        self.assertFalse(p.accepts(K('g',99*H,2),102*H))
        self.assertFalse(p.accepts(K('g',99*H,1),102*H))
        self.assertFalse(P('b','assess',100*H,130*H).accepts(K('g',99*H,2),102*H))

    def test_strict_clocks_and_lead_arithmetic(self):
        k=K('g',1700002800,18)
        self.assertEqual(k.valid_epoch,1700002800+64800)
        for issue in (1,0.0,True):
            with self.assertRaises(ValueError): K('g',issue,1)
        for lead in (0,19,True,1.0):
            with self.assertRaises(ValueError): K('g',0,lead)
        with self.assertRaises(ValueError): P('x','fit',0,H)
        with self.assertRaises(ValueError): P('x','fit',0,H,label_available_by=0)
        with self.assertRaises(ValueError): P('x','assess',0,H).accepts(K('g',0,1),0)

    def test_minimum_delay_and_three_six_hour_sensitivities(self):
        row=O('g',10*H,10*H,2.,True,'assumed')
        with self.assertRaises(ValueError): issue_available_value(row,10*H)
        self.assertEqual(issue_available_value(row,11*H)['effective_available_epoch'],11*H)
        for delay in (3,6):
            with self.assertRaises(ValueError): issue_available_value(row,(10+delay-1)*H,delay)
            self.assertEqual(issue_available_value(row,(10+delay)*H,delay)['age_hours'],delay)
        p=P('a','fit',0,12*H,label_available_by=12*H)
        self.assertTrue(p.accepts(K('g',10*H,1),11*H))
        self.assertFalse(p.accepts(K('g',10*H,1),11*H,3))
        for delay in (0,True,1.0):
            with self.assertRaises(ValueError): issue_available_value(row,20*H,delay)

    def test_future_value_rejected_and_missing_not_imputed(self):
        o=O('g',10*H,11*H,3.,True,'source')
        self.assertEqual(issue_available_value(o,11*H)['age_hours'],1)
        with self.assertRaises(ValueError): issue_available_value(o,10*H)
        result=issue_available_value(replace(o,value=None),11*H)
        self.assertFalse(result['mask']);self.assertIsNone(result['value'])
        self.assertFalse(issue_available_value(replace(o,approved=False),11*H)['mask'])
        with self.assertRaises(ValueError): replace(o,value=math.nan)


class ScaleTests(unittest.TestCase):
    def setUp(self):
        self.p=P('a','fit',0,10*H,label_available_by=10*H)
        self.rows=[O('g',H,2*H,1.,True,'s'),O('g',2*H,3*H,3.,True,'s')]

    def test_unique_fit_scaling_and_masks(self):
        s=fit_unique_hour_scale(self.rows,self.p)
        self.assertEqual((s.count,s.mean,s.std),(2,2.,1.))
        tail=O('g',9*H,11*H,1e12,True,'s')
        masked=O('g',3*H,4*H,1e12,False,'s')
        self.assertEqual(fit_unique_hour_scale(self.rows+[tail,masked],self.p),s)

    def test_delay_excludes_tail_from_training_scale(self):
        tail=O('g',8*H,8*H,1e6,True,'s')
        default=fit_unique_hour_scale(self.rows+[tail],self.p)
        delayed=fit_unique_hour_scale(self.rows+[tail],self.p,3)
        self.assertEqual(default.count,3);self.assertEqual(delayed.count,2)
        self.assertEqual(delayed.mean,2.)

    def test_future_extreme_and_duplicate_rows_cannot_enter(self):
        for rows in (self.rows+[self.rows[0]],self.rows+[O('g',11*H,12*H,1e12,False,'s')]):
            with self.assertRaises(ValueError): fit_unique_hour_scale(rows,self.p)
        with self.assertRaises(ValueError): fit_unique_hour_scale(self.rows,replace(self.p,role='select'))
        with self.assertRaises(ValueError): fit_unique_hour_scale([self.rows[0],replace(self.rows[1],value=1.)],self.p)


class StackingTests(unittest.TestCase):
    def setUp(self):
        a=P('a','fit',0,10*H,label_available_by=10*H)
        b=P('b','select',10*H,20*H,label_available_by=20*H)
        self.c=P('c','fit',20*H,30*H,label_available_by=30*H)
        self.lineage=L('m','s','c','sc','replay',(a,),(b,),(a,),20*H,True)
        self.key=K('g',20*H,1)
        self.row=S(self.key,2.,self.lineage)

    def test_disjoint_lineage_and_support(self):
        out=validate_disjoint_stack([self.row],[self.key],self.c)
        self.assertEqual(out['method'],'disjoint_chronological_blocks')
        self.assertFalse(out['native_replay_authenticated'])

    def test_fit_selection_scaler_overlap_all_rejected(self):
        for field,role in (('fit_support','fit'),('selection_support','select'),('scaler_support','fit')):
            with self.subTest(field=field):
                p=P('bad',role,20*H,22*H,label_available_by=22*H)
                with self.assertRaises(ValueError):
                    lineage=replace(self.lineage,**{field:(p,), 'frozen_before_epoch':22*H})
                    validate_disjoint_stack([replace(self.row,upstream=lineage)],[self.key],self.c)

    def test_scaler_on_selection_and_overlapping_upstream_labels_rejected(self):
        # Both precede outlet C: mere disjointness from C would wrongly pass.
        wrong_scale=P('sc','fit',10*H,18*H,label_available_by=18*H)
        with self.assertRaises(ValueError): replace(self.lineage,scaler_support=(wrong_scale,))
        overlap=P('overlap','select',9*H,20*H,label_available_by=20*H)
        with self.assertRaises(ValueError): replace(self.lineage,selection_support=(overlap,))

    def test_changed_model_or_replay_identity_and_missing_rows(self):
        key2=K('g',21*H,1)
        for field in ('model_id','configuration_id','scaler_id','source_id','replay_id'):
            other=S(key2,2.,replace(self.lineage,**{field:'changed'}))
            with self.assertRaises(ValueError): validate_disjoint_stack([self.row,other],[self.key,key2],self.c)
        with self.assertRaises(ValueError): validate_disjoint_stack([self.row],[self.key,key2],self.c)
        with self.assertRaises(ValueError): validate_disjoint_stack([self.row,self.row],[self.key],self.c)
        with self.assertRaises(ValueError): replace(self.lineage,frozen_weight_replay=False)
        with self.assertRaises(ValueError): replace(self.lineage,replay_id='')

    def test_crossing_stack_targets_rejected(self):
        key=K('g',29*H,1)
        with self.assertRaises(ValueError): validate_disjoint_stack([S(key,2.,self.lineage)],[key],self.c)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.keys=[K('g',0,1),K('g',H,1),K('g',0,2)]
        self.obs=dict.fromkeys(self.keys,1.)

    def test_neutral_zero_and_equal_lead_weight_not_row_weight(self):
        self.assertEqual(score_paired(self.keys,self.obs,self.obs,{'g':2})['primary'],0)
        pred={self.keys[0]:1.,self.keys[1]:1.,self.keys[2]:5.}
        result=score_paired(self.keys,self.obs,pred,{'g':2})
        self.assertEqual(result['primary'],1.) # mean(0/2,4/2), not pooled RMS
        self.assertEqual(result['cells'][1]['pbias_percent'],400)

    def test_missing_labels_preserved_and_candidate_failure_not_deleted(self):
        obs=dict(self.obs);obs[self.keys[0]]=None
        result=score_paired(self.keys,obs,self.obs,{'g':1})
        self.assertEqual(result['missing_target_rows'],1)
        self.assertEqual(result['scored_rows'],2)
        for pred in ({self.keys[0]:1.},dict(self.obs,**{})):
            if len(pred)==len(self.obs): pred[self.keys[0]]=math.nan
            with self.assertRaises(ValueError): score_paired(self.keys,obs,pred,{'g':1})
        with self.assertRaises(ValueError): score_paired(self.keys+self.keys,self.obs,self.obs,{'g':1})

    def test_scale_zero_support_and_zero_observation_sum(self):
        for scale in (0,-1,math.nan):
            with self.assertRaises(ValueError): score_paired(self.keys,self.obs,self.obs,{'g':scale})
        obs=dict(self.obs);obs[self.keys[2]]=None
        with self.assertRaises(ValueError): score_paired(self.keys,obs,self.obs,{'g':1})
        zeros=dict.fromkeys(self.keys,0.)
        result=score_paired(self.keys,zeros,zeros,{'g':1})
        self.assertIsNone(result['cells'][0]['pbias_percent'])
        self.assertEqual(result['cells'][0]['pbias_undefined'],'zero_observation_sum')

    def test_nonfinite_derived_metrics_fail_visibly(self):
        key=K('g',0,1)
        cases=[(1.,1e308,1.),(-1e308,1e308,1.),(1e-308,1.,1.),(1.,2.,1e-320)]
        for observed,predicted,scale in cases:
            with self.subTest(case=(observed,predicted,scale)):
                with self.assertRaises(ValueError): score_paired([key],{key:observed},{key:predicted},{'g':scale})
        huge=dict.fromkeys(self.keys,1e308)
        with self.assertRaises(ValueError): score_paired(self.keys,huge,huge,{'g':1})

    def test_equal_gauge_then_equal_lead_aggregation(self):
        key=K('h',0,1);keys=self.keys+[key]
        obs={**self.obs,key:1.};pred={**self.obs,key:5.}
        result=score_paired(keys,obs,pred,{'g':1,'h':2})
        self.assertEqual(result['primary'],1.) # gauge g=0, gauge h=2


if __name__=='__main__': unittest.main()
