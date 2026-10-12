"""Generated values only. Assertions test event order, ownership and toy algebra."""
from dataclasses import FrozenInstanceError, replace
import math
import unittest

from hydra_correctors.process_boundary import (HOUR, DEMAND, LIQUID, UNITS, ContractError, ToyModel, Identity,
                      Input, Sentinel, Readback, UpdateResult, Decision, MockPhysical,
                      Controller, ToyObjectiveSpec, ToyEvaluator, bounded_multiplier, symmetric_pair)


def policy(frame, state, model):
    return Decision(frame.identity, bounded_multiplier(model.parameters[0]),
                    Sentinel(replace(frame.identity, epoch=frame.identity.epoch+HOUR),
                             tuple(v+1.0 for v in state.values)))


def create(*, model=None, catchment="cat-1", trajectory="parent", physical_type=MockPhysical, callback=policy):
    model = model or ToyModel("synthetic-bounded-head", (0.0,))
    identity = Identity(catchment, trajectory, 0, 0, model.model_id, "fixed-unit-scale-no-fit", "generated-inputs", "generated-support")
    physical = physical_type(identity, 0.4)
    return Controller(physical, model, identity.scaler_id, callback), physical


def stage(controller, *, demand=1e-6, liquid=2.0, reverse=False):
    names = (LIQUID, DEMAND) if reverse else (DEMAND, LIQUID)
    for name in names:
        controller.set_input(Input(controller.identity, name, demand if name == DEMAND else liquid, UNITS[name]))


class BoundaryTests(unittest.TestCase):
    def test_order_and_repeated_reads_are_causally_identical_with_one_decision(self):
        frames_a, frames_b = [], []
        def recording(log):
            def fn(frame, state, model):
                log.append(frame)
                return policy(frame, state, model)
            return fn
        a, pa = create(callback=recording(frames_a))
        b, pb = create(callback=recording(frames_b))
        stage(a); stage(b, reverse=True)
        for _ in range(9):
            self.assertEqual(a.frame(), b.frame())
            self.assertEqual(a.frame().values, (1e-6, 2.0, 0.4))
        self.assertEqual(frames_a+frames_b, [])
        a.step(); b.step()
        self.assertEqual(frames_a, frames_b)
        self.assertEqual(pa.snapshot(), pb.snapshot())
        self.assertEqual(a.sentinel, b.sentinel)
        self.assertEqual(a.audit()["policy_calls"], 1)
        self.assertEqual(a.audit()["physical_update_calls"], 1)
        self.assertEqual(a.sentinel.values, (1.0,)*32)

    def test_raw_demand_is_not_previous_scaled_or_post_update_value(self):
        model=ToyModel("synthetic-bounded-head", (0.8,)); c, physical=create(model=model)
        stage(c,demand=0.00001,liquid=3.)
        frame=c.frame();self.assertEqual(frame.pre_update_soil_m,0.4)
        result=c.step();expected_factor=math.exp(math.log(2)*math.tanh(.8))
        self.assertEqual(physical.snapshot()["history"][0]["demand_m_s"],0.00001*expected_factor)
        self.assertNotEqual(frame.pre_update_soil_m,result.soil_m)
        stage(c,demand=0.00002,liquid=0.)
        self.assertEqual(c.frame().raw_demand_m_s,0.00002)
        self.assertEqual(c.frame().pre_update_soil_m,result.soil_m)
        c.step();self.assertEqual(c.audit()["policy_calls"],2)

    def test_neutral_submission_is_exact_and_toy_algebra_is_independent(self):
        c,p=create();stage(c,demand=0.0000123456789,liquid=5.)
        c.step();self.assertEqual(p.snapshot()["history"][0]["demand_m_s"],0.0000123456789)
        self.assertAlmostEqual(p.snapshot()["soil_m"],0.405-0.0000123456789*3600,places=15)
        self.assertEqual(bounded_multiplier(0.),1.)
        for raw in (-1e300,-10.,-.2,.2,10.,1e300):
            self.assertGreaterEqual(bounded_multiplier(raw),.5)
            self.assertLessEqual(bounded_multiplier(raw),2.)

    def test_missing_stale_duplicate_and_cross_owner_inputs_fail(self):
        for kind in ("missing","duplicate","stale","catchment","trajectory","model","scaler","support","input"):
            with self.subTest(kind=kind):
                c,_=create()
                if kind=="missing":
                    c.set_input(Input(c.identity,DEMAND,1e-6,UNITS[DEMAND]))
                    action=c.step
                elif kind=="duplicate":
                    packet=Input(c.identity,DEMAND,1e-6,UNITS[DEMAND]);c.set_input(packet)
                    action=lambda:c.set_input(packet)
                else:
                    field={"stale":"epoch","catchment":"catchment_id","trajectory":"trajectory_id",
                           "model":"model_id","scaler":"scaler_id","support":"support_id","input":"input_id"}[kind]
                    wrong=replace(c.identity,**{field:HOUR if field=="epoch" else "other"})
                    action=lambda:c.set_input(Input(wrong,DEMAND,1e-6,UNITS[DEMAND]))
                with self.assertRaises(ContractError):action()
                self.assertEqual(c.audit()["policy_calls"],0)
                self.assertEqual(c.audit()["physical_update_calls"],0)
                self.assertEqual(c.audit()["phase"],"POISONED")

    def test_invalid_input_metadata_and_nonfinite_values_poison(self):
        variants=[{"status":"FAILED"},{"units":"m"},{"dtype":"float"},{"itemsize":4},{"nbytes":16},
                  {"value":-1.},{"value":math.nan},{"value":math.inf},{"value":True},{"variable":"RAIN_RATE"}]
        for change in variants:
            with self.subTest(change=change):
                c,_=create();packet=replace(Input(c.identity,DEMAND,1e-6,UNITS[DEMAND]),**change)
                with self.assertRaises(ContractError):c.set_input(packet)
                with self.assertRaises(ContractError):c.step()

    def test_second_step_without_fresh_inputs_fails_and_no_second_decision(self):
        c,_=create();stage(c);c.step()
        with self.assertRaises(ContractError):c.step()
        self.assertEqual(c.audit()["policy_calls"],1)
        self.assertEqual(c.audit()["physical_update_calls"],1)

    def test_pre_update_getter_contract_rejects_wrong_status_time_units_and_nan(self):
        for change in ({"status":"FAILED"},{"units":"mm"},{"value":math.nan},{"nbytes":16},{"dtype":"float"}):
            with self.subTest(change=change):
                class BadRead(MockPhysical):
                    def read_soil(self):return replace(super().read_soil(),**change)
                c,_=create(physical_type=BadRead);stage(c)
                with self.assertRaises(ContractError):c.step()
                self.assertEqual(c.audit()["policy_calls"],0)
        class WrongClock(MockPhysical):
            def read_soil(self):
                out=super().read_soil();return replace(out,identity=replace(out.identity,epoch=HOUR))
        c,_=create(physical_type=WrongClock);stage(c)
        with self.assertRaisesRegex(ContractError,"soil status/units/type/clock"):c.step()

    def test_bad_policy_factor_and_state_never_reach_physical_update(self):
        for bad in (-1.,.49,2.01,math.nan,math.inf,True):
            with self.subTest(factor=bad):
                def bad_policy(frame,state,model):return replace(policy(frame,state,model),multiplier=bad)
                c,_=create(callback=bad_policy);stage(c)
                with self.assertRaises(ContractError):c.step()
                self.assertEqual(c.audit()["physical_update_calls"],0)
        def wrong_state(frame,state,model):return Decision(frame.identity,1.,state)
        c,_=create(callback=wrong_state);stage(c)
        with self.assertRaisesRegex(ContractError,"sentinel"):c.step()
        with self.assertRaises(ContractError):c.fork("child")

    def test_mutating_then_failed_update_poison_preserves_partial_state(self):
        class Failed(MockPhysical):
            def update(self,*args):
                super().update(*args)
                self._soil_m=.123  # deliberately mutate before reporting failure
                return UpdateResult(self._identity,.123,"FAILED")
        c,p=create(physical_type=Failed);stage(c);before=c.sentinel
        with self.assertRaisesRegex(ContractError,"update status"):c.step()
        self.assertEqual(p.snapshot()["soil_m"],.123)
        self.assertEqual(p.snapshot()["identity"].epoch,HOUR)
        self.assertEqual(c.sentinel,before)
        self.assertEqual(c.audit()["identity"].epoch,0)
        self.assertFalse(c.audit()["rollback_claim"])
        for action in (c.step,c.frame,lambda:c.fork("child"),lambda:stage(c)):
            with self.assertRaisesRegex(ContractError,"poisoned"):action()

    def test_exception_invalid_return_clock_and_state_poison(self):
        for mode in ("exception","clock","nan","mismatch","negative","type"):
            with self.subTest(mode=mode):
                class BadUpdate(MockPhysical):
                    def update(self,*args):
                        value=super().update(*args)
                        if mode=="exception":raise RuntimeError("after mutation")
                        if mode=="clock":return replace(value,identity=replace(value.identity,epoch=2*HOUR))
                        if mode=="nan":return replace(value,soil_m=math.nan)
                        if mode=="mismatch":return replace(value,soil_m=.999)
                        if mode=="negative":return replace(value,soil_m=-.1)
                        return None
                c,_=create(physical_type=BadUpdate);stage(c)
                with self.assertRaises((ContractError,RuntimeError)):c.step()
                self.assertEqual(c.audit()["phase"],"POISONED")
                self.assertEqual(c.sentinel.identity.epoch,0)

    def test_policy_physical_mutation_is_rejected_before_update(self):
        holder={}
        def bad(frame,state,model):
            holder["physical"]._soil_m+=1
            return policy(frame,state,model)
        c,p=create(callback=bad);holder["physical"]=p;stage(c)
        with self.assertRaisesRegex(ContractError,"policy mutated"):c.step()
        self.assertEqual(c.audit()["physical_update_calls"],0)

    def test_double_correction_owner_is_rejected(self):
        c,p=create()
        with self.assertRaisesRegex(ContractError,"duplicate correction"):
            Controller(p,c.model,c.identity.scaler_id,policy)

    def test_identity_model_and_sentinel_are_immutable(self):
        c,_=create()
        for obj,attr,value in ((c.identity,"catchment_id","other"),(c.model,"parameters",(.1,)),(c.sentinel,"values",(1.,)*32)):
            with self.assertRaises(FrozenInstanceError):setattr(obj,attr,value)
        with self.assertRaises(AttributeError):c.model=ToyModel("other",(.1,))
        with self.assertRaises(ContractError):Sentinel(c.identity,[0.]*32)
        with self.assertRaises(ContractError):Sentinel(c.identity,(0.,)*31)
        with self.assertRaises(ContractError):ToyModel("x",(math.nan,))

    def test_fork_only_clean_and_full_independent_physical_controller_state(self):
        parent,p=create();parent.set_input(Input(parent.identity,LIQUID,2.,UNITS[LIQUID]))
        with self.assertRaisesRegex(ContractError,"clean boundary"):parent.fork("early")
        parent.set_input(Input(parent.identity,DEMAND,1e-6,UNITS[DEMAND]));parent.step()
        original=parent.audit();a=parent.fork("a");b=parent.fork("b")
        self.assertIsNot(a._physical,p);self.assertIsNot(a._physical._history,p._history)
        self.assertIsNot(a._physical._history,b._physical._history)
        self.assertIsNot(a.sentinel,parent.sentinel)
        self.assertEqual(a.sentinel.values,parent.sentinel.values)
        stage(a,demand=2e-6,liquid=25.);a.step()
        self.assertEqual(parent.audit(),original)
        self.assertEqual(b.sentinel.values,(1.,)*32)
        self.assertEqual(a.sentinel.values,(2.,)*32)
        self.assertEqual(b._physical.snapshot()["soil_m"],p.snapshot()["soil_m"])
        a._physical._history[0]["demand_m_s"]=999.
        self.assertNotEqual(a._physical._history,p._history)
        with self.assertRaisesRegex(ContractError,"duplicate branch"):parent.fork("a")

    def test_siblings_same_inputs_match_and_catchments_do_not_share_memory(self):
        parent,_=create();stage(parent);parent.step();a=parent.fork("a");b=parent.fork("b")
        stage(a);stage(b,reverse=True);a.step();b.step()
        self.assertEqual(a._physical.snapshot()["soil_m"],b._physical.snapshot()["soil_m"])
        self.assertEqual(a.sentinel.values,b.sentinel.values)
        other,_=create(catchment="cat-2");before=other.audit()
        stage(a);a.step();self.assertEqual(other.audit(),before)

    def test_model_or_scaler_change_requires_original_replay(self):
        c,p=create();stage(c);c.step();new=ToyModel(c.model.policy_id,(.5,))
        with self.assertRaisesRegex(ContractError,"origin replay"):c.fork("changed",model=new)
        with self.assertRaisesRegex(ContractError,"origin replay"):c.fork("changed",scaler_id="new-scaler")
        with self.assertRaisesRegex(ContractError,"fresh origin replay"):Controller(p,new,"new-scaler",policy)
        fresh,_=create(model=new);self.assertEqual(fresh.identity.epoch,fresh.identity.origin_epoch)
        self.assertEqual(fresh.sentinel.values,(0.,)*32)

    def test_replay_from_origin_reproduces_mock_forward_state(self):
        a,_=create();b,_=create()
        for i in range(5):
            stage(a,demand=(i+1)*1e-6,liquid=float(i));a.step()
        for i in range(5):
            stage(b,demand=(i+1)*1e-6,liquid=float(i),reverse=True);b.step()
        self.assertEqual(a.audit(),b.audit())


class PairTests(unittest.TestCase):
    def setUp(self):self.spec=ToyObjectiveSpec(0,"generated-inputs","generated-support","fixed-no-fit")

    def test_scalar_symmetric_estimate_matches_exact_quadratic_derivative(self):
        made=[]
        def factory():
            e=ToyEvaluator(self.spec,(1.,));made.append(e);return e
        result=symmetric_pair((3.,),(1,),.25,self.spec,factory)
        self.assertEqual(result["J_plus"],2.25**2)
        self.assertEqual(result["J_minus"],1.75**2)
        self.assertEqual(result["symmetric_estimate"],(4.,))
        self.assertEqual(result["objective_calls"],2);self.assertEqual(result["optimizer_steps"],0)
        self.assertIsNot(made[0],made[1]);self.assertIsNot(made[0].state,made[1].state)
        self.assertEqual([e.calls for e in made],[1,1]);self.assertTrue(all(e.spec==self.spec for e in made))

    def test_simultaneous_estimate_is_not_claimed_as_exact_multidimensional_gradient(self):
        result=symmetric_pair((2.,3.),(1,-1),.5,self.spec,lambda:ToyEvaluator(self.spec,(0.,0.)))
        self.assertEqual(result["J_plus"],12.5);self.assertEqual(result["J_minus"],14.5)
        self.assertEqual(result["symmetric_estimate"],(-2.,2.))
        self.assertNotEqual(result["symmetric_estimate"],(4.,6.))
        self.assertFalse(result["native_gradient_claim"])

    def test_same_used_or_shared_mutable_evaluator_is_rejected(self):
        one=ToyEvaluator(self.spec,(0.,))
        with self.assertRaisesRegex(ContractError,"independent"):symmetric_pair((1.,),(1,),.1,self.spec,lambda:one)
        self.assertEqual(one.calls,0)
        used=ToyEvaluator(self.spec,(0.,));used.evaluate((1.,));fresh=ToyEvaluator(self.spec,(0.,));pool=iter([used,fresh])
        with self.assertRaisesRegex(ContractError,"fresh"):symmetric_pair((1.,),(1,),.1,self.spec,lambda:next(pool))
        a=ToyEvaluator(self.spec,(0.,));b=ToyEvaluator(self.spec,(0.,));b.state=a.state;pool=iter([a,b])
        with self.assertRaisesRegex(ContractError,"independent"):symmetric_pair((1.,),(1,),.1,self.spec,lambda:next(pool))

    def test_changed_input_support_scaler_origin_or_objective_fails_before_calls(self):
        for key,value in (("input_id","changed"),("support_id","changed"),("scaler_id","changed"),("origin_epoch",HOUR)):
            with self.subTest(key=key):
                a=ToyEvaluator(self.spec,(0.,));b=ToyEvaluator(replace(self.spec,**{key:value}),(0.,));pool=iter([a,b])
                with self.assertRaisesRegex(ContractError,"identical objective"):symmetric_pair((1.,),(1,),.1,self.spec,lambda:next(pool))
                self.assertEqual(a.calls+b.calls,0)
        pool=iter([ToyEvaluator(self.spec,(0.,)),ToyEvaluator(self.spec,(1.,))])
        with self.assertRaises(ContractError):symmetric_pair((1.,),(1,),.1,self.spec,lambda:next(pool))

    def test_invalid_perturbation_nonfinite_and_runtime_substitution_fail(self):
        factory=lambda:ToyEvaluator(self.spec,(0.,))
        for theta,delta,radius in (((math.nan,),(1,),.1),((1.,),(0,),.1),((1.,),(True,),.1),((1.,),(1,),0.),((1.,),(1,),math.inf)):
            with self.subTest(theta=theta,delta=delta,radius=radius),self.assertRaises(ContractError):
                symmetric_pair(theta,delta,radius,self.spec,factory)
        with self.assertRaisesRegex(ContractError,"analytic toy"):symmetric_pair((1.,),(1,),.1,self.spec,lambda:object())
        with self.assertRaisesRegex(ContractError,"toy arithmetic"):ToyObjectiveSpec(0,"x","y","z",True)


if __name__ == "__main__":
    unittest.main()
