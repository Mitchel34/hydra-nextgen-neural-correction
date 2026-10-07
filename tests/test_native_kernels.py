"""Generated untrained arithmetic fixtures; no ML backend or physical model."""
import ctypes as ct
from dataclasses import FrozenInstanceError
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import unittest

from hydra_correctors import kernels as r

COUNTS={"C_success":0,"C_rejected":0,"Python_success":0,"Python_rejected":0}
EVIDENCE={}
MAX_CALLS=4096
TOL={"absolute":2e-14,"relative":2e-13}
ROSTER={"test_cases":15,"successful_forward_calls_max":MAX_CALLS,"agreement_tolerance":TOL,
        "models":["mlp16 d3,81 parameters","gru32 d3,3489 parameters,R/Z/N reset-after,bhn fixed0"],
        "source_inference_only":True,"fits":0,"real_data":False,"MLX_GPU_BMI_CFE_VM_calls":0}


def count(name):
    COUNTS[name]+=1
    if COUNTS["C_success"]+COUNTS["Python_success"]>MAX_CALLS:raise RuntimeError("Prospective inference-call ceiling")


def before_forward():
    if COUNTS["C_success"]+COUNTS["Python_success"]>=MAX_CALLS:raise RuntimeError("No successful-forward allowance remains")


def python(model,raw,h=None):
    before_forward()
    try:out=model.forward(raw,h)
    except (ValueError,ArithmeticError,OverflowError):count("Python_rejected");raise
    count("Python_success");return out


def model(arch,params=None,scaler=None):
    # Fixtures supply explicit immutable scalers; the model API has no defaults.
    if scaler is None:scaler=r.Scaler((0.0,0.0,0.0),(1.0,1.0,1.0))
    if params is None:params=(0.0,)*r.layout(arch)["parameter_count"]
    return r.Model(arch,tuple(params),scaler)


def array(values,extra=0):return (ct.c_double*(len(values)+extra))(*values)


class Native:
    def __init__(self):
        path=Path(os.environ["HYDRA_KERNEL_LIBRARY"])
        expected=os.environ["HYDRA_KERNEL_LIBRARY_SHA256"]
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise RuntimeError("Built library changed before ctypes load")
        self.lib=ct.CDLL(str(path))
        EVIDENCE["library_sha256_before_load"]=expected
        pair=[ct.POINTER(ct.c_double),ct.c_size_t]
        for arch,pairs in (("mlp16",5),("gru32",6)):
            fn=getattr(self.lib,"hk_"+arch);fn.argtypes=pair*pairs;fn.restype=ct.c_int
    def call(self,arch,arrays,lengths=None,out=None,out_n=None):
        before_forward()
        sizes=[len(x) for x in arrays] if lengths is None else lengths
        n=18 if arch=="mlp16" else 34
        if out is None:out=array((12345.0,)*n)
        before=[bytes(x) for x in arrays if x is not None]
        args=[]
        for values,size in zip(arrays,sizes):args.extend([values,size])
        args.extend([out,n if out_n is None else out_n])
        status=getattr(self.lib,"hk_"+arch)(*args)
        count("C_success" if status==0 else "C_rejected")
        if before!=[bytes(x) for x in arrays if x is not None]:raise AssertionError("C changed a const input")
        return status,out
    def forward(self,model,raw,h=None):
        arrays=[array(raw),array(model.scaler.means),array(model.scaler.scales),array(model.parameters)]
        if model.architecture=="gru32":arrays.append(array(h))
        status,out=self.call(model.architecture,arrays)
        if status:raise ArithmeticError("C kernel status "+str(status))
        return r.Output(out[0],out[1],tuple(out[2:]))


class KernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.native=Native()
    def equal(self,a,b):
        for left,right in zip((a.head,a.factor,*a.hidden),(b.head,b.factor,*b.hidden)):
            self.assertTrue(math.isclose(left,right,abs_tol=TOL["absolute"],rel_tol=TOL["relative"]),(left,right))
    def test_01_zero_head_exact_neutral_with_nonzero_activations(self):
        for arch,offset in (("mlp16",48),("gru32",3360+64)):
            params=[0.0]*r.layout(arch)["parameter_count"];params[offset]=0.7
            m=model(arch,params);h=None if arch=="mlp16" else r.zero_state()
            c=self.native.forward(m,(1.0,0.0,0.5),h);p=python(m,(1.0,0.0,0.5),h)
            self.equal(c,p);self.assertEqual(c.factor,1.0);self.assertEqual(p.factor,1.0)
            self.assertNotEqual(c.hidden[0],0.0)
    def test_02_nontrivial_mlp_analytic_oracle(self):
        params=[0.0]*81;params[:3]=[2.0,-1.0,0.5];params[48]=0.25;params[64]=-0.75;params[80]=0.1
        scale=r.Scaler((1.0,3.0,0.5),(2.0,2.0,0.5));m=model("mlp16",params,scale)
        hidden=math.tanh(3.25);head=0.1-0.75*hidden
        expected=r.Output(head,math.exp(math.log(2.0)*math.tanh(head)),(hidden,)+(0.0,)*15)
        self.equal(self.native.forward(m,(3.0,2.0,1.0)),expected);self.equal(python(m,(3.0,2.0,1.0)),expected)
        EVIDENCE["mlp_analytic"]={"normalized_input":[1.0,-0.5,1.0],"hidden0":hidden,"head":head}
    def test_03_gru_reset_after_offdiagonal_oracle(self):
        params=[0.0]*3489;params[3360]=math.log(3.0);params[3361]=-math.log(3.0)
        params[288+64*32+1]=2.0;params[288+65*32]=-1.0
        params[3360+64]=0.1;params[3360+65]=-0.05;params[3456]=1.0;params[3457]=2.0;params[3488]=0.02
        h=(0.4,-0.2)+(0.0,)*30;m=model("gru32",params)
        after=(0.5*math.tanh(-0.2)+0.2,0.5*math.tanh(-0.15)-0.1)+(0.0,)*30
        before=(0.2,0.5*math.tanh(-0.35)-0.1)+(0.0,)*30
        head=after[0]+2*after[1]+0.02;expected=r.Output(head,r.factor(head),after)
        self.equal(self.native.forward(m,(0.0,0.0,0.0),h),expected);self.equal(python(m,(0.0,0.0,0.0),h),expected)
        self.assertGreater(abs(after[0]-before[0]),0.09);self.assertNotEqual(head,0.02)
        EVIDENCE["reset_convention_oracle"]={"reset_after_h01":after[:2],"reset_before_h01":before[:2],"head_uses_candidate":head}
    def test_04_signed_standardized_inputs_and_hidden_are_valid(self):
        scaler=r.Scaler((2.0,2.0,2.0),(1.0,2.0,4.0))
        self.assertEqual(scaler.normalize((1.0,0.0,0.0)),(-1.0,-1.0,-0.5))
        m=model("gru32",scaler=scaler);h=(-0.5,)*32
        c=self.native.forward(m,(1.0,0.0,0.0),h);p=python(m,(1.0,0.0,0.0),h)
        self.equal(c,p);self.assertEqual(c.hidden,(-0.25,)*32)
    def test_05_factor_bounds_and_extreme_finite_head(self):
        for arch in ("mlp16","gru32"):
            for bias in (-1e300,-2.0,0.0,2.0,1e300):
                params=[0.0]*r.layout(arch)["parameter_count"];params[-1]=bias;m=model(arch,params)
                h=None if arch=="mlp16" else r.zero_state()
                c=self.native.forward(m,(0.0,0.0,0.0),h);self.equal(c,python(m,(0.0,0.0,0.0),h))
                self.assertTrue(0.5<=c.factor<=2.0)
    def test_06_generated_multistep_cold_replay_and_independent_branches(self):
        scale=r.Scaler((2e-8,2.0,0.4),(1e-7,3.0,0.5))
        params=tuple(0.08*math.sin(i*0.17) for i in range(3489));m=model("gru32",params,scale)
        raw=[((k%5)*2e-8,(k%7)*0.75,0.2+0.03*(k%9)) for k in range(24)]
        def replay(sequence,origin):
            cstate=tuple(origin);pstate=tuple(origin);result=[]
            for x in sequence:
                cout=self.native.forward(m,x,cstate);pout=python(m,x,pstate);self.equal(cout,pout)
                cstate=cout.hidden;pstate=pout.hidden;result.append(cout)
            return tuple(result)
        before=r.dumps(m);a=replay(raw,r.zero_state());b=replay(raw,r.zero_state());self.assertEqual(a,b)
        origin=a[7].hidden;unchanged=tuple(origin)
        normal=replay(raw[8:12],origin);pulse=replay(((raw[8][0],20.0,raw[8][2]),*raw[9:12]),origin)
        repeat=replay(raw[8:12],origin)
        self.assertEqual(normal,repeat);self.assertNotEqual(normal,pulse);self.assertEqual(origin,unchanged)
        self.assertEqual(r.dumps(m),before)
        EVIDENCE["replay"]={"cold_steps":24,"branch_steps":4,"cold_exact":True,"sibling_exact":True,"changed_input_changes_candidate":True}
    def test_07_generated_mlp_agreement(self):
        m=model("mlp16",tuple(0.15*math.cos(i*0.11) for i in range(81)),r.Scaler((0.5,1.0,0.3),(0.2,2.0,0.5)))
        for k in range(12):
            raw=(k/20.0,(k%4)*0.5,0.1+k/100.0);self.equal(self.native.forward(m,raw),python(m,raw))
    def arrays(self,arch):
        arrays=[array((0.1,0.2,0.3)),array((0.0,0.0,0.0)),array((1.0,1.0,1.0)),array((0.0,)*r.layout(arch)["parameter_count"])]
        if arch=="gru32":arrays.append(array(r.zero_state()))
        return arrays
    def test_08_invalid_scaling_raw_params_and_state_atomic(self):
        for arch in ("mlp16","gru32"):
            scenarios=[(0,0,-1.0,4),(0,1,math.inf,2),(1,0,math.nan,2),(2,0,0.0,3),(2,1,-1.0,3),
                       (3,7,math.nan,2),(3,0,math.inf,2)]
            if arch=="gru32":scenarios.append((4,31,math.nan,2))
            for arr,i,value,status in scenarios:
                args=self.arrays(arch);args[arr][i]=value;n=18 if arch=="mlp16" else 34;out=array((789.0,)*n);before=bytes(out)
                actual,_=self.native.call(arch,args,out=out);self.assertEqual(actual,status);self.assertEqual(bytes(out),before)
    def test_09_intermediate_overflow_atomic_despite_zero_head(self):
        for arch in ("mlp16","gru32"):
            args=self.arrays(arch);args[0][0]=1e308;args[3][0]=1e308
            out=array((456.0,)*(18 if arch=="mlp16" else 34));before=bytes(out)
            self.assertEqual(self.native.call(arch,args,out=out)[0],6);self.assertEqual(bytes(out),before)
            params=[0.0]*r.layout(arch)["parameter_count"];params[0]=1e308;m=model(arch,params)
            with self.assertRaises(ArithmeticError):python(m,(1e308,0.0,0.0),None if arch=="mlp16" else r.zero_state())
            # Dot-from-zero must reject overflow before a later bias could
            # mathematically cancel it, and allow cancellation inside the dot.
            bias_index=48 if arch=="mlp16" else 3360
            params=[0.0]*r.layout(arch)["parameter_count"]
            params[0]=params[1]=1e308;params[bias_index]=-1e308;m=model(arch,params)
            h=None if arch=="mlp16" else r.zero_state()
            with self.assertRaises(ArithmeticError):python(m,(1.0,1.0,0.0),h)
            with self.assertRaises(ArithmeticError):self.native.forward(m,(1.0,1.0,0.0),h)
            params[1]=-1e308;params[bias_index]=1e308;m=model(arch,params)
            self.equal(self.native.forward(m,(1.0,1.0,0.0),h),python(m,(1.0,1.0,0.0),h))
    def test_10_shape_null_and_output_length_rejected_atomic(self):
        for arch in ("mlp16","gru32"):
            args=self.arrays(arch);sizes=[len(a) for a in args];n=18 if arch=="mlp16" else 34
            for k in range(len(args)):
                sizes_bad=sizes.copy();sizes_bad[k]-=1;out=array((321.0,)*n);before=bytes(out)
                self.assertEqual(self.native.call(arch,args,sizes_bad,out)[0],1);self.assertEqual(bytes(out),before)
            null=args.copy();null[0]=None
            self.assertEqual(self.native.call(arch,null,sizes)[0],1)
            out=array((321.0,)*n);before=bytes(out)
            self.assertEqual(self.native.call(arch,args,out=out,out_n=n-1)[0],1);self.assertEqual(bytes(out),before)
    def test_11_overlap_every_input_rejects_without_mutation(self):
        for arch in ("mlp16","gru32"):
            n=18 if arch=="mlp16" else 34
            for index in range(4 if arch=="mlp16" else 5):
                original=self.arrays(arch);lengths=[len(a) for a in original]
                backing=array(tuple(original[index]),extra=n+2);original[index]=backing
                out=ct.cast(ct.byref(backing,ct.sizeof(ct.c_double)),ct.POINTER(ct.c_double))
                self.assertEqual(self.native.call(arch,original,lengths,out)[0],5)
    def test_12_immutable_model_scaler_and_strict_roundtrip_identities(self):
        for arch in ("mlp16","gru32"):
            m=model(arch);raw=r.dumps(m);clone=r.loads(raw)
            self.assertEqual(clone,m);self.assertEqual(clone.id,m.id);self.assertEqual(r.dumps(clone),raw)
            with self.assertRaises(FrozenInstanceError):m.parameters=(1.0,)
            with self.assertRaises(FrozenInstanceError):m.scaler.means=(1.0,1.0,1.0)
            with self.assertRaises(TypeError):m.parameters[0]=1.0
            changed=model(arch,scaler=r.Scaler((1.0,0.0,0.0),(1.0,1.0,1.0)))
            self.assertNotEqual(m.id,changed.id);self.assertNotEqual(m.scaler.id,changed.scaler.id)
    def test_13_strict_serialization_rejects_corruption_duplicates_and_wrong_layout(self):
        base=model("gru32").document()
        mutations=[lambda d:d.update(extra=True),lambda d:d.update(dtype="float32"),lambda d:d.update(version=True),
                   lambda d:d["parameters"].__setitem__(0,True),lambda d:d["parameters"].append(0.0),
                   lambda d:d["layout"].update(reset="before"),lambda d:d["scaler"]["scales"].__setitem__(0,0.0),
                   lambda d:d["features"][0].update(units="mm h-1"),lambda d:d.update(model_id="0"*64),
                   lambda d:d["scaler"].update(bhn=[0.0]*32)]
        for mutate in mutations:
            d=json.loads(r.canonical(base));mutate(d)
            with self.assertRaises(ValueError):r.loads(r.canonical(d))
        raw=r.dumps(model("mlp16"))
        with self.assertRaises(ValueError):r.loads(raw.replace(b'{',b'{"schema":"duplicate",',1))
        with self.assertRaises(ValueError):r.loads(raw.replace(b'0.0',b'NaN',1))
        with self.assertRaises(ValueError):r.loads(b" "+raw)
        with self.assertRaises(ValueError):r.loads(b"0"*(r.MAX_ARTIFACT_BYTES+1))
    def test_14_python_validation_no_implicit_scaling_state_or_dtype(self):
        with self.assertRaises(ValueError):r.Scaler((0.0,0.0,0.0),(1.0,0.0,1.0))
        with self.assertRaises(ValueError):r.Model("mlp16",(0.0,)*80,r.Scaler((0.0,)*3,(1.0,)*3))
        with self.assertRaises(ValueError):r.Model("mlp16",(0.0,)*81,None)
        m=model("gru32")
        for raw,h in [((-1.0,0.0,0.0),r.zero_state()),((0.0,0.0,0.0),None),((True,0.0,0.0),r.zero_state()),
                      ((0.0,0.0,0.0),(math.nan,)+(0.0,)*31)]:
            with self.assertRaises(ValueError):python(m,raw,h)
    def test_15_scaling_subtraction_overflow_has_no_fallback(self):
        for arch in ("mlp16","gru32"):
            m=model(arch,scaler=r.Scaler((-1e308,0.0,0.0),(1.0,1.0,1.0)))
            with self.assertRaises(ArithmeticError):python(m,(1e308,0.0,0.0),None if arch=="mlp16" else r.zero_state())
            args=self.arrays(arch);args[0][0]=1e308;args[1][0]=-1e308
            out=array((987.0,)*(18 if arch=="mlp16" else 34));before=bytes(out)
            self.assertEqual(self.native.call(arch,args,out=out)[0],6);self.assertEqual(bytes(out),before)


def main():
    result=unittest.TextTestRunner(verbosity=2,failfast=True).run(unittest.defaultTestLoader.loadTestsFromTestCase(KernelTests))
    library_after=hashlib.sha256(Path(os.environ["HYDRA_KERNEL_LIBRARY"]).read_bytes()).hexdigest()
    ok=(result.wasSuccessful() and result.testsRun==ROSTER["test_cases"]
        and library_after==os.environ["HYDRA_KERNEL_LIBRARY_SHA256"])
    report=dict(schema="hydra-internal-neural-kernel-fixture-result/v1",status="PASSED" if ok else "FAILED",roster=ROSTER,
                tests_run=result.testsRun,actual_calls=COUNTS,evidence=EVIDENCE,untrained_generated_parameters=True,
                library_sha256_after=library_after,
                physical_or_neural_integration_proven=False,training_signal_or_forecast_skill_proven=False)
    with Path(os.environ["HYDRA_KERNEL_REPORT"]).open("x") as f:json.dump(report,f,indent=2,allow_nan=False);f.write("\n")
    return ok


if __name__=="__main__":raise SystemExit(0 if main() else 1)
