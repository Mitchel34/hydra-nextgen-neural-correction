"""Immutable untrained inference artifacts and independent stdlib-double math."""
from dataclasses import dataclass
import hashlib
import json
import math

FEATURES=(("water_potential_evaporation_flux","m s-1"),
          ("atmosphere_water__liquid_equivalent_precipitation_rate","mm h-1"),
          ("SOIL_STORAGE","m"))
SCHEMA="hydra-internal-neural-kernel/v1"
MAX_ARTIFACT_BYTES=1024*1024


def require(ok,message):
    if not ok:raise ValueError(message)


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()+b"\n"


def identity(value):return hashlib.sha256(canonical(value)).hexdigest()


def features():return [{"name":name,"units":unit} for name,unit in FEATURES]


def layout(architecture):
    if architecture=="mlp16":
        return {"name":"mlp16-row-major-v1","parameter_count":81,"hidden":16,"activation":"tanh",
                "segments":[["Wx",[16,3],0,48],["b",[16],48,16],["head_w",[16],64,16],["head_b",[1],80,1]]}
    if architecture=="gru32":
        return {"name":"gru32-reset-after-row-major-v1","parameter_count":3489,"hidden":32,
                "gate_order":["R","Z","N"],"reset":"after","bhn":"fixed_zero_not_serialized","head_state":"hnext",
                "segments":[["Wx",[96,3],0,288],["Wh",[96,32],288,3072],["b",[96],3360,96],
                            ["head_w",[32],3456,32],["head_b",[1],3488,1]]}
    raise ValueError("Only fixed mlp16/gru32 architectures are supported")


def vector(value,size,label,nonnegative=False,positive=False):
    require(type(value) is tuple and len(value)==size,label+" requires exact immutable tuple shape")
    require(all(type(x) is float and math.isfinite(x) for x in value),label+" requires finite float64 values, not booleans/integers")
    if nonnegative:require(all(x>=0 for x in value),label+" must be nonnegative")
    if positive:require(all(x>0 for x in value),label+" must be strictly positive")
    return value


def checked(value):
    if not math.isfinite(value):raise ArithmeticError("Nonfinite intermediate; no output or identity fallback")
    return value


def dot(weights,values):
    require(len(weights)==len(values),"Dot shape mismatch")
    result=0.0
    for weight,value in zip(weights,values):result=checked(result+checked(weight*value))
    return result


def sigmoid(value):
    if value>=0:
        x=math.exp(-value);return 1.0/(1.0+x)
    x=math.exp(value);return x/(1.0+x)


def factor(head):
    checked(head)
    if head==0.0:return 1.0
    value=math.exp(math.log(2.0)*math.tanh(head))
    require(0.5<=value<=2.0,"Engineering factor bound violated")
    return value


@dataclass(frozen=True,slots=True)
class Scaler:
    means:tuple
    scales:tuple
    def __post_init__(self):
        vector(self.means,3,"means");vector(self.scales,3,"scales",positive=True)
    def contract(self):
        return {"means":list(self.means),"scales":list(self.scales),"feature_id":identity(features())}
    @property
    def id(self):return identity(self.contract())
    def normalize(self,raw):
        vector(raw,3,"raw",nonnegative=True)
        return tuple(checked(checked(value-mean)/scale) for value,mean,scale in zip(raw,self.means,self.scales))


@dataclass(frozen=True,slots=True)
class Output:
    head:float
    factor:float
    hidden:tuple


@dataclass(frozen=True,slots=True)
class Model:
    architecture:str
    parameters:tuple
    scaler:Scaler
    def __post_init__(self):
        require(type(self.architecture) is str,"Architecture string required")
        vector(self.parameters,layout(self.architecture)["parameter_count"],"parameters")
        require(type(self.scaler) is Scaler,"Explicit immutable Scaler required; no fitting/defaults")
    def forward(self,raw,hidden=None):
        x=self.scaler.normalize(raw);w=self.parameters
        if self.architecture=="mlp16":
            require(hidden is None,"MLP has no recurrent input state")
            h=tuple(math.tanh(checked(dot(w[3*i:3*i+3],x)+w[48+i])) for i in range(16))
            head=checked(dot(w[64:80],h)+w[80])
        else:
            vector(hidden,32,"hidden")
            projected=[]
            for gate in range(3):
                rows=[]
                for i in range(32):
                    row=gate*32+i
                    a=checked(dot(w[row*3:row*3+3],x)+w[3360+row])
                    b=dot(w[288+row*32:288+row*32+32],hidden)
                    rows.append((a,b))
                projected.append(tuple(rows))
            r=tuple(sigmoid(checked(a+b)) for a,b in projected[0])
            z=tuple(sigmoid(checked(a+b)) for a,b in projected[1])
            n=tuple(math.tanh(checked(a+checked(reset*b))) for (a,b),reset in zip(projected[2],r))
            h=tuple(checked(checked((1.0-gate)*candidate)+checked(gate*previous)) for gate,candidate,previous in zip(z,n,hidden))
            head=checked(dot(w[3456:3488],h)+w[3488])
        return Output(head,factor(head),h)
    def document(self):
        spec=layout(self.architecture);scale=self.scaler.contract();scale["id"]=self.scaler.id
        result={"schema":SCHEMA,"version":1,"architecture":self.architecture,"dtype":"float64",
                "features":features(),"feature_id":identity(features()),"layout":spec,"layout_id":identity(spec),
                "scaler":scale,"parameters":list(self.parameters)}
        result["model_id"]=identity(result);return result
    @property
    def id(self):return self.document()["model_id"]


def dumps(model):
    require(type(model) is Model,"Exact immutable Model required")
    return canonical(model.document())


def loads(raw):
    require(type(raw) is bytes and 0<len(raw)<=MAX_ARTIFACT_BYTES,"Bounded JSON bytes required")
    def unique(pairs):
        result={}
        for key,value in pairs:
            require(key not in result,"Duplicate JSON field: "+key);result[key]=value
        return result
    def invalid_constant(value):raise ValueError("Nonfinite JSON constant: "+value)
    document=json.loads(raw,object_pairs_hook=unique,parse_constant=invalid_constant)
    require(type(document) is dict and set(document)=={"schema","version","architecture","dtype","features","feature_id",
            "layout","layout_id","scaler","parameters","model_id"},"Exact artifact fields required")
    require(document["schema"]==SCHEMA and type(document["version"]) is int and document["version"]==1
            and document["dtype"]=="float64","Schema/version/dtype mismatch")
    require(document["features"]==features() and document["feature_id"]==identity(features()),"Feature identity mismatch")
    spec=layout(document["architecture"])
    require(document["layout"]==spec and document["layout_id"]==identity(spec),"Fixed architecture/layout mismatch")
    scale=document["scaler"]
    require(type(scale) is dict and set(scale)=={"means","scales","feature_id","id"}
            and scale["feature_id"]==identity(features()),"Exact scaling fields/feature identity required")
    require(type(scale["means"]) is list and type(scale["scales"]) is list,"Scaling arrays required")
    scaler=Scaler(tuple(scale["means"]),tuple(scale["scales"]))
    require(scale["id"]==scaler.id,"Scaler identity mismatch")
    require(type(document["parameters"]) is list,"Flat parameter array required")
    model=Model(document["architecture"],tuple(document["parameters"]),scaler)
    require(model.document()==document and model.id==document["model_id"],"Model content identity mismatch")
    require(dumps(model)==raw,"Artifact must use exact canonical JSON bytes")
    return model


def zero_state():return (0.0,)*32
