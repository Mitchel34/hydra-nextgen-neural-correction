"""Strict generated-model JSON↔binary exporter; no fitting or native loading."""
import hashlib
from pathlib import Path
import struct

REFERENCE=Path(__file__).with_name("kernels.py")
REFERENCE_SHA="9b991bffeb4027c4e27c55c12e82c9b107c2854f2fff3f645c34dccc19ba5bc3"
if hashlib.sha256(REFERENCE.read_bytes()).hexdigest()!=REFERENCE_SHA:raise ValueError("Immutable inference reference changed")
from . import kernels as ref
MAGIC=b"HYDRANN1"
HEADER=struct.Struct("<8s9I64s64s64s64s")
VERSION,DTYPE=1,1


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def export_bytes(canonical_json):
    model=ref.loads(canonical_json)
    spec=ref.layout(model.architecture);arch=1 if model.architecture=="mlp16" else 2
    count=spec["parameter_count"];values=(*model.scaler.means,*model.scaler.scales,*model.parameters)
    tags=(ref.identity(ref.features()),ref.identity(spec),model.id,model.scaler.id)
    header=HEADER.pack(MAGIC,VERSION,HEADER.size,arch,3,spec["hidden"],count,DTYPE,0 if arch==1 else 1,6+count,
                       *(tag.encode("ascii") for tag in tags))
    return header+struct.pack("<"+str(len(values))+"d",*values)


def decode_bytes(raw):
    require(type(raw) is bytes and len(raw)>=HEADER.size,"Complete binary header required")
    fields=HEADER.unpack(raw[:HEADER.size])
    magic,version,header,arch,d,h,p,dtype,gate,n,*tags=fields
    require(magic==MAGIC and version==VERSION and header==HEADER.size and dtype==DTYPE and d==3,"Fixed binary version/dtype/input contract required")
    require(arch in (1,2),"Fixed architecture required")
    architecture="mlp16" if arch==1 else "gru32";layout=ref.layout(architecture)
    require(h==layout["hidden"] and p==layout["parameter_count"] and gate==(0 if arch==1 else 1) and n==6+p,
            "Architecture/shape/convention mismatch")
    require(len(raw)==HEADER.size+8*n,"Exact binary size/EOF required")
    decoded=[]
    for tag in tags:
        require(all(c in b"0123456789abcdef" for c in tag),"Lowercase64hex identity required")
        decoded.append(tag.decode("ascii"))
    values=struct.unpack("<"+str(n)+"d",raw[HEADER.size:])
    model=ref.Model(architecture,tuple(values[6:]),ref.Scaler(tuple(values[:3]),tuple(values[3:6])))
    expected=[ref.identity(ref.features()),ref.identity(layout),model.id,model.scaler.id]
    require(decoded==expected,"Binary feature/layout/model/scaler content identity mismatch")
    require(export_bytes(ref.dumps(model))==raw,"Exact binary representation required")
    return model


def export_files(canonical_json,json_path,binary_path):
    """Write only fresh artifacts supplied by the caller, never overwrite."""
    payload=export_bytes(canonical_json);model=decode_bytes(payload)
    json_path,binary_path=Path(json_path),Path(binary_path)
    require(json_path.is_absolute() and binary_path.is_absolute() and json_path!=binary_path
            and not json_path.exists() and not binary_path.exists(),"Fresh distinct absolute output files required")
    with json_path.open("xb") as f:f.write(canonical_json)
    with binary_path.open("xb") as f:f.write(payload)
    return {"json":{"path":str(json_path),"sha256":sha(canonical_json)},
            "binary":{"path":str(binary_path),"sha256":sha(payload)},"model_id":model.id,"scaler_id":model.scaler.id,
            "architecture":model.architecture,"wire_version":VERSION,"binary_bytes":len(payload)}


def verify_files(binding):
    require(set(binding)=={"json","binary","model_id","scaler_id","architecture","wire_version","binary_bytes"},"Exact caller artifact binding required")
    blobs={}
    for name in ("json","binary"):
        item=binding[name];require(set(item)=={"path","sha256"},"Exact artifact path/hash binding required")
        path=Path(item["path"]);require(path.is_file() and not path.is_symlink(),"Regular generated artifact required")
        require(0<path.stat().st_size<=(ref.MAX_ARTIFACT_BYTES if name=="json" else 28260),"Artifact byte ceiling")
        raw=path.read_bytes();require(sha(raw)==item["sha256"],"Artifact byte identity changed")
        blobs[name]=raw
    model=ref.loads(blobs["json"])
    require(export_bytes(blobs["json"])==blobs["binary"] and decode_bytes(blobs["binary"])==model,"Canonical JSON/binary mismatch")
    require(model.id==binding["model_id"] and model.scaler.id==binding["scaler_id"] and model.architecture==binding["architecture"]
            and type(binding["wire_version"]) is int and binding["wire_version"]==VERSION
            and type(binding["binary_bytes"]) is int and len(blobs["binary"])==binding["binary_bytes"],"Caller model/scaler identity mismatch")
    return model
