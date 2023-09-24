_IIIlIllIlIIIIllll = "converted_tts"
_IlIllIIlIIlIIIlII = "Replacing old dropdown file..."
_IllIIIlIIllIIlIII = "emb_g.weight"
_IIllIIIllIIlIIlII = "sample_rate"
_IlIlIlIlIllIlIlIl = "Index not used."
_IllIlIllIlIllllII = "Using index:%s."
_IlIlIIlIlIIIllIll = "You need to upload an audio"
_IIllIlIIIIlIIlIll = "pretrained"
_IlllIIIlllIlIlIII = "EXTRACT-MODEL"
_IIIIIIlIlIIIIIIII = "TRAIN-FEATURE"
_IllIlIIIllIlIIlll = "TRAIN"
_IlllIlllIIIllIlIl = "EXTRACT-FEATURE"
_IllIIIlllIIllIlll = "PRE-PROCESS"
_IIIIlllllllllIIII = "INFER"
_IIlIIIlIIlIllIlII = "HOME"
_IllIllllllIIIlIll = "_v2"
_IllllllIIlIlIIIII = "clean_empty_cache"
_IllIIIIllIlIIllll = "Bark-tts"
_IIIllIlIlIIIIIllI = "MDX"
_IlllIlIllIlIIIIlI = ".onnx"
_IIIlIlllllIlIllIl = "aac"
_IIllllIIIlIIlIlll = "ogg"
_IllIIIIIIIlIIIllI = "csvdb/stop.csv"
_IlIllIllIllIIllIl = "audio-others"
_IlllllllIIIIlllIl = "pm"
_IlllIllIlllIlIllI = "weight"
_IlIlIIlIlIlIIIlll = "cpu"
_IllIllllIIIIIIIIl = "rmvpe_onnx"
_IllIllIIIIlIIllII = "Edge-tts"
_IIIIllIlIllIIIIlI = "VR"
_IlIIIlIlllIIIlIIl = "datasets"
_IllIIlIlIlIlllllI = "32k"
_IIllIIlllllIllIII = "version"
_IlIIIIIIIIIIllIlI = ".index"
_IIIlIllllIIllIlII = "m4a"
_IllIlllIlIIIIllII = "mp3"
_IlIIIlIIIIlllllII = "48k"
_IIIllIllllIIlIIIl = ".pth"
_IIllIIIIIIIIIIIlI = "flac"
_IlllllIIllllllIlI = "logs"
_IIllIIllIIlIIlIIl = "D"
_IIlIllIlllIlllIIl = "G"
_IlllIIIIIIIIlIlll = "f0"
_IlIIIllIlIlIIlIIl = "."
_IlIIIIIIIIlIIlllI = "trained"
_IIlIllIllIlIIIIIl = "wav"
_IlIlIIllllllIlIII = "v2"
_IlllIlIIIIIlIIIIl = "r"
_IIIlIlIIIIIlIIllI = "config"
_IIIIIIlllIlIIIlIl = 1.0
_IllllIIIIllIIlIII = "audios"
_IIlllllllIIIIlIII = "40k"
_IlIlIlIlllllIIlIl = '"'
_IlllIlIIllIIIllII = "audio-outputs"
_IlIIlIlIlIllllllI = "rmvpe"
_IIlIllllIlIlIllII = " "
_IIlIIllllllllllll = "choices"
_IIllIIIIIIllllIll = "value"
_IlIlllllIlllIlIIl = "v1"
_IIlIlllllIIIIIIll = "\n"
_IlIlIllIIIIlIlIll = "visible"
_IIlIIllIIIlIlIIII = None
_IIIIlllIIllIIlIlI = "update"
_IIllIlIIlIlIIIIIl = "__type__"
_IIllIlIlIIllIlIII = False
_IIIllIllIIlIIIIIl = True
import sys
from shutil import rmtree
import shutil, json, datetime, unicodedata
from glob import glob1
from signal import SIGTERM
import librosa, os

IIIIIlIlllllIIIlI = os.getcwd()
sys.path.append(IIIIIlIlllllIIIlI)
import lib.globals.globals as rvc_globals
from LazyImport import lazyload
import mdx
from mdx_processing_script import get_model_list, id_to_ptm, prepare_mdx, run_mdx

IllIlllIIlllIllIl = lazyload("math")
import traceback, warnings

IIlIIlllIlIIIIlll = lazyload("tensorlowest")
import faiss

IllllIllIIllIIlIl = lazyload("ffmpeg")
import nltk

nltk.download("punkt", quiet=_IIIllIllIIlIIIIIl)
from nltk.tokenize import sent_tokenize
from bark import generate_audio, SAMPLE_RATE

IlIIIlIlllIlIlllI = lazyload("numpy")
IlIIIlIIllIlllIIl = lazyload("torch")
IlIIllIllIlIllIlI = lazyload("regex")
os.environ["TF_CPP_MIN_LIG_LEVEL"] = "3"
os.environ["IPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
from random import shuffle
from subprocess import Popen
import easy_infer, audioEffects

IlIIlIlIIllIIIIII = lazyload("gradio")
IIIIIIIllIlIIIIlI = lazyload("soundfile")
IlIIllllIlIIIIlll = IIIIIIIllIlIIIIlI.write
from config import Config
import fairseq
from i18n import I18nAuto
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_, _audio_pre_new
from MDXNet import MDXNetDereverb
from my_utils import load_audio
from train.process_ckpt import change_info, extract_small_model, merge, show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans
import time, threading
from shlex import quote as SQuote

IIIIlllllIIllIlll = ""
IlIllllIllIllllII = lambda IlIIlIIIlIlIllIII: SQuote(str(IlIIlIIIlIlIllIII))
IlllIIlllIlllllIl = os.path.join(IIIIIlIlllllIIIlI, "TEMP")
IIIllIlllIIlIIlII = os.path.join(IIIIIlIlllllIIIlI, "runtime/Lib/site-packages")
IIllIIllIIIlllIIl = [
    _IlllllIIllllllIlI,
    _IllllIIIIllIIlIII,
    _IlIIIlIlllIIIlIIl,
    "weights",
]
rmtree(IlllIIlllIlllllIl, ignore_errors=_IIIllIllIIlIIIIIl)
rmtree(os.path.join(IIIllIlllIIlIIlII, "infer_pack"), ignore_errors=_IIIllIllIIlIIIIIl)
rmtree(os.path.join(IIIllIlllIIlIIlII, "uvr5_pack"), ignore_errors=_IIIllIllIIlIIIIIl)
os.makedirs(
    os.path.join(IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII), exist_ok=_IIIllIllIIlIIIIIl
)
os.makedirs(
    os.path.join(IIIIIlIlllllIIIlI, _IlIllIllIllIIllIl), exist_ok=_IIIllIllIIlIIIIIl
)
os.makedirs(IlllIIlllIlllllIl, exist_ok=_IIIllIllIIlIIIIIl)
for IlllllIlIlllIlllI in IIllIIllIIIlllIIl:
    os.makedirs(
        os.path.join(IIIIIlIlllllIIIlI, IlllllIlIlllIlllI), exist_ok=_IIIllIllIIlIIIIIl
    )
os.environ["TEMP"] = IlllIIlllIlllllIl
warnings.filterwarnings("ignore")
IlIIIlIIllIlllIIl.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)
try:
    IIIlIllIlIlIIIlII = open(_IllIIIIIIIlIIIllI, "x")
    IIIlIllIlIlIIIlII.close()
except FileExistsError:
    pass
global IIlIlIIIIlIIllIII, IlllIllIIlllIIlIl, IlllIIIIlIlIIIIII
IIlIlIIIIlIIllIII = rvc_globals.DoFormant
IlllIllIIlllIIlIl = rvc_globals.Quefrency
IlllIIIIlIlIIIIII = rvc_globals.Timbre
IIlIlllllIlllIIlI = Config()
if IIlIlllllIlllIIlI.dml == _IIIllIllIIlIIIIIl:

    def IIlllIlIIllllIIlI(IlllllIIlIlIlIlll, IllIIIIlIIlIIIllI, IlIlIllIlllIIllII):
        IlllllIIlIlIlIlll.scale = IlIlIllIlllIIllII
        IlIlIIIllIlIIlIIl = IllIIIIlIIlIIIllI.clone().detach()
        return IlIlIIIllIlIIlIIl

    fairseq.modules.grad_multiply.GradMultiply.forward = IIlllIlIIllllIIlI
IlIllIllIIllIlIII = I18nAuto()
IlIllIllIIllIlIII.print()
IlIllIllllIIlIllI = IlIIIlIIllIlllIIl.cuda.device_count()
IIIIllllllIIIlllI = []
IllIIlIllIlIIIlIl = []
IIIIIllllIIllIIlI = _IIllIlIlIIllIlIII
IlIllllIlllIIlIlI = [
    "10",
    "16",
    "20",
    "30",
    "40",
    "A2",
    "A3",
    "A4",
    "P4",
    "A50",
    "500",
    "A60",
    "70",
    "80",
    "90",
    "M4",
    "T4",
    "TITAN",
]
if IlIIIlIIllIlllIIl.cuda.is_available() or IlIllIllllIIlIllI != 0:
    for IlIIIIIIllIlllIII in range(IlIllIllllIIlIllI):
        IIIlIllllIIlIlllI = IlIIIlIIllIlllIIl.cuda.get_device_name(
            IlIIIIIIllIlllIII
        ).upper()
        if any(
            IlIlIIIlIlIIllIlI in IIIlIllllIIlIlllI
            for IlIlIIIlIlIIllIlI in IlIllllIlllIIlIlI
        ):
            IIIIIllllIIllIIlI = _IIIllIllIIlIIIIIl
            IIIIllllllIIIlllI.append("%s\t%s" % (IlIIIIIIllIlllIII, IIIlIllllIIlIlllI))
            IllIIlIllIlIIIlIl.append(
                int(
                    IlIIIlIIllIlllIIl.cuda.get_device_properties(
                        IlIIIIIIllIlllIII
                    ).total_memory
                    / 1e9
                    + 0.4
                )
            )
IllIIlIIIlllllllI = (
    _IIlIlllllIIIIIIll.join(IIIIllllllIIIlllI)
    if IIIIIllllIIllIIlI and IIIIllllllIIIlllI
    else "Unfortunately, there is no compatible GPU available to support your training."
)
IlIIIlIIIIIIIlIlI = (
    min(IllIIlIllIlIIIlIl) // 2 if IIIIIllllIIllIIlI and IIIIllllllIIIlllI else 1
)
IIIIIIlllIIlllIlI = "-".join(
    IlIIIIlIlIIlIIlIl[0] for IlIIIIlIlIIlIIlIl in IIIIllllllIIIlllI
)
IIlIlIIIIlllIIlIl = _IIlIIllIIIlIlIIII


def IlIlIlIIlllllIIIl():
    global IIlIlIIIIlllIIlIl
    (
        IIlIIlllIlllIIIlI,
        _IllIIlIIllIllllll,
        _IllIIlIIllIllllll,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ["/kaggle/input/ax-rmf/hubert_base.pt"], suffix=""
    )
    IIlIlIIIIlllIIlIl = IIlIIlllIlllIIIlI[0].to(IIlIlllllIlllIIlI.device)
    if IIlIlllllIlllIIlI.is_half:
        IIlIlIIIIlllIIlIl = IIlIlIIIIlllIIlIl.half()
    IIlIlIIIIlllIIlIl.eval()


IIlllIlIlllllIlll = _IlIIIlIlllIIIlIIl
IlIIlIlIlIlllllll = "weights"
IllllIIIIIIIlIlIl = "uvr5_weights"
IllIlIIlIlIIlIlII = _IlllllIIllllllIlI
IlIIlIllIlllIlIlI = "formantshiftcfg"
IllIlIIIllllIlIlI = _IllllIIIIllIIlIII
IIlIIIllIIllIlIll = _IlIllIllIllIIllIl
IIlIIlllIlllIIIll = {
    _IIlIllIllIlIIIIIl,
    _IllIlllIlIIIIllII,
    _IIllIIIIIIIIIIIlI,
    _IIllllIIIlIIlIlll,
    "opus",
    _IIIlIllllIIllIlII,
    "mp4",
    _IIIlIlllllIlIllIl,
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}
IlIIIlIlIlllllIIl = [
    os.path.join(IlIlIlIllIlIllllI, IllIllIlllIIlIIII)
    for (IlIlIlIllIlIllllI, _IllllIIllIllIlllI, IIIIlIIIIlIIllIIl) in os.walk(
        IlIIlIlIlIlllllll
    )
    for IllIllIlllIIlIIII in IIIIlIIIIlIIllIIl
    if IllIllIlllIIlIIII.endswith((_IIIllIllllIIlIIIl, _IlllIlIllIlIIIIlI))
]
IllllIIllIlllllII = [
    os.path.join(IllIIllIIIIlllllI, IlIIIllIlIlIlllll)
    for (IllIIllIIIIlllllI, _IllllllIllIllIlIl, IlIllIlIlIllllIlI) in os.walk(
        IllIlIIlIlIIlIlII, topdown=_IIllIlIlIIllIlIII
    )
    for IlIIIllIlIlIlllll in IlIllIlIlIllllIlI
    if IlIIIllIlIlIlllll.endswith(_IlIIIIIIIIIIllIlI)
    and _IlIIIIIIIIlIIlllI not in IlIIIllIlIlIlllll
]
IlllIIIllIllIlllI = [
    os.path.join(IlIIIIIIllIIlllII, IIIIIllIIlIIlIlll)
    for (IlIIIIIIllIIlllII, _IIIllIlIlllIlIllI, IIlIllllllIIIIlll) in os.walk(
        IllIlIIIllllIlIlI, topdown=_IIllIlIlIIllIlIII
    )
    for IIIIIllIIlIIlIlll in IIlIllllllIIIIlll
    if IIIIIllIIlIIlIlll.endswith(tuple(IIlIIlllIlllIIIll))
    if IIIIIllIIlIIlIlll.endswith(tuple(IIlIIlllIlllIIIll))
    and not IIIIIllIIlIIlIlll.endswith(".gitignore")
]
IlIIIIlIIIlIlllll = [
    os.path.join(IlllIllIllIlllIIl, IlIllIIllIlIllIlI)
    for (IlllIllIllIlllIIl, _IlIIIlllIIlIIIIII, IllIIIlIlIIIllIII) in os.walk(
        IIlIIIllIIllIlIll, topdown=_IIllIlIlIIllIlIII
    )
    for IlIllIIllIlIllIlI in IllIIIlIlIIIllIII
    if IlIllIIllIlIllIlI.endswith(tuple(IIlIIlllIlllIIIll))
]
IllIlIllIIIllIlII = [
    IIllllIIIIlllIIll.replace(_IIIllIllllIIlIIIl, "")
    for IIllllIIIIlllIIll in os.listdir(IllllIIIIIIIlIlIl)
    if IIllllIIIIlllIIll.endswith(_IIIllIllllIIlIIIl) or "onnx" in IIllllIIIIlllIIll
]
IIIlIllIIllIIIIII = lambda: sorted(IlIIIlIlIlllllIIl)[0] if IlIIIlIlIlllllIIl else ""
IlIlIIlIllIllIIll = []
for IllIllIIIlllllIll in os.listdir(os.path.join(IIIIIlIlllllIIIlI, IIlllIlIlllllIlll)):
    if _IlIIIllIlIlIIlIIl not in IllIllIIIlllllIll:
        IlIlIIlIllIllIIll.append(
            os.path.join(
                easy_infer.find_folder_parent(_IlIIIllIlIlIIlIIl, _IIllIlIIIIlIIlIll),
                _IlIIIlIlllIIIlIIl,
                IllIllIIIlllllIll,
            )
        )


def IlIllIlllIIIllIII():
    if len(IlIlIIlIllIllIIll) > 0:
        return sorted(IlIlIIlIllIllIIll)[0]
    else:
        return ""


def IIlllIllIlIIIIIII(IIIIIllllIlllllll):
    IllIIIIIllIllIIII = get_model_list()
    IIIlIIlIIlIIIIIll = list(IllIIIIIllIllIIII)
    if IIIIIllllIlllllll == _IIIIllIlIllIIIIlI:
        return {
            _IIlIIllllllllllll: IllIlIllIIIllIlII,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
    elif IIIIIllllIlllllll == _IIIllIlIlIIIIIllI:
        return {
            _IIlIIllllllllllll: IIIlIIlIIlIIIIIll,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }


IlIIIlIIllIlIllIl = easy_infer.get_bark_voice()
IlIIlllIllIllIlll = easy_infer.get_edge_voice()


def IlIIIIIIIllIIllII(IIIIIIllIlIlllIIl):
    if IIIIIIllIlIlllIIl == _IllIllIIIIlIIllII:
        return {
            _IIlIIllllllllllll: IlIIlllIllIllIlll,
            _IIllIIIIIIllllIll: "",
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
    elif IIIIIIllIlIlllIIl == _IllIIIIllIlIIllll:
        return {
            _IIlIIllllllllllll: IlIIIlIIllIlIllIl,
            _IIllIIIIIIllllIll: "",
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }


def IIllIIlIlIIIllIII(IlIIIIlIlIllllIIl):
    IlllllIIIIIIIIlIl = []
    for IlllllIIllllIIIIl in os.listdir(
        os.path.join(IIIIIlIlllllIIIlI, IIlllIlIlllllIlll)
    ):
        if _IlIIIllIlIlIIlIIl not in IlllllIIllllIIIIl:
            IlllllIIIIIIIIlIl.append(
                os.path.join(
                    easy_infer.find_folder_parent(
                        _IlIIIllIlIlIIlIIl, _IIllIlIIIIlIIlIll
                    ),
                    _IlIIIlIlllIIIlIIl,
                    IlllllIIllllIIIIl,
                )
            )
    return IlIIlIlIIllIIIIII.Dropdown.update(choices=IlllllIIIIIIIIlIl)


def IIllIIIllllIIllll():
    IlIlIIIIIIllIlIlI = [
        os.path.join(IlllllllIlIlllIII, IIIIllIlIIIIlIllI)
        for (IlllllllIlIlllIII, _IIIIIIlllllIIIIll, IlIllIIIlIIIIIllI) in os.walk(
            IllIlIIlIlIIlIlII
        )
        for IIIIllIlIIIIlIllI in IlIllIIIlIIIIIllI
        if IIIIllIlIIIIlIllI.endswith(_IlIIIIIIIIIIllIlI)
        and _IlIIIIIIIIlIIlllI not in IIIIllIlIIIIlIllI
    ]
    return IlIlIIIIIIllIlIlI if IlIlIIIIIIllIlIlI else ""


def IIllIlllIIlIIIllI():
    IIIIIlIlIIIlIIlIl = [
        os.path.join(IIlIIIlIIllIlllII, IlIIllllIlIlllIlI)
        for (IIlIIIlIIllIlllII, _IIIIllIlIlIIllIlI, IIIlIIlllIlIlIIll) in os.walk(
            IlIIlIllIlllIlIlI
        )
        for IlIIllllIlIlllIlI in IIIlIIlllIlIlIIll
        if IlIIllllIlIlllIlI.endswith(".txt")
    ]
    return IIIIIlIlIIIlIIlIl if IIIIIlIlIIIlIIlIl else ""


import soundfile as sf


def IllllIlIIIIIIIIIl(IlllIIIIlIlIIIlll, IllIlllIllIIIIlIl, IIIIIlIIlIIlllIIl):
    IllIIllllIIIIIllI = 1
    while _IIIllIllIIlIIIIIl:
        IIIllllIlllIlllIl = os.path.join(
            IlllIIIIlIlIIIlll,
            f"{IllIlllIllIIIIlIl}_{IllIIllllIIIIIllI}.{IIIIIlIIlIIlllIIl}",
        )
        if not os.path.exists(IIIllllIlllIlllIl):
            return IIIllllIlllIlllIl
        IllIIllllIIIIIllI += 1


def IlllllIlIIIIIIIII(
    IIIlllllllIllIlIl,
    IlIIIllIllllIIIII,
    IllIlIlIIlIIlIIIl,
    IIIIlllIIlIIllllI,
    IlIlIIIIIIIlIIIIl,
):
    IIIIlIllllIIIllIl, IIlllllIIIIllIIIl = librosa.load(
        IIIlllllllIllIlIl, sr=_IIlIIllIIIlIlIIII
    )
    IIlIllIllIIIllIII, IlIlllIlIllIIllIl = librosa.load(
        IlIIIllIllllIIIII, sr=_IIlIIllIIIlIlIIII
    )
    if IIlllllIIIIllIIIl != IlIlllIlIllIIllIl:
        if IIlllllIIIIllIIIl > IlIlllIlIllIIllIl:
            IIlIllIllIIIllIII = librosa.resample(
                IIlIllIllIIIllIII,
                orig_sr=IlIlllIlIllIIllIl,
                target_sr=IIlllllIIIIllIIIl,
            )
        else:
            IIIIlIllllIIIllIl = librosa.resample(
                IIIIlIllllIIIllIl,
                orig_sr=IIlllllIIIIllIIIl,
                target_sr=IlIlllIlIllIIllIl,
            )
    IIlllIIIlIIlIIlIl = min(len(IIIIlIllllIIIllIl), len(IIlIllIllIIIllIII))
    IIIIlIllllIIIllIl = librosa.util.fix_length(IIIIlIllllIIIllIl, IIlllIIIlIIlIIlIl)
    IIlIllIllIIIllIII = librosa.util.fix_length(IIlIllIllIIIllIII, IIlllIIIlIIlIIlIl)
    if IIIIlllIIlIIllllI != _IIIIIIlllIlIIIlIl:
        IIIIlIllllIIIllIl *= IIIIlllIIlIIllllI
    if IlIlIIIIIIIlIIIIl != _IIIIIIlllIlIIIlIl:
        IIlIllIllIIIllIII *= IlIlIIIIIIIlIIIIl
    IIIlllIlIIIIIlllI = IIIIlIllllIIIllIl + IIlIllIllIIIllIII
    sf.write(IllIlIlIIlIIlIIIl, IIIlllIlIIIIIlllI, IIlllllIIIIllIIIl)


def IIllIIIlllllIlIlI(
    IIIllIlllIIIllllI,
    IlIIlIIlIlIIlllII,
    IllIIlllllIIlIlll=_IIIIIIlllIlIIIlIl,
    IIlIlIlllllllIlIl=_IIIIIIlllIlIIIlIl,
    IllllIllIIlIIlIIl=_IIllIlIlIIllIlIII,
    IlllIIllIIIlIIlll=_IIllIlIlIIllIlIII,
    IIlIIIllllIIIlIll=_IIllIlIlIIllIlIII,
):
    IIIlIIlllIIllIIII = "Conversion complete!"
    IIIllIIlIIIIIIlIl = "combined_audio"
    IlIIIlIlIlIlIlIll = os.path.join(IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII)
    os.makedirs(IlIIIlIlIlIlIlIll, exist_ok=_IIIllIllIIlIIIIIl)
    IlIlllIllIIIlllII = IIIllIIlIIIIIIlIl
    IlIIllIIIllllIlll = _IIlIllIllIlIIIIIl
    IIIlIIlIIllllIIII = IllllIlIIIIIIIIIl(
        IlIIIlIlIlIlIlIll, IlIlllIllIIIlllII, IlIIllIIIllllIlll
    )
    print(IllllIllIIlIIlIIl)
    print(IlllIIllIIIlIIlll)
    print(IIlIIIllllIIIlIll)
    if IllllIllIIlIIlIIl or IlllIIllIIIlIIlll or IIlIIIllllIIIlIll:
        IlIlllIllIIIlllII = "effect_audio"
        IIIlIIlIIllllIIII = IllllIlIIIIIIIIIl(
            IlIIIlIlIlIlIlIll, IlIlllIllIIIlllII, IlIIllIIIllllIlll
        )
        IIIIllIIlllIlIlll = audioEffects.process_audio(
            IlIIlIIlIlIIlllII,
            IIIlIIlIIllllIIII,
            IllllIllIIlIIlIIl,
            IlllIIllIIIlIIlll,
            IIlIIIllllIIIlIll,
        )
        IlIlllIllIIIlllII = IIIllIIlIIIIIIlIl
        IIIlIIlIIllllIIII = IllllIlIIIIIIIIIl(
            IlIIIlIlIlIlIlIll, IlIlllIllIIIlllII, IlIIllIIIllllIlll
        )
        IlllllIlIIIIIIIII(
            IIIllIlllIIIllllI,
            IIIIllIIlllIlIlll,
            IIIlIIlIIllllIIII,
            IllIIlllllIIlIlll,
            IIlIlIlllllllIlIl,
        )
        return IlIllIllIIllIlIII(IIIlIIlllIIllIIII), IIIlIIlIIllllIIII
    else:
        IlIlllIllIIIlllII = IIIllIIlIIIIIIlIl
        IIIlIIlIIllllIIII = IllllIlIIIIIIIIIl(
            IlIIIlIlIlIlIlIll, IlIlllIllIIIlllII, IlIIllIIIllllIlll
        )
        IlllllIlIIIIIIIII(
            IIIllIlllIIIllllI,
            IlIIlIIlIlIIlllII,
            IIIlIIlIIllllIIII,
            IllIIlllllIIlIlll,
            IIlIlIlllllllIlIl,
        )
        return IlIllIllIIllIlIII(IIIlIIlllIIllIIII), IIIlIIlIIllllIIII


def IlIlIIIIlIlIIlIIl(
    IIlIIIllIIlIIIlIl,
    IIIlIIIllllIllIlI,
    IIIIIIlIIIIIlIllI,
    IlIlIIIIlIlIlllII,
    IllIIIlIIllIIlllI,
    IIIlIlIIIIlIlllIl,
    IIIlIIlIllllIIIlI,
    IIlIllllIlllllIII,
    IllllIIIIIlIIlIIl,
    IlIIIlIlIllIlllll,
    IlllIllIIIllIIlII,
    IlIllIlllllIlllIl,
    IIIlllIllIIlIlIll,
    IIllIlIIllIllIlIl,
    IlllIIlIlIIllIIIl,
    IllIIlIIIlIIlIlll,
    IIIIlIIlIlIlIIllI,
    IIlIIlIIIllIIIlII,
    IIIlIllIllIIlIlIl,
):
    global IIllIIlIIIIllIlII
    IIllIIlIIIIllIlII = 0
    IllIIIlIIIIllIlll = time.time()
    global IlIllIIIIIIlIlIll, IllIIllIlIIlIIlII, IIIIlIIIIIlIIIIlI, IIlIlIIIIlllIIlIl, IIlIIIlIIlIlllllI
    IIIllllllIIlIlIII = (
        _IIIllIllIIlIIIIIl
        if IIIlIlIIIIlIlllIl == _IllIllllIIIIIIIIl
        else _IIllIlIlIIllIlIII
    )
    if not IIIlIIIllllIllIlI and not IIIIIIlIIIIIlIllI:
        return _IlIlIIlIlIIIllIll, _IIlIIllIIIlIlIIII
    if not os.path.exists(IIIlIIIllllIllIlI) and not os.path.exists(
        os.path.join(IIIIIlIlllllIIIlI, IIIlIIIllllIllIlI)
    ):
        return "Audio was not properly selected or doesn't exist", _IIlIIllIIIlIlIIII
    IIIIIIlIIIIIlIllI = IIIIIIlIIIIIlIllI or IIIlIIIllllIllIlI
    print(f"\nStarting inference for '{os.path.basename(IIIIIIlIIIIIlIllI)}'")
    print("-------------------")
    IlIlIIIIlIlIlllII = int(IlIlIIIIlIlIlllII)
    if rvc_globals.NotesIrHertz and IIIlIlIIIIlIlllIl != _IlIIlIlIlIllllllI:
        IlllIIlIlIIllIIIl = (
            IIIIlllIlllIIlIIl(IllIIlIIIlIIlIlll) if IllIIlIIIlIIlIlll else 50
        )
        IIIIlIIlIlIlIIllI = (
            IIIIlllIlllIIlIIl(IIlIIlIIIllIIIlII) if IIlIIlIIIllIIIlII else 1100
        )
        print(
            f"Converted Min pitch: freq - {IlllIIlIlIIllIIIl}\nConverted Max pitch: freq - {IIIIlIIlIlIlIIllI}"
        )
    else:
        IlllIIlIlIIllIIIl = IlllIIlIlIIllIIIl or 50
        IIIIlIIlIlIlIIllI = IIIIlIIlIlIlIIllI or 1100
    try:
        IIIIIIlIIIIIlIllI = IIIIIIlIIIIIlIllI or IIIlIIIllllIllIlI
        print(f"Attempting to load {IIIIIIlIIIIIlIllI}....")
        IllIIIlllIlIIIlII = load_audio(
            IIIIIIlIIIIIlIllI,
            16000,
            DoFormant=rvc_globals.DoFormant,
            Quefrency=rvc_globals.Quefrency,
            Timbre=rvc_globals.Timbre,
        )
        IllIIIllIIIIIIIIl = IlIIIlIlllIlIlllI.abs(IllIIIlllIlIIIlII).max() / 0.95
        if IllIIIllIIIIIIIIl > 1:
            IllIIIlllIlIIIlII /= IllIIIllIIIIIIIIl
        IIIlllIIIllllIlll = [0, 0, 0]
        if not IIlIlIIIIlllIIlIl:
            print("Loading hubert for the first time...")
            IlIlIlIIlllllIIIl()
        try:
            IlIIlIllIIIIlIlll = IlIIIlIIlIllIIlII.get(_IlllIIIIIIIIlIlll, 1)
        except NameError:
            IIIllIIIIllllllII = "Model was not properly selected"
            print(IIIllIIIIllllllII)
            return IIIllIIIIllllllII, _IIlIIllIIIlIlIIII
        IIIlIIlIllllIIIlI = (
            IIIlIIlIllllIIIlI.strip(_IIlIllllIlIlIllII)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIlllllIIIIIIll)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIllllIlIlIllII)
            .replace(_IlIIIIIIIIlIIlllI, "added")
            if IIIlIIlIllllIIIlI != ""
            else IIlIllllIlllllIII
        )
        try:
            IIllIIIllllIIIlII = IIIIlIIIIIlIIIIlI.pipeline(
                IIlIlIIIIlllIIlIl,
                IllIIllIlIIlIIlII,
                IIlIIIllIIlIIIlIl,
                IllIIIlllIlIIIlII,
                IIIIIIlIIIIIlIllI,
                IIIlllIIIllllIlll,
                IlIlIIIIlIlIlllII,
                IIIlIlIIIIlIlllIl,
                IIIlIIlIllllIIIlI,
                IllllIIIIIlIIlIIl,
                IlIIlIllIIIIlIlll,
                IlIIIlIlIllIlllll,
                IlIllIIIIIIlIlIll,
                IlllIllIIIllIIlII,
                IlIllIlllllIlllIl,
                IIlIIIlIIlIlllllI,
                IIIlllIllIIlIlIll,
                IIllIlIIllIllIlIl,
                IIIlIllIllIIlIlIl,
                IIIllllllIIlIlIII,
                f0_file=IllIIIlIIllIIlllI,
                f0_min=IlllIIlIlIIllIIIl,
                f0_max=IIIIlIIlIlIlIIllI,
            )
        except AssertionError:
            IIIllIIIIllllllII = (
                "Mismatching index version detected (v1 with v2, or v2 with v1)."
            )
            print(IIIllIIIIllllllII)
            return IIIllIIIIllllllII, _IIlIIllIIIlIlIIII
        except NameError:
            IIIllIIIIllllllII = (
                "RVC libraries are still loading. Please try again in a few seconds."
            )
            print(IIIllIIIIllllllII)
            return IIIllIIIIllllllII, _IIlIIllIIIlIlIIII
        if IlIllIIIIIIlIlIll != IlllIllIIIllIIlII >= 16000:
            IlIllIIIIIIlIlIll = IlllIllIIIllIIlII
        IlIIIllllIIIlIllI = (
            _IllIlIllIlIllllII % IIIlIIlIllllIIIlI
            if os.path.exists(IIIlIIlIllllIIIlI)
            else _IlIlIlIlIllIlIlIl
        )
        IlIIllllIIllIIIlI = time.time()
        IIllIIlIIIIllIlII = IlIIllllIIllIIIlI - IllIIIlIIIIllIlll
        IIIIlIlIllIIllIII = _IlllIlIIllIIIllII
        os.makedirs(IIIIlIlIllIIllIII, exist_ok=_IIIllIllIIlIIIIIl)
        IlIlIIlIIllIllIlI = "generated_audio_{}.wav"
        IllIIlIIlIlIIIlIl = 1
        while _IIIllIllIIlIIIIIl:
            IlIIIlIIllIllllII = os.path.join(
                IIIIlIlIllIIllIII, IlIlIIlIIllIllIlI.format(IllIIlIIlIlIIIlIl)
            )
            if not os.path.exists(IlIIIlIIllIllllII):
                break
            IllIIlIIlIlIIIlIl += 1
        wavfile.write(IlIIIlIIllIllllII, IlIllIIIIIIlIlIll, IIllIIIllllIIIlII)
        print(f"Generated audio saved to: {IlIIIlIIllIllllII}")
        return (
            f"Success.\n {IlIIIllllIIIlIllI}\nTime:\n npy:{IIIlllIIIllllIlll[0]}, f0:{IIIlllIIIllllIlll[1]}, infer:{IIIlllIIIllllIlll[2]}\nTotal Time: {IIllIIlIIIIllIlII} seconds",
            (IlIllIIIIIIlIlIll, IIllIIIllllIIIlII),
        )
    except:
        IIIIIllIllIIllIlI = traceback.format_exc()
        print(IIIIIllIllIIllIlI)
        return IIIIIllIllIIllIlI, (_IIlIIllIIIlIlIIII, _IIlIIllIIIlIlIIII)


def IllIllIIIllllIlIl(
    IllIlIlIllllIlIlI,
    IIllIIIIIlIIlllII,
    IIIlIIIllIllIlIIl,
    IlIlIlIIlIllIIlIl,
    IlllIllIIIlIlllll,
    IlIlIIIlIlIIIlllI,
    IlIIllllIIlIlIllI,
    IIllllIIlllIlllIl,
    IIIIIlIIIlIllllII,
    IIlIIIIIlIIlIllll,
    IllIllIIIIlIIllll,
    IlIIIIllIIlIIIllI,
    IIIllllllIllIIlIl,
    IIlIlIIIIIllIIllI,
    IIlIlllIlIlIllIIl,
    IIlIllllIIlIIllII,
    IIIIllIIlllllIIlI,
    IIIIlIllIlIIlIIIl,
    IIlIIllllllllIllI,
    IIlIlIIlIIllIIlII,
):
    if rvc_globals.NotesIrHertz and IlIlIIIlIlIIIlllI != _IlIIlIlIlIllllllI:
        IIlIllllIIlIIllII = (
            IIIIlllIlllIIlIIl(IIIIllIIlllllIIlI) if IIIIllIIlllllIIlI else 50
        )
        IIIIlIllIlIIlIIIl = (
            IIIIlllIlllIIlIIl(IIlIIllllllllIllI) if IIlIIllllllllIllI else 1100
        )
        print(
            f"Converted Min pitch: freq - {IIlIllllIIlIIllII}\nConverted Max pitch: freq - {IIIIlIllIlIIlIIIl}"
        )
    else:
        IIlIllllIIlIIllII = IIlIllllIIlIIllII or 50
        IIIIlIllIlIIlIIIl = IIIIlIllIlIIlIIIl or 1100
    try:
        IIllIIIIIlIIlllII, IIIlIIIllIllIlIIl = [
            IllllIIllllIllIll.strip(_IIlIllllIlIlIllII)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIlllllIIIIIIll)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIllllIlIlIllII)
            for IllllIIllllIllIll in [IIllIIIIIlIIlllII, IIIlIIIllIllIlIIl]
        ]
        os.makedirs(IIIlIIIllIllIlIIl, exist_ok=_IIIllIllIIlIIIIIl)
        IlIlIlIIlIllIIlIl = (
            [
                os.path.join(IIllIIIIIlIIlllII, IlllllIlIllIIllII)
                for IlllllIlIllIIllII in os.listdir(IIllIIIIIlIIlllII)
            ]
            if IIllIIIIIlIIlllII
            else [IIlIIIIllIIIllllI.name for IIlIIIIllIIIllllI in IlIlIlIIlIllIIlIl]
        )
        IIIIIllIIIIIlIIll = []
        for IlllllIIlllllIIII in IlIlIlIIlIllIIlIl:
            IlllIlllIllIIllll, IlIIIIlIIIIllIlII = IlIlIIIIlIlIIlIIl(
                IllIlIlIllllIlIlI,
                IlllllIIlllllIIII,
                _IIlIIllIIIlIlIIII,
                IlllIllIIIlIlllll,
                _IIlIIllIIIlIlIIII,
                IlIlIIIlIlIIIlllI,
                IlIIllllIIlIlIllI,
                IIllllIIlllIlllIl,
                IIIIIlIIIlIllllII,
                IIlIIIIIlIIlIllll,
                IllIllIIIIlIIllll,
                IlIIIIllIIlIIIllI,
                IIIllllllIllIIlIl,
                IIlIlllIlIlIllIIl,
                IIlIllllIIlIIllII,
                IIIIllIIlllllIIlI,
                IIIIlIllIlIIlIIIl,
                IIlIIllllllllIllI,
                IIlIlIIlIIllIIlII,
            )
            if "Success" in IlllIlllIllIIllll:
                try:
                    IIIlllIIlIllllllI, IIIlIlIlIllllIllI = IlIIIIlIIIIllIlII
                    IIlIIllllIIIlllIl = os.path.splitext(
                        os.path.basename(IlllllIIlllllIIII)
                    )[0]
                    IlIIIIllIlllIIllI = (
                        f"{IIIlIIIllIllIlIIl}/{IIlIIllllIIIlllIl}.{IIlIlIIIIIllIIllI}"
                    )
                    IlllllIIlllllIIII, IIIIllIIlIllllIIl = (
                        IlIIIIllIlllIIllI,
                        IIlIlIIIIIllIIllI,
                    )
                    IlllllIIlllllIIII, IIIIllIIlIllllIIl = (
                        IlIIIIllIlllIIllI
                        if IIlIlIIIIIllIIllI
                        in [
                            _IIlIllIllIlIIIIIl,
                            _IIllIIIIIIIIIIIlI,
                            _IllIlllIlIIIIllII,
                            _IIllllIIIlIIlIlll,
                            _IIIlIlllllIlIllIl,
                            _IIIlIllllIIllIlII,
                        ]
                        else f"{IlIIIIllIlllIIllI}.wav",
                        IIlIlIIIIIllIIllI,
                    )
                    IlIIllllIlIIIIlll(
                        IlllllIIlllllIIII, IIIlIlIlIllllIllI, IIIlllIIlIllllllI
                    )
                    if os.path.exists(IlllllIIlllllIIII) and IIIIllIIlIllllIIl not in [
                        _IIlIllIllIlIIIIIl,
                        _IIllIIIIIIIIIIIlI,
                        _IllIlllIlIIIIllII,
                        _IIllllIIIlIIlIlll,
                        _IIIlIlllllIlIllIl,
                        _IIIlIllllIIllIlII,
                    ]:
                        sys.stdout.write(
                            f"Running command: ffmpeg -i {IlIllllIllIllllII(IlllllIIlllllIIII)} -vn {IlIllllIllIllllII(IlllllIIlllllIIII[:-4]+_IlIIIllIlIlIIlIIl+IIIIllIIlIllllIIl)} -q:a 2 -y"
                        )
                        os.system(
                            f"ffmpeg -i {IlIllllIllIllllII(IlllllIIlllllIIII)} -vn {IlIllllIllIllllII(IlllllIIlllllIIII[:-4]+_IlIIIllIlIlIIlIIl+IIIIllIIlIllllIIl)} -q:a 2 -y"
                        )
                except:
                    IlllIlllIllIIllll += traceback.format_exc()
                    print(f"\nException encountered: {IlllIlllIllIIllll}")
            IIIIIllIIIIIlIIll.append(
                f"{os.path.basename(IlllllIIlllllIIII)}->{IlllIlllIllIIllll}"
            )
            yield _IIlIlllllIIIIIIll.join(IIIIIllIIIIIlIIll)
        yield _IIlIlllllIIIIIIll.join(IIIIIllIIIIIlIIll)
    except:
        yield traceback.format_exc()


def IIIlIlIlIIIIlIIll(
    IIIIllllIlIIllIlI,
    IIlIllIIlIlIllllI,
    IIIIIIIIIlIIlIIlI,
    IlllllIIllIlIlIIl,
    IlllIlIIlIlIIlIll,
    IlIlllIIIllIIlIII,
    IlllllllIIIlIlllI,
    IIlIlllIlIllIlIll,
):
    IlIIllllIIIllIlll = "streams"
    IIIIllIllllllIlII = "onnx_dereverb_By_FoxJoy"
    IIIIllIIIIIlIlIlI = []
    if IIlIlllIlIllIlIll == _IIIIllIlIllIIIIlI:
        try:
            IIlIllIIlIlIllllI, IIIIIIIIIlIIlIIlI, IlllIlIIlIlIIlIll = [
                IllllIIlIIlIllIll.strip(_IIlIllllIlIlIllII)
                .strip(_IlIlIlIlllllIIlIl)
                .strip(_IIlIlllllIIIIIIll)
                .strip(_IlIlIlIlllllIIlIl)
                .strip(_IIlIllllIlIlIllII)
                for IllllIIlIIlIllIll in [
                    IIlIllIIlIlIllllI,
                    IIIIIIIIIlIIlIIlI,
                    IlllIlIIlIlIIlIll,
                ]
            ]
            IIlIlIIlllIllllIl = [
                os.path.join(IIlIllIIlIlIllllI, IIIlIllllIIIIllII)
                for IIIlIllllIIIIllII in os.listdir(IIlIllIIlIlIllllI)
                if IIIlIllllIIIIllII.endswith(tuple(IIlIIlllIlllIIIll))
            ]
            IlIIlIlIIllIllIlI = (
                MDXNetDereverb(15)
                if IIIIllllIlIIllIlI == IIIIllIllllllIlII
                else (
                    _audio_pre_ if "DeEcho" not in IIIIllllIlIIllIlI else _audio_pre_new
                )(
                    agg=int(IlIlllIIIllIIlIII),
                    model_path=os.path.join(
                        IllllIIIIIIIlIlIl, IIIIllllIlIIllIlI + _IIIllIllllIIlIIIl
                    ),
                    device=IIlIlllllIlllIIlI.device,
                    is_half=IIlIlllllIlllIIlI.is_half,
                )
            )
            try:
                if IlllllIIllIlIlIIl != _IIlIIllIIIlIlIIII:
                    IlllllIIllIlIlIIl = [
                        IIIlIlllllIIIIIlI.name
                        for IIIlIlllllIIIIIlI in IlllllIIllIlIlIIl
                    ]
                else:
                    IlllllIIllIlIlIIl = IIlIlIIlllIllllIl
            except:
                traceback.print_exc()
                IlllllIIllIlIlIIl = IIlIlIIlllIllllIl
            print(IlllllIIllIlIlIIl)
            for IIlIlIIIllIIIIIII in IlllllIIllIlIlIIl:
                IIllIIIIlllIIIlII = os.path.join(IIlIllIIlIlIllllI, IIlIlIIIllIIIIIII)
                IllIIIIIlllllIIIl, IlIlIIIIIlllIlllI = 1, 0
                try:
                    IlllIlIIlIlllllII = IllllIllIIllIIlIl.probe(
                        IIllIIIIlllIIIlII, cmd="ffprobe"
                    )
                    if (
                        IlllIlIIlIlllllII[IlIIllllIIIllIlll][0]["channels"] == 2
                        and IlllIlIIlIlllllII[IlIIllllIIIllIlll][0][_IIllIIIllIIlIIlII]
                        == "44100"
                    ):
                        IllIIIIIlllllIIIl = 0
                        IlIIlIlIIllIllIlI._path_audio_(
                            IIllIIIIlllIIIlII,
                            IlllIlIIlIlIIlIll,
                            IIIIIIIIIlIIlIIlI,
                            IlllllllIIIlIlllI,
                        )
                        IlIlIIIIIlllIlllI = 1
                except:
                    traceback.print_exc()
                if IllIIIIIlllllIIIl:
                    IlIlllllllIlIlIIl = f"{IlllIIlllIlllllIl}/{os.path.basename(IlIllllIllIllllII(IIllIIIIlllIIIlII))}.reformatted.wav"
                    os.system(
                        f"ffmpeg -i {IlIllllIllIllllII(IIllIIIIlllIIIlII)} -vn -acodec pcm_s16le -ac 2 -ar 44100 {IlIllllIllIllllII(IlIlllllllIlIlIIl)} -y"
                    )
                    IIllIIIIlllIIIlII = IlIlllllllIlIlIIl
                try:
                    if not IlIlIIIIIlllIlllI:
                        IlIIlIlIIllIllIlI._path_audio_(
                            IIllIIIIlllIIIlII,
                            IlllIlIIlIlIIlIll,
                            IIIIIIIIIlIIlIIlI,
                            IlllllllIIIlIlllI,
                        )
                    IIIIllIIIIIlIlIlI.append(
                        f"{os.path.basename(IIllIIIIlllIIIlII)}->Success"
                    )
                    yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
                except:
                    IIIIllIIIIIlIlIlI.append(
                        f"{os.path.basename(IIllIIIIlllIIIlII)}->{traceback.format_exc()}"
                    )
                    yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
        except:
            IIIIllIIIIIlIlIlI.append(traceback.format_exc())
            yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
        finally:
            try:
                if IIIIllllIlIIllIlI == IIIIllIllllllIlII:
                    del IlIIlIlIIllIllIlI.pred.model
                    del IlIIlIlIIllIllIlI.pred.model_
                else:
                    del IlIIlIlIIllIllIlI.model
                del IlIIlIlIIllIllIlI
            except:
                traceback.print_exc()
            print(_IllllllIIlIlIIIII)
            if IlIIIlIIllIlllIIl.cuda.is_available():
                IlIIIlIIllIlllIIl.cuda.empty_cache()
        yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
    elif IIlIlllIlIllIlIll == _IIIllIlIlIIIIIllI:
        try:
            IIIIllIIIIIlIlIlI.append(
                IlIllIllIIllIlIII(
                    "Starting audio conversion... (This might take a moment)"
                )
            )
            yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
            IIlIllIIlIlIllllI, IIIIIIIIIlIIlIIlI, IlllIlIIlIlIIlIll = [
                IllIIllIIIIlIllIl.strip(_IIlIllllIlIlIllII)
                .strip(_IlIlIlIlllllIIlIl)
                .strip(_IIlIlllllIIIIIIll)
                .strip(_IlIlIlIlllllIIlIl)
                .strip(_IIlIllllIlIlIllII)
                for IllIIllIIIIlIllIl in [
                    IIlIllIIlIlIllllI,
                    IIIIIIIIIlIIlIIlI,
                    IlllIlIIlIlIIlIll,
                ]
            ]
            IIlIlIIlllIllllIl = [
                os.path.join(IIlIllIIlIlIllllI, IIllIIlIIIIllIIIl)
                for IIllIIlIIIIllIIIl in os.listdir(IIlIllIIlIlIllllI)
                if IIllIIlIIIIllIIIl.endswith(tuple(IIlIIlllIlllIIIll))
            ]
            try:
                if IlllllIIllIlIlIIl != _IIlIIllIIIlIlIIII:
                    IlllllIIllIlIlIIl = [
                        IIIllIlIIlIIllllI.name
                        for IIIllIlIIlIIllllI in IlllllIIllIlIlIIl
                    ]
                else:
                    IlllllIIllIlIlIIl = IIlIlIIlllIllllIl
            except:
                traceback.print_exc()
                IlllllIIllIlIlIIl = IIlIlIIlllIllllIl
            print(IlllllIIllIlIlIIl)
            IIlIlllIIIlIIllll = _IIIllIllIIlIIIIIl
            IlllllIIlIIIIIIII = _IIIllIllIIlIIIIIl
            IlIIlllllllIllIlI = _IIIllIllIIlIIIIIl
            IlIIlIIIIllIlIllI = 3072
            IlIIllllIIIIIllll = 256
            IlllIIlllIlIlIlll = 7680
            IIlIlIIIIIIlIIIll = _IIIllIllIIlIIIIIl
            IIIlllIIIlllllllI = 1.025
            IIllllIllIllIIIlI = "Vocals_custom"
            IIlIIllllllIllIIl = "Instrumental_custom"
            IIlIlIIllIIlllIlI = _IIIllIllIIlIIIIIl
            IlIlIlIIIllllllIl = id_to_ptm(IIIIllllIlIIllIlI)
            IIIlllIIIlllllllI = (
                IIIlllIIIlllllllI
                if IIlIlIIIIIIlIIIll or IlIIlllllllIllIlI
                else _IIlIIllIIIlIlIIII
            )
            IIIlIlIIllIIllIlI = prepare_mdx(
                IlIlIlIIIllllllIl,
                IlIIlllllllIllIlI,
                IlIIlIIIIllIlIllI,
                IlIIllllIIIIIllll,
                IlllIIlllIlIlIlll,
                compensation=IIIlllIIIlllllllI,
            )
            for IIlIlIIIllIIIIIII in IlllllIIllIlIlIIl:
                IlIlIIIllllllllIl = (
                    IIllllIllIllIIIlI if IlIIlllllllIllIlI else _IIlIIllIIIlIlIIII
                )
                IllIIlllllIllllIl = (
                    IIlIIllllllIllIIl if IlIIlllllllIllIlI else _IIlIIllIIIlIlIIII
                )
                run_mdx(
                    IlIlIlIIIllllllIl,
                    IIIlIlIIllIIllIlI,
                    IIlIlIIIllIIIIIII,
                    IlllllllIIIlIlllI,
                    diff=IIlIlllIIIlIIllll,
                    suffix=IlIlIIIllllllllIl,
                    diff_suffix=IllIIlllllIllllIl,
                    denoise=IlllllIIlIIIIIIII,
                )
            if IIlIlIIllIIlllIlI:
                print()
                print("[MDX-Net_Colab settings used]")
                print(f"Model used: {IlIlIlIIIllllllIl}")
                print(f"Model MD5: {mdx.MDX.get_hash(IlIlIlIIIllllllIl)}")
                print(f"Model parameters:")
                print(f"    -dim_f: {IIIlIlIIllIIllIlI.dim_f}")
                print(f"    -dim_t: {IIIlIlIIllIIllIlI.dim_t}")
                print(f"    -n_fft: {IIIlIlIIllIIllIlI.n_fft}")
                print(f"    -compensation: {IIIlIlIIllIIllIlI.compensation}")
                print()
                print("[Input file]")
                print("filename(s): ")
                for IIlIllIIllIllIIII in IlllllIIllIlIlIIl:
                    print(f"    -{IIlIllIIllIllIIII}")
                    IIIIllIIIIIlIlIlI.append(
                        f"{os.path.basename(IIlIllIIllIllIIII)}->Success"
                    )
                    yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
        except:
            IIIIllIIIIIlIlIlI.append(traceback.format_exc())
            yield _IIlIlllllIIIIIIll.join(IIIIllIIIIIlIlIlI)
        finally:
            try:
                del IIIlIlIIllIIllIlI
            except:
                traceback.print_exc()
            print(_IllllllIIlIlIIIII)
            if IlIIIlIIllIlllIIl.cuda.is_available():
                IlIIIlIIllIlllIIl.cuda.empty_cache()


def IIlIIIIlllllllIll(IIIIIlIIIlIlIIlIl, IllIIIllIlllIIlIl, IIIIIIIIlIIIIIllI):
    global IIIIIllIIlIllIlll, IlIllIIIIIIlIlIll, IllIIllIlIIlIIlII, IIIIlIIIIIlIIIIlI, IlIIIlIIlIllIIlII, IIlIIIlIIlIlllllI, IIlIlIIIIlllIIlIl
    if not IIIIIlIIIlIlIIlIl:
        if IIlIlIIIIlllIIlIl is not _IIlIIllIIIlIlIIII:
            print(_IllllllIIlIlIIIII)
            del (
                IllIIllIlIIlIIlII,
                IIIIIllIIlIllIlll,
                IIIIlIIIIIlIIIIlI,
                IIlIlIIIIlllIIlIl,
                IlIllIIIIIIlIlIll,
            )
            IIlIlIIIIlllIIlIl = (
                IllIIllIlIIlIIlII
            ) = (
                IIIIIllIIlIllIlll
            ) = (
                IIIIlIIIIIlIIIIlI
            ) = IIlIlIIIIlllIIlIl = IlIllIIIIIIlIlIll = _IIlIIllIIIlIlIIII
            if IlIIIlIIllIlllIIl.cuda.is_available():
                IlIIIlIIllIlllIIl.cuda.empty_cache()
            IIIIlIIIllIllllII, IIlIIIlIIlIlllllI = IlIIIlIIlIllIIlII.get(
                _IlllIIIIIIIIlIlll, 1
            ), IlIIIlIIlIllIIlII.get(_IIllIIlllllIllIII, _IlIlllllIlllIlIIl)
            IllIIllIlIIlIIlII = (
                (
                    SynthesizerTrnMs256NSFsid
                    if IIlIIIlIIlIlllllI == _IlIlllllIlllIlIIl
                    else SynthesizerTrnMs768NSFsid
                )(
                    *IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI],
                    is_half=IIlIlllllIlllIIlI.is_half,
                )
                if IIIIlIIIllIllllII == 1
                else (
                    SynthesizerTrnMs256NSFsid_nono
                    if IIlIIIlIIlIlllllI == _IlIlllllIlllIlIIl
                    else SynthesizerTrnMs768NSFsid_nono
                )(*IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI])
            )
            del IllIIllIlIIlIIlII, IlIIIlIIlIllIIlII
            if IlIIIlIIllIlllIIl.cuda.is_available():
                IlIIIlIIllIlllIIl.cuda.empty_cache()
            IlIIIlIIlIllIIlII = _IIlIIllIIIlIlIIII
        return (
            {
                _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
        ) * 3
    print(f"loading {IIIIIlIIIlIlIIlIl}")
    IlIIIlIIlIllIIlII = IlIIIlIIllIlllIIl.load(
        IIIIIlIIIlIlIIlIl, map_location=_IlIlIIlIlIlIIIlll
    )
    IlIllIIIIIIlIlIll = IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI][-1]
    IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI][-3] = IlIIIlIIlIllIIlII[_IlllIllIlllIlIllI][
        _IllIIIlIIllIIlIII
    ].shape[0]
    if IlIIIlIIlIllIIlII.get(_IlllIIIIIIIIlIlll, 1) == 0:
        IllIIIllIlllIIlIl = IIIIIIIIlIIIIIllI = {
            _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
            _IIllIIIIIIllllIll: 0.5,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
    else:
        IllIIIllIlllIIlIl = {
            _IlIlIllIIIIlIlIll: _IIIllIllIIlIIIIIl,
            _IIllIIIIIIllllIll: IllIIIllIlllIIlIl,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
        IIIIIIIIlIIIIIllI = {
            _IlIlIllIIIIlIlIll: _IIIllIllIIlIIIIIl,
            _IIllIIIIIIllllIll: IIIIIIIIlIIIIIllI,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
    IIlIIIlIIlIlllllI = IlIIIlIIlIllIIlII.get(_IIllIIlllllIllIII, _IlIlllllIlllIlIIl)
    IllIIllIlIIlIIlII = (
        (
            SynthesizerTrnMs256NSFsid
            if IIlIIIlIIlIlllllI == _IlIlllllIlllIlIIl
            else SynthesizerTrnMs768NSFsid
        )(*IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI], is_half=IIlIlllllIlllIIlI.is_half)
        if IlIIIlIIlIllIIlII.get(_IlllIIIIIIIIlIlll, 1) == 1
        else (
            SynthesizerTrnMs256NSFsid_nono
            if IIlIIIlIIlIlllllI == _IlIlllllIlllIlIIl
            else SynthesizerTrnMs768NSFsid_nono
        )(*IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI])
    )
    del IllIIllIlIIlIIlII.enc_q
    print(
        IllIIllIlIIlIIlII.load_state_dict(
            IlIIIlIIlIllIIlII[_IlllIllIlllIlIllI], strict=_IIllIlIlIIllIlIII
        )
    )
    IllIIllIlIIlIIlII.eval().to(IIlIlllllIlllIIlI.device)
    IllIIllIlIIlIIlII = (
        IllIIllIlIIlIIlII.half()
        if IIlIlllllIlllIIlI.is_half
        else IllIIllIlIIlIIlII.float()
    )
    IIIIlIIIIIlIIIIlI = VC(IlIllIIIIIIlIlIll, IIlIlllllIlllIIlI)
    IIIIIllIIlIllIlll = IlIIIlIIlIllIIlII[_IIIlIlIIIIIlIIllI][-3]
    return (
        {
            _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
            "maximum": IIIIIllIIlIllIlll,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
        IllIIIllIlllIIlIl,
        IIIIIIIIlIIIIIllI,
    )


def IIllIlllIIIIIIIll():
    IIlIIIIIlIIlIIIlI = [
        os.path.join(IlIlIIIIIlIlllllI, IlIlllIIIlIlIIllI)
        for (IlIlIIIIIlIlllllI, _IIIIIIIIllIIIIIll, IIIIlllllllllIllI) in os.walk(
            IlIIlIlIlIlllllll
        )
        for IlIlllIIIlIlIIllI in IIIIlllllllllIllI
        if IlIlllIIIlIlIIllI.endswith((_IIIllIllllIIlIIIl, _IlllIlIllIlIIIIlI))
    ]
    IIlIllIIIIIIlIIII = [
        os.path.join(IllllllllIIIlIIII, IIIllIIIlIIIIIIII)
        for (IllllllllIIIlIIII, _IIIIIIlIIIIIIIllI, IlIlIIllIllIIllIl) in os.walk(
            IllIlIIlIlIIlIlII, topdown=_IIllIlIlIIllIlIII
        )
        for IIIllIIIlIIIIIIII in IlIlIIllIllIIllIl
        if IIIllIIIlIIIIIIII.endswith(_IlIIIIIIIIIIllIlI)
        and _IlIIIIIIIIlIIlllI not in IIIllIIIlIIIIIIII
    ]
    IlllIlIlllIllIlll = [
        os.path.join(IllIlIIIllllIlIlI, IllIlIlIIlllIlIIl)
        for IllIlIlIIlllIlIIl in os.listdir(
            os.path.join(IIIIIlIlllllIIIlI, _IllllIIIIllIIlIII)
        )
    ]
    return (
        {
            _IIlIIllllllllllll: sorted(IIlIIIIIlIIlIIIlI),
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
        {
            _IIlIIllllllllllll: sorted(IIlIllIIIIIIlIIII),
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
        {
            _IIlIIllllllllllll: sorted(IlllIlIlllIllIlll),
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
    )


def IIlIIlIIIlIIllIlI():
    IIIlIllIIIIIlIIlI = [
        os.path.join(IlIIIlIllIIllIlll, IlIIlIIIllIlllllI)
        for (IlIIIlIllIIllIlll, _IllIllllIIlIllllI, IIlIlIIIIIllllIII) in os.walk(
            IlIIlIlIlIlllllll
        )
        for IlIIlIIIllIlllllI in IIlIlIIIIIllllIII
        if IlIIlIIIllIlllllI.endswith((_IIIllIllllIIlIIIl, _IlllIlIllIlIIIIlI))
    ]
    IIlllllIIllllIIll = [
        os.path.join(IlllIIIIIlIIIllIl, IIIIllIllIllIllll)
        for (IlllIIIIIlIIIllIl, _IllIIlIlIIIIlIlIl, IIIlIIlIIlllIIlll) in os.walk(
            IllIlIIlIlIIlIlII, topdown=_IIllIlIlIIllIlIII
        )
        for IIIIllIllIllIllll in IIIlIIlIIlllIIlll
        if IIIIllIllIllIllll.endswith(_IlIIIIIIIIIIllIlI)
        and _IlIIIIIIIIlIIlllI not in IIIIllIllIllIllll
    ]
    return {
        _IIlIIllllllllllll: sorted(IIIlIllIIIIIlIIlI),
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }, {
        _IIlIIllllllllllll: sorted(IIlllllIIllllIIll),
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }


def IlIlIIlIIlllIllIl():
    IIllllllllIIlIIll = [
        os.path.join(IllIlIIIllllIlIlI, IlIlllIIIIlllIIIl)
        for IlIlllIIIIlllIIIl in os.listdir(
            os.path.join(IIIIIlIlllllIIIlI, _IllllIIIIllIIlIII)
        )
    ]
    IIIlIIllllllIIlll = [
        os.path.join(IIlIIIllIIllIlIll, IIIlllllIIlllllll)
        for IIIlllllIIlllllll in os.listdir(
            os.path.join(IIIIIlIlllllIIIlI, _IlIllIllIllIIllIl)
        )
    ]
    return {
        _IIlIIllllllllllll: sorted(IIIlIIllllllIIlll),
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }, {
        _IIlIIllllllllllll: sorted(IIllllllllIIlIIll),
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }


IIllllllIIIIIIlll = {
    _IllIIlIlIlIlllllI: 32000,
    _IIlllllllIIIIlIII: 40000,
    _IlIIIlIIIIlllllII: 48000,
}


def IlIllIIlIIIllIIlI(IlIllllIlIIIIlIIl, IIIIIllllIlIIllII):
    while IIIIIllllIlIIllII.poll() is _IIlIIllIIIlIlIIII:
        time.sleep(0.5)
    IlIllllIlIIIIlIIl[0] = _IIIllIllIIlIIIIIl


def IIlIlIIlIIIIlIlIl(IIlIlllIlIlllIIIl, IlllllIIIllIlIllI):
    while not all(
        IllIIIllIlIlllIlI.poll() is not _IIlIIllIIIlIlIIII
        for IllIIIllIlIlllIlI in IlllllIIIllIlIllI
    ):
        time.sleep(0.5)
    IIlIlllIlIlllIIIl[0] = _IIIllIllIIlIIIIIl


def IIIllIlIIlIIIlIIl(IlIIllIIlIlllIIII, IIllIlIlIlIlllIII, IllIlIIlllIllIIll):
    global IIlIlIIIIlIIllIII, IlllIllIIlllIIlIl, IlllIIIIlIlIIIIII
    IIlIlIIIIlIIllIII = IlIIllIIlIlllIIII
    IlllIllIIlllIIlIl = IIllIlIlIlIlllIII
    IlllIIIIlIlIIIIII = IllIlIIlllIllIIll
    rvc_globals.DoFormant = IlIIllIIlIlllIIII
    rvc_globals.Quefrency = IIllIlIlIlIlllIII
    rvc_globals.Timbre = IllIlIIlllIllIIll
    IIIlIlIllIIlIlIIl = {
        _IlIlIllIIIIlIlIll: IIlIlIIIIlIIllIII,
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }
    return (
        {_IIllIIIIIIllllIll: IIlIlIIIIlIIllIII, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
    ) + (IIIlIlIllIIlIlIIl,) * 6


def IIlIlIlllIIIIlIIl(IlIlllIlIlllIIIII, IIlIIlIIlIIlIIlIl):
    global IlllIllIIlllIIlIl, IlllIIIIlIlIIIIII, IIlIlIIIIlIIllIII
    IlllIllIIlllIIlIl = IlIlllIlIlllIIIII
    IlllIIIIlIlIIIIII = IIlIIlIIlIIlIIlIl
    IIlIlIIIIlIIllIII = _IIIllIllIIlIIIIIl
    rvc_globals.DoFormant = _IIIllIllIIlIIIIIl
    rvc_globals.Quefrency = IlIlllIlIlllIIIII
    rvc_globals.Timbre = IIlIIlIIlIIlIIlIl
    return {
        _IIllIIIIIIllllIll: IlllIllIIlllIIlIl,
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }, {_IIllIIIIIIllllIll: IlllIIIIlIlIIIIII, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI}


def IllIIIIIllIIlIIIl(IIlIIlIlIIIlIIIll, IIIIIllllIlIlIIlI, IllIlIlIIIIIllIll):
    if IIlIIlIlIIIlIIIll:
        with open(IIlIIlIlIIIlIIIll, _IlllIlIIIIIlIIIIl) as IlIlIlllIIlllIIll:
            IlllIllIllllIIlll = IlIlIlllIIlllIIll.readlines()
            IIIIIllllIlIlIIlI, IllIlIlIIIIIllIll = (
                IlllIllIllllIIlll[0].strip(),
                IlllIllIllllIIlll[1],
            )
        IIlIlIlllIIIIlIIl(IIIIIllllIlIlIIlI, IllIlIlIIIIIllIll)
    else:
        IIIIIllllIlIlIIlI, IllIlIlIIIIIllIll = IIIlllIlIlllIllll(
            IIlIIlIlIIIlIIIll, IIIIIllllIlIlIIlI, IllIlIlIIIIIllIll
        )
    return (
        {
            _IIlIIllllllllllll: IIllIlllIIlIIIllI(),
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
        {_IIllIIIIIIllllIll: IIIIIllllIlIlIIlI, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
        {_IIllIIIIIIllllIll: IllIlIlIIIIIllIll, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
    )


def IIllIllllIIIlIIlI(
    IIIIIllIllIllIlll, IllIIlIIIlllIllll, IIlllIllIllIIIlll, IlllIlIIIlllIllIl
):
    IIlllIllIllIIIlll = IIllllllIIIIIIlll[IIlllIllIllIIIlll]
    IllIlIIIlIIIIllll = os.path.join(
        IIIIIlIlllllIIIlI, _IlllllIIllllllIlI, IllIIlIIIlllIllll
    )
    IlIIlllllIIIIIllI = os.path.join(IllIlIIIlIIIIllll, "preprocess.log")
    os.makedirs(IllIlIIIlIIIIllll, exist_ok=_IIIllIllIIlIIIIIl)
    with open(IlIIlllllIIIIIllI, "w") as IIIllllllIIIIlIlI:
        0
    IIIllIIIIIIlIlIll = f"{IIlIlllllIlllIIlI.python_cmd} trainset_preprocess_pipeline_print.py {IIIIIllIllIllIlll} {IlIllllIllIllllII(IIlllIllIllIIIlll)} {IlIllllIllIllllII(IlllIlIIIlllIllIl)} {IllIlIIIlIIIIllll} {IlIllllIllIllllII(IIlIlllllIlllIIlI.noparallel)}"
    print(IIIllIIIIIIlIlIll)
    IllllIIIIlllIlIII = Popen(IIIllIIIIIIlIlIll, shell=_IIIllIllIIlIIIIIl)
    IllIlIIlllllIlIII = [_IIllIlIlIIllIlIII]
    threading.Thread(
        target=IlIllIIlIIIllIIlI, args=(IllIlIIlllllIlIII, IllllIIIIlllIlIII)
    ).start()
    while not IllIlIIlllllIlIII[0]:
        with open(IlIIlllllIIIIIllI, _IlllIlIIIIIlIIIIl) as IIIllllllIIIIlIlI:
            yield IIIllllllIIIIlIlI.read()
        time.sleep(1)
    with open(IlIIlllllIIIIIllI, _IlllIlIIIIIlIIIIl) as IIIllllllIIIIlIlI:
        IlllIIllIIIlllllI = IIIllllllIIIIlIlI.read()
    print(IlllIIllIIIlllllI)
    yield IlllIIllIIIlllllI


def IlIllIllIllllIIIl(
    IIIIIIlIlllIlllll,
    IIIlIIIlllIlIlIIl,
    IIIIIIIlIlIIIIlll,
    IIlIIIIllIllllIlI,
    IllllIlIIIlIlIllI,
    IlIllIlllIIlIIIII,
    IIIIllIIlIIIIIlII,
):
    IIIIIIlIlllIlllll = IIIIIIlIlllIlllll.split("-")
    IIIlIIlIIlIlIllll = f"{IIIIIlIlllllIIIlI}/logs/{IllllIlIIIlIlIllI}"
    IIlIIIIIIIIllllIl = f"{IIIlIIlIIlIlIllll}/extract_fl_feature.log"
    os.makedirs(IIIlIIlIIlIlIllll, exist_ok=_IIIllIllIIlIIIIIl)
    with open(IIlIIIIIIIIllllIl, "w") as IllllllllIIIIIIII:
        0
    if IIlIIIIllIllllIlI:
        IllllIIllllIIIIll = f"{IIlIlllllIlllIIlI.python_cmd} extract_fl_print.py {IIIlIIlIIlIlIllll} {IlIllllIllIllllII(IIIlIIIlllIlIlIIl)} {IlIllllIllIllllII(IIIIIIIlIlIIIIlll)} {IlIllllIllIllllII(IIIIllIIlIIIIIlII)}"
        print(IllllIIllllIIIIll)
        IIIIllllllIllIlIl = Popen(
            IllllIIllllIIIIll, shell=_IIIllIllIIlIIIIIl, cwd=IIIIIlIlllllIIIlI
        )
        IlIlIlIIIllIllIIl = [_IIllIlIlIIllIlIII]
        threading.Thread(
            target=IlIllIIlIIIllIIlI, args=(IlIlIlIIIllIllIIl, IIIIllllllIllIlIl)
        ).start()
        while not IlIlIlIIIllIllIIl[0]:
            with open(IIlIIIIIIIIllllIl, _IlllIlIIIIIlIIIIl) as IllllllllIIIIIIII:
                yield IllllllllIIIIIIII.read()
            time.sleep(1)
    IllIIlIllllIIllII = len(IIIIIIlIlllIlllll)
    IllIIIIlIlIllIllI = []
    for IlIIlllIIlllllllI, IIIllIlIIllIllllI in enumerate(IIIIIIlIlllIlllll):
        IllllIIllllIIIIll = f"{IIlIlllllIlllIIlI.python_cmd} extract_feature_print.py {IlIllllIllIllllII(IIlIlllllIlllIIlI.device)} {IlIllllIllIllllII(IllIIlIllllIIllII)} {IlIllllIllIllllII(IlIIlllIIlllllllI)} {IlIllllIllIllllII(IIIllIlIIllIllllI)} {IIIlIIlIIlIlIllll} {IlIllllIllIllllII(IlIllIlllIIlIIIII)}"
        print(IllllIIllllIIIIll)
        IIIIllllllIllIlIl = Popen(
            IllllIIllllIIIIll, shell=_IIIllIllIIlIIIIIl, cwd=IIIIIlIlllllIIIlI
        )
        IllIIIIlIlIllIllI.append(IIIIllllllIllIlIl)
    IlIlIlIIIllIllIIl = [_IIllIlIlIIllIlIII]
    threading.Thread(
        target=IIlIlIIlIIIIlIlIl, args=(IlIlIlIIIllIllIIl, IllIIIIlIlIllIllI)
    ).start()
    while not IlIlIlIIIllIllIIl[0]:
        with open(IIlIIIIIIIIllllIl, _IlllIlIIIIIlIIIIl) as IllllllllIIIIIIII:
            yield IllllllllIIIIIIII.read()
        time.sleep(1)
    with open(IIlIIIIIIIIllllIl, _IlllIlIIIIIlIIIIl) as IllllllllIIIIIIII:
        IIIIlIlllllIlIlII = IllllllllIIIIIIII.read()
    print(IIIIlIlllllIlIlII)
    yield IIIIlIlllllIlIlII


def IIIIlIIIlIIllllIl(IIIIlIllIlIIIIlII, IllIIIlIlllIlIIlI, IllIIIlIllIlIIIIl):
    IIlIIIllIIIIIIlIl = (
        "" if IllIIIlIllIlIIIIl == _IlIlllllIlllIlIIl else _IllIllllllIIIlIll
    )
    IIlIllIlIlIIlIIIl = _IlllIIIIIIIIlIlll if IllIIIlIlllIlIIlI else ""
    IIllIllIIIIllIllI = {_IIlIllIlllIlllIIl: "", _IIllIIllIIlIIlIIl: ""}
    for IllIIlIIIIlIIlIlI in IIllIllIIIIllIllI:
        IIlllllIllIIIIIll = f"/kaggle/input/ax-rmf/pretrained{IIlIIIllIIIIIIlIl}/{IIlIllIlIlIIlIIIl}{IllIIlIIIIlIIlIlI}{IIIIlIllIlIIIIlII}.pth"
        if os.access(IIlllllIllIIIIIll, os.F_OK):
            IIllIllIIIIllIllI[IllIIlIIIIlIIlIlI] = IIlllllIllIIIIIll
        else:
            print(f"{IIlllllIllIIIIIll} doesn't exist, will not use pretrained model.")
    return IIllIllIIIIllIllI[_IIlIllIlllIlllIIl], IIllIllIIIIllIllI[_IIllIIllIIlIIlIIl]


def IlIlIIIlllllIllIl(IIlllIIlIIIllIIlI, IIIIIlIIIIIIllIll, IlllIIlIllIlIIlll):
    IIlIIIIIlIlIIIlIl = (
        "" if IlllIIlIllIlIIlll == _IlIlllllIlllIlIIl else _IllIllllllIIIlIll
    )
    IIlllIIlIIIllIIlI = (
        _IIlllllllIIIIlIII
        if IIlllIIlIIIllIIlI == _IllIIlIlIlIlllllI
        and IlllIIlIllIlIIlll == _IlIlllllIlllIlIIl
        else IIlllIIlIIIllIIlI
    )
    IIllIIIIllIlIIlIl = (
        {
            _IIlIIllllllllllll: [_IIlllllllIIIIlIII, _IlIIIlIIIIlllllII],
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            _IIllIIIIIIllllIll: IIlllIIlIIIllIIlI,
        }
        if IlllIIlIllIlIIlll == _IlIlllllIlllIlIIl
        else {
            _IIlIIllllllllllll: [
                _IIlllllllIIIIlIII,
                _IlIIIlIIIIlllllII,
                _IllIIlIlIlIlllllI,
            ],
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            _IIllIIIIIIllllIll: IIlllIIlIIIllIIlI,
        }
    )
    IlIIIlIIIllIIllll = _IlllIIIIIIIIlIlll if IIIIIlIIIIIIllIll else ""
    IIIllllIIIIlIIlII = {_IIlIllIlllIlllIIl: "", _IIllIIllIIlIIlIIl: ""}
    for IIlIIIlllIIIIIIII in IIIllllIIIIlIIlII:
        IllIIlllIIllIIlIl = f"/kaggle/input/ax-rmf/pretrained{IIlIIIIIlIlIIIlIl}/{IlIIIlIIIllIIllll}{IIlIIIlllIIIIIIII}{IIlllIIlIIIllIIlI}.pth"
        if os.access(IllIIlllIIllIIlIl, os.F_OK):
            IIIllllIIIIlIIlII[IIlIIIlllIIIIIIII] = IllIIlllIIllIIlIl
        else:
            print(f"{IllIIlllIIllIIlIl} doesn't exist, will not use pretrained model.")
    return (
        IIIllllIIIIlIIlII[_IIlIllIlllIlllIIl],
        IIIllllIIIIlIIlII[_IIllIIllIIlIIlIIl],
        IIllIIIIllIlIIlIl,
    )


def IlIIlIlllllIllIlI(IlIlIlIlIlIIIIIII, IllllIIllllIIIlII, IlIIlIlIlIlIlIllI):
    IlllIIIIIllIIIIlI = (
        "" if IlIIlIlIlIlIlIllI == _IlIlllllIlllIlIIl else _IllIllllllIIIlIll
    )
    IIIlIlIIllIIlIIll = "/kaggle/input/ax-rmf/pretrained%s/f0%s%s.pth"
    IIllIllIIIIIIlllI = {_IIlIllIlllIlllIIl: "", _IIllIIllIIlIIlIIl: ""}
    for IlIIIIIllIIlIIlll in IIllIllIIIIIIlllI:
        IlIlIIlllllIllllI = IIIlIlIIllIIlIIll % (
            IlllIIIIIllIIIIlI,
            IlIIIIIllIIlIIlll,
            IllllIIllllIIIlII,
        )
        if os.access(IlIlIIlllllIllllI, os.F_OK):
            IIllIllIIIIIIlllI[IlIIIIIllIIlIIlll] = IlIlIIlllllIllllI
        else:
            print(IlIlIIlllllIllllI, "doesn't exist, will not use pretrained model")
    return (
        {_IlIlIllIIIIlIlIll: IlIlIlIlIlIIIIIII, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
        IIllIllIIIIIIlllI[_IIlIllIlllIlllIIl],
        IIllIllIIIIIIlllI[_IIllIIllIIlIIlIIl],
        {_IlIlIllIIIIlIlIll: IlIlIlIlIlIIIIIII, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
    )


global IllllIIIIIIIIIIlI


def IlIllIlIllIllIlIl(IlllllIlIllIlllIl, IllllllIlllllIlIl):
    IlIlIIIIlIIlIlIlI = 1
    IIIIlIllllllIllIl = os.path.join(IlllllIlIllIlllIl, "1_16k_wavs")
    if os.path.isdir(IIIIlIllllllIllIl):
        IIIIlIllIlllllIll = len(glob1(IIIIlIllllllIllIl, "*.wav"))
        if IIIIlIllIlllllIll > 0:
            IlIlIIIIlIIlIlIlI = IllIlllIIlllIllIl.ceil(
                IIIIlIllIlllllIll / IllllllIlllllIlIl
            )
            if IlIlIIIIlIIlIlIlI > 1:
                IlIlIIIIlIIlIlIlI += 1
    return IlIlIIIIlIIlIlIlI


global IlIIIllIlllllIlIl, IllIllIIIIIlIlIII


def IIllIIIllIIllIlII(
    IlIIlIllIllllIIII,
    IlIlIlIllIllIIlII,
    IlIlIlIlIIlIlIIIl,
    IlIlllIIllIIIIIll,
    IlllIlIIlIIlllIII,
    IIllIlIlIIlIIllIl,
    IlllIIIlIlIIlllIl,
    IIIlIIIlIlIIlIIll,
    IIIIlIIIllIIIllll,
    IlIlIlIIllIIIIIlI,
    IIIIIlIIlIIllIIll,
    IIlllIllIIlIllIll,
    IIIIlIllIIIllllll,
    IllIIlIlllllIllll,
):
    with open(_IllIIIIIIIlIIIllI, "w+") as IllIIIIlllIIIlIlI:
        IllIIIIlllIIIlIlI.write("False")
    IIlIIllIlllIIIIII = os.path.join(
        IIIIIlIlllllIIIlI, _IlllllIIllllllIlI, IlIIlIllIllllIIII
    )
    os.makedirs(IIlIIllIlllIIIIII, exist_ok=_IIIllIllIIlIIIIIl)
    IIIlllIIIlllllIII = os.path.join(IIlIIllIlllIIIIII, "0_gt_wavs")
    IIlllIllIlllIIlll = "256" if IllIIlIlllllIllll == _IlIlllllIlllIlIIl else "768"
    IIIllIIIlIllIIIIl = os.path.join(IIlIIllIlllIIIIII, f"3_feature{IIlllIllIlllIIlll}")
    IIlIIIllllIlIIllI = IlIllIlIllIllIlIl(IIlIIllIlllIIIIII, IlllIIIlIlIIlllIl)
    IlIllIIIIIIIIIllI = [IIIlllIIIlllllIII, IIIllIIIlIllIIIIl]
    if IlIlIlIlIIlIlIIIl:
        IlIIIIIIIIllllllI = f"{IIlIIllIlllIIIIII}/2a_f0"
        IlIlIIllllIIlIIll = f"{IIlIIllIlllIIIIII}/2b-f0nsf"
        IlIllIIIIIIIIIllI.extend([IlIIIIIIIIllllllI, IlIlIIllllIIlIIll])
    IlllIllllIIlIIlll = set(
        IIlllIlllIllIIlII.split(_IlIIIllIlIlIIlIIl)[0]
        for IIlIIIIllIIIlIIlI in IlIllIIIIIIIIIllI
        for IIlllIlllIllIIlII in os.listdir(IIlIIIIllIIIlIIlI)
    )

    def IlIlIlIllIIIIIllI(IllIlIllIIlIIIlII):
        IIlIIIIIIllIlIlll = [IIIlllIIIlllllIII, IIIllIIIlIllIIIIl]
        if IlIlIlIlIIlIlIIIl:
            IIlIIIIIIllIlIlll.extend([IlIIIIIIIIllllllI, IlIlIIllllIIlIIll])
        return "|".join(
            [
                IllllIlIllllllllI.replace("\\", "\\\\")
                + "/"
                + IllIlIllIIlIIIlII
                + (
                    ".wav.npy"
                    if IllllIlIllllllllI in [IlIIIIIIIIllllllI, IlIlIIllllIIlIIll]
                    else ".wav"
                    if IllllIlIllllllllI == IIIlllIIIlllllIII
                    else ".npy"
                )
                for IllllIlIllllllllI in IIlIIIIIIllIlIlll
            ]
        )

    IIllIIlIIIIlllIII = [
        f"{IlIlIlIllIIIIIllI(IlIIIIlllIIIIIllI)}|{IlIlllIIllIIIIIll}"
        for IlIIIIlllIIIIIllI in IlllIllllIIlIIlll
    ]
    IlllIlIIlIllIllII = f"{IIIIIlIlllllIIIlI}/logs/mute"
    for _IlIlIlllllllllIll in range(2):
        IlIIlIIIIIIlIlIII = f"{IlllIlIIlIllIllII}/0_gt_wavs/mute{IlIlIlIllIllIIlII}.wav|{mute_dir}/3_feature{IIlllIllIlllIIlll}/mute.npy"
        if IlIlIlIlIIlIlIIIl:
            IlIIlIIIIIIlIlIII += f"|{IlllIlIIlIllIllII}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy"
        IIllIIlIIIIlllIII.append(IlIIlIIIIIIlIlIII + f"|{IlIlllIIllIIIIIll}")
    shuffle(IIllIIlIIIIlllIII)
    with open(f"{IIlIIllIlllIIIIII}/filelist.txt", "w") as IIllllIlIlIlIIlIl:
        IIllllIlIlIlIIlIl.write(_IIlIlllllIIIIIIll.join(IIllIIlIIIIlllIII))
    print("write filelist done")
    print("use gpus:", IIIIIlIIlIIllIIll)
    if IIIIlIIIllIIIllll == "":
        print("no pretrained Generator")
    if IlIlIlIIllIIIIIlI == "":
        print("no pretrained Discriminator")
    IllllIlIllllllIll = f"-pg {IIIIlIIIllIIIllll}" if IIIIlIIIllIIIllll else ""
    IIIIllllllllIIIIl = f"-pd {IlIlIlIIllIIIIIlI}" if IlIlIlIIllIIIIIlI else ""
    IllIllIIllIlIIIIl = f"{IIlIlllllIlllIIlI.python_cmd} train_nsf_sim_cache_sid_load_pretrain.py -e {IlIIlIllIllllIIII} -sr {IlIlIlIllIllIIlII} -f0 {int(IlIlIlIlIIlIlIIIl)} -bs {IlllIIIlIlIIlllIl} -g {IIIIIlIIlIIllIIll if IIIIIlIIlIIllIIll is not _IIlIIllIIIlIlIIII else''} -te {IIllIlIlIIlIIllIl} -se {IlllIlIIlIIlllIII} {IllllIlIllllllIll} {IIIIllllllllIIIIl} -l {int(IIIlIIIlIlIIlIIll)} -c {int(IIlllIllIIlIllIll)} -sw {int(IIIIlIllIIIllllll)} -v {IllIIlIlllllIllll} -li {IIlIIIllllIlIIllI}"
    print(IllIllIIllIlIIIIl)
    global IIIIIIIllllIIIllI
    IIIIIIIllllIIIllI = Popen(
        IllIllIIllIlIIIIl, shell=_IIIllIllIIlIIIIIl, cwd=IIIIIlIlllllIIIlI
    )
    global IlIIIllIlllllIlIl
    IlIIIllIlllllIlIl = IIIIIIIllllIIIllI.pid
    IIIIIIIllllIIIllI.wait()
    return (
        IlIllIllIIllIlIII("Training is done, check train.log"),
        {
            _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
        {
            _IlIlIllIIIIlIlIll: _IIIllIllIIlIIIIIl,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        },
    )


def IIIllllIlIlIlllll(IIIIlIIllIIIIIIII, IlllIlIIlllIlllll):
    IIlIIlIlIIlIIlIlI = os.path.join(
        IIIIIlIlllllIIIlI, _IlllllIIllllllIlI, IIIIlIIllIIIIIIII
    )
    os.makedirs(IIlIIlIlIIlIIlIlI, exist_ok=_IIIllIllIIlIIIIIl)
    IllIlIlllIIIIIlIl = "256" if IlllIlIIlllIlllll == _IlIlllllIlllIlIIl else "768"
    IllllIllIIIlIIlIl = os.path.join(IIlIIlIlIIlIIlIlI, f"3_feature{IllIlIlllIIIIIlIl}")
    if not os.path.exists(IllllIllIIIlIIlIl) or len(os.listdir(IllllIllIIIlIIlIl)) == 0:
        return "请先进行特征提取!"
    IlllIIllIIIlIlIlI = [
        IlIIIlIlllIlIlllI.load(os.path.join(IllllIllIIIlIIlIl, IlIIIllllIlllIlIl))
        for IlIIIllllIlllIlIl in sorted(os.listdir(IllllIllIIIlIIlIl))
    ]
    IIllIIIIIllIlIIII = IlIIIlIlllIlIlllI.concatenate(IlllIIllIIIlIlIlI, 0)
    IlIIIlIlllIlIlllI.random.shuffle(IIllIIIIIllIlIIII)
    IlllIIllllIlIlIII = []
    if IIllIIIIIllIlIIII.shape[0] > 2 * 10**5:
        IlllIIllllIlIlIII.append(
            "Trying doing kmeans %s shape to 10k centers." % IIllIIIIIllIlIIII.shape[0]
        )
        yield _IIlIlllllIIIIIIll.join(IlllIIllllIlIlIII)
        try:
            IIllIIIIIllIlIIII = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=_IIIllIllIIlIIIIIl,
                    batch_size=256 * IIlIlllllIlllIIlI.n_cpu,
                    compute_labels=_IIllIlIlIIllIlIII,
                    init="random",
                )
                .fit(IIllIIIIIllIlIIII)
                .cluster_centers_
            )
        except Exception as IIIlllIIIIlllIIII:
            IlllIIllllIlIlIII.append(str(IIIlllIIIIlllIIII))
            yield _IIlIlllllIIIIIIll.join(IlllIIllllIlIlIII)
    IlIIIlIlllIlIlllI.save(
        os.path.join(IIlIIlIlIIlIIlIlI, "total_fea.npy"), IIllIIIIIllIlIIII
    )
    IIllIIIlIIIIIlIII = min(
        int(16 * IlIIIlIlllIlIlllI.sqrt(IIllIIIIIllIlIIII.shape[0])),
        IIllIIIIIllIlIIII.shape[0] // 39,
    )
    IlllIIllllIlIlIII.append("%s,%s" % (IIllIIIIIllIlIIII.shape, IIllIIIlIIIIIlIII))
    yield _IIlIlllllIIIIIIll.join(IlllIIllllIlIlIII)
    IIIlIIlIIIlIIIlll = faiss.index_factory(
        int(IllIlIlllIIIIIlIl), f"IVF{IIllIIIlIIIIIlIII},Flat"
    )
    IllIIIlllIIIIIIIl = faiss.extract_index_ivf(IIIlIIlIIIlIIIlll)
    IllIIIlllIIIIIIIl.nprobe = 1
    IIIlIIlIIIlIIIlll.train(IIllIIIIIllIlIIII)
    IIIIlIIIlIIIIlIlI = f"{IIlIIlIlIIlIIlIlI}/trained_IVF{IIllIIIlIIIIIlIII}_Flat_nprobe_{IllIIIlllIIIIIIIl.nprobe}_{IIIIlIIllIIIIIIII}_{IlllIlIIlllIlllll}.index"
    faiss.write_index(IIIlIIlIIIlIIIlll, IIIIlIIIlIIIIlIlI)
    IlllIIllllIlIlIII.append("adding")
    yield _IIlIlllllIIIIIIll.join(IlllIIllllIlIlIII)
    IllIlIlIllIlIlIlI = 8192
    for IlIIIIlIIlIIlllIl in range(0, IIllIIIIIllIlIIII.shape[0], IllIlIlIllIlIlIlI):
        IIIlIIlIIIlIIIlll.add(
            IIllIIIIIllIlIIII[IlIIIIlIIlIIlllIl : IlIIIIlIIlIIlllIl + IllIlIlIllIlIlIlI]
        )
    IIIIlIIIlIIIIlIlI = f"{IIlIIlIlIIlIIlIlI}/added_IVF{IIllIIIlIIIIIlIII}_Flat_nprobe_{IllIIIlllIIIIIIIl.nprobe}_{IIIIlIIllIIIIIIII}_{IlllIlIIlllIlllll}.index"
    faiss.write_index(IIIlIIlIIIlIIIlll, IIIIlIIIlIIIIlIlI)
    IlllIIllllIlIlIII.append(
        f"Successful Index Construction，added_IVF{IIllIIIlIIIIIlIII}_Flat_nprobe_{IllIIIlllIIIIIIIl.nprobe}_{IIIIlIIllIIIIIIII}_{IlllIlIIlllIlllll}.index"
    )
    yield _IIlIlllllIIIIIIll.join(IlllIIllllIlIlIII)


def IIIIIllIlIlIIllIl(IIIIIIIIIllllIllI):
    IllllIIllIIIIlllI = os.path.join(os.path.dirname(IIIIIIIIIllllIllI), "train.log")
    if not os.path.exists(IllllIIllIIIIlllI):
        return (
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
        )
    try:
        with open(IllllIIllIIIIlllI, _IlllIlIIIIIlIIIIl) as IIIIlllIlIIlIllIl:
            IIIIIIlIIIIIIllII = next(IIIIlllIlIIlIllIl).strip()
            IIlIIlIIIIIlIlllI = eval(IIIIIIlIIIIIIllII.split("\t")[-1])
            IIIIlIIIIIlIIlllI, IIIlIIIllIIIIIlII = IIlIIlIIIIIlIlllI.get(
                _IIllIIIllIIlIIlII
            ), IIlIIlIIIIIlIlllI.get("if_f0")
            IllllllIIIlllllll = (
                _IlIlIIllllllIlIII
                if IIlIIlIIIIIlIlllI.get(_IIllIIlllllIllIII) == _IlIlIIllllllIlIII
                else _IlIlllllIlllIlIIl
            )
            return IIIIlIIIIIlIIlllI, str(IIIlIIIllIIIIIlII), IllllllIIIlllllll
    except Exception as IIllIllIlllllllll:
        print(
            f"Exception occurred: {str(IIllIllIlllllllll)}, Traceback: {traceback.format_exc()}"
        )
        return (
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
            {_IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI},
        )


def IllllIllllIIlIIll(IIIIlIIlIlIllIIll, IllllllIIIIlIllll):
    IIlIIlIlIIllIIlll = "rnd"
    IIlllIlIIIllIlIlI = "pitchf"
    IIlIIIIIlllIIIllI = "pitch"
    IIIIlIIllllIIlIIl = "phone"
    IIIIlIlIllIIIIIII = IlIIIlIIllIlllIIl.device(_IlIlIIlIlIlIIIlll)
    IIlllIIIlIIlIlllI = IlIIIlIIllIlllIIl.load(
        IIIIlIIlIlIllIIll, map_location=IIIIlIlIllIIIIIII
    )
    IIllIlIlIIIIIlIIl = (
        256
        if IIlllIIIlIIlIlllI.get(_IIllIIlllllIllIII, _IlIlllllIlllIlIIl)
        == _IlIlllllIlllIlIIl
        else 768
    )
    IlIlIlIllllIIllIl = {
        IIIIlIIllllIIlIIl: IlIIIlIIllIlllIIl.rand(1, 200, IIllIlIlIIIIIlIIl),
        "phone_lengths": IlIIIlIIllIlllIIl.LongTensor([200]),
        IIlIIIIIlllIIIllI: IlIIIlIIllIlllIIl.randint(5, 255, (1, 200)),
        IIlllIlIIIllIlIlI: IlIIIlIIllIlllIIl.rand(1, 200),
        "ds": IlIIIlIIllIlllIIl.zeros(1).long(),
        IIlIIlIlIIllIIlll: IlIIIlIIllIlllIIl.rand(1, 192, 200),
    }
    IIlllIIIlIIlIlllI[_IIIlIlIIIIIlIIllI][-3] = IIlllIIIlIIlIlllI[_IlllIllIlllIlIllI][
        _IllIIIlIIllIIlIII
    ].shape[0]
    IIlllllllIIIlIIII = SynthesizerTrnMsNSFsidM(
        *IIlllIIIlIIlIlllI[_IIIlIlIIIIIlIIllI],
        is_half=_IIllIlIlIIllIlIII,
        version=IIlllIIIlIIlIlllI.get(_IIllIIlllllIllIII, _IlIlllllIlllIlIIl),
    )
    IIlllllllIIIlIIII.load_state_dict(
        IIlllIIIlIIlIlllI[_IlllIllIlllIlIllI], strict=_IIllIlIlIIllIlIII
    )
    IIlllllllIIIlIIII = IIlllllllIIIlIIII.to(IIIIlIlIllIIIIIII)
    IllIllIIllIIIlIll = {
        IIIIlIIllllIIlIIl: [1],
        IIlIIIIIlllIIIllI: [1],
        IIlllIlIIIllIlIlI: [1],
        IIlIIlIlIIllIIlll: [2],
    }
    IlIIIlIIllIlllIIl.onnx.export(
        IIlllllllIIIlIIII,
        tuple(
            IIllIIIlllIIlIlIl.to(IIIIlIlIllIIIIIII)
            for IIllIIIlllIIlIlIl in IlIlIlIllllIIllIl.values()
        ),
        IllllllIIIIlIllll,
        dynamic_axes=IllIllIIllIIIlIll,
        do_constant_folding=_IIllIlIlIIllIlIII,
        opset_version=13,
        verbose=_IIllIlIlIIllIlIII,
        input_names=list(IlIlIlIllllIIllIl.keys()),
        output_names=["audio"],
    )
    return "Finished"


import scipy.io.wavfile as wavfile

IlIlIlllIlIIIIIIl = _IIlIIIlIIlIllIlII


def IIIlIlIIIlIlIIIII(IIlllllllIIlllllI):
    IIIllIIIllIlIIIlI = '(?:(?<=\\s)|^)"(.*?)"(?=\\s|$)|(\\S+)'
    IllllllIIIIlIIIlI = IlIIllIllIlIllIlI.findall(IIIllIIIllIlIIIlI, IIlllllllIIlllllI)
    IllllllIIIIlIIIlI = [
        IllIlIIIlIllIlIlI[0] if IllIlIIIlIllIlIlI[0] else IllIlIIIlIllIlIlI[1]
        for IllIlIIIlIllIlIlI in IllllllIIIIlIIIlI
    ]
    return IllllllIIIIlIIIlI


IIllllIlIIIlIlIll = lambda IIIlIIIIllIllllIl: all(
    IlllIIIllIIllllIl is not _IIlIIllIIIlIlIIII
    for IlllIIIllIIllllIl in IIIlIIIIllIllllIl
)


def IlIlllIlIIIIlIIll(IIIIlIlIlIlIllIII):
    (
        IIIIlIIIlIlIllIIl,
        IlIlIlIlIIIIlIllI,
        IIIlllIllIlIIIlIl,
        IIIIIlIIIIllllllI,
        IlIlllllIllIllIll,
        IIIlIlIIIlIllllll,
        IIlllIllIllIlIIII,
        IIIIIlllIIIIIllll,
        IlIIIllIlllllIlll,
        IIlllIIIlIlIIIlll,
        IlIlIIIIlllIlIIll,
        IIIIIllllIlIlIIIl,
        IIlIllllIIIlIIIll,
        _IlIIIllIIIlIllIlI,
        IlllIIllIIllIllII,
        IIIIlllllIlIIlIll,
        IIlIIIIIlIlllllIl,
    ) = IIIlIlIIIlIlIIIII(IIIIlIlIlIlIllIII)[:17]
    IlIlllllIllIllIll, IIIIIlllIIIIIllll, IlIIIllIlllllIlll, IIlllIIIlIlIIIlll = map(
        int,
        [IlIlllllIllIllIll, IIIIIlllIIIIIllll, IlIIIllIlllllIlll, IIlllIIIlIlIIIlll],
    )
    IIIlIlIIIlIllllll, IlIlIIIIlllIlIIll, IIIIIllllIlIlIIIl, IIlIllllIIIlIIIll = map(
        float,
        [IIIlIlIIIlIllllll, IlIlIIIIlllIlIIll, IIIIIllllIlIlIIIl, IIlIllllIIIlIIIll],
    )
    if IIlIIIIIlIlllllIl.lower() == "false":
        IllIlIIlIIllIIlIl = _IIIIIIlllIlIIIlIl
        IIIllIIIIIIlIIllI = _IIIIIIlllIlIIIlIl
    else:
        IllIlIIlIIllIIlIl, IIIllIIIIIIlIIllI = map(
            float, IIIlIlIIIlIlIIIII(IIIIlIlIlIlIllIII)[17:19]
        )
    rvc_globals.DoFormant = IIlIIIIIlIlllllIl.lower() == "true"
    rvc_globals.Quefrency = IllIlIIlIIllIIlIl
    rvc_globals.Timbre = IIIllIIIIIIlIIllI
    IIIlIlllIIlIIIlll = "Infer-CLI:"
    IllllIIlllIlIllII = f"audio-others/{IIIlllIllIlIIIlIl}"
    print(f"{IIIlIlllIIlIIIlll} Starting the inference...")
    IllIIIlIllIlllIII = IIlIIIIlllllllIll(
        IIIIlIIIlIlIllIIl, IIlIllllIIIlIIIll, IIlIllllIIIlIIIll
    )
    print(IllIIIlIllIlllIII)
    print(f"{IIIlIlllIIlIIIlll} Performing inference...")
    IIlIlIIIllllIIlll = IlIlIIIIlIlIIlIIl(
        IlIlllllIllIllIll,
        IlIlIlIlIIIIlIllI,
        IlIlIlIlIIIIlIllI,
        IIIlIlIIIlIllllll,
        _IIlIIllIIIlIlIIII,
        IIlllIllIllIlIIII,
        IIIIIlIIIIllllllI,
        IIIIIlIIIIllllllI,
        IIIIIllllIlIlIIIl,
        IlIIIllIlllllIlll,
        IIlllIIIlIlIIIlll,
        IlIlIIIIlllIlIIll,
        IIlIllllIIIlIIIll,
        IIIIIlllIIIIIllll,
        f0_min=IlllIIllIIllIllII,
        note_min=_IIlIIllIIIlIlIIII,
        f0_max=IIIIlllllIlIIlIll,
        note_max=_IIlIIllIIIlIlIIII,
        fl_autotune=_IIllIlIlIIllIlIII,
    )
    if "Success." in IIlIlIIIllllIIlll[0]:
        print(
            f"{IIIlIlllIIlIIIlll} Inference succeeded. Writing to {IllllIIlllIlIllII}..."
        )
        wavfile.write(
            IllllIIlllIlIllII, IIlIlIIIllllIIlll[1][0], IIlIlIIIllllIIlll[1][1]
        )
        print(f"{IIIlIlllIIlIIIlll} Finished! Saved output to {IllllIIlllIlIllII}")
    else:
        print(
            f"{IIIlIlllIIlIIIlll} Inference failed. Here's the traceback: {IIlIlIIIllllIIlll[0]}"
        )


def IIlIIIIlIlIIlIllI(IlIIIlIlIIIllIlIl):
    print("Pre-process: Starting...")
    IIllllIlIIIlIlIll(
        IIllIllllIIIlIIlI(
            *IIIlIlIIIlIlIIIII(IlIIIlIlIIIllIlIl)[:3],
            int(IIIlIlIIIlIlIIIII(IlIIIlIlIIIllIlIl)[3]),
        )
    )
    print("Pre-process: Finished")


def IIllIIIlIlIIlIlll(IIlIIllllllIlIlII):
    (
        IIllIllIIIlIlIllI,
        IllllIlIIIIlIIIll,
        IllllIlIllIIlIlIl,
        IIIIlIIIIIIIlIlll,
        IIlllIIIIllIIIllI,
        IIllIlIIlIllIllIl,
        IllIIIlIllIlIIIlI,
    ) = IIIlIlIIIlIlIIIII(IIlIIllllllIlIlII)
    IllllIlIllIIlIlIl = int(IllllIlIllIIlIlIl)
    IIIIlIIIIIIIlIlll = bool(int(IIIIlIIIIIIIlIlll))
    IIllIlIIlIllIllIl = int(IIllIlIIlIllIllIl)
    print(
        f"Extract Feature Has Pitch: {IIIIlIIIIIIIlIlll}Extract Feature Version: {IllIIIlIllIlIIIlI}Feature Extraction: Starting..."
    )
    IllIllIIlllIIIIII = IlIllIllIllllIIIl(
        IllllIlIIIIlIIIll,
        IllllIlIllIIlIlIl,
        IIlllIIIIllIIIllI,
        IIIIlIIIIIIIlIlll,
        IIllIllIIIlIlIllI,
        IllIIIlIllIlIIIlI,
        IIllIlIIlIllIllIl,
    )
    IIllllIlIIIlIlIll(IllIllIIlllIIIIII)
    print("Feature Extraction: Finished")


def IlllllIllIIlIlIlI(IIllllllIIIIllllI):
    IIllllllIIIIllllI = IIIlIlIIIlIlIIIII(IIllllllIIIIllllI)
    IllllIIlIIllllIll = IIllllllIIIIllllI[0]
    IIlIIlllIllIlllII = IIllllllIIIIllllI[1]
    IIlIIlllIlIlIlllI = [
        bool(int(IlIlllllllIIIIllI)) for IlIlllllllIIIIllI in IIllllllIIIIllllI[2:11]
    ]
    IIlllIllIlIlIIlll = IIllllllIIIIllllI[11]
    IIIlllllIlIIlIIIl = (
        "/kaggle/input/ax-rmf/pretrained/"
        if IIlllIllIlIlIIlll == _IlIlllllIlllIlIIl
        else "/kaggle/input/ax-rmf/pretrained_v2/"
    )
    IIlIllIlIIIllllll = f"{IIIlllllIlIIlIIIl}f0G{IIlIIlllIllIlllII}.pth"
    IlIIllIllllIlllIl = f"{IIIlllllIlIIlIIIl}f0D{IIlIIlllIllIlllII}.pth"
    print("Train-CLI: Training...")
    IIllIIIllIIllIlII(
        IllllIIlIIllllIll,
        IIlIIlllIllIlllII,
        *IIlIIlllIlIlIlllI,
        IIlIllIlIIIllllll,
        IlIIllIllllIlllIl,
        IIlllIllIlIlIIlll,
    )


def IIIllllllIIllIIlI(IllIlIlllIIllllIl):
    IllIIIlIIlIlIllll = "Train Feature Index-CLI"
    print(f"{IllIIIlIIlIlIllll}: Training... Please wait")
    IIllllIlIIIlIlIll(IIIllllIlIlIlllll(*IIIlIlIIIlIlIIIII(IllIlIlllIIllllIl)))
    print(f"{IllIIIlIIlIlIllll}: Done!")


def IlIlIIIIlllIIIIII(IlIllllIIlIllllIl):
    IlllIIllIlllIIIlI = extract_small_model(*IIIlIlIIIlIlIIIII(IlIllllIIlIllllIl))
    print(
        "Extract Small Model: Success!"
        if IlllIIllIlllIIIlI == "Success."
        else f"{IlllIIllIlllIIIlI}\nExtract Small Model: Failed!"
    )


def IIIlllIlIlllIllll(IIIlIlIIIlIIIlIll, IlIIIIlIllIIIIIlI, IIllllIlIIlIlllIl):
    if IIIlIlIIIlIIIlIll:
        try:
            with open(IIIlIlIIIlIIIlIll, _IlllIlIIIIIlIIIIl) as IlIlIllIllIIIlIlI:
                IIIlIllIIIIIIllll = IlIlIllIllIIIlIlI.read().splitlines()
            IlIIIIlIllIIIIIlI, IIllllIlIIlIlllIl = (
                IIIlIllIIIIIIllll[0],
                IIIlIllIIIIIIllll[1],
            )
            IIlIlIlllIIIIlIIl(IlIIIIlIllIIIIIlI, IIllllIlIIlIlllIl)
        except IndexError:
            print("Error: File does not have enough lines to read 'qfer' and 'tmbr'")
        except FileNotFoundError:
            print("Error: File does not exist")
        except Exception as IlIIIlllllIIlIlIl:
            print("An unexpected error occurred", IlIIIlllllIIlIlIl)
    return {
        _IIllIIIIIIllllIll: IlIIIIlIllIIIIIlI,
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }, {_IIllIIIIIIllllIll: IIllllIlIIlIlllIl, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI}


def IIIllIlIIlIllIlIl():
    IlIIlIllllIIIIlll = {
        _IIlIIIlIIlIllIlII: "\n    go home            : Takes you back to home with a navigation list.\n    go infer           : Takes you to inference command execution.\n    go pre-process     : Takes you to training step.1) pre-process command execution.\n    go extract-feature : Takes you to training step.2) extract-feature command execution.\n    go train           : Takes you to training step.3) being or continue training command execution.\n    go train-feature   : Takes you to the train feature index command execution.\n    go extract-model   : Takes you to the extract small model command execution.",
        _IIIIlllllllllIIII: "\n    arg 1) model name with .pth in ./weights: mi-test.pth\n    arg 2) source audio path: myFolder\\MySource.wav\n    arg 3) output file name to be placed in './audio-others': MyTest.wav\n    arg 4) feature index file path: logs/mi-test/added_IVF3l42_Flat_nprobe_1.index\n    arg 5) speaker id: 0\n    arg 6) transposition: 0\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)\n    arg 8) crepe hop length: 160\n    arg 9) harvest median filter radius: 3 (0-7)\n    arg 10) post resample rate: 0\n    arg 11) mix volume envelope: 1\n    arg 12) feature index ratio: 0.78 (0-1)\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2",
        _IllIIIlllIIllIlll: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Number of CPU threads to use: 8 \n\nExample: mi-test mydataset 40k 24",
        _IlllIlllIIIllIlIl: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 3) Number of CPU threads to use: 8\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)\n    arg 6) Crepe hop length: 128\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n\nExample: mi-test 0 24 1 harvest 128 v2",
        _IllIlIIIllIlIIlll: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Sample rate: 40k (32k, 40k, 48k)\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 4) speaker id: 0\n    arg 5) Save epoch iteration: 50\n    arg 6) Total epochs: 10000\n    arg 7) Batch size: 8\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2",
        _IIIIIIlIlIIIIIIII: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test v2",
        _IlllIIIlllIlIlIII: '\n    arg 1) Model Path: logs/mi-test/G_168000.pth\n    arg 2) Model save name: MyModel\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) Model information: "My Model"\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2',
    }
    print(IlIIlIllllIIIIlll.get(IlIlIlllIlIIIIIIl, "Invalid page"))


def IIlllllIlIlIlIIII(IIIllIIllIIIIlIll):
    global IlIlIlllIlIIIIIIl
    IlIlIlllIlIIIIIIl = IIIllIIllIIIIlIll
    return 0


def IIIlIlIIlIIlIIIlI(IlIlIlllIlIIlIlll):
    IIlIIllIIIlIlllIl = {
        "go home": _IIlIIIlIIlIllIlII,
        "go infer": _IIIIlllllllllIIII,
        "go pre-process": _IllIIIlllIIllIlll,
        "go extract-feature": _IlllIlllIIIllIlIl,
        "go train": _IllIlIIIllIlIIlll,
        "go train-feature": _IIIIIIlIlIIIIIIII,
        "go extract-model": _IlllIIIlllIlIlIII,
    }
    IlIIllIIIllIIIIIl = {
        _IIIIlllllllllIIII: IlIlllIlIIIIlIIll,
        _IllIIIlllIIllIlll: IIlIIIIlIlIIlIllI,
        _IlllIlllIIIllIlIl: IIllIIIlIlIIlIlll,
        _IllIlIIIllIlIIlll: IlllllIllIIlIlIlI,
        _IIIIIIlIlIIIIIIII: IIIllllllIIllIIlI,
        _IlllIIIlllIlIlIII: IlIlIIIIlllIIIIII,
    }
    if IlIlIlllIlIIlIlll in IIlIIllIIIlIlllIl:
        return IIlllllIlIlIlIIII(IIlIIllIIIlIlllIl[IlIlIlllIlIIlIlll])
    if IlIlIlllIlIIlIlll[:3] == "go ":
        print(f"page '{IlIlIlllIlIIlIlll[3:]}' does not exist!")
        return 0
    if IlIlIlllIlIIIIIIl in IlIIllIIIllIIIIIl:
        IlIIllIIIllIIIIIl[IlIlIlllIlIIIIIIl](IlIlIlllIlIIlIlll)


def IIllllllIlllIIIll():
    while _IIIllIllIIlIIIIIl:
        print(f"\nYou are currently in '{IlIlIlllIlIIIIIIl}':")
        IIIllIlIIlIllIlIl()
        print(f"{IlIlIlllIlIIIIIIl}: ", end="")
        try:
            IIIlIlIIlIIlIIIlI(input())
        except Exception as IlIlllllIIllIlllI:
            print(f"An error occurred: {traceback.format_exc()}")


if IIlIlllllIlllIIlI.is_cli:
    print(
        "\n\nMangio-RVC-Fork v2 CLI App!\nWelcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n"
    )
    IIllllllIlllIIIll()
"\ndef get_presets():\n    data = None\n    with open('../inference-presets.json', 'r') as file:\n        data = json.load(file)\n    preset_names = []\n    for preset in data['presets']:\n        preset_names.append(preset['name'])\n    \n    return preset_names\n"


def IIIIIIIIlIIIllllI(IIlIIlllIIllllIIl):
    IlIIIlIllIlllllll = IIlIIlllIIllllIIl != _IlIIlIlIlIllllllI
    if rvc_globals.NotesIrHertz:
        return (
            {
                _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: IlIIIlIllIlllllll,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: IlIIIlIllIlllllll,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
        )
    else:
        return (
            {
                _IlIlIllIIIIlIlIll: IlIIIlIllIlllllll,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: IlIIIlIllIlllllll,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
            {
                _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
                _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
            },
        )


def IIlllllllIIlIIlll(IIIllIlllIIIlIlIl):
    IlIIIlIIIIlIlIIll = IlIIllIllIlIllIlI.sub("\\.pth|\\.onnx$", "", IIIllIlllIIIlIlIl)
    IIlIllIIllIllIllI = os.path.split(IlIIIlIIIIlIlIIll)[-1]
    if IlIIllIllIlIllIlI.match(".+_e\\d+_s\\d+$", IIlIllIIllIllIllI):
        IllIlllllIllIIIIl = IIlIllIIllIllIllI.rsplit("_", 2)[0]
    else:
        IllIlllllIllIIIIl = IIlIllIIllIllIllI
    IlIllIIllIlIIlIII = os.path.join(IllIlIIlIlIIlIlII, IllIlllllIllIIIIl)
    IIlIlIlllIIIlllll = [IlIllIIllIlIIlIII] if os.path.exists(IlIllIIllIlIIlIII) else []
    IIlIlIlllIIIlllll.append(IllIlIIlIlIIlIlII)
    IIIIlIIlIlIIIlIIl = []
    for IIIIlIIIllIlllIIl in IIlIlIlllIIIlllll:
        for IIlIIIlIIIlllllll in os.listdir(IIIIlIIIllIlllIIl):
            if (
                IIlIIIlIIIlllllll.endswith(_IlIIIIIIIIIIllIlI)
                and _IlIIIIIIIIlIIlllI not in IIlIIIlIIIlllllll
            ):
                IlllllIIIIlIIlIIl = any(
                    IIIIIlIllIIlIlIIl.lower() in IIlIIIlIIIlllllll.lower()
                    for IIIIIlIllIIlIlIIl in [IIlIllIIllIllIllI, IllIlllllIllIIIIl]
                )
                IIIIIIIlIlIIlIIII = IIIIlIIIllIlllIIl == IlIllIIllIlIIlIII
                if IlllllIIIIlIIlIIl or IIIIIIIlIlIIlIIII:
                    IlIIlIlllIlllIIII = os.path.join(
                        IIIIlIIIllIlllIIl, IIlIIIlIIIlllllll
                    )
                    if IlIIlIlllIlllIIII in IllllIIllIlllllII:
                        IIIIlIIlIlIIIlIIl.append(
                            (
                                IlIIlIlllIlllIIII,
                                os.path.getsize(IlIIlIlllIlllIIII),
                                _IIlIllllIlIlIllII not in IIlIIIlIIIlllllll,
                            )
                        )
    if IIIIlIIlIlIIIlIIl:
        IIIIlIIlIlIIIlIIl.sort(
            key=lambda IIllIIIIlIIlllIll: (-IIllIIIIlIIlllIll[2], -IIllIIIIlIIlllIll[1])
        )
        IlIllIlIIlIIIlIII = IIIIlIIlIlIIIlIIl[0][0]
        return IlIllIlIIlIIIlIII, IlIllIlIIlIIIlIII
    return "", ""


def IllllllIIIlIlIlII(IIIlIlIIIIllIIIIl):
    if IIIlIlIIIIllIIIIl:
        try:
            with open(_IllIIIIIIIlIIIllI, "w+") as IlIIIIlIIIllllIlI:
                IlIIIIlIIIllllIlI.write("True")
            os.kill(IlIIIllIlllllIlIl, SIGTERM)
        except Exception as IIIIlIIIlIllIIlll:
            print(f"Couldn't click due to {IIIIlIIIlIllIIlll}")
        return {
            _IlIlIllIIIIlIlIll: _IIIllIllIIlIIIIIl,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }, {
            _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
        }
    return {
        _IlIlIllIIIIlIlIll: _IIllIlIlIIllIlIII,
        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
    }, {_IlIlIllIIIIlIlIll: _IIIllIllIIlIIIIIl, _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI}


IlIllllIIIIlllIII = "weights/"


def IIIIlllIlllIIlIIl(IllIlllIlllIIllIl):
    IlIIllIlIIlIIIlIl = {
        "C": -9,
        "C#": -8,
        _IIllIIllIIlIIlIIl: -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        _IIlIllIlllIlllIIl: -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    IIlIIIIIlIIlllIII, IlllIlIIlIllIllll = IllIlllIlllIIllIl[:-1], int(
        IllIlllIlllIIllIl[-1]
    )
    IlllIllIllIIllllI = IlIIllIlIIlIIIlIl[IIlIIIIIlIIlllIII]
    IIIllIlIllIIllIll = 12 * (IlllIlIIlIllIllll - 4) + IlllIllIllIIllllI
    IlllIlllIlIllllll = 44e1 * (2.0 ** (_IIIIIIlllIlIIIlIl / 12)) ** IIIllIlIllIIllIll
    return IlllIlllIlIllllll


def IllIIIlIlllIlllII(IlIlIlIIIllIllllI):
    if IlIlIlIIIllIllllI is _IIlIIllIIIlIlIIII:
        0
    else:
        IlllllIIlIlIlIllI = IlIlIlIIIllIllllI
        IIIlIlllIIIIIllIl = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        )
        IllIlIllIIIlIllII = "./audios/" + IIIlIlllIIIIIllIl
        shutil.move(IlllllIIlIlIlIllI, IllIlIllIIIlIllII)
        return IIIlIlllIIIIIllIl


def IIIllIIIlIIIlIlII(IIIllllIIlllllllI):
    if IIIllllIIlllllllI is _IIlIIllIIIlIlIIII:
        0
    else:
        IIIlIlIllllIIllIl = IIIllllIIlllllllI.name
        IIIlIIlIlIIlIllIl = os.path.join(
            _IllllIIIIllIIlIII, os.path.basename(IIIlIlIllllIIllIl)
        )
        if os.path.exists(IIIlIIlIlIIlIllIl):
            os.remove(IIIlIIlIlIIlIllIl)
            print(_IlIllIIlIIlIIIlII)
        shutil.move(IIIlIlIllllIIllIl, IIIlIIlIlIIlIllIl)


def IIIllllIIIlIIlllI(IIllllIlIllIIIllI):
    IIIllIIIIlllIllll = IIllllIlIllIIIllI.name
    IlllIlIlIIIlllIll = os.path.join(
        _IllllIIIIllIIlIII, os.path.basename(IIIllIIIIlllIllll)
    )
    if os.path.exists(IlllIlIlIIIlllIll):
        os.remove(IlllIlIlIIIlllIll)
        print(_IlIllIIlIIlIIIlII)
    shutil.move(IIIllIIIIlllIllll, IlllIlIlIIIlllIll)
    return IlllIlIlIIIlllIll


from gtts import gTTS
import edge_tts, asyncio


def IIIllIIlIIllllIlI(
    IllIllIIIIllIIlll,
    IIIIIlIlIllIlIlll,
    IIllIllIllIllIlll,
    IIIIllllIIlIIlIIl,
    IIIIIIlIllllIlIll,
    IlllIlllIIllIlIII,
    IIIlllIIIlIIIIIll,
    IIlllllllIllIIIlI,
    IllIlllIllIlIIlII,
    IIIIlIIlIlIIlIIIl,
    IlIllIIIlIIIllllI,
    IllIIIllIIlIlllll,
    IlIIlIlIlIIlIIlll,
    IIIllIIIIlIIIIIIl,
    IIIlllIlIllIIlIIl,
):
    global IlIllIIIIIIlIlIll, IllIIllIlIIlIIlII, IIIIlIIIIIlIIIIlI, IIlIlIIIIlllIIlIl, IIlIIIlIIlIlllllI, IlIIIlIIlIllIIlII
    if IIIIIlIlIllIlIlll is _IIlIIllIIIlIlIIII:
        return _IlIlIIlIlIIIllIll, _IIlIIllIIIlIlIIII
    IIllIllIllIllIlll = int(IIllIllIllIllIlll)
    try:
        IIlllIlIIIllllIlI = load_audio(IIIIIlIlIllIlIlll, 16000)
        IllIlIlIIlIIIlIlI = IlIIIlIlllIlIlllI.abs(IIlllIlIIIllllIlI).max() / 0.95
        if IllIlIlIIlIIIlIlI > 1:
            IIlllIlIIIllllIlI /= IllIlIlIIlIIIlIlI
        IIIlllIIIIIIIlIll = [0, 0, 0]
        if not IIlIlIIIIlllIIlIl:
            IlIlIlIIlllllIIIl()
        IIIIIIllIIllIlIIl = IlIIIlIIlIllIIlII.get(_IlllIIIIIIIIlIlll, 1)
        IlllIlllIIllIlIII = (
            IlllIlllIIllIlIII.strip(_IIlIllllIlIlIllII)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIlllllIIIIIIll)
            .strip(_IlIlIlIlllllIIlIl)
            .strip(_IIlIllllIlIlIllII)
            .replace(_IlIIIIIIIIlIIlllI, "added")
            if IlllIlllIIllIlIII != ""
            else IIIlllIIIlIIIIIll
        )
        IlIlIIIIIllIllIll = IIIIlIIIIIlIIIIlI.pipeline(
            IIlIlIIIIlllIIlIl,
            IllIIllIlIIlIIlII,
            IllIllIIIIllIIlll,
            IIlllIlIIIllllIlI,
            IIIIIlIlIllIlIlll,
            IIIlllIIIIIIIlIll,
            IIllIllIllIllIlll,
            IIIIIIlIllllIlIll,
            IlllIlllIIllIlIII,
            IIlllllllIllIIIlI,
            IIIIIIllIIllIlIIl,
            IllIlllIllIlIIlII,
            IlIllIIIIIIlIlIll,
            IIIIlIIlIlIIlIIIl,
            IlIllIIIlIIIllllI,
            IIlIIIlIIlIlllllI,
            IllIIIllIIlIlllll,
            IlIIlIlIlIIlIIlll,
            IIIllIIIIlIIIIIIl,
            IIIlllIlIllIIlIIl,
            f0_file=IIIIllllIIlIIlIIl,
        )
        if IlIllIIIIIIlIlIll != IIIIlIIlIlIIlIIIl >= 16000:
            IlIllIIIIIIlIlIll = IIIIlIIlIlIIlIIIl
        IlIIIIIlllIllIlIl = (
            _IllIlIllIlIllllII % IlllIlllIIllIlIII
            if os.path.exists(IlllIlllIIllIlIII)
            else _IlIlIlIlIllIlIlIl
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            IlIIIIIlllIllIlIl,
            IIIlllIIIIIIIlIll[0],
            IIIlllIIIIIIIlIll[1],
            IIIlllIIIIIIIlIll[2],
        ), (IlIllIIIIIIlIlIll, IlIlIIIIIllIllIll)
    except:
        IIlIllIIllIllIlll = traceback.format_exc()
        print(IIlIllIIllIllIlll)
        return IIlIllIIllIllIlll, (_IIlIIllIIIlIlIIII, _IIlIIllIIIlIlIIII)


def IIIIIllIlIlIllIII(
    _IllIlllIlIIIlIlII,
    IIlIlIlIIIIIIlIIl,
    IlllIlIlllllIlIIl="",
    IIlIIIIlIlIIIllII=0,
    IllIllIIIIlIIllIl=_IlllllllIIIIlllIl,
    IIlIIIllllllllllI=float(0.66),
    IIlIIllIIlIIIllll=float(64),
    IllllIIlllIlIIllI=_IIllIlIlIIllIlIII,
    IIlIIIIlIIIllIIlI=_IIllIlIlIIllIlIII,
    IlIllIIlIIIlllIlI="",
    IIIIlIlIllllIlIll="",
):
    IIlIIIIlllllllIll(
        sid=IlllIlIlllllIlIIl, to_return_protectl=0.33, to_return_protect1=0.33
    )
    for _IllIlIlIllIlIIlII in _IllIlllIlIIIlIlII:
        IlllIlIIlllIllIIl = (
            "audio2/" + IIlIlIlIIIIIIlIIl[_IllIlIlIllIlIIlII]
            if _IllIlIlIllIlIIlII != _IIIlIllIlIIIIllll
            else IIlIlIlIIIIIIlIIl[0]
        )
        try:
            print(IIlIlIlIIIIIIlIIl[_IllIlIlIllIlIIlII], IlllIlIlllllIlIIl)
        except:
            pass
        IllIIlllIlIllIIIl, (IlIIIIIIlllIIIIlI, IlIlIIlIllIlIIIII) = IIIllIIlIIllllIlI(
            sid=0,
            input_audio_path=IlllIlIIlllIllIIl,
            f0_up_key=IIlIIIIlIlIIIllII,
            f0_file=_IIlIIllIIIlIlIIII,
            f0_method=IllIllIIIIlIIllIl,
            file_index=IlIllIIlIIIlllIlI,
            file_index2=IIIIlIlIllllIlIll,
            index_rate=IIlIIIllllllllllI,
            filter_radius=int(3),
            resample_sr=int(0),
            rms_mix_rate=float(0.25),
            protect=float(0.33),
            crepe_hop_length=IIlIIllIIlIIIllll,
            fl_autotune=IllllIIlllIlIIllI,
            rmvpe_onnx=IIlIIIIlIIIllIIlI,
        )
        sf.write(
            file=IlllIlIIlllIllIIl, samplerate=IlIIIIIIlllIIIIlI, data=IlIlIIlIllIlIIIII
        )


def IIllIllIIlIIIIlIl(IllIIlllIlIlIllll, IIIlIIIllIIlIIIIl):
    try:
        return IllIIlllIlIlIllll.to(IIIlIIIllIIlIIIIl)
    except Exception as IIlIllIlllllIIllI:
        print(IIlIllIlllllIIllI)
        return IllIIlllIlIlIllll


def __bark__(IIllllIIlIlllIlll, IllIllIlllIlIIIIl):
    IIIlllllllllllIIl = "tts"
    IIllIIllllIlIIlll = "suno/bark"
    os.makedirs(
        os.path.join(IIIIIlIlllllIIIlI, IIIlllllllllllIIl), exist_ok=_IIIllIllIIlIIIIIl
    )
    from transformers import AutoProcessor, BarkModel

    IlIIlIIIIlIllIlII = (
        "cuda:0" if IlIIIlIIllIlllIIl.cuda.is_available() else _IlIlIIlIlIlIIIlll
    )
    IIIlIIIlIIIlllIII = (
        IlIIIlIIllIlllIIl.float32
        if _IlIlIIlIlIlIIIlll in IlIIlIIIIlIllIlII
        else IlIIIlIIllIlllIIl.float16
    )
    IlllIIIllllllllll = AutoProcessor.from_pretrained(
        IIllIIllllIlIIlll,
        cache_dir=os.path.join(IIIIIlIlllllIIIlI, IIIlllllllllllIIl, IIllIIllllIlIIlll),
        torch_dtype=IIIlIIIlIIIlllIII,
    )
    IllIllIIIllllllII = BarkModel.from_pretrained(
        IIllIIllllIlIIlll,
        cache_dir=os.path.join(IIIIIlIlllllIIIlI, IIIlllllllllllIIl, IIllIIllllIlIIlll),
        torch_dtype=IIIlIIIlIIIlllIII,
    ).to(IlIIlIIIIlIllIlII)
    IlIlllIIlIIlllIII = IlllIIIllllllllll(
        text=[IIllllIIlIlllIlll], return_tensors="pt", voice_preset=IllIllIlllIlIIIIl
    )
    IlIllllIIlIIlIllI = {
        IIlIIIllIlllIIIII: IIllIllIIlIIIIlIl(IIllIIllllIlIllII, IlIIlIIIIlIllIlII)
        if hasattr(IIllIIllllIlIllII, "to")
        else IIllIIllllIlIllII
        for (IIlIIIllIlllIIIII, IIllIIllllIlIllII) in IlIlllIIlIIlllIII.items()
    }
    IIIlllIllIlIIlIIl = IllIllIIIllllllII.generate(
        **IlIllllIIlIIlIllI, do_sample=_IIIllIllIIlIIIIIl
    )
    IlIlIIlllIlllIIIl = IllIllIIIllllllII.generation_config.sample_rate
    IlIlIIlIllIlIlIIl = IIIlllIllIlIIlIIl.cpu().numpy().squeeze()
    return IlIlIIlIllIlIlIIl, IlIlIIlllIlllIIIl


def IIlllIllIllIIIIIl(
    IIllllIIIllIIllIl,
    IllIllIIllIIlIlII,
    IIIlIIIlIlIlIllll,
    IlIIlllIIlIlIIIIl,
    IIIIllIIIlIllIIlI,
    IIIIIIlllllllIIIl,
    IIllIIllIIIlIllII,
    IIIllIIllllIllIII,
    IIIIIIIllIIlIIllI,
    IIIlllIIIllIIIIll,
):
    IIlllIIIIlIlllllI = "converted_bark.wav"
    IIllIllIlIIIIlIll = "bark_out.wav"
    IIIIllIlIlIlIlIIl = "converted_tts.wav"
    if IllIllIIllIIlIlII == _IIlIIllIIIlIlIIII:
        return
    IIIIllIllIlIIllII = os.path.join(
        IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIIIllIlIlIlIlIIl
    )
    IlllllIlIIllIIlIl = (
        _IIIllIllIIlIIIIIl
        if IIIIIIlllllllIIIl == _IllIllllIIIIIIIIl
        else _IIllIlIlIIllIlIII
    )
    if "SET_LIMIT" == os.getenv("DEMO"):
        if len(IIllllIIIllIIllIl) > 60:
            IIllllIIIllIIllIl = IIllllIIIllIIllIl[:60]
            print("DEMO; limit to 60 characters")
    IlIIIllIIIIlIlllI = IllIllIIllIIlIlII[:2]
    if IIIlllIIIllIIIIll == _IllIllIIIIlIIllII:
        try:
            asyncio.run(
                edge_tts.Communicate(
                    IIllllIIIllIIllIl, "-".join(IllIllIIllIIlIlII.split("-")[:-1])
                ).save(IIIIllIllIlIIllII)
            )
        except:
            try:
                IIIlIIIllIlllllll = gTTS(IIllllIIIllIIllIl, lang=IlIIIllIIIIlIlllI)
                IIIlIIIllIlllllll.save(IIIIllIllIlIIllII)
                IIIlIIIllIlllllll.save
                print(
                    f"No audio was received. Please change the tts voice for {IllIllIIllIIlIlII}. USING gTTS."
                )
            except:
                IIIlIIIllIlllllll = gTTS("a", lang=IlIIIllIIIIlIlllI)
                IIIlIIIllIlllllll.save(IIIIllIllIlIIllII)
                print("Error: Audio will be replaced.")
        os.system("cp audio-outputs/converted_tts.wav audio-outputs/real_tts.wav")
        IIIIIllIlIlIllIII(
            [_IIIlIllIlIIIIllll],
            ["audio-outputs/converted_tts.wav"],
            model_voice_path=IIIlIIIlIlIlIllll,
            transpose=IIIIllIIIlIllIIlI,
            f0method=IIIIIIlllllllIIIl,
            index_rate_=IIllIIllIIIlIllII,
            crepe_hop_length_=IIIllIIllllIllIII,
            fl_autotune=IIIIIIIllIIlIIllI,
            rmvpe_onnx=IlllllIlIIllIIlIl,
            file_index="",
            file_index2=IlIIlllIIlIlIIIIl,
        )
        return os.path.join(
            IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIIIllIlIlIlIlIIl
        ), os.path.join(IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, "real_tts.wav")
    elif IIIlllIIIllIIIIll == _IllIIIIllIlIIllll:
        try:
            IIlIIIIlllllllIll(
                sid=IIIlIIIlIlIlIllll, to_return_protectl=0.33, to_return_protect1=0.33
            )
            IIIIIllIlIIIllIlI = IIllllIIIllIIllIl.replace(
                _IIlIlllllIIIIIIll, _IIlIllllIlIlIllII
            ).strip()
            IIIlIIIlIllIlIlII = sent_tokenize(IIIIIllIlIIIllIlI)
            print(IIIlIIIlIllIlIlII)
            IIIlIllIlIllIIllI = IlIIIlIlllIlIlllI.zeros(int(0.25 * SAMPLE_RATE))
            IllIIIlIIlllllllI = []
            IlllIIlIIIIIllIII = os.path.join(
                IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIllIllIlIIIIlIll
            )
            for IlllllIlllIIlIIlI in IIIlIIIlIllIlIlII:
                IllllIlIllllllIIl, _IlIlllIIllIlIllII = __bark__(
                    IlllllIlllIIlIIlI, IllIllIIllIIlIlII.split("-")[0]
                )
                IllIIIlIIlllllllI += [IllllIlIllllllIIl, IIIlIllIlIllIIllI.copy()]
            sf.write(
                file=IlllIIlIIIIIllIII,
                samplerate=SAMPLE_RATE,
                data=IlIIIlIlllIlIlllI.concatenate(IllIIIlIIlllllllI),
            )
            IllIllIlIIlIIlIll, (
                IIlIIlllllIIllIIl,
                IlIllIIIIlllIIIlI,
            ) = IIIllIIlIIllllIlI(
                sid=0,
                input_audio_path=os.path.join(
                    IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIllIllIlIIIIlIll
                ),
                f0_up_key=IIIIllIIIlIllIIlI,
                f0_file=_IIlIIllIIIlIlIIII,
                f0_method=IIIIIIlllllllIIIl,
                file_index="",
                file_index2=IlIIlllIIlIlIIIIl,
                index_rate=IIllIIllIIIlIllII,
                filter_radius=int(3),
                resample_sr=int(0),
                rms_mix_rate=float(0.25),
                protect=float(0.33),
                crepe_hop_length=IIIllIIllllIllIII,
                fl_autotune=IIIIIIIllIIlIIllI,
                rmvpe_onnx=IlllllIlIIllIIlIl,
            )
            wavfile.write(
                os.path.join(IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIlllIIIIlIlllllI),
                rate=IIlIIlllllIIllIIl,
                data=IlIllIIIIlllIIIlI,
            )
            return (
                os.path.join(IIIIIlIlllllIIIlI, _IlllIlIIllIIIllII, IIlllIIIIlIlllllI),
                IlllIIlIIIIIllIII,
            )
        except Exception as IlllIIllIIIlllIlI:
            print(f"{IlllIIllIIIlllIlI}")
            return _IIlIIllIIIlIlIIII, _IIlIIllIIIlIlIIII


def IIIIllllllIIIIIlI(IllIIlllIlIlllIll=IlIIlIlIIllIIIIII.themes.Soft()):
    IllIlIIIIIIIllIll = "Model information to be placed:"
    IIIlIllIlIIlllIll = "Model architecture version:"
    IIlIlIllllIIIllII = "Name:"
    IlIlIIlIIIlIIIlIl = "-0.5"
    IIlIlIllIllIlllll = "Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:"
    IIIIlIIIIlIIIIllI = "You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."
    IlIlllIllIIllIlll = (
        "Export audio (click on the three dots in the lower right corner to download)"
    )
    IIIlllllllIlllIll = "Default value is 1.0"
    IllIlllllllIIIlll = "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:"
    IllIIlIIllIIIllll = "Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:"
    IIIlIIlIIIIlIlIIl = "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:"
    IllIIlIllllIllIIl = "Feature search database file path:"
    IIIllIlllIIIllIlI = "Max pitch:"
    IIlIlIIlIllIIIlII = "Min pitch:"
    IlllIlIlIlIlIIIlI = "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
    IIIIllIIllIlIIIII = "Mangio-Crepe Hop Length (Only applies to mangio-crepe): Hop length refers to the time it takes for the speaker to jump to a dramatic pitch. Lower hop lengths take more time to infer but are more pitch accurate."
    IIIlllIlIlIllllII = "Enable autotune"
    IllIllIIlIllIlIll = "crepe-tiny"
    IIllIlllIIllllIIl = "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
    IIIllllIIllIIllIl = "Search feature ratio:"
    IllIIIIlIlIlIIlll = "Auto-detect index path and select from the dropdown:"
    IllIllllIllIIlIll = "filepath"
    IIllIIllIlIllIIIl = "Drag your audio here:"
    IlIlllIIllIIlIIII = "Refresh"
    IllIlIIlIlllIIlII = "Path to Model:"
    IIIIllllIIllIlIlI = "Model information to be placed"
    IIIIllllIIIlllllI = "Name for saving"
    IlIIIIlIlIlIIIllI = "Whether the model has pitch guidance."
    IllIlllllllIIlIlI = "Target sample rate:"
    IIlllIlIIIllIlllI = "multiple"
    IlllllllllIIlIIlI = "opt"
    IlllIIllllIlIlllI = "crepe"
    IIIllIlllIlIIIIlI = "dio"
    Illllllllllllllll = "harvest"
    IIllIlIlIIlIIllll = "Select the pitch extraction algorithm:"
    IlIIllIIlIIIlIIlI = "Advanced Settings"
    IllIIlllllIIllllI = "Convert"
    IllIIlIllIIlIlIII = "rmvpe+"
    IlIIIlIllIllIllll = "Path to model"
    IlIllIlIIlIlIlIlI = "mangio-crepe-tiny"
    IIllllIlIlIIIIIll = "mangio-crepe"
    IllllIIIlllIlIIII = "Output information:"
    IllIllIIIIlIllIll = "primary"
    IIIIIIIllIIlIllIl = IlIIIlIlIlllllIIl[0] if IlIIIlIlIlllllIIl else ""
    with IlIIlIlIIllIIIIII.Blocks(
        theme="JohnSmith9982/small_and_pretty", title="AX-RVC"
    ) as IIllIIlIllIlllIIl:
        IlIIlIlIIllIIIIII.HTML("<h1> 🍏 AX-RVC (Mangio-RVC-Fork) </h1>")
        with IlIIlIlIIllIIIIII.Tabs():
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Model Inference")):
                with IlIIlIlIIllIIIIII.Row():
                    IlIlIIlIlllIllIIl = IlIIlIlIIllIIIIII.Dropdown(
                        label=IlIllIllIIllIlIII("Inferencing voice:"),
                        choices=sorted(IlIIIlIlIlllllIIl),
                        value=IIIIIIIllIIlIllIl,
                    )
                    IIIlIllIIIllIIIll = IlIIlIlIIllIIIIII.Button(
                        IlIllIllIIllIlIII(IlIlllIIllIIlIIII), variant=IllIllIIIIlIllIll
                    )
                    IlIIlllIlIllllllI = IlIIlIlIIllIIIIII.Button(
                        IlIllIllIIllIlIII("Unload voice to save GPU memory"),
                        variant=IllIllIIIIlIllIll,
                    )
                    IlIIlllIlIllllllI.click(
                        fn=lambda: {
                            _IIllIIIIIIllllIll: "",
                            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                        },
                        inputs=[],
                        outputs=[IlIlIIlIlllIllIIl],
                    )
                with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Single")):
                    with IlIIlIlIIllIIIIII.Row():
                        IIIlllllIIIlIlIll = IlIIlIlIIllIIIIII.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=IlIllIllIIllIlIII("Select Speaker/Singer ID:"),
                            value=0,
                            visible=_IIllIlIlIIllIlIII,
                            interactive=_IIIllIllIIlIIIIIl,
                        )
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Row():
                            with IlIIlIlIIllIIIIII.Column():
                                IIlIIlIIlIllIIlll = IlIIlIlIIllIIIIII.File(
                                    label=IlIllIllIIllIlIII(IIllIIllIlIllIIIl)
                                )
                                IIlllllllIllllllI = IlIIlIlIIllIIIIII.Audio(
                                    source="microphone",
                                    label=IlIllIllIIllIlIII("Or record an audio:"),
                                    type=IllIllllIllIIlIll,
                                )
                                IIIlIIlIIllIIIlII = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(
                                        "Manual path to the audio file to be processed"
                                    ),
                                    value=os.path.join(
                                        IIIIIlIlllllIIIlI,
                                        _IllllIIIIllIIlIII,
                                        "someguy.mp3",
                                    ),
                                    visible=_IIllIlIlIIllIlIII,
                                )
                                IIIlllIIllIIlIIIl = IlIIlIlIIllIIIIII.Dropdown(
                                    label=IlIllIllIIllIlIII(
                                        "Auto detect audio path and select from the dropdown:"
                                    ),
                                    choices=sorted(IlllIIIllIllIlllI),
                                    value="",
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIlllIIllIIlIIIl.select(
                                    fn=lambda: "",
                                    inputs=[],
                                    outputs=[IIIlIIlIIllIIIlII],
                                )
                                IIIlIIlIIllIIIlII.input(
                                    fn=lambda: "",
                                    inputs=[],
                                    outputs=[IIIlllIIllIIlIIIl],
                                )
                                IIlIIlIIlIllIIlll.upload(
                                    fn=IIIllllIIIlIIlllI,
                                    inputs=[IIlIIlIIlIllIIlll],
                                    outputs=[IIIlIIlIIllIIIlII],
                                )
                                IIlIIlIIlIllIIlll.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IIIlllIIllIIlIIIl],
                                )
                                IIlllllllIllllllI.change(
                                    fn=IllIIIlIlllIlllII,
                                    inputs=[IIlllllllIllllllI],
                                    outputs=[IIIlIIlIIllIIIlII],
                                )
                                IIlllllllIllllllI.change(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IIIlllIIllIIlIIIl],
                                )
                            IlllIlIIlllIIIlIl, _IlIIlllIllIlIllIl = IIlllllllIIlIIlll(
                                IlIlIIlIlllIllIIl.value
                            )
                            with IlIIlIlIIllIIIIII.Column():
                                IlIllIIlIlIIlIlll = IlIIlIlIIllIIIIII.Dropdown(
                                    label=IlIllIllIIllIlIII(IllIIIIlIlIlIIlll),
                                    choices=IIllIIIllllIIllll(),
                                    value=IlllIlIIlllIIIlIl,
                                    interactive=_IIIllIllIIlIIIIIl,
                                    allow_custom_value=_IIIllIllIIlIIIIIl,
                                )
                                IIlIlIIIlllIlIlIl = IlIIlIlIIllIIIIII.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=IlIllIllIIllIlIII(IIIllllIIllIIllIl),
                                    value=0.75,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIlIllIIIllIIIll.click(
                                    fn=IIllIlllIIIIIIIll,
                                    inputs=[],
                                    outputs=[
                                        IlIlIIlIlllIllIIl,
                                        IlIllIIlIlIIlIlll,
                                        IIIlllIIllIIlIIIl,
                                    ],
                                )
                                with IlIIlIlIIllIIIIII.Column():
                                    IllIlIIlIlIIllllI = IlIIlIlIIllIIIIII.Number(
                                        label=IlIllIllIIllIlIII(IIllIlllIIllllIIl),
                                        value=0,
                                    )
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII(IlIIllIIlIIIlIIlI),
                            open=_IIllIlIlIIllIlIII,
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                with IlIIlIlIIllIIIIII.Column():
                                    IllIIIIIlIIlIllII = IlIIlIlIIllIIIIII.Radio(
                                        label=IlIllIllIIllIlIII(IIllIlIlIIlIIllll),
                                        choices=[
                                            _IlllllllIIIIlllIl,
                                            Illllllllllllllll,
                                            IIIllIlllIlIIIIlI,
                                            IlllIIllllIlIlllI,
                                            IllIllIIlIllIlIll,
                                            IIllllIlIlIIIIIll,
                                            IlIllIlIIlIlIlIlI,
                                            _IlIIlIlIlIllllllI,
                                            _IllIllllIIIIIIIIl,
                                            IllIIlIllIIlIlIII,
                                        ],
                                        value=IllIIlIllIIlIlIII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IllIIlllllIllIIll = IlIIlIlIIllIIIIII.Checkbox(
                                        label=IIIlllIlIlIllllII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIIlIIIIIIIIIIII = IlIIlIlIIllIIIIII.Checkbox(
                                        value=bool(IIlIlIIIIlIIllIII),
                                        label=IlIllIllIIllIlIII(
                                            "Formant shift inference audio"
                                        ),
                                        info=IlIllIllIIllIlIII(
                                            "Used for male to female and vice-versa conversions"
                                        ),
                                        interactive=_IIIllIllIIlIIIIIl,
                                        visible=_IIIllIllIIlIIIIIl,
                                    )
                                    IIllIIIlllIllllIl = IlIIlIlIIllIIIIII.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label=IlIllIllIIllIlIII(IIIIllIIllIlIIIII),
                                        value=120,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        visible=_IIllIlIlIIllIlIII,
                                    )
                                    IIIIlllIlllIlllIl = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=IlIllIllIIllIlIII(IlllIlIlIlIlIIIlI),
                                        value=3,
                                        step=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIllIllllIIllIIl = IlIIlIlIIllIIIIII.Slider(
                                        label=IlIllIllIIllIlIII(IIlIlIIlIllIIIlII),
                                        info=IlIllIllIIllIlIII(
                                            "Specify minimal pitch for inference [HZ]"
                                        ),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=50,
                                        maximum=16000,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        visible=not rvc_globals.NotesIrHertz
                                        and IllIIIIIlIIlIllII.value
                                        != _IlIIlIlIlIllllllI,
                                    )
                                    IlIlIIllllIlIIIII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IIlIlIIlIllIIIlII),
                                        info=IlIllIllIIllIlIII(
                                            "Specify minimal pitch for inference [NOTE][OCTAVE]"
                                        ),
                                        placeholder="C5",
                                        visible=rvc_globals.NotesIrHertz
                                        and IllIIIIIlIIlIllII.value
                                        != _IlIIlIlIlIllllllI,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIllllllIllIIIllI = IlIIlIlIIllIIIIII.Slider(
                                        label=IlIllIllIIllIlIII(IIIllIlllIIIllIlI),
                                        info=IlIllIllIIllIlIII(
                                            "Specify max pitch for inference [HZ]"
                                        ),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=1100,
                                        maximum=16000,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        visible=not rvc_globals.NotesIrHertz
                                        and IllIIIIIlIIlIllII.value
                                        != _IlIIlIlIlIllllllI,
                                    )
                                    IIIlllIlllIlIllll = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IIIllIlllIIIllIlI),
                                        info=IlIllIllIIllIlIII(
                                            "Specify max pitch for inference [NOTE][OCTAVE]"
                                        ),
                                        placeholder="C6",
                                        visible=rvc_globals.NotesIrHertz
                                        and IllIIIIIlIIlIllII.value
                                        != _IlIIlIlIlIllllllI,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIlIlllIllllIIlIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIIlIllllIllIIl),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                IllIIIIIlIIlIllII.change(
                                    fn=lambda IIllllIllIllIIIIl: {
                                        _IlIlIllIIIIlIlIll: IIllllIllIllIIIIl
                                        in [IIllllIlIlIIIIIll, IlIllIlIIlIlIlIlI],
                                        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                                    },
                                    inputs=[IllIIIIIlIIlIllII],
                                    outputs=[IIllIIIlllIllllIl],
                                )
                                IllIIIIIlIIlIllII.change(
                                    fn=IIIIIIIIlIIIllllI,
                                    inputs=[IllIIIIIlIIlIllII],
                                    outputs=[
                                        IlIllIllllIIllIIl,
                                        IlIlIIllllIlIIIII,
                                        IIllllllIllIIIllI,
                                        IIIlllIlllIlIllll,
                                    ],
                                )
                                with IlIIlIlIIllIIIIII.Column():
                                    IlIllIlllllIllIll = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=IlIllIllIIllIlIII(IIIlIIlIIIIlIlIIl),
                                        value=0,
                                        step=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIIIlIlIIIlllIlI = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IlIllIllIIllIlIII(IllIIlIIllIIIllll),
                                        value=0.25,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIlIIIllIIIIIllII = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=IlIllIllIIllIlIII(IllIlllllllIIIlll),
                                        value=0.33,
                                        step=0.01,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IllIIIlIIllIllIlI = IlIIlIlIIllIIIIII.File(
                                        label=IlIllIllIIllIlIII(
                                            "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:"
                                        )
                                    )
                                    IIIIIllIIIIlIllll = IlIIlIlIIllIIIIII.Dropdown(
                                        value="",
                                        choices=IIllIlllIIlIIIllI(),
                                        label=IlIllIllIIllIlIII(
                                            "Browse presets for formanting"
                                        ),
                                        info=IlIllIllIIllIlIII(
                                            "Presets are located in formantshiftcfg/ folder"
                                        ),
                                        visible=bool(IIlIlIIIIlIIllIII),
                                    )
                                    IIlllIlIlllIlIlIl = IlIIlIlIIllIIIIII.Button(
                                        value="🔄",
                                        visible=bool(IIlIlIIIIlIIllIII),
                                        variant=IllIllIIIIlIllIll,
                                    )
                                    IllllIlllIIllIlIl = IlIIlIlIIllIIIIII.Slider(
                                        value=IlllIllIIlllIIlIl,
                                        info=IlIllIllIIllIlIII(IIIlllllllIlllIll),
                                        label=IlIllIllIIllIlIII(
                                            "Quefrency for formant shifting"
                                        ),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(IIlIlIIIIlIIllIII),
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIIIlIlllIlIIIIII = IlIIlIlIIllIIIIII.Slider(
                                        value=IlllIIIIlIlIIIIII,
                                        info=IlIllIllIIllIlIII(IIIlllllllIlllIll),
                                        label=IlIllIllIIllIlIII(
                                            "Timbre for formant shifting"
                                        ),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(IIlIlIIIIlIIllIII),
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIllIIlllIllIIIl = IlIIlIlIIllIIIIII.Button(
                                        IlIllIllIIllIlIII("Apply"),
                                        variant=IllIllIIIIlIllIll,
                                        visible=bool(IIlIlIIIIlIIllIII),
                                    )
                                IIIIIllIIIIlIllll.change(
                                    fn=IIIlllIlIlllIllll,
                                    inputs=[
                                        IIIIIllIIIIlIllll,
                                        IllllIlllIIllIlIl,
                                        IIIIlIlllIlIIIIII,
                                    ],
                                    outputs=[IllllIlllIIllIlIl, IIIIlIlllIlIIIIII],
                                )
                                IlIIlIIIIIIIIIIII.change(
                                    fn=IIIllIlIIlIIIlIIl,
                                    inputs=[
                                        IlIIlIIIIIIIIIIII,
                                        IllllIlllIIllIlIl,
                                        IIIIlIlllIlIIIIII,
                                    ],
                                    outputs=[
                                        IlIIlIIIIIIIIIIII,
                                        IllllIlllIIllIlIl,
                                        IIIIlIlllIlIIIIII,
                                        IlIllIIlllIllIIIl,
                                        IIIIIllIIIIlIllll,
                                        IIlllIlIlllIlIlIl,
                                    ],
                                )
                                IlIllIIlllIllIIIl.click(
                                    fn=IIlIlIlllIIIIlIIl,
                                    inputs=[IllllIlllIIllIlIl, IIIIlIlllIlIIIIII],
                                    outputs=[IllllIlllIIllIlIl, IIIIlIlllIlIIIIII],
                                )
                                IIlllIlIlllIlIlIl.click(
                                    fn=IllIIIIIllIIlIIIl,
                                    inputs=[
                                        IIIIIllIIIIlIllll,
                                        IllllIlllIIllIlIl,
                                        IIIIlIlllIlIIIIII,
                                    ],
                                    outputs=[
                                        IIIIIllIIIIlIllll,
                                        IllllIlllIIllIlIl,
                                        IIIIlIlllIlIIIIII,
                                    ],
                                )
                    with IlIIlIlIIllIIIIII.Row():
                        IlllllIIllIIlIlII = IlIIlIlIIllIIIIII.Textbox(
                            label=IlIllIllIIllIlIII(IllllIIIlllIlIIII)
                        )
                        IIlIlllIIIllIllIl = IlIIlIlIIllIIIIII.Audio(
                            label=IlIllIllIIllIlIII(IlIlllIllIIllIlll)
                        )
                    IIlIIIlIlIllIIlll = IlIIlIlIIllIIIIII.Button(
                        IlIllIllIIllIlIII(IllIIlllllIIllllI), variant=IllIllIIIIlIllIll
                    ).style(full_width=_IIIllIllIIlIIIIIl)
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Row():
                            IIlIIIlIlIllIIlll.click(
                                IlIlIIIIlIlIIlIIl,
                                [
                                    IIIlllllIIIlIlIll,
                                    IIIlIIlIIllIIIlII,
                                    IIIlllIIllIIlIIIl,
                                    IllIlIIlIlIIllllI,
                                    IllIIIlIIllIllIlI,
                                    IllIIIIIlIIlIllII,
                                    IIlIlllIllllIIlIl,
                                    IlIllIIlIlIIlIlll,
                                    IIlIlIIIlllIlIlIl,
                                    IIIIlllIlllIlllIl,
                                    IlIllIlllllIllIll,
                                    IlIIIlIlIIIlllIlI,
                                    IIlIIIllIIIIIllII,
                                    IIllIIIlllIllllIl,
                                    IlIllIllllIIllIIl,
                                    IlIlIIllllIlIIIII,
                                    IIllllllIllIIIllI,
                                    IIIlllIlllIlIllll,
                                    IllIIlllllIllIIll,
                                ],
                                [IlllllIIllIIlIlII, IIlIlllIIIllIllIl],
                            )
                with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Batch")):
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Row():
                            with IlIIlIlIIllIIIIII.Column():
                                IIIIllIlIlIIIllII = IlIIlIlIIllIIIIII.Number(
                                    label=IlIllIllIIllIlIII(IIllIlllIIllllIIl), value=0
                                )
                                IllIIllIllIllIllI = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII("Specify output folder:"),
                                    value=IlllllllllIIlIIlI,
                                )
                                IlllllIIIIlIIlIlI = IlIIlIlIIllIIIIII.Dropdown(
                                    label=IlIllIllIIllIlIII(IllIIIIlIlIlIIlll),
                                    choices=IIllIIIllllIIllll(),
                                    value=IlllIlIIlllIIIlIl,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIlIIlIlllIllIIl.select(
                                    fn=IIlllllllIIlIIlll,
                                    inputs=[IlIlIIlIlllIllIIl],
                                    outputs=[IlIllIIlIlIIlIlll, IlllllIIIIlIIlIlI],
                                )
                                IIIlIllIIIllIIIll.click(
                                    fn=lambda: IIllIlllIIIIIIIll()[1],
                                    inputs=[],
                                    outputs=IlllllIIIIlIIlIlI,
                                )
                                IIIIIIllIlllIIIIl = IlIIlIlIIllIIIIII.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=IlIllIllIIllIlIII(IIIllllIIllIIllIl),
                                    value=0.75,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                            with IlIIlIlIIllIIIIII.Column():
                                IllIIIIIlIIlIllll = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(
                                        "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"
                                    ),
                                    value=os.path.join(
                                        IIIIIlIlllllIIIlI, _IllllIIIIllIIlIII
                                    ),
                                    lines=2,
                                )
                                IIIllIIIIlIIllIII = IlIIlIlIIllIIIIII.File(
                                    file_count=IIlllIlIIIllIlllI,
                                    label=IlIllIllIIllIlIII(IIIIlIIIIlIIIIllI),
                                )
                    with IlIIlIlIIllIIIIII.Row():
                        with IlIIlIlIIllIIIIII.Column():
                            IlIlllIlIIIIIIIll = IlIIlIlIIllIIIIII.Checkbox(
                                value=_IIllIlIlIIllIlIII,
                                label=IlIllIllIIllIlIII(IlIIllIIlIIIlIIlI),
                                interactive=_IIIllIllIIlIIIIIl,
                            )
                            with IlIIlIlIIllIIIIII.Row(
                                visible=_IIllIlIlIIllIlIII
                            ) as IIIIlIlllIllIIIIl:
                                with IlIIlIlIIllIIIIII.Row(
                                    label=IlIllIllIIllIlIII(IlIIllIIlIIIlIIlI),
                                    open=_IIllIlIlIIllIlIII,
                                ):
                                    with IlIIlIlIIllIIIIII.Column():
                                        IllllIIIIllIIlIlI = IlIIlIlIIllIIIIII.Textbox(
                                            label=IlIllIllIIllIlIII(IllIIlIllllIllIIl),
                                            value="",
                                            interactive=_IIIllIllIIlIIIIIl,
                                        )
                                        IIIIIIlIlllllIllI = IlIIlIlIIllIIIIII.Radio(
                                            label=IlIllIllIIllIlIII(IIllIlIlIIlIIllll),
                                            choices=[
                                                _IlllllllIIIIlllIl,
                                                Illllllllllllllll,
                                                IIIllIlllIlIIIIlI,
                                                IlllIIllllIlIlllI,
                                                IllIllIIlIllIlIll,
                                                IIllllIlIlIIIIIll,
                                                IlIllIlIIlIlIlIlI,
                                                _IlIIlIlIlIllllllI,
                                                _IllIllllIIIIIIIIl,
                                                IllIIlIllIIlIlIII,
                                            ],
                                            value=IllIIlIllIIlIlIII,
                                            interactive=_IIIllIllIIlIIIIIl,
                                        )
                                        IllIIlllllIllIIll = IlIIlIlIIllIIIIII.Checkbox(
                                            label=IIIlllIlIlIllllII,
                                            interactive=_IIIllIllIIlIIIIIl,
                                        )
                                        IIlIIlIlIIlIIIIlI = IlIIlIlIIllIIIIII.Radio(
                                            label=IlIllIllIIllIlIII(
                                                "Export file format"
                                            ),
                                            choices=[
                                                _IIlIllIllIlIIIIIl,
                                                _IIllIIIIIIIIIIIlI,
                                                _IllIlllIlIIIIllII,
                                                _IIIlIllllIIllIlII,
                                            ],
                                            value=_IIlIllIllIlIIIIIl,
                                            interactive=_IIIllIllIIlIIIIIl,
                                        )
                                with IlIIlIlIIllIIIIII.Column():
                                    IIlIllIlIIllllllI = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=IlIllIllIIllIlIII(IIIlIIlIIIIlIlIIl),
                                        value=0,
                                        step=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlllIIIlIIlIIllII = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IlIllIllIIllIlIII(IllIIlIIllIIIllll),
                                        value=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlllllIlllIlIIlIl = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=IlIllIllIIllIlIII(IllIlllllllIIIlll),
                                        value=0.33,
                                        step=0.01,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIIlIllIlIIlllllI = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=IlIllIllIIllIlIII(IlllIlIlIlIlIIIlI),
                                        value=3,
                                        step=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                            IlIllIlIIlIlIllll = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(IllllIIIlllIlIIII)
                            )
                            IllIIlIIIIllllIlI = IlIIlIlIIllIIIIII.Button(
                                IlIllIllIIllIlIII(IllIIlllllIIllllI),
                                variant=IllIllIIIIlIllIll,
                            )
                            IllIIlIIIIllllIlI.click(
                                IllIllIIIllllIlIl,
                                [
                                    IIIlllllIIIlIlIll,
                                    IllIIIIIlIIlIllll,
                                    IllIIllIllIllIllI,
                                    IIIllIIIIlIIllIII,
                                    IIIIllIlIlIIIllII,
                                    IIIIIIlIlllllIllI,
                                    IllllIIIIllIIlIlI,
                                    IlllllIIIIlIIlIlI,
                                    IIIIIIllIlllIIIIl,
                                    IIIlIllIlIIlllllI,
                                    IIlIllIlIIllllllI,
                                    IlllIIIlIIlIIllII,
                                    IlllllIlllIlIIlIl,
                                    IIlIIlIlIIlIIIIlI,
                                    IIllIIIlllIllllIl,
                                    IlIllIllllIIllIIl
                                    if not rvc_globals.NotesIrHertz
                                    else IlIlIIllllIlIIIII,
                                    IIllllllIllIIIllI
                                    if not rvc_globals.NotesIrHertz
                                    else IIIlllIlllIlIllll,
                                    IllIIlllllIllIIll,
                                ],
                                [IlIllIlIIlIlIllll],
                            )
                    IlIlIIlIlllIllIIl.change(
                        fn=IIlIIIIlllllllIll,
                        inputs=[
                            IlIlIIlIlllIllIIl,
                            IIlIIIllIIIIIllII,
                            IlllllIlllIlIIlIl,
                        ],
                        outputs=[
                            IIIlllllIIIlIlIll,
                            IIlIIIllIIIIIllII,
                            IlllllIlllIlIIlIl,
                        ],
                    )
                    (
                        IIIlllllIIIlIlIll,
                        IIlIIIllIIIIIllII,
                        IlllllIlllIlIIlIl,
                    ) = IIlIIIIlllllllIll(
                        IlIlIIlIlllIllIIl.value, IIlIIIllIIIIIllII, IlllllIlllIlIIlIl
                    )

                    def IlIIIIlIlIIIIlllI(IIIlIlIIIllIlllll):
                        return {
                            _IlIlIllIIIIlIlIll: IIIlIlIIIllIlllll,
                            _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                        }

                    IlIlllIlIIIIIIIll.change(
                        fn=IlIIIIlIlIIIIlllI,
                        inputs=[IlIlllIlIIIIIIIll],
                        outputs=[IIIIlIlllIllIIIIl],
                    )
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Train")):
                with IlIIlIlIIllIIIIII.Accordion(
                    label=IlIllIllIIllIlIII("Step 1: Processing data")
                ):
                    with IlIIlIlIIllIIIIII.Row():
                        IIllllIlIIIllllII = IlIIlIlIIllIIIIII.Textbox(
                            label=IlIllIllIIllIlIII("Enter the model name:"),
                            value=IlIllIllIIllIlIII("Model_Name"),
                        )
                        IllllllIllIlIllIl = IlIIlIlIIllIIIIII.Radio(
                            label=IlIllIllIIllIlIII(IllIlllllllIIlIlI),
                            choices=[
                                _IIlllllllIIIIlIII,
                                _IlIIIlIIIIlllllII,
                                _IllIIlIlIlIlllllI,
                            ],
                            value=_IIlllllllIIIIlIII,
                            interactive=_IIIllIllIIlIIIIIl,
                        )
                        IIIllIlIlIIIlllll = IlIIlIlIIllIIIIII.Checkbox(
                            label=IlIllIllIIllIlIII(IlIIIIlIlIlIIIllI),
                            value=_IIIllIllIIlIIIIIl,
                            interactive=_IIIllIllIIlIIIIIl,
                        )
                        IIIlIIIllIIIlIIII = IlIIlIlIIllIIIIII.Radio(
                            label=IlIllIllIIllIlIII("Version:"),
                            choices=[_IlIlllllIlllIlIIl, _IlIlIIllllllIlIII],
                            value=_IlIlIIllllllIlIII,
                            interactive=_IIIllIllIIlIIIIIl,
                            visible=_IIIllIllIIlIIIIIl,
                        )
                        IllIlIIIlIllllllI = IlIIlIlIIllIIIIII.Slider(
                            minimum=0,
                            maximum=IIlIlllllIlllIIlI.n_cpu,
                            step=1,
                            label=IlIllIllIIllIlIII("Number of CPU processes:"),
                            value=int(
                                IlIIIlIlllIlIlllI.ceil(IIlIlllllIlllIIlI.n_cpu / 1.5)
                            ),
                            interactive=_IIIllIllIIlIIIIIl,
                        )
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Accordion(
                        label=IlIllIllIIllIlIII("Step 2: Skipping pitch extraction")
                    ):
                        with IlIIlIlIIllIIIIII.Row():
                            with IlIIlIlIIllIIIIII.Column():
                                IllIllllIllIIIIlI = IlIIlIlIIllIIIIII.Dropdown(
                                    choices=sorted(IlIlIIlIllIllIIll),
                                    label=IlIllIllIIllIlIII("Select your dataset:"),
                                    value=IlIllIlllIIIllIII(),
                                )
                                IllIlIIIIIllllIlI = IlIIlIlIIllIIIIII.Button(
                                    IlIllIllIIllIlIII("Update list"),
                                    variant=IllIllIIIIlIllIll,
                                )
                            IlIIIIIlIIIIIIIII = IlIIlIlIIllIIIIII.Slider(
                                minimum=0,
                                maximum=4,
                                step=1,
                                label=IlIllIllIIllIlIII("Specify the model ID:"),
                                value=0,
                                interactive=_IIIllIllIIlIIIIIl,
                            )
                            IllIlIIIIIllllIlI.click(
                                easy_infer.update_dataset_list,
                                [IlIIIIIlIIIIIIIII],
                                IllIllllIllIIIIlI,
                            )
                            IllIIlIIIIllllIlI = IlIIlIlIIllIIIIII.Button(
                                IlIllIllIIllIlIII("Process data"),
                                variant=IllIllIIIIlIllIll,
                            )
                            IIlIIIIllIIIllIII = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(IllllIIIlllIlIIII), value=""
                            )
                            IllIIlIIIIllllIlI.click(
                                IIllIllllIIIlIIlI,
                                [
                                    IllIllllIllIIIIlI,
                                    IIllllIlIIIllllII,
                                    IllllllIllIlIllIl,
                                    IllIlIIIlIllllllI,
                                ],
                                [IIlIIIIllIIIllIII],
                            )
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Accordion(
                        label=IlIllIllIIllIlIII("Step 3: Extracting features")
                    ):
                        with IlIIlIlIIllIIIIII.Row():
                            with IlIIlIlIIllIIIIII.Column():
                                IIIllIIlIlllIIIII = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(IIlIlIllIllIlllll),
                                    value=IIIIIIlllIIlllIlI,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII("GPU Information:"),
                                    value=IllIIlIIIlllllllI,
                                )
                            with IlIIlIlIIllIIIIII.Column():
                                IIlIlIIlIlllllIIl = IlIIlIlIIllIIIIII.Radio(
                                    label=IlIllIllIIllIlIII(IIllIlIlIIlIIllll),
                                    choices=[
                                        _IlllllllIIIIlllIl,
                                        Illllllllllllllll,
                                        IIIllIlllIlIIIIlI,
                                        IlllIIllllIlIlllI,
                                        IIllllIlIlIIIIIll,
                                        _IlIIlIlIlIllllllI,
                                    ],
                                    value=_IlIIlIlIlIllllllI,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IllllllIIlllIlIll = IlIIlIlIIllIIIIII.Slider(
                                    minimum=1,
                                    maximum=512,
                                    step=1,
                                    label=IlIllIllIIllIlIII(IIIIllIIllIlIIIII),
                                    value=64,
                                    interactive=_IIIllIllIIlIIIIIl,
                                    visible=_IIllIlIlIIllIlIII,
                                )
                                IIlIlIIlIlllllIIl.change(
                                    fn=lambda IIIlIlIIlIllIlIIl: {
                                        _IlIlIllIIIIlIlIll: IIIlIlIIlIllIlIIl
                                        in [IIllllIlIlIIIIIll, IlIllIlIIlIlIlIlI],
                                        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                                    },
                                    inputs=[IIlIlIIlIlllllIIl],
                                    outputs=[IllllllIIlllIlIll],
                                )
                            IIlIIIIIllIlIlllI = IlIIlIlIIllIIIIII.Button(
                                IlIllIllIIllIlIII("Feature extraction"),
                                variant=IllIllIIIIlIllIll,
                            )
                            IIIIlIlIlIllIIIII = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                value="",
                                max_lines=8,
                                interactive=_IIllIlIlIIllIlIII,
                            )
                            IIlIIIIIllIlIlllI.click(
                                IlIllIllIllllIIIl,
                                [
                                    IIIllIIlIlllIIIII,
                                    IllIlIIIlIllllllI,
                                    IIlIlIIlIlllllIIl,
                                    IIIllIlIlIIIlllll,
                                    IIllllIlIIIllllII,
                                    IIIlIIIllIIIlIIII,
                                    IllllllIIlllIlIll,
                                ],
                                [IIIIlIlIlIllIIIII],
                            )
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Row():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII("Step 4: Model training started")
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                IllIIIIlIIllIlllI = IlIIlIlIIllIIIIII.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    label=IlIllIllIIllIlIII("Save frequency:"),
                                    value=10,
                                    interactive=_IIIllIllIIlIIIIIl,
                                    visible=_IIIllIllIIlIIIIIl,
                                )
                                IlIlIIlllIllIIllI = IlIIlIlIIllIIIIII.Slider(
                                    minimum=1,
                                    maximum=10000,
                                    step=2,
                                    label=IlIllIllIIllIlIII("Training epochs:"),
                                    value=750,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIIIlllllIllIIII = IlIIlIlIIllIIIIII.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    label=IlIllIllIIllIlIII("Batch size per GPU:"),
                                    value=IlIIIlIIIIIIIlIlI,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                            with IlIIlIlIIllIIIIII.Row():
                                IIIlIlIIlIlIIIIlI = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII(
                                        "Whether to save only the latest .ckpt file to save hard drive space"
                                    ),
                                    value=_IIIllIllIIlIIIIIl,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIllIlIlIllIlIIII = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII(
                                        "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training"
                                    ),
                                    value=_IIllIlIlIIllIlIII,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIIIlllllIlIIlII = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII(
                                        "Save a small final model to the 'weights' folder at each save point"
                                    ),
                                    value=_IIIllIllIIlIIIIIl,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                            with IlIIlIlIIllIIIIII.Row():
                                IIIllIlIlIIlIllIl = IlIIlIlIIllIIIIII.Textbox(
                                    lines=4,
                                    label=IlIllIllIIllIlIII(
                                        "Load pre-trained base model G path:"
                                    ),
                                    value="/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth",
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlllllllllIllIIII = IlIIlIlIIllIIIIII.Textbox(
                                    lines=4,
                                    label=IlIllIllIIllIlIII(
                                        "Load pre-trained base model D path:"
                                    ),
                                    value="/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth",
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIllIlllIIllIllII = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(IIlIlIllIllIlllll),
                                    value=IIIIIIlllIIlllIlI,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IllllllIllIlIllIl.change(
                                    IIIIlIIIlIIllllIl,
                                    [
                                        IllllllIllIlIllIl,
                                        IIIllIlIlIIIlllll,
                                        IIIlIIIllIIIlIIII,
                                    ],
                                    [IIIllIlIlIIlIllIl, IlllllllllIllIIII],
                                )
                                IIIlIIIllIIIlIIII.change(
                                    IlIlIIIlllllIllIl,
                                    [
                                        IllllllIllIlIllIl,
                                        IIIllIlIlIIIlllll,
                                        IIIlIIIllIIIlIIII,
                                    ],
                                    [
                                        IIIllIlIlIIlIllIl,
                                        IlllllllllIllIIII,
                                        IllllllIllIlIllIl,
                                    ],
                                )
                                IIIllIlIlIIIlllll.change(
                                    fn=IlIIlIlllllIllIlI,
                                    inputs=[
                                        IIIllIlIlIIIlllll,
                                        IllllllIllIlIllIl,
                                        IIIlIIIllIIIlIIII,
                                    ],
                                    outputs=[
                                        IIlIlIIlIlllllIIl,
                                        IIIllIlIlIIlIllIl,
                                        IlllllllllIllIIII,
                                    ],
                                )
                                IIIllIlIlIIIlllll.change(
                                    fn=lambda IlllIlIllIlIlIIII: {
                                        _IlIlIllIIIIlIlIll: IlllIlIllIlIlIIII
                                        in [IIllllIlIlIIIIIll, IlIllIlIIlIlIlIlI],
                                        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                                    },
                                    inputs=[IIlIlIIlIlllllIIl],
                                    outputs=[IllllllIIlllIlIll],
                                )
                                IlIIIlIlIIIIIIllI = IlIIlIlIIllIIIIII.Button(
                                    IlIllIllIIllIlIII("Stop training"),
                                    variant=IllIllIIIIlIllIll,
                                    visible=_IIllIlIlIIllIlIII,
                                )
                                IlIIlIlIIIIIlIlII = IlIIlIlIIllIIIIII.Button(
                                    IlIllIllIIllIlIII("Train model"),
                                    variant=IllIllIIIIlIllIll,
                                    visible=_IIIllIllIIlIIIIIl,
                                )
                                IlIIlIlIIIIIlIlII.click(
                                    fn=IllllllIIIlIlIlII,
                                    inputs=[
                                        IlIIlIlIIllIIIIII.Number(
                                            value=0, visible=_IIllIlIlIIllIlIII
                                        )
                                    ],
                                    outputs=[IlIIlIlIIIIIlIlII, IlIIIlIlIIIIIIllI],
                                )
                                IlIIIlIlIIIIIIllI.click(
                                    fn=IllllllIIIlIlIlII,
                                    inputs=[
                                        IlIIlIlIIllIIIIII.Number(
                                            value=1, visible=_IIllIlIlIIllIlIII
                                        )
                                    ],
                                    outputs=[IlIIlIlIIIIIlIlII, IlIIIlIlIIIIIIllI],
                                )
                                with IlIIlIlIIllIIIIII.Column():
                                    IllllIlIllIllllII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                        value="",
                                        max_lines=4,
                                    )
                                    IIllIlIlllIlIIllI = IlIIlIlIIllIIIIII.Dropdown(
                                        label=IlIllIllIIllIlIII("Save type"),
                                        choices=[
                                            IlIllIllIIllIlIII("Save all"),
                                            IlIllIllIIllIlIII("Save D and G"),
                                            IlIllIllIIllIlIII("Save voice"),
                                        ],
                                        value=IlIllIllIIllIlIII("Choose the method"),
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlllllIIlIllIlIlI = IlIIlIlIIllIIIIII.Button(
                                        IlIllIllIIllIlIII("Train feature index"),
                                        variant=IllIllIIIIlIllIll,
                                    )
                                    IlIIlIIlIlllIIIII = IlIIlIlIIllIIIIII.Button(
                                        IlIllIllIIllIlIII("Save model"),
                                        variant=IllIllIIIIlIllIll,
                                    )
                                IlIIIlllllIlIIlII.change(
                                    fn=lambda IIllIIlIIllIlIllI: {
                                        _IlIlIllIIIIlIlIll: IIllIIlIIllIlIllI,
                                        _IIllIlIIlIlIIIIIl: _IIIIlllIIllIIlIlI,
                                    },
                                    inputs=[IlIIIlllllIlIIlII],
                                    outputs=[IllIIIIlIIllIlllI],
                                )
                            IlIIlIlIIIIIlIlII.click(
                                IIllIIIllIIllIlII,
                                [
                                    IIllllIlIIIllllII,
                                    IllllllIllIlIllIl,
                                    IIIllIlIlIIIlllll,
                                    IlIIIIIlIIIIIIIII,
                                    IllIIIIlIIllIlllI,
                                    IlIlIIlllIllIIllI,
                                    IlIIIlllllIllIIII,
                                    IIIlIlIIlIlIIIIlI,
                                    IIIllIlIlIIlIllIl,
                                    IlllllllllIllIIII,
                                    IIllIlllIIllIllII,
                                    IIllIlIlIllIlIIII,
                                    IlIIIlllllIlIIlII,
                                    IIIlIIIllIIIlIIII,
                                ],
                                [
                                    IllllIlIllIllllII,
                                    IlIIIlIlIIIIIIllI,
                                    IlIIlIlIIIIIlIlII,
                                ],
                            )
                            IlllllIIlIllIlIlI.click(
                                IIIllllIlIlIlllll,
                                [IIllllIlIIIllllII, IIIlIIIllIIIlIIII],
                                IllllIlIllIllllII,
                            )
                            IlIIlIIlIlllIIIII.click(
                                easy_infer.save_model,
                                [IIllllIlIIIllllII, IIllIlIlllIlIIllI],
                                IllllIlIllIllllII,
                            )
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Row():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII(
                                "Step 5: Export lowest points on a graph of the model"
                            )
                        ):
                            IllIIIllllllllIlI = IlIIlIlIIllIIIIII.Textbox(
                                visible=_IIllIlIlIIllIlIII
                            )
                            IIlllIllIlIlIlIII = IlIIlIlIIllIIIIII.Textbox(
                                visible=_IIllIlIlIIllIlIII
                            )
                            IllIllIllIIllIIll = IlIIlIlIIllIIIIII.Textbox(
                                visible=_IIllIlIlIIllIlIII, value=IlIllllIIIIlllIII
                            )
                            with IlIIlIlIIllIIIIII.Row():
                                IIIllllllIIlllllI = IlIIlIlIIllIIIIII.Slider(
                                    minimum=1,
                                    maximum=25,
                                    label=IlIllIllIIllIlIII(
                                        "How many lowest points to save:"
                                    ),
                                    value=3,
                                    step=1,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIIIIlIllllllIll = IlIIlIlIIllIIIIII.Button(
                                    value=IlIllIllIIllIlIII(
                                        "Export lowest points of a model"
                                    ),
                                    variant=IllIllIIIIlIllIll,
                                )
                                IIllllIllIIIllIlI = IlIIlIlIIllIIIIII.File(
                                    file_count=IIlllIlIIIllIlllI,
                                    label=IlIllIllIIllIlIII("Output models:"),
                                    interactive=_IIllIlIlIIllIlIII,
                                )
                            with IlIIlIlIIllIIIIII.Row():
                                IlIIlIlllIlIIIIlI = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                    value="",
                                    max_lines=10,
                                )
                                IllIllIIIlllIIlIl = IlIIlIlIIllIIIIII.Dataframe(
                                    label=IlIllIllIIllIlIII(
                                        "Stats of selected models:"
                                    ),
                                    datatype="number",
                                    type="pandas",
                                )
                            IIIIIIlIllllllIll.click(
                                lambda IIIIIIIIIIlIIllII: os.path.join(
                                    _IlllllIIllllllIlI, IIIIIIIIIIlIIllII, "lowestvals"
                                ),
                                inputs=[IIllllIlIIIllllII],
                                outputs=[IllIIIllllllllIlI],
                            )
                            IIIIIIlIllllllIll.click(
                                fn=IIlIIlllIlIIIIlll.main,
                                inputs=[
                                    IIllllIlIIIllllII,
                                    IllIIIIlIIllIlllI,
                                    IIIllllllIIlllllI,
                                ],
                                outputs=[IIlllIllIlIlIlIII],
                            )
                            IIlllIllIlIlIlIII.change(
                                fn=IIlIIlllIlIIIIlll.selectweights,
                                inputs=[
                                    IIllllIlIIIllllII,
                                    IIlllIllIlIlIlIII,
                                    IllIllIllIIllIIll,
                                    IllIIIllllllllIlI,
                                ],
                                outputs=[
                                    IlIIlIlllIlIIIIlI,
                                    IIllllIllIIIllIlI,
                                    IllIllIIIlllIIlIl,
                                ],
                            )
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("UVR5")):
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Row():
                        with IlIIlIlIIllIIIIII.Column():
                            IIlIIlIIIlIllIIlI = IlIIlIlIIllIIIIII.Radio(
                                label=IlIllIllIIllIlIII("Model Architecture:"),
                                choices=[_IIIIllIlIllIIIIlI, _IIIllIlIlIIIIIllI],
                                value=_IIIIllIlIllIIIIlI,
                                interactive=_IIIllIllIIlIIIIIl,
                            )
                            IlIIlIIlIIIlIIIIl = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(
                                    "Enter the path of the audio folder to be processed:"
                                ),
                                value=os.path.join(
                                    IIIIIlIlllllIIIlI, _IllllIIIIllIIlIII
                                ),
                            )
                            IllllIlIlIlIIIllI = IlIIlIlIIllIIIIII.File(
                                file_count=IIlllIlIIIllIlllI,
                                label=IlIllIllIIllIlIII(IIIIlIIIIlIIIIllI),
                            )
                        with IlIIlIlIIllIIIIII.Column():
                            IIlllIIIIIlllllII = IlIIlIlIIllIIIIII.Dropdown(
                                label=IlIllIllIIllIlIII("Model:"),
                                choices=IllIlIllIIIllIlII,
                            )
                            IIlIlIlIlIIIlIlIl = IlIIlIlIIllIIIIII.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                label="Vocal Extraction Aggressive",
                                value=10,
                                interactive=_IIIllIllIIlIIIIIl,
                                visible=_IIllIlIlIIllIlIII,
                            )
                            IIIlllllIIIIlIllI = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(
                                    "Specify the output folder for vocals:"
                                ),
                                value=IlllllllllIIlIIlI,
                            )
                            IllllIIIlIlIllllI = IlIIlIlIIllIIIIII.Textbox(
                                label=IlIllIllIIllIlIII(
                                    "Specify the output folder for accompaniment:"
                                ),
                                value=IlllllllllIIlIIlI,
                            )
                            IIllIIIlIllllIlll = IlIIlIlIIllIIIIII.Radio(
                                label=IlIllIllIIllIlIII("Export file format:"),
                                choices=[
                                    _IIlIllIllIlIIIIIl,
                                    _IIllIIIIIIIIIIIlI,
                                    _IllIlllIlIIIIllII,
                                    _IIIlIllllIIllIlII,
                                ],
                                value=_IIllIIIIIIIIIIIlI,
                                interactive=_IIIllIllIIlIIIIIl,
                            )
                        IIlIIlIIIlIllIIlI.change(
                            fn=IIlllIllIlIIIIIII,
                            inputs=IIlIIlIIIlIllIIlI,
                            outputs=IIlllIIIIIlllllII,
                        )
                        IIlIIIIIllIlIlllI = IlIIlIlIIllIIIIII.Button(
                            IlIllIllIIllIlIII(IllIIlllllIIllllI),
                            variant=IllIllIIIIlIllIll,
                        )
                        IlIlIlIIIIIlIllII = IlIIlIlIIllIIIIII.Textbox(
                            label=IlIllIllIIllIlIII(IllllIIIlllIlIIII)
                        )
                        IIlIIIIIllIlIlllI.click(
                            IIIlIlIlIIIIlIIll,
                            [
                                IIlllIIIIIlllllII,
                                IlIIlIIlIIIlIIIIl,
                                IIIlllllIIIIlIllI,
                                IllllIlIlIlIIIllI,
                                IllllIIIlIlIllllI,
                                IIlIlIlIlIIIlIlIl,
                                IIllIIIlIllllIlll,
                                IIlIIlIIIlIllIIlI,
                            ],
                            [IlIlIlIIIIIlIllII],
                        )
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("TTS")):
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Column():
                        IllIIIlIIIlIlIIII = IlIIlIlIIllIIIIII.Textbox(
                            label=IlIllIllIIllIlIII("Text:"),
                            placeholder=IlIllIllIIllIlIII(
                                "Enter the text you want to convert to voice..."
                            ),
                            lines=6,
                        )
                with IlIIlIlIIllIIIIII.Group():
                    with IlIIlIlIIllIIIIII.Row():
                        with IlIIlIlIIllIIIIII.Column():
                            IIIIIlIllllIIIIlI = [_IllIllIIIIlIIllII, _IllIIIIllIlIIllll]
                            IllIIIlIIllllIlII = IlIIlIlIIllIIIIII.Dropdown(
                                IIIIIlIllllIIIIlI,
                                value=_IllIllIIIIlIIllII,
                                label=IlIllIllIIllIlIII("TTS Method:"),
                                visible=_IIIllIllIIlIIIIIl,
                            )
                            IlIIlIIIIlllIIlIl = IlIIlIlIIllIIIIII.Dropdown(
                                IlIIlllIllIllIlll,
                                label=IlIllIllIIllIlIII("TTS Model:"),
                                visible=_IIIllIllIIlIIIIIl,
                            )
                            IllIIIlIIllllIlII.change(
                                fn=IlIIIIIIIllIIllII,
                                inputs=IllIIIlIIllllIlII,
                                outputs=IlIIlIIIIlllIIlIl,
                            )
                        with IlIIlIlIIllIIIIII.Column():
                            IlllIlIIIlIIIIlII = IlIIlIlIIllIIIIII.Dropdown(
                                label=IlIllIllIIllIlIII("RVC Model:"),
                                choices=sorted(IlIIIlIlIlllllIIl),
                                value=IIIIIIIllIIlIllIl,
                            )
                            IlllIlIIlllIIIlIl, _IlIIlllIllIlIllIl = IIlllllllIIlIIlll(
                                IlllIlIIIlIIIIlII.value
                            )
                            IIIIIIIllIIIIIlIl = IlIIlIlIIllIIIIII.Dropdown(
                                label=IlIllIllIIllIlIII("Select the .index file:"),
                                choices=IIllIIIllllIIllll(),
                                value=IlllIlIIlllIIIlIl,
                                interactive=_IIIllIllIIlIIIIIl,
                                allow_custom_value=_IIIllIllIIlIIIIIl,
                            )
                with IlIIlIlIIllIIIIII.Row():
                    IIIlllllIlIIIIIll = IlIIlIlIIllIIIIII.Button(
                        IlIllIllIIllIlIII(IlIlllIIllIIlIIII), variant=IllIllIIIIlIllIll
                    )
                    IIIlllllIlIIIIIll.click(
                        fn=IIlIIlIIIlIIllIlI,
                        inputs=[],
                        outputs=[IlllIlIIIlIIIIlII, IIIIIIIllIIIIIlIl],
                    )
                with IlIIlIlIIllIIIIII.Row():
                    IllIlIllIllIllIlI = IlIIlIlIIllIIIIII.Audio(
                        label=IlIllIllIIllIlIII("Audio TTS:")
                    )
                    IIIIIllIllIIIIllI = IlIIlIlIIllIIIIII.Audio(
                        label=IlIllIllIIllIlIII("Audio RVC:")
                    )
                with IlIIlIlIIllIIIIII.Row():
                    IllIIlIlIlIIIlIll = IlIIlIlIIllIIIIII.Button(
                        IlIllIllIIllIlIII(IllIIlllllIIllllI), variant=IllIllIIIIlIllIll
                    )
                IllIIlIlIlIIIlIll.click(
                    IIlllIllIllIIIIIl,
                    inputs=[
                        IllIIIlIIIlIlIIII,
                        IlIIlIIIIlllIIlIl,
                        IlllIlIIIlIIIIlII,
                        IIIIIIIllIIIIIlIl,
                        IllIlIIlIlIIllllI,
                        IIlIlIIlIlllllIIl,
                        IIlIlIIIlllIlIlIl,
                        IIllIIIlllIllllIl,
                        IllIIlllllIllIIll,
                        IllIIIlIIllllIlII,
                    ],
                    outputs=[IIIIIllIllIIIIllI, IllIlIllIllIllIlI],
                )
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Resources")):
                easy_infer.download_model()
                easy_infer.download_backup()
                easy_infer.download_dataset(IllIllllIllIIIIlI)
                easy_infer.download_audio()
                easy_infer.youtube_separator()
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Extra")):
                IlIIlIlIIllIIIIII.Markdown(
                    value=IlIllIllIIllIlIII(
                        "This section contains some extra utilities that often may be in experimental phases"
                    )
                )
                with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Merge Audios")):
                    with IlIIlIlIIllIIIIII.Group():
                        IlIIlIlIIllIIIIII.Markdown(
                            value="## "
                            + IlIllIllIIllIlIII(
                                "Merge your generated audios with the instrumental"
                            )
                        )
                        IlIIlIlIIllIIIIII.Markdown(
                            value="",
                            scale=IlIlIIlIIIlIIIlIl,
                            visible=_IIIllIllIIlIIIIIl,
                        )
                        IlIIlIlIIllIIIIII.Markdown(
                            value="",
                            scale=IlIlIIlIIIlIIIlIl,
                            visible=_IIIllIllIIlIIIIIl,
                        )
                        with IlIIlIlIIllIIIIII.Row():
                            with IlIIlIlIIllIIIIII.Column():
                                IIlIIlIIlIllIIlll = IlIIlIlIIllIIIIII.File(
                                    label=IlIllIllIIllIlIII(IIllIIllIlIllIIIl)
                                )
                                IlIIlIlIIllIIIIII.Markdown(
                                    value=IlIllIllIIllIlIII(
                                        "### Instrumental settings:"
                                    )
                                )
                                IIIlllIIllIIlIIIl = IlIIlIlIIllIIIIII.Dropdown(
                                    label=IlIllIllIIllIlIII(
                                        "Choose your instrumental:"
                                    ),
                                    choices=sorted(IlIIIIlIIIlIlllll),
                                    value="",
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIllIlIllllllIIl = IlIIlIlIIllIIIIII.Slider(
                                    minimum=0,
                                    maximum=10,
                                    label=IlIllIllIIllIlIII(
                                        "Volume of the instrumental audio:"
                                    ),
                                    value=_IIIIIIlllIlIIIlIl,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIIlIlIIllIIIIII.Markdown(
                                    value=IlIllIllIIllIlIII("### Audio settings:")
                                )
                                IllllIIIlIIllIIll = IlIIlIlIIllIIIIII.Dropdown(
                                    label=IlIllIllIIllIlIII(
                                        "Select the generated audio"
                                    ),
                                    choices=sorted(IlllIIIllIllIlllI),
                                    value="",
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                with IlIIlIlIIllIIIIII.Row():
                                    IIlIIIIIIIlIlIlIl = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=10,
                                        label=IlIllIllIIllIlIII(
                                            "Volume of the generated audio:"
                                        ),
                                        value=_IIIIIIlllIlIIIlIl,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                IlIIlIlIIllIIIIII.Markdown(
                                    value=IlIllIllIIllIlIII("### Add the effects:")
                                )
                                IlIllllllllIIIIIl = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII("Reverb"),
                                    value=_IIllIlIlIIllIlIII,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIlIIllIlllIIlII = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII("Compressor"),
                                    value=_IIllIlIlIIllIlIII,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IIIlIlIIlIIlIlIlI = IlIIlIlIIllIIIIII.Checkbox(
                                    label=IlIllIllIIllIlIII("Noise Gate"),
                                    value=_IIllIlIlIIllIlIII,
                                    interactive=_IIIllIllIIlIIIIIl,
                                )
                                IlIIIIllIIIlIIIll = IlIIlIlIIllIIIIII.Button(
                                    IlIllIllIIllIlIII("Merge"),
                                    variant=IllIllIIIIlIllIll,
                                ).style(full_width=_IIIllIllIIlIIIIIl)
                                IlllllIIllIIlIlII = IlIIlIlIIllIIIIII.Textbox(
                                    label=IlIllIllIIllIlIII(IllllIIIlllIlIIII)
                                )
                                IIlIlllIIIllIllIl = IlIIlIlIIllIIIIII.Audio(
                                    label=IlIllIllIIllIlIII(IlIlllIllIIllIlll),
                                    type=IllIllllIllIIlIll,
                                )
                                IIlIIlIIlIllIIlll.upload(
                                    fn=IIIllllIIIlIIlllI,
                                    inputs=[IIlIIlIIlIllIIlll],
                                    outputs=[IIIlllIIllIIlIIIl],
                                )
                                IIlIIlIIlIllIIlll.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IIIlllIIllIIlIIIl],
                                )
                                IIIlIllIIIllIIIll.click(
                                    fn=lambda: IlIlIIlIIlllIllIl(),
                                    inputs=[],
                                    outputs=[IIIlllIIllIIlIIIl, IllllIIIlIIllIIll],
                                )
                                IlIIIIllIIIlIIIll.click(
                                    fn=IIllIIIlllllIlIlI,
                                    inputs=[
                                        IIIlllIIllIIlIIIl,
                                        IllllIIIlIIllIIll,
                                        IIIllIlIllllllIIl,
                                        IIlIIIIIIIlIlIlIl,
                                        IlIllllllllIIIIIl,
                                        IIIlIIllIlllIIlII,
                                        IIIlIlIIlIIlIlIlI,
                                    ],
                                    outputs=[IlllllIIllIIlIlII, IIlIlllIIIllIllIl],
                                )
                with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Processing")):
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII(
                                "Model fusion, can be used to test timbre fusion"
                            )
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                with IlIIlIlIIllIIIIII.Column():
                                    IIllllIIIIllIIlII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IIlIlIllllIIIllII),
                                        value="",
                                        max_lines=1,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIIlllllI
                                        ),
                                    )
                                    IIIlllllIllllIlIl = IlIIlIlIIllIIIIII.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IlIllIllIIllIlIII("Weight for Model A:"),
                                        value=0.5,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIIlllIlIIIIIlll = IlIIlIlIIllIIIIII.Checkbox(
                                        label=IlIllIllIIllIlIII(IlIIIIlIlIlIIIllI),
                                        value=_IIIllIllIIlIIIIIl,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIIIIllIIIllIlIl = IlIIlIlIIllIIIIII.Radio(
                                        label=IlIllIllIIllIlIII(IIIlIllIlIIlllIll),
                                        choices=[
                                            _IlIlllllIlllIlIIl,
                                            _IlIlIIllllllIlIII,
                                        ],
                                        value=_IlIlIIllllllIlIII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IllIIlllIIIIllIll = IlIIlIlIIllIIIIII.Radio(
                                        label=IlIllIllIIllIlIII(IllIlllllllIIlIlI),
                                        choices=[
                                            _IIlllllllIIIIlIII,
                                            _IlIIIlIIIIlllllII,
                                        ],
                                        value=_IIlllllllIIIIlIII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                with IlIIlIlIIllIIIIII.Column():
                                    IIIIlllIlIIllIlll = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII("Path to Model A:"),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IlIIIlIllIllIllll
                                        ),
                                    )
                                    IIlIllIIlIllIIlIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII("Path to Model B:"),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IlIIIlIllIllIllll
                                        ),
                                    )
                                    IllllIlIlIIIIlIll = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIlIIIIIIIllIll),
                                        value="",
                                        max_lines=8,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIllIlIlI
                                        ),
                                    )
                                    IIIlIIIIlIllIllIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                        value="",
                                        max_lines=8,
                                    )
                            IllIIIIlIIIIlllll = IlIIlIlIIllIIIIII.Button(
                                IlIllIllIIllIlIII("Fusion"), variant=IllIllIIIIlIllIll
                            )
                            IllIIIIlIIIIlllll.click(
                                merge,
                                [
                                    IIIIlllIlIIllIlll,
                                    IIlIllIIlIllIIlIl,
                                    IIIlllllIllllIlIl,
                                    IllIIlllIIIIllIll,
                                    IlIIlllIlIIIIIlll,
                                    IllllIlIlIIIIlIll,
                                    IIllllIIIIllIIlII,
                                    IlIIIIllIIIllIlIl,
                                ],
                                IIIlIIIIlIllIllIl,
                            )
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII("Modify model information")
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                with IlIIlIlIIllIIIIII.Column():
                                    IlIIIIIIlIIIIlIIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIlIIlIlllIIlII),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IlIIIlIllIllIllll
                                        ),
                                    )
                                    IIllllIIIlllIIlIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(
                                            "Model information to be modified:"
                                        ),
                                        value="",
                                        max_lines=8,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIllIlIlI
                                        ),
                                    )
                                with IlIIlIlIIllIIIIII.Column():
                                    IIIlIllllllIIlllI = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII("Save file name:"),
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIIlllllI
                                        ),
                                        value="",
                                        max_lines=8,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IlIIlIIIlIlllIlII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                        value="",
                                        max_lines=8,
                                    )
                            IlIIlIIlIlllIIIII = IlIIlIlIIllIIIIII.Button(
                                IlIllIllIIllIlIII("Modify"), variant=IllIllIIIIlIllIll
                            )
                            IlIIlIIlIlllIIIII.click(
                                change_info,
                                [
                                    IlIIIIIIlIIIIlIIl,
                                    IIllllIIIlllIIlIl,
                                    IIIlIllllllIIlllI,
                                ],
                                IlIIlIIIlIlllIlII,
                            )
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII("View model information")
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                with IlIIlIlIIllIIIIII.Column():
                                    IlllIlllIllIlIIIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIlIIlIlllIIlII),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IlIIIlIllIllIllll
                                        ),
                                    )
                                    IIIIIIIIlIllIlIIl = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                        value="",
                                        max_lines=8,
                                    )
                                    IllIIIlllIIllIIll = IlIIlIlIIllIIIIII.Button(
                                        IlIllIllIIllIlIII("View"),
                                        variant=IllIllIIIIlIllIll,
                                    )
                            IllIIIlllIIllIIll.click(
                                show_info, [IlllIlllIllIlIIIl], IIIIIIIIlIllIlIIl
                            )
                    with IlIIlIlIIllIIIIII.Group():
                        with IlIIlIlIIllIIIIII.Accordion(
                            label=IlIllIllIIllIlIII("Model extraction")
                        ):
                            with IlIIlIlIIllIIIIII.Row():
                                with IlIIlIlIIllIIIIII.Column():
                                    IllIIlIlllIlIIIII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IIlIlIllllIIIllII),
                                        value="",
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIIlllllI
                                        ),
                                    )
                                    IIlIIIIllIIlllIIl = IlIIlIlIIllIIIIII.Checkbox(
                                        label=IlIllIllIIllIlIII(IlIIIIlIlIlIIIllI),
                                        value=_IIIllIllIIlIIIIIl,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIIIIlllllIIllIIl = IlIIlIlIIllIIIIII.Radio(
                                        label=IlIllIllIIllIlIII(IIIlIllIlIIlllIll),
                                        choices=[
                                            _IlIlllllIlllIlIIl,
                                            _IlIlIIllllllIlIII,
                                        ],
                                        value=_IlIlIIllllllIlIII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IllIIIIIIIllIIIIl = IlIIlIlIIllIIIIII.Radio(
                                        label=IlIllIllIIllIlIII(IllIlllllllIIlIlI),
                                        choices=[
                                            _IllIIlIlIlIlllllI,
                                            _IIlllllllIIIIlIII,
                                            _IlIIIlIIIIlllllII,
                                        ],
                                        value=_IIlllllllIIIIlIII,
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                with IlIIlIlIIllIIIIII.Column():
                                    IIIllllIIlIIllIll = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIlIIlIlllIIlII),
                                        placeholder=IlIllIllIIllIlIII(
                                            IlIIIlIllIllIllll
                                        ),
                                        interactive=_IIIllIllIIlIIIIIl,
                                    )
                                    IIlIIIIIIIIIIlIII = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllIlIIIIIIIllIll),
                                        value="",
                                        max_lines=8,
                                        interactive=_IIIllIllIIlIIIIIl,
                                        placeholder=IlIllIllIIllIlIII(
                                            IIIIllllIIllIlIlI
                                        ),
                                    )
                                    IIlllllIllIlIlIlI = IlIIlIlIIllIIIIII.Textbox(
                                        label=IlIllIllIIllIlIII(IllllIIIlllIlIIII),
                                        value="",
                                        max_lines=8,
                                    )
                            with IlIIlIlIIllIIIIII.Row():
                                IIllIllIlIllIIIIl = IlIIlIlIIllIIIIII.Button(
                                    IlIllIllIIllIlIII("Extract"),
                                    variant=IllIllIIIIlIllIll,
                                )
                                IIIllllIIlIIllIll.change(
                                    IIIIIllIlIlIIllIl,
                                    [IIIllllIIlIIllIll],
                                    [
                                        IllIIIIIIIllIIIIl,
                                        IIlIIIIllIIlllIIl,
                                        IIIIIlllllIIllIIl,
                                    ],
                                )
                            IIllIllIlIllIIIIl.click(
                                extract_small_model,
                                [
                                    IIIllllIIlIIllIll,
                                    IllIIlIlllIlIIIII,
                                    IllIIIIIIIllIIIIl,
                                    IIlIIIIllIIlllIIl,
                                    IIlIIIIIIIIIIlIII,
                                    IIIIIlllllIIllIIl,
                                ],
                                IIlllllIllIlIlIlI,
                            )
            with IlIIlIlIIllIIIIII.TabItem(IlIllIllIIllIlIII("Settings")):
                with IlIIlIlIIllIIIIII.Row():
                    IlIIlIlIIllIIIIII.Markdown(
                        value=IlIllIllIIllIlIII("Pitch settings")
                    )
                    IlIllIllIIlIlIlII = IlIIlIlIIllIIIIII.Checkbox(
                        label=IlIllIllIIllIlIII(
                            "Whether to use note names instead of their hertz value. E.G. [C5, D6] instead of [523.25, 1174.66]Hz"
                        ),
                        value=rvc_globals.NotesIrHertz,
                        interactive=_IIIllIllIIlIIIIIl,
                    )
            IlIllIllIIlIlIlII.change(
                fn=lambda IIllIlIIIIIllllll: rvc_globals.__setattr__(
                    "NotesIrHertz", IIllIlIIIIIllllll
                ),
                inputs=[IlIllIllIIlIlIlII],
                outputs=[],
            )
            IlIllIllIIlIlIlII.change(
                fn=IIIIIIIIlIIIllllI,
                inputs=[IllIIIIIlIIlIllII],
                outputs=[
                    IlIllIllllIIllIIl,
                    IlIlIIllllIlIIIII,
                    IIllllllIllIIIllI,
                    IIIlllIlllIlIllll,
                ],
            )
        return IIllIIlIllIlllIIl


def IIIlIllllIlllllll(IIIIIIIlIlllIllll):
    IIlIIllIlllIIIllI = "0.0.0.0"
    IIIlllllIlIIlIlll = IIlIlllllIlllIIlI.iscolab or IIlIlllllIlllIIlI.paperspace
    IIIIIlIIlIlIIlIlI = 511
    IIlllllIIIlIIIIIl = 1022
    if IIlIlllllIlllIIlI.iscolab or IIlIlllllIlllIIlI.paperspace:
        IIIIIIIlIlllIllll.queue(
            concurrency_count=IIIIIlIIlIlIIlIlI, max_size=IIlllllIIIlIIIIIl
        ).launch(
            server_name=IIlIIllIlllIIIllI,
            inbrowser=not IIlIlllllIlllIIlI.noautoopen,
            server_port=IIlIlllllIlllIIlI.listen_port,
            quiet=_IIIllIllIIlIIIIIl,
            favicon_path="./images/icon.png",
            share=_IIllIlIlIIllIlIII,
        )
    else:
        IIIIIIIlIlllIllll.queue(
            concurrency_count=IIIIIlIIlIlIIlIlI, max_size=IIlllllIIIlIIIIIl
        ).launch(
            server_name=IIlIIllIlllIIIllI,
            inbrowser=not IIlIlllllIlllIIlI.noautoopen,
            server_port=IIlIlllllIlllIIlI.listen_port,
            quiet=_IIIllIllIIlIIIIIl,
            favicon_path=".\\images\\icon.png",
            share=IIIlllllIlIIlIlll,
        )


if __name__ == "__main__":
    if os.name == "nt":
        print(
            IlIllIllIIllIlIII(
                "Any ConnectionResetErrors post-conversion are irrelevant and purely visual; they can be ignored.\n"
            )
        )
    IlIIIllIllIllllIl = IIIIllllllIIIIIlI(UTheme=IIlIlllllIlllIIlI.grtheme)
    IIIlIllllIlllllll(IlIIIllIllIllllIl)
