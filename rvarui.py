_IlIIIllIlIIlIIIlI = "converted_tts"
_IIlIIlllIllIIIIlI = "Replacing old dropdown file..."
_IllIIIIlIIllIIlIl = "emb_g.weight"
_IIIIIlIlIlIIIlIll = "sample_rate"
_IlIIlIIllllIIIlIl = "Index not used."
_IIIlIllIIIlllIllI = "Using index:%s."
_IIIIllIIllIllIIII = "You need to upload an audio"
_IIIIlIIIIllllllII = "pretrained"
_IIllllIllIIllllIl = "EXTRACT-MODEL"
_IllIIIlIIIIIIlIIl = "TRAIN-FEATURE"
_IIIllIllIIlllIlll = "TRAIN"
_IlIlllIlllIIlllIl = "EXTRACT-FEATURE"
_IllIIlllIIIllIIIl = "PRE-PROCESS"
_IlIlIlIlIIlllIlII = "INFER"
_IlIIIlIIlIlIIlIII = "HOME"
_IIlllIIllIIlllIll = "_v2"
_IlIlIlIIlllIIIllI = "clean_empty_cache"
_IIIlllIlllIIIIlII = "Bark-tts"
_IIIIlIIllIIlllIll = "MDX"
_IlllllIIllllIlIII = ".onnx"
_IlIIllllIlIIlllII = "aac"
_IlllllIllIIlIlIIl = "ogg"
_IlIllIlllIIIIIIIl = "csvdb/stop.csv"
_IlIIlllllIlllIlIl = "audio-others"
_IIIlIIllIlIlIllII = "pm"
_IIIlllIlIlIIIlIll = "weight"
_IIlllIIIIIlllIIlI = "cpu"
_IIIlIlIIllllIIIlI = "rmvpe_onnx"
_IIlIIllIlIIIIlIll = "Edge-tts"
_IlIIIIIllllIlIlII = "VR"
_IIIlllllIlIIIIIll = "datasets"
_IllIIIIIIlIIIIlll = "32k"
_IllIllIllIllllIIl = "version"
_IlIIllIllIllIIIIl = ".index"
_IlIIllIllllIlIIII = "m4a"
_IIIlIllllIIlIIIIl = "mp3"
_IlIlllllIIIIIlIlI = "48k"
_IlIlIIlIllllIIlIl = ".pth"
_IlIIlllIlIllIIlll = "flac"
_IIIlIlIlIIIIllllI = "logs"
_IIIlIIIlIIIIlllll = "D"
_IlIIIIIIllllIIlll = "G"
_IllIIIlIllllIIllI = "f0"
_IIIIlllIlIIIlllll = "."
_IlIllIlIIIlllIIIl = "trained"
_IIlIlIlIIllIIIIll = "wav"
_IIIllllllIllIIIII = "v2"
_IlIIIIlllIlIIllll = "r"
_IIlIIlIIlIlIIIlII = "config"
_IlIlllIlIlIlIlIll = 1.0
_IIlIIIIIlIlIIIllI = "audios"
_IIIllIIIlIIllIlII = "40k"
_IIlllllllllIllIll = '"'
_IIlIIIlIIIllIIIll = "audio-outputs"
_IlIIllllIIIlIlIlI = "rmvpe"
_IlIlIIIlllIllIlll = " "
_IIllIIIlIIlllIlII = "choices"
_IlllIIllllIIIIlIl = "value"
_IlIIIIIIlIlIllIIl = "v1"
_IIlIlIIllllIlIIlI = "\n"
_IIIllIllllIlIllIl = "visible"
_IllIIIllIIIllllIl = None
_IlllllIlIIlIlIlII = "update"
_IIllIIllllllIllII = "__type__"
_IIIIlIlIIIlIlIlll = False
_IlllIIIlllIllIllI = True
import sys
from shutil import rmtree
import shutil, json, datetime, unicodedata
from glob import glob1
from signal import SIGTERM
import librosa, os

IlIIIllllllIllIll = os.getcwd()
sys.path.append(IlIIIllllllIllIll)
import lib.globals.globals as rvc_globals
from LazyImport import lazyload
import mdx
from mdx_processing_script import get_model_list, id_to_ptm, prepare_mdx, run_mdx

IlIlllIIIlIlIIlll = lazyload("math")
import traceback, warnings

IIlllllllIIIIIlIl = lazyload("tensorlowest")
import faiss

IIIIIIllIIIlllIIl = lazyload("ffmpeg")
import nltk

nltk.download("punkt", quiet=_IlllIIIlllIllIllI)
from nltk.tokenize import sent_tokenize
from bark import generate_audio, SAMPLE_RATE

IIlIlllllIlIlIIlI = lazyload("numpy")
IIIIIIlIIIIIIIIII = lazyload("torch")
IIIIlIllIlIlIlllI = lazyload("regex")
os.environ["TF_CPP_MIN_LIG_LEVEL"] = "3"
os.environ["IPENBLAS_NUM_THREADS"] = "1"
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import logging
from random import shuffle
from subprocess import Popen
import easy_infer, audioEffects

IllIlIlIIllllIlIl = lazyload("gradio")
IIllIIIIlIIIIIIll = lazyload("soundfile")
IlIlIIlIlIIllIIIl = IIllIIIIlIIIIIIll.write
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

IIIIlIIlllllIlllI = ""
IlIIIIlllllIllIIl = lambda IlIllIlIlIIlIllII: SQuote(str(IlIllIlIlIIlIllII))
IlllllllIIlIlIlIl = os.path.join(IlIIIllllllIllIll, "TEMP")
IlIlIlllIIIlIIIlI = os.path.join(IlIIIllllllIllIll, "runtime/Lib/site-packages")
IlIlIIIlllIlIIIlI = [
    _IIIlIlIlIIIIllllI,
    _IIlIIIIIlIlIIIllI,
    _IIIlllllIlIIIIIll,
    "weights",
]
rmtree(IlllllllIIlIlIlIl, ignore_errors=_IlllIIIlllIllIllI)
rmtree(os.path.join(IlIlIlllIIIlIIIlI, "infer_pack"), ignore_errors=_IlllIIIlllIllIllI)
rmtree(os.path.join(IlIlIlllIIIlIIIlI, "uvr5_pack"), ignore_errors=_IlllIIIlllIllIllI)
os.makedirs(
    os.path.join(IlIIIllllllIllIll, _IIlIIIlIIIllIIIll), exist_ok=_IlllIIIlllIllIllI
)
os.makedirs(
    os.path.join(IlIIIllllllIllIll, _IlIIlllllIlllIlIl), exist_ok=_IlllIIIlllIllIllI
)
os.makedirs(IlllllllIIlIlIlIl, exist_ok=_IlllIIIlllIllIllI)
for IllIlIllIIlllIlIl in IlIlIIIlllIlIIIlI:
    os.makedirs(
        os.path.join(IlIIIllllllIllIll, IllIlIllIIlllIlIl), exist_ok=_IlllIIIlllIllIllI
    )
os.environ["TEMP"] = IlllllllIIlIlIlIl
warnings.filterwarnings("ignore")
IIIIIIlIIIIIIIIII.manual_seed(114514)
logging.getLogger("numba").setLevel(logging.WARNING)
try:
    IIIIlllIIlIlIIlII = open(_IlIllIlllIIIIIIIl, "x")
    IIIIlllIIlIlIIlII.close()
except FileExistsError:
    pass
global IlllIIIllIIIlIllI, IIIIllIIIIIlIlllI, IlIIIIlllIIIllllI
IlllIIIllIIIlIllI = rvc_globals.DoFormant
IIIIllIIIIIlIlllI = rvc_globals.Quefrency
IlIIIIlllIIIllllI = rvc_globals.Timbre
IlIllIlIlIIlIlIll = Config()
if IlIllIlIlIIlIlIll.dml == _IlllIIIlllIllIllI:

    def IlllIlIIlIIllIIll(IlIIlIllIlllIlIII, IIIlIlIIllIIllllI, IlIIllllIlIlIIIll):
        IlIIlIllIlllIlIII.scale = IlIIllllIlIlIIIll
        IIlIIlIIIIlIIlIIl = IIIlIlIIllIIllllI.clone().detach()
        return IIlIIlIIIIlIIlIIl

    fairseq.modules.grad_multiply.GradMultiply.forward = IlllIlIIlIIllIIll
IIIllIIllIIIIIIlI = I18nAuto()
IIIllIIllIIIIIIlI.print()
IIlIlIllIlIIlllII = IIIIIIlIIIIIIIIII.cuda.device_count()
IIIllIIIIlllIlllI = []
IlIIlIlIlllllIlll = []
IIIlllIIlIIIlllII = _IIIIlIlIIIlIlIlll
IllIlllIlIlIIllIl = [
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
if IIIIIIlIIIIIIIIII.cuda.is_available() or IIlIlIllIlIIlllII != 0:
    for IIlIlllllIllllIll in range(IIlIlIllIlIIlllII):
        IIIIIllllIIlIllIl = IIIIIIlIIIIIIIIII.cuda.get_device_name(
            IIlIlllllIllllIll
        ).upper()
        if any(
            IIlIlIIlIlIlIIIlI in IIIIIllllIIlIllIl
            for IIlIlIIlIlIlIIIlI in IllIlllIlIlIIllIl
        ):
            IIIlllIIlIIIlllII = _IlllIIIlllIllIllI
            IIIllIIIIlllIlllI.append("%s\t%s" % (IIlIlllllIllllIll, IIIIIllllIIlIllIl))
            IlIIlIlIlllllIlll.append(
                int(
                    IIIIIIlIIIIIIIIII.cuda.get_device_properties(
                        IIlIlllllIllllIll
                    ).total_memory
                    / 1e9
                    + 0.4
                )
            )
IIlIIlllIlIlIIIll = (
    _IIlIlIIllllIlIIlI.join(IIIllIIIIlllIlllI)
    if IIIlllIIlIIIlllII and IIIllIIIIlllIlllI
    else "Unfortunately, there is no compatible GPU available to support your training."
)
IIlIllIllIllIIIIl = (
    min(IlIIlIlIlllllIlll) // 2 if IIIlllIIlIIIlllII and IIIllIIIIlllIlllI else 1
)
IIIlIlllllIllIlII = "-".join(
    IIIlIlllIIllIIlll[0] for IIIlIlllIIllIIlll in IIIllIIIIlllIlllI
)
IIIlIlllIIIIlllIl = _IllIIIllIIIllllIl


def IlIIlIllIIIIIIlIl():
    global IIIlIlllIIIIlllIl
    (
        IlIlllIIllIIIIlII,
        _IIllIlIIlllIlIIIl,
        _IIllIlIIlllIlIIIl,
    ) = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        ["/kaggle/input/ax-rmf/hubert_base.pt"], suffix=""
    )
    IIIlIlllIIIIlllIl = IlIlllIIllIIIIlII[0].to(IlIllIlIlIIlIlIll.device)
    if IlIllIlIlIIlIlIll.is_half:
        IIIlIlllIIIIlllIl = IIIlIlllIIIIlllIl.half()
    IIIlIlllIIIIlllIl.eval()


IlIllllIlIllIlIlI = _IIIlllllIlIIIIIll
IllIIllIlllIlIlIl = "weights"
IIllllIlIlllIIllI = "uvr5_weights"
IlIlIIIllIIllIlll = _IIIlIlIlIIIIllllI
IlIIIllIllllIIIll = "formantshiftcfg"
IllIIlIIIllIllIII = _IIlIIIIIlIlIIIllI
IllIIIIllIIIIlIll = _IlIIlllllIlllIlIl
IIlIIIlIIlIllllII = {
    _IIlIlIlIIllIIIIll,
    _IIIlIllllIIlIIIIl,
    _IlIIlllIlIllIIlll,
    _IlllllIllIIlIlIIl,
    "opus",
    _IlIIllIllllIlIIII,
    "mp4",
    _IlIIllllIlIIlllII,
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}
IIlIIIlIIIlIIlIII = [
    os.path.join(IlIllllIIIIIIlIII, IlllIIIllIIIIIIII)
    for (IlIllllIIIIIIlIII, _IllIlIllIIllIllIl, IIlIllIIlIlIlllll) in os.walk(
        IllIIllIlllIlIlIl
    )
    for IlllIIIllIIIIIIII in IIlIllIIlIlIlllll
    if IlllIIIllIIIIIIII.endswith((_IlIlIIlIllllIIlIl, _IlllllIIllllIlIII))
]
IllIlIIlIlIIIIIlI = [
    os.path.join(IIIIlIllllIlIlIlI, IllIlllllllIlllll)
    for (IIIIlIllllIlIlIlI, _IIlIIIIIIlIIIllIl, IllIlIlIllIlIllIl) in os.walk(
        IlIlIIIllIIllIlll, topdown=_IIIIlIlIIIlIlIlll
    )
    for IllIlllllllIlllll in IllIlIlIllIlIllIl
    if IllIlllllllIlllll.endswith(_IlIIllIllIllIIIIl)
    and _IlIllIlIIIlllIIIl not in IllIlllllllIlllll
]
IIIlIIllllIIIIIII = [
    os.path.join(IIllIllIlIIIIllII, IlIIllIllIIIIIllI)
    for (IIllIllIlIIIIllII, _IlllIIIIIIlIllllI, IIlllllllIIIlllIl) in os.walk(
        IllIIlIIIllIllIII, topdown=_IIIIlIlIIIlIlIlll
    )
    for IlIIllIllIIIIIllI in IIlllllllIIIlllIl
    if IlIIllIllIIIIIllI.endswith(tuple(IIlIIIlIIlIllllII))
    if IlIIllIllIIIIIllI.endswith(tuple(IIlIIIlIIlIllllII))
    and not IlIIllIllIIIIIllI.endswith(".gitignore")
]
IIllIlllIIllIIllI = [
    os.path.join(IlIlIlIIIIllIIlII, IIlIllIlllIlllIIl)
    for (IlIlIlIIIIllIIlII, _IIlllIlllllIllIll, IlllIIIllIIlllIlI) in os.walk(
        IllIIIIllIIIIlIll, topdown=_IIIIlIlIIIlIlIlll
    )
    for IIlIllIlllIlllIIl in IlllIIIllIIlllIlI
    if IIlIllIlllIlllIIl.endswith(tuple(IIlIIIlIIlIllllII))
]
IllllIIlIIllIIlll = [
    IlIIIlIIlIIllIllI.replace(_IlIlIIlIllllIIlIl, "")
    for IlIIIlIIlIIllIllI in os.listdir(IIllllIlIlllIIllI)
    if IlIIIlIIlIIllIllI.endswith(_IlIlIIlIllllIIlIl) or "onnx" in IlIIIlIIlIIllIllI
]
IlllIllIlIIlIlIll = lambda: sorted(IIlIIIlIIIlIIlIII)[0] if IIlIIIlIIIlIIlIII else ""
IlIIIIIIIIllIIlIl = []
for IIllIllIIIlIlIIII in os.listdir(os.path.join(IlIIIllllllIllIll, IlIllllIlIllIlIlI)):
    if _IIIIlllIlIIIlllll not in IIllIllIIIlIlIIII:
        IlIIIIIIIIllIIlIl.append(
            os.path.join(
                easy_infer.find_folder_parent(_IIIIlllIlIIIlllll, _IIIIlIIIIllllllII),
                _IIIlllllIlIIIIIll,
                IIllIllIIIlIlIIII,
            )
        )


def IlIIIllIlIlIllllI():
    if len(IlIIIIIIIIllIIlIl) > 0:
        return sorted(IlIIIIIIIIllIIlIl)[0]
    else:
        return ""


def IIlllIIllllllllll(IIlIIllIllIIlIllI):
    IlIlIIllIIlllIIIl = get_model_list()
    IIIIIIIllIllIlIIl = list(IlIlIIllIIlllIIIl)
    if IIlIIllIllIIlIllI == _IlIIIIIllllIlIlII:
        return {
            _IIllIIIlIIlllIlII: IllllIIlIIllIIlll,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
    elif IIlIIllIllIIlIllI == _IIIIlIIllIIlllIll:
        return {
            _IIllIIIlIIlllIlII: IIIIIIIllIllIlIIl,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }


IIllIlIIlIlIIllII = easy_infer.get_bark_voice()
IlIIIlllIIIlIIIII = easy_infer.get_edge_voice()


def IlIlIlllIlllIIIlI(IllllllllIlIllIll):
    if IllllllllIlIllIll == _IIlIIllIlIIIIlIll:
        return {
            _IIllIIIlIIlllIlII: IlIIIlllIIIlIIIII,
            _IlllIIllllIIIIlIl: "",
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
    elif IllllllllIlIllIll == _IIIlllIlllIIIIlII:
        return {
            _IIllIIIlIIlllIlII: IIllIlIIlIlIIllII,
            _IlllIIllllIIIIlIl: "",
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }


def IlIIllIllIlIIIlII(IlllllllllllIlIlI):
    IllIlllIIIllIlIlI = []
    for IlIIlllIlIIlIIlll in os.listdir(
        os.path.join(IlIIIllllllIllIll, IlIllllIlIllIlIlI)
    ):
        if _IIIIlllIlIIIlllll not in IlIIlllIlIIlIIlll:
            IllIlllIIIllIlIlI.append(
                os.path.join(
                    easy_infer.find_folder_parent(
                        _IIIIlllIlIIIlllll, _IIIIlIIIIllllllII
                    ),
                    _IIIlllllIlIIIIIll,
                    IlIIlllIlIIlIIlll,
                )
            )
    return IllIlIlIIllllIlIl.Dropdown.update(choices=IllIlllIIIllIlIlI)


def IIlllIlIIIlIIlIll():
    IlIlllIIllIlIlIII = [
        os.path.join(IIlIlIIlIlIllllIl, IIIIIIlIlllllIIlI)
        for (IIlIlIIlIlIllllIl, _IllIIIIIllIIIIIIl, IllIllIlIlIllIIII) in os.walk(
            IlIlIIIllIIllIlll
        )
        for IIIIIIlIlllllIIlI in IllIllIlIlIllIIII
        if IIIIIIlIlllllIIlI.endswith(_IlIIllIllIllIIIIl)
        and _IlIllIlIIIlllIIIl not in IIIIIIlIlllllIIlI
    ]
    return IlIlllIIllIlIlIII if IlIlllIIllIlIlIII else ""


def IIIllllIIlIlllIlI():
    IIlIlIllIIIlIlllI = [
        os.path.join(IllIIlIIllIIlIIlI, IlllIIlIIIIIIllIl)
        for (IllIIlIIllIIlIIlI, _IllIlIIIIlIIlIlIl, IlIlIIllIlIIIIIlI) in os.walk(
            IlIIIllIllllIIIll
        )
        for IlllIIlIIIIIIllIl in IlIlIIllIlIIIIIlI
        if IlllIIlIIIIIIllIl.endswith(".txt")
    ]
    return IIlIlIllIIIlIlllI if IIlIlIllIIIlIlllI else ""


import soundfile as sf


def IIllIIlIIIIIIlIll(IlIlIIlllllIIIlll, IllllIIIlIllllIlI, IIIlIlIIllIllllIl):
    IIIIIllIIlIlIllIl = 1
    while _IlllIIIlllIllIllI:
        IlIIIlIlIlllIlllI = os.path.join(
            IlIlIIlllllIIIlll,
            f"{IllllIIIlIllllIlI}_{IIIIIllIIlIlIllIl}.{IIIlIlIIllIllllIl}",
        )
        if not os.path.exists(IlIIIlIlIlllIlllI):
            return IlIIIlIlIlllIlllI
        IIIIIllIIlIlIllIl += 1


def IIIlIIllllIllIllI(
    IIllllIIllIIIIlIl,
    IlllIlIIIllIIIlll,
    IIllIlIlIIIllllll,
    IllllIIIIllIllIlI,
    IlIIllIIIlIllIlII,
):
    IlllllIlIlIlIllIl, IIlllllIlIlllIlll = librosa.load(
        IIllllIIllIIIIlIl, sr=_IllIIIllIIIllllIl
    )
    IlIlIlIIlIIIllllI, IlIlllllIlIlllIIl = librosa.load(
        IlllIlIIIllIIIlll, sr=_IllIIIllIIIllllIl
    )
    if IIlllllIlIlllIlll != IlIlllllIlIlllIIl:
        if IIlllllIlIlllIlll > IlIlllllIlIlllIIl:
            IlIlIlIIlIIIllllI = librosa.resample(
                IlIlIlIIlIIIllllI,
                orig_sr=IlIlllllIlIlllIIl,
                target_sr=IIlllllIlIlllIlll,
            )
        else:
            IlllllIlIlIlIllIl = librosa.resample(
                IlllllIlIlIlIllIl,
                orig_sr=IIlllllIlIlllIlll,
                target_sr=IlIlllllIlIlllIIl,
            )
    IIlIlIIlIlIllIlIl = min(len(IlllllIlIlIlIllIl), len(IlIlIlIIlIIIllllI))
    IlllllIlIlIlIllIl = librosa.util.fix_length(IlllllIlIlIlIllIl, IIlIlIIlIlIllIlIl)
    IlIlIlIIlIIIllllI = librosa.util.fix_length(IlIlIlIIlIIIllllI, IIlIlIIlIlIllIlIl)
    if IllllIIIIllIllIlI != _IlIlllIlIlIlIlIll:
        IlllllIlIlIlIllIl *= IllllIIIIllIllIlI
    if IlIIllIIIlIllIlII != _IlIlllIlIlIlIlIll:
        IlIlIlIIlIIIllllI *= IlIIllIIIlIllIlII
    IIIIlIlllllllIIlI = IlllllIlIlIlIllIl + IlIlIlIIlIIIllllI
    sf.write(IIllIlIlIIIllllll, IIIIlIlllllllIIlI, IIlllllIlIlllIlll)


def IlllIIlIIIIlllllI(
    IlIllIlIIlIllllII,
    IllIlIIlIIIIlIlIl,
    IIlIIIlIlIlIIllll=_IlIlllIlIlIlIlIll,
    IIIIIlIIllIIIIIII=_IlIlllIlIlIlIlIll,
    IlllllIIllIIlIlII=_IIIIlIlIIIlIlIlll,
    IlIllIlIIlllllIIl=_IIIIlIlIIIlIlIlll,
    IIlIIlllIlIIlIlIl=_IIIIlIlIIIlIlIlll,
):
    IIIlIlIlIlllIlllI = "Conversion complete!"
    IllIlIIllIIlIIlII = "combined_audio"
    IIIllllIIIIlIIlII = os.path.join(IlIIIllllllIllIll, _IIlIIIlIIIllIIIll)
    os.makedirs(IIIllllIIIIlIIlII, exist_ok=_IlllIIIlllIllIllI)
    IllIlIIIlIIlIIllI = IllIlIIllIIlIIlII
    IIlIIIIIlIIllIllI = _IIlIlIlIIllIIIIll
    IIIlIIllIIIlIlIll = IIllIIlIIIIIIlIll(
        IIIllllIIIIlIIlII, IllIlIIIlIIlIIllI, IIlIIIIIlIIllIllI
    )
    print(IlllllIIllIIlIlII)
    print(IlIllIlIIlllllIIl)
    print(IIlIIlllIlIIlIlIl)
    if IlllllIIllIIlIlII or IlIllIlIIlllllIIl or IIlIIlllIlIIlIlIl:
        IllIlIIIlIIlIIllI = "effect_audio"
        IIIlIIllIIIlIlIll = IIllIIlIIIIIIlIll(
            IIIllllIIIIlIIlII, IllIlIIIlIIlIIllI, IIlIIIIIlIIllIllI
        )
        IIIIIIlIllllIIlIl = audioEffects.process_audio(
            IllIlIIlIIIIlIlIl,
            IIIlIIllIIIlIlIll,
            IlllllIIllIIlIlII,
            IlIllIlIIlllllIIl,
            IIlIIlllIlIIlIlIl,
        )
        IllIlIIIlIIlIIllI = IllIlIIllIIlIIlII
        IIIlIIllIIIlIlIll = IIllIIlIIIIIIlIll(
            IIIllllIIIIlIIlII, IllIlIIIlIIlIIllI, IIlIIIIIlIIllIllI
        )
        IIIlIIllllIllIllI(
            IlIllIlIIlIllllII,
            IIIIIIlIllllIIlIl,
            IIIlIIllIIIlIlIll,
            IIlIIIlIlIlIIllll,
            IIIIIlIIllIIIIIII,
        )
        return IIIllIIllIIIIIIlI(IIIlIlIlIlllIlllI), IIIlIIllIIIlIlIll
    else:
        IllIlIIIlIIlIIllI = IllIlIIllIIlIIlII
        IIIlIIllIIIlIlIll = IIllIIlIIIIIIlIll(
            IIIllllIIIIlIIlII, IllIlIIIlIIlIIllI, IIlIIIIIlIIllIllI
        )
        IIIlIIllllIllIllI(
            IlIllIlIIlIllllII,
            IllIlIIlIIIIlIlIl,
            IIIlIIllIIIlIlIll,
            IIlIIIlIlIlIIllll,
            IIIIIlIIllIIIIIII,
        )
        return IIIllIIllIIIIIIlI(IIIlIlIlIlllIlllI), IIIlIIllIIIlIlIll


def IlIIlIIlIIlllIIII(
    IlllIllIlllIlllIl,
    IIlIIIIllIlIIllll,
    IIlIlllIlIIIIllIl,
    IllIIlIlIIIIllIIl,
    IlIllIlllIllIIIIl,
    IIlIIllllllIIIlIl,
    IlllIIIlllIllIIll,
    IIIlIIlIIIIlIIlll,
    IIlIllIlIIlllIIIl,
    IIllIIIIlllllIllI,
    IIllIlIIIIIlIllll,
    IIlllIIIIIlIllIIl,
    IIIlllIllIllIIlll,
    IIIllIllIlIllllIl,
    IlIIlllIlIIlIIlII,
    IllllIllIIlIlIllI,
    IlIIllIlIlIIIllIl,
    IIllIlIlIlIlIlIII,
    IIIIIlIIIllIIIIll,
):
    global IIlIIIIlIIlIlIIll
    IIlIIIIlIIlIlIIll = 0
    IllIIlIlIlIIllllI = time.time()
    global IIllIllIlIlIlIlII, IIllIIIllIIllllll, IIIIIlIIIlIllllII, IIIlIlllIIIIlllIl, IllIIIIIlllllIlIl
    IlIllIlIllIIlIlll = (
        _IlllIIIlllIllIllI
        if IIlIIllllllIIIlIl == _IIIlIlIIllllIIIlI
        else _IIIIlIlIIIlIlIlll
    )
    if not IIlIIIIllIlIIllll and not IIlIlllIlIIIIllIl:
        return _IIIIllIIllIllIIII, _IllIIIllIIIllllIl
    if not os.path.exists(IIlIIIIllIlIIllll) and not os.path.exists(
        os.path.join(IlIIIllllllIllIll, IIlIIIIllIlIIllll)
    ):
        return "Audio was not properly selected or doesn't exist", _IllIIIllIIIllllIl
    IIlIlllIlIIIIllIl = IIlIlllIlIIIIllIl or IIlIIIIllIlIIllll
    print(f"\nStarting inference for '{os.path.basename(IIlIlllIlIIIIllIl)}'")
    print("-------------------")
    IllIIlIlIIIIllIIl = int(IllIIlIlIIIIllIIl)
    if rvc_globals.NotesIrHertz and IIlIIllllllIIIlIl != _IlIIllllIIIlIlIlI:
        IlIIlllIlIIlIIlII = (
            IIIllIIIlIIIlllII(IllllIllIIlIlIllI) if IllllIllIIlIlIllI else 50
        )
        IlIIllIlIlIIIllIl = (
            IIIllIIIlIIIlllII(IIllIlIlIlIlIlIII) if IIllIlIlIlIlIlIII else 1100
        )
        print(
            f"Converted Min pitch: freq - {IlIIlllIlIIlIIlII}\nConverted Max pitch: freq - {IlIIllIlIlIIIllIl}"
        )
    else:
        IlIIlllIlIIlIIlII = IlIIlllIlIIlIIlII or 50
        IlIIllIlIlIIIllIl = IlIIllIlIlIIIllIl or 1100
    try:
        IIlIlllIlIIIIllIl = IIlIlllIlIIIIllIl or IIlIIIIllIlIIllll
        print(f"Attempting to load {IIlIlllIlIIIIllIl}....")
        IIlIIIllllIIlIIII = load_audio(
            IIlIlllIlIIIIllIl,
            16000,
            DoFormant=rvc_globals.DoFormant,
            Quefrency=rvc_globals.Quefrency,
            Timbre=rvc_globals.Timbre,
        )
        IlIIIlIIIIlIIllII = IIlIlllllIlIlIIlI.abs(IIlIIIllllIIlIIII).max() / 0.95
        if IlIIIlIIIIlIIllII > 1:
            IIlIIIllllIIlIIII /= IlIIIlIIIIlIIllII
        IIIllIIlIlIIIIIll = [0, 0, 0]
        if not IIIlIlllIIIIlllIl:
            print("Loading hubert for the first time...")
            IlIIlIllIIIIIIlIl()
        try:
            IIIIlIIIlIIlIllll = IIIIlIIlIIlllIIIl.get(_IllIIIlIllllIIllI, 1)
        except NameError:
            IIllIllIIlIlIIIII = "Model was not properly selected"
            print(IIllIllIIlIlIIIII)
            return IIllIllIIlIlIIIII, _IllIIIllIIIllllIl
        IlllIIIlllIllIIll = (
            IlllIIIlllIllIIll.strip(_IlIlIIIlllIllIlll)
            .strip(_IIlllllllllIllIll)
            .strip(_IIlIlIIllllIlIIlI)
            .strip(_IIlllllllllIllIll)
            .strip(_IlIlIIIlllIllIlll)
            .replace(_IlIllIlIIIlllIIIl, "added")
            if IlllIIIlllIllIIll != ""
            else IIIlIIlIIIIlIIlll
        )
        try:
            IIlIlIIlllllIIIll = IIIIIlIIIlIllllII.pipeline(
                IIIlIlllIIIIlllIl,
                IIllIIIllIIllllll,
                IlllIllIlllIlllIl,
                IIlIIIllllIIlIIII,
                IIlIlllIlIIIIllIl,
                IIIllIIlIlIIIIIll,
                IllIIlIlIIIIllIIl,
                IIlIIllllllIIIlIl,
                IlllIIIlllIllIIll,
                IIlIllIlIIlllIIIl,
                IIIIlIIIlIIlIllll,
                IIllIIIIlllllIllI,
                IIllIllIlIlIlIlII,
                IIllIlIIIIIlIllll,
                IIlllIIIIIlIllIIl,
                IllIIIIIlllllIlIl,
                IIIlllIllIllIIlll,
                IIIllIllIlIllllIl,
                IIIIIlIIIllIIIIll,
                IlIllIlIllIIlIlll,
                f0_file=IlIllIlllIllIIIIl,
                f0_min=IlIIlllIlIIlIIlII,
                f0_max=IlIIllIlIlIIIllIl,
            )
        except AssertionError:
            IIllIllIIlIlIIIII = (
                "Mismatching index version detected (v1 with v2, or v2 with v1)."
            )
            print(IIllIllIIlIlIIIII)
            return IIllIllIIlIlIIIII, _IllIIIllIIIllllIl
        except NameError:
            IIllIllIIlIlIIIII = (
                "RVC libraries are still loading. Please try again in a few seconds."
            )
            print(IIllIllIIlIlIIIII)
            return IIllIllIIlIlIIIII, _IllIIIllIIIllllIl
        if IIllIllIlIlIlIlII != IIllIlIIIIIlIllll >= 16000:
            IIllIllIlIlIlIlII = IIllIlIIIIIlIllll
        IllIllllllIIIllIl = (
            _IIIlIllIIIlllIllI % IlllIIIlllIllIIll
            if os.path.exists(IlllIIIlllIllIIll)
            else _IlIIlIIllllIIIlIl
        )
        IlllIIIIlIlIlIlIl = time.time()
        IIlIIIIlIIlIlIIll = IlllIIIIlIlIlIlIl - IllIIlIlIlIIllllI
        IlIlIlIIlllIlIIll = _IIlIIIlIIIllIIIll
        os.makedirs(IlIlIlIIlllIlIIll, exist_ok=_IlllIIIlllIllIllI)
        IlIIlIIlIIIIlIIll = "generated_audio_{}.wav"
        IlIllIlllllIIIIII = 1
        while _IlllIIIlllIllIllI:
            IllIlIlIIlIIIlIII = os.path.join(
                IlIlIlIIlllIlIIll, IlIIlIIlIIIIlIIll.format(IlIllIlllllIIIIII)
            )
            if not os.path.exists(IllIlIlIIlIIIlIII):
                break
            IlIllIlllllIIIIII += 1
        wavfile.write(IllIlIlIIlIIIlIII, IIllIllIlIlIlIlII, IIlIlIIlllllIIIll)
        print(f"Generated audio saved to: {IllIlIlIIlIIIlIII}")
        return (
            f"Success.\n {IllIllllllIIIllIl}\nTime:\n npy:{IIIllIIlIlIIIIIll[0]}, f0:{IIIllIIlIlIIIIIll[1]}, infer:{IIIllIIlIlIIIIIll[2]}\nTotal Time: {IIlIIIIlIIlIlIIll} seconds",
            (IIllIllIlIlIlIlII, IIlIlIIlllllIIIll),
        )
    except:
        IIllIIIlIIIIIIIIl = traceback.format_exc()
        print(IIllIIIlIIIIIIIIl)
        return IIllIIIlIIIIIIIIl, (_IllIIIllIIIllllIl, _IllIIIllIIIllllIl)


def IlllIllIIIIlllllI(
    IlllllllIIllIIIlI,
    IlllIIIIlIIIlIlII,
    IlIIIIIllIIlIIllI,
    IllIIIlIlIlIIllIl,
    IllIllIlIIlllllIl,
    IIlIlIllllIlIlIIl,
    IIIIIlllllIlIlIIl,
    IIIlIIIIIlllIIIll,
    IIllIlIIIllIlIIIl,
    IlIIIIlIIIllllIlI,
    IlIlllIlIIIllIIll,
    IIIlIlIIIllIlllIl,
    IIIIlllIlIlIlIIIl,
    IIIlllIllllIlllII,
    IIlIlIIlIIIlIllll,
    IIIllIlIIIlIllllI,
    IlIllIIIIlIlIlIIl,
    IllIlIIlllIIIlIIl,
    IlIlllllIlIllllIl,
    IlIlIIlIIIIIllIll,
):
    if rvc_globals.NotesIrHertz and IIlIlIllllIlIlIIl != _IlIIllllIIIlIlIlI:
        IIIllIlIIIlIllllI = (
            IIIllIIIlIIIlllII(IlIllIIIIlIlIlIIl) if IlIllIIIIlIlIlIIl else 50
        )
        IllIlIIlllIIIlIIl = (
            IIIllIIIlIIIlllII(IlIlllllIlIllllIl) if IlIlllllIlIllllIl else 1100
        )
        print(
            f"Converted Min pitch: freq - {IIIllIlIIIlIllllI}\nConverted Max pitch: freq - {IllIlIIlllIIIlIIl}"
        )
    else:
        IIIllIlIIIlIllllI = IIIllIlIIIlIllllI or 50
        IllIlIIlllIIIlIIl = IllIlIIlllIIIlIIl or 1100
    try:
        IlllIIIIlIIIlIlII, IlIIIIIllIIlIIllI = [
            IllIlIlIIllIlIIlI.strip(_IlIlIIIlllIllIlll)
            .strip(_IIlllllllllIllIll)
            .strip(_IIlIlIIllllIlIIlI)
            .strip(_IIlllllllllIllIll)
            .strip(_IlIlIIIlllIllIlll)
            for IllIlIlIIllIlIIlI in [IlllIIIIlIIIlIlII, IlIIIIIllIIlIIllI]
        ]
        os.makedirs(IlIIIIIllIIlIIllI, exist_ok=_IlllIIIlllIllIllI)
        IllIIIlIlIlIIllIl = (
            [
                os.path.join(IlllIIIIlIIIlIlII, IlllIllIllIlllIll)
                for IlllIllIllIlllIll in os.listdir(IlllIIIIlIIIlIlII)
            ]
            if IlllIIIIlIIIlIlII
            else [IIIlIIlIIllIIllll.name for IIIlIIlIIllIIllll in IllIIIlIlIlIIllIl]
        )
        IlIlIIIlIIIIIIllI = []
        for IllllIlIlIIIIIlll in IllIIIlIlIlIIllIl:
            IlIIlIllIllIllIIl, IIIlIlIlIIlIllIII = IlIIlIIlIIlllIIII(
                IlllllllIIllIIIlI,
                IllllIlIlIIIIIlll,
                _IllIIIllIIIllllIl,
                IllIllIlIIlllllIl,
                _IllIIIllIIIllllIl,
                IIlIlIllllIlIlIIl,
                IIIIIlllllIlIlIIl,
                IIIlIIIIIlllIIIll,
                IIllIlIIIllIlIIIl,
                IlIIIIlIIIllllIlI,
                IlIlllIlIIIllIIll,
                IIIlIlIIIllIlllIl,
                IIIIlllIlIlIlIIIl,
                IIlIlIIlIIIlIllll,
                IIIllIlIIIlIllllI,
                IlIllIIIIlIlIlIIl,
                IllIlIIlllIIIlIIl,
                IlIlllllIlIllllIl,
                IlIlIIlIIIIIllIll,
            )
            if "Success" in IlIIlIllIllIllIIl:
                try:
                    IlIIlIlIlllllIIlI, IlIlllIIlIlIlIlII = IIIlIlIlIIlIllIII
                    IllIIIlIIlIlllIll = os.path.splitext(
                        os.path.basename(IllllIlIlIIIIIlll)
                    )[0]
                    IllIIIlIIIIIllllI = (
                        f"{IlIIIIIllIIlIIllI}/{IllIIIlIIlIlllIll}.{IIIlllIllllIlllII}"
                    )
                    IllllIlIlIIIIIlll, IlIIIIIlIllIIIIIl = (
                        IllIIIlIIIIIllllI,
                        IIIlllIllllIlllII,
                    )
                    IllllIlIlIIIIIlll, IlIIIIIlIllIIIIIl = (
                        IllIIIlIIIIIllllI
                        if IIIlllIllllIlllII
                        in [
                            _IIlIlIlIIllIIIIll,
                            _IlIIlllIlIllIIlll,
                            _IIIlIllllIIlIIIIl,
                            _IlllllIllIIlIlIIl,
                            _IlIIllllIlIIlllII,
                            _IlIIllIllllIlIIII,
                        ]
                        else f"{IllIIIlIIIIIllllI}.wav",
                        IIIlllIllllIlllII,
                    )
                    IlIlIIlIlIIllIIIl(
                        IllllIlIlIIIIIlll, IlIlllIIlIlIlIlII, IlIIlIlIlllllIIlI
                    )
                    if os.path.exists(IllllIlIlIIIIIlll) and IlIIIIIlIllIIIIIl not in [
                        _IIlIlIlIIllIIIIll,
                        _IlIIlllIlIllIIlll,
                        _IIIlIllllIIlIIIIl,
                        _IlllllIllIIlIlIIl,
                        _IlIIllllIlIIlllII,
                        _IlIIllIllllIlIIII,
                    ]:
                        sys.stdout.write(
                            f"Running command: ffmpeg -i {IlIIIIlllllIllIIl(IllllIlIlIIIIIlll)} -vn {IlIIIIlllllIllIIl(IllllIlIlIIIIIlll[:-4]+_IIIIlllIlIIIlllll+IlIIIIIlIllIIIIIl)} -q:a 2 -y"
                        )
                        os.system(
                            f"ffmpeg -i {IlIIIIlllllIllIIl(IllllIlIlIIIIIlll)} -vn {IlIIIIlllllIllIIl(IllllIlIlIIIIIlll[:-4]+_IIIIlllIlIIIlllll+IlIIIIIlIllIIIIIl)} -q:a 2 -y"
                        )
                except:
                    IlIIlIllIllIllIIl += traceback.format_exc()
                    print(f"\nException encountered: {IlIIlIllIllIllIIl}")
            IlIlIIIlIIIIIIllI.append(
                f"{os.path.basename(IllllIlIlIIIIIlll)}->{IlIIlIllIllIllIIl}"
            )
            yield _IIlIlIIllllIlIIlI.join(IlIlIIIlIIIIIIllI)
        yield _IIlIlIIllllIlIIlI.join(IlIlIIIlIIIIIIllI)
    except:
        yield traceback.format_exc()


def IIIIllIllIllllIIl(
    IlllIIIllIllIIlll,
    IlIIllllllllIIIIl,
    IlIlllIlllIIIlIlI,
    IIIlIIIllIllllIlI,
    IlIIlIllllIIIIlIl,
    IlllIIllIIIlIIIll,
    IIlllllIIlIIIIllI,
    IIlIllIlIIIlIlIII,
):
    IIIIIIlIIlIlIIlII = "streams"
    IIIIIlllIlIIlllIl = "onnx_dereverb_By_FoxJoy"
    IIlIIlIIIlIIlllII = []
    if IIlIllIlIIIlIlIII == _IlIIIIIllllIlIlII:
        try:
            IlIIllllllllIIIIl, IlIlllIlllIIIlIlI, IlIIlIllllIIIIlIl = [
                IIllIIllIlllIlIlI.strip(_IlIlIIIlllIllIlll)
                .strip(_IIlllllllllIllIll)
                .strip(_IIlIlIIllllIlIIlI)
                .strip(_IIlllllllllIllIll)
                .strip(_IlIlIIIlllIllIlll)
                for IIllIIllIlllIlIlI in [
                    IlIIllllllllIIIIl,
                    IlIlllIlllIIIlIlI,
                    IlIIlIllllIIIIlIl,
                ]
            ]
            IlIIIlIlIlIIlIIIl = [
                os.path.join(IlIIllllllllIIIIl, IlIIlllIIIIIIIlll)
                for IlIIlllIIIIIIIlll in os.listdir(IlIIllllllllIIIIl)
                if IlIIlllIIIIIIIlll.endswith(tuple(IIlIIIlIIlIllllII))
            ]
            IIIlIllIlIllIIIlI = (
                MDXNetDereverb(15)
                if IlllIIIllIllIIlll == IIIIIlllIlIIlllIl
                else (
                    _audio_pre_ if "DeEcho" not in IlllIIIllIllIIlll else _audio_pre_new
                )(
                    agg=int(IlllIIllIIIlIIIll),
                    model_path=os.path.join(
                        IIllllIlIlllIIllI, IlllIIIllIllIIlll + _IlIlIIlIllllIIlIl
                    ),
                    device=IlIllIlIlIIlIlIll.device,
                    is_half=IlIllIlIlIIlIlIll.is_half,
                )
            )
            try:
                if IIIlIIIllIllllIlI != _IllIIIllIIIllllIl:
                    IIIlIIIllIllllIlI = [
                        IIllIIllIlIIIIlIl.name
                        for IIllIIllIlIIIIlIl in IIIlIIIllIllllIlI
                    ]
                else:
                    IIIlIIIllIllllIlI = IlIIIlIlIlIIlIIIl
            except:
                traceback.print_exc()
                IIIlIIIllIllllIlI = IlIIIlIlIlIIlIIIl
            print(IIIlIIIllIllllIlI)
            for IIIIlllIlllIlllII in IIIlIIIllIllllIlI:
                IIIlIlllllIlIIlII = os.path.join(IlIIllllllllIIIIl, IIIIlllIlllIlllII)
                IIllllIIlllIllIII, IIlllIIlIlIlIlIIl = 1, 0
                try:
                    IIlIlIIlIIlllllII = IIIIIIllIIIlllIIl.probe(
                        IIIlIlllllIlIIlII, cmd="ffprobe"
                    )
                    if (
                        IIlIlIIlIIlllllII[IIIIIIlIIlIlIIlII][0]["channels"] == 2
                        and IIlIlIIlIIlllllII[IIIIIIlIIlIlIIlII][0][_IIIIIlIlIlIIIlIll]
                        == "44100"
                    ):
                        IIllllIIlllIllIII = 0
                        IIIlIllIlIllIIIlI._path_audio_(
                            IIIlIlllllIlIIlII,
                            IlIIlIllllIIIIlIl,
                            IlIlllIlllIIIlIlI,
                            IIlllllIIlIIIIllI,
                        )
                        IIlllIIlIlIlIlIIl = 1
                except:
                    traceback.print_exc()
                if IIllllIIlllIllIII:
                    IIIIllllIIIIIlIIl = f"{IlllllllIIlIlIlIl}/{os.path.basename(IlIIIIlllllIllIIl(IIIlIlllllIlIIlII))}.reformatted.wav"
                    os.system(
                        f"ffmpeg -i {IlIIIIlllllIllIIl(IIIlIlllllIlIIlII)} -vn -acodec pcm_s16le -ac 2 -ar 44100 {IlIIIIlllllIllIIl(IIIIllllIIIIIlIIl)} -y"
                    )
                    IIIlIlllllIlIIlII = IIIIllllIIIIIlIIl
                try:
                    if not IIlllIIlIlIlIlIIl:
                        IIIlIllIlIllIIIlI._path_audio_(
                            IIIlIlllllIlIIlII,
                            IlIIlIllllIIIIlIl,
                            IlIlllIlllIIIlIlI,
                            IIlllllIIlIIIIllI,
                        )
                    IIlIIlIIIlIIlllII.append(
                        f"{os.path.basename(IIIlIlllllIlIIlII)}->Success"
                    )
                    yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
                except:
                    IIlIIlIIIlIIlllII.append(
                        f"{os.path.basename(IIIlIlllllIlIIlII)}->{traceback.format_exc()}"
                    )
                    yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
        except:
            IIlIIlIIIlIIlllII.append(traceback.format_exc())
            yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
        finally:
            try:
                if IlllIIIllIllIIlll == IIIIIlllIlIIlllIl:
                    del IIIlIllIlIllIIIlI.pred.model
                    del IIIlIllIlIllIIIlI.pred.model_
                else:
                    del IIIlIllIlIllIIIlI.model
                del IIIlIllIlIllIIIlI
            except:
                traceback.print_exc()
            print(_IlIlIlIIlllIIIllI)
            if IIIIIIlIIIIIIIIII.cuda.is_available():
                IIIIIIlIIIIIIIIII.cuda.empty_cache()
        yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
    elif IIlIllIlIIIlIlIII == _IIIIlIIllIIlllIll:
        try:
            IIlIIlIIIlIIlllII.append(
                IIIllIIllIIIIIIlI(
                    "Starting audio conversion... (This might take a moment)"
                )
            )
            yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
            IlIIllllllllIIIIl, IlIlllIlllIIIlIlI, IlIIlIllllIIIIlIl = [
                IlllIlllIllIllIll.strip(_IlIlIIIlllIllIlll)
                .strip(_IIlllllllllIllIll)
                .strip(_IIlIlIIllllIlIIlI)
                .strip(_IIlllllllllIllIll)
                .strip(_IlIlIIIlllIllIlll)
                for IlllIlllIllIllIll in [
                    IlIIllllllllIIIIl,
                    IlIlllIlllIIIlIlI,
                    IlIIlIllllIIIIlIl,
                ]
            ]
            IlIIIlIlIlIIlIIIl = [
                os.path.join(IlIIllllllllIIIIl, IlIIlllIlIIllIIIl)
                for IlIIlllIlIIllIIIl in os.listdir(IlIIllllllllIIIIl)
                if IlIIlllIlIIllIIIl.endswith(tuple(IIlIIIlIIlIllllII))
            ]
            try:
                if IIIlIIIllIllllIlI != _IllIIIllIIIllllIl:
                    IIIlIIIllIllllIlI = [
                        IIIlIllIIIlllllIl.name
                        for IIIlIllIIIlllllIl in IIIlIIIllIllllIlI
                    ]
                else:
                    IIIlIIIllIllllIlI = IlIIIlIlIlIIlIIIl
            except:
                traceback.print_exc()
                IIIlIIIllIllllIlI = IlIIIlIlIlIIlIIIl
            print(IIIlIIIllIllllIlI)
            IllIIIllllIlllIIl = _IlllIIIlllIllIllI
            IlIIIIIlllIIIIIll = _IlllIIIlllIllIllI
            IIlIlIIlIlIlIlIIl = _IlllIIIlllIllIllI
            IIllIIlIlllllIllI = 3072
            IlIIlIIIlIlIIIlII = 256
            IIIIlIIIIllIlIIlI = 7680
            IIlIlllIIlllllIII = _IlllIIIlllIllIllI
            IIlllIlllIlIlllIl = 1.025
            IlIIlIlIIIlIIllII = "Vocals_custom"
            IlIIIlllIlIIIlllI = "Instrumental_custom"
            IlIlIIlIllllIIIll = _IlllIIIlllIllIllI
            IlIlllIIIIIIIllll = id_to_ptm(IlllIIIllIllIIlll)
            IIlllIlllIlIlllIl = (
                IIlllIlllIlIlllIl
                if IIlIlllIIlllllIII or IIlIlIIlIlIlIlIIl
                else _IllIIIllIIIllllIl
            )
            IIlIIIllllIIlIIIl = prepare_mdx(
                IlIlllIIIIIIIllll,
                IIlIlIIlIlIlIlIIl,
                IIllIIlIlllllIllI,
                IlIIlIIIlIlIIIlII,
                IIIIlIIIIllIlIIlI,
                compensation=IIlllIlllIlIlllIl,
            )
            for IIIIlllIlllIlllII in IIIlIIIllIllllIlI:
                IIIIIlIIIllIlllll = (
                    IlIIlIlIIIlIIllII if IIlIlIIlIlIlIlIIl else _IllIIIllIIIllllIl
                )
                IIlIIIlIlIIlIlIll = (
                    IlIIIlllIlIIIlllI if IIlIlIIlIlIlIlIIl else _IllIIIllIIIllllIl
                )
                run_mdx(
                    IlIlllIIIIIIIllll,
                    IIlIIIllllIIlIIIl,
                    IIIIlllIlllIlllII,
                    IIlllllIIlIIIIllI,
                    diff=IllIIIllllIlllIIl,
                    suffix=IIIIIlIIIllIlllll,
                    diff_suffix=IIlIIIlIlIIlIlIll,
                    denoise=IlIIIIIlllIIIIIll,
                )
            if IlIlIIlIllllIIIll:
                print()
                print("[MDX-Net_Colab settings used]")
                print(f"Model used: {IlIlllIIIIIIIllll}")
                print(f"Model MD5: {mdx.MDX.get_hash(IlIlllIIIIIIIllll)}")
                print(f"Model parameters:")
                print(f"    -dim_f: {IIlIIIllllIIlIIIl.dim_f}")
                print(f"    -dim_t: {IIlIIIllllIIlIIIl.dim_t}")
                print(f"    -n_fft: {IIlIIIllllIIlIIIl.n_fft}")
                print(f"    -compensation: {IIlIIIllllIIlIIIl.compensation}")
                print()
                print("[Input file]")
                print("filename(s): ")
                for IlIIIIIlIIlIIlIII in IIIlIIIllIllllIlI:
                    print(f"    -{IlIIIIIlIIlIIlIII}")
                    IIlIIlIIIlIIlllII.append(
                        f"{os.path.basename(IlIIIIIlIIlIIlIII)}->Success"
                    )
                    yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
        except:
            IIlIIlIIIlIIlllII.append(traceback.format_exc())
            yield _IIlIlIIllllIlIIlI.join(IIlIIlIIIlIIlllII)
        finally:
            try:
                del IIlIIIllllIIlIIIl
            except:
                traceback.print_exc()
            print(_IlIlIlIIlllIIIllI)
            if IIIIIIlIIIIIIIIII.cuda.is_available():
                IIIIIIlIIIIIIIIII.cuda.empty_cache()


def IIIlllIllllIIIIll(IlllIIIllIlllIlIl, IIIIIlIIIIlllIIII, IlllIllIlIlIlIllI):
    global IIlIlIIIIllIlIlIl, IIllIllIlIlIlIlII, IIllIIIllIIllllll, IIIIIlIIIlIllllII, IIIIlIIlIIlllIIIl, IllIIIIIlllllIlIl, IIIlIlllIIIIlllIl
    if not IlllIIIllIlllIlIl:
        if IIIlIlllIIIIlllIl is not _IllIIIllIIIllllIl:
            print(_IlIlIlIIlllIIIllI)
            del (
                IIllIIIllIIllllll,
                IIlIlIIIIllIlIlIl,
                IIIIIlIIIlIllllII,
                IIIlIlllIIIIlllIl,
                IIllIllIlIlIlIlII,
            )
            IIIlIlllIIIIlllIl = (
                IIllIIIllIIllllll
            ) = (
                IIlIlIIIIllIlIlIl
            ) = (
                IIIIIlIIIlIllllII
            ) = IIIlIlllIIIIlllIl = IIllIllIlIlIlIlII = _IllIIIllIIIllllIl
            if IIIIIIlIIIIIIIIII.cuda.is_available():
                IIIIIIlIIIIIIIIII.cuda.empty_cache()
            IllllIlIlIIllIlII, IllIIIIIlllllIlIl = IIIIlIIlIIlllIIIl.get(
                _IllIIIlIllllIIllI, 1
            ), IIIIlIIlIIlllIIIl.get(_IllIllIllIllllIIl, _IlIIIIIIlIlIllIIl)
            IIllIIIllIIllllll = (
                (
                    SynthesizerTrnMs256NSFsid
                    if IllIIIIIlllllIlIl == _IlIIIIIIlIlIllIIl
                    else SynthesizerTrnMs768NSFsid
                )(
                    *IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII],
                    is_half=IlIllIlIlIIlIlIll.is_half,
                )
                if IllllIlIlIIllIlII == 1
                else (
                    SynthesizerTrnMs256NSFsid_nono
                    if IllIIIIIlllllIlIl == _IlIIIIIIlIlIllIIl
                    else SynthesizerTrnMs768NSFsid_nono
                )(*IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII])
            )
            del IIllIIIllIIllllll, IIIIlIIlIIlllIIIl
            if IIIIIIlIIIIIIIIII.cuda.is_available():
                IIIIIIlIIIIIIIIII.cuda.empty_cache()
            IIIIlIIlIIlllIIIl = _IllIIIllIIIllllIl
        return (
            {
                _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
        ) * 3
    print(f"loading {IlllIIIllIlllIlIl}")
    IIIIlIIlIIlllIIIl = IIIIIIlIIIIIIIIII.load(
        IlllIIIllIlllIlIl, map_location=_IIlllIIIIIlllIIlI
    )
    IIllIllIlIlIlIlII = IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII][-1]
    IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII][-3] = IIIIlIIlIIlllIIIl[_IIIlllIlIlIIIlIll][
        _IllIIIIlIIllIIlIl
    ].shape[0]
    if IIIIlIIlIIlllIIIl.get(_IllIIIlIllllIIllI, 1) == 0:
        IIIIIlIIIIlllIIII = IlllIllIlIlIlIllI = {
            _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
            _IlllIIllllIIIIlIl: 0.5,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
    else:
        IIIIIlIIIIlllIIII = {
            _IIIllIllllIlIllIl: _IlllIIIlllIllIllI,
            _IlllIIllllIIIIlIl: IIIIIlIIIIlllIIII,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
        IlllIllIlIlIlIllI = {
            _IIIllIllllIlIllIl: _IlllIIIlllIllIllI,
            _IlllIIllllIIIIlIl: IlllIllIlIlIlIllI,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
    IllIIIIIlllllIlIl = IIIIlIIlIIlllIIIl.get(_IllIllIllIllllIIl, _IlIIIIIIlIlIllIIl)
    IIllIIIllIIllllll = (
        (
            SynthesizerTrnMs256NSFsid
            if IllIIIIIlllllIlIl == _IlIIIIIIlIlIllIIl
            else SynthesizerTrnMs768NSFsid
        )(*IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII], is_half=IlIllIlIlIIlIlIll.is_half)
        if IIIIlIIlIIlllIIIl.get(_IllIIIlIllllIIllI, 1) == 1
        else (
            SynthesizerTrnMs256NSFsid_nono
            if IllIIIIIlllllIlIl == _IlIIIIIIlIlIllIIl
            else SynthesizerTrnMs768NSFsid_nono
        )(*IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII])
    )
    del IIllIIIllIIllllll.enc_q
    print(
        IIllIIIllIIllllll.load_state_dict(
            IIIIlIIlIIlllIIIl[_IIIlllIlIlIIIlIll], strict=_IIIIlIlIIIlIlIlll
        )
    )
    IIllIIIllIIllllll.eval().to(IlIllIlIlIIlIlIll.device)
    IIllIIIllIIllllll = (
        IIllIIIllIIllllll.half()
        if IlIllIlIlIIlIlIll.is_half
        else IIllIIIllIIllllll.float()
    )
    IIIIIlIIIlIllllII = VC(IIllIllIlIlIlIlII, IlIllIlIlIIlIlIll)
    IIlIlIIIIllIlIlIl = IIIIlIIlIIlllIIIl[_IIlIIlIIlIlIIIlII][-3]
    return (
        {
            _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
            "maximum": IIlIlIIIIllIlIlIl,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
        IIIIIlIIIIlllIIII,
        IlllIllIlIlIlIllI,
    )


def IlllIIIllllIlllIl():
    IllIlIIIlIIIlIlll = [
        os.path.join(IIIllllIlIlIllIll, IIllIIlIlIIIIIllI)
        for (IIIllllIlIlIllIll, _IlllllIlllIIIIlll, IIllIllllIIIIllIl) in os.walk(
            IllIIllIlllIlIlIl
        )
        for IIllIIlIlIIIIIllI in IIllIllllIIIIllIl
        if IIllIIlIlIIIIIllI.endswith((_IlIlIIlIllllIIlIl, _IlllllIIllllIlIII))
    ]
    IIlllIlllIIllllll = [
        os.path.join(IIlIIIIlIIIlllIIl, IlIIIIlllIIlllIlI)
        for (IIlIIIIlIIIlllIIl, _IIllIlllIllllllll, IlllIIIllIIIIlllI) in os.walk(
            IlIlIIIllIIllIlll, topdown=_IIIIlIlIIIlIlIlll
        )
        for IlIIIIlllIIlllIlI in IlllIIIllIIIIlllI
        if IlIIIIlllIIlllIlI.endswith(_IlIIllIllIllIIIIl)
        and _IlIllIlIIIlllIIIl not in IlIIIIlllIIlllIlI
    ]
    IlIIlllIIlIIIIIll = [
        os.path.join(IllIIlIIIllIllIII, IlllIlIIIllIIIllI)
        for IlllIlIIIllIIIllI in os.listdir(
            os.path.join(IlIIIllllllIllIll, _IIlIIIIIlIlIIIllI)
        )
    ]
    return (
        {
            _IIllIIIlIIlllIlII: sorted(IllIlIIIlIIIlIlll),
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
        {
            _IIllIIIlIIlllIlII: sorted(IIlllIlllIIllllll),
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
        {
            _IIllIIIlIIlllIlII: sorted(IlIIlllIIlIIIIIll),
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
    )


def IlIllIllIIIIllIII():
    IIIllIIIlIIllIIIl = [
        os.path.join(IlIIlIlIIlllIlIII, IIlIIIIlIIllIIIIl)
        for (IlIIlIlIIlllIlIII, _IIIIIIlIllIllIIIl, IIIIllIIIlllIlIII) in os.walk(
            IllIIllIlllIlIlIl
        )
        for IIlIIIIlIIllIIIIl in IIIIllIIIlllIlIII
        if IIlIIIIlIIllIIIIl.endswith((_IlIlIIlIllllIIlIl, _IlllllIIllllIlIII))
    ]
    IlIIIllIlIIlIIlll = [
        os.path.join(IlIlIlIIlIlIIIlII, IIIIIlIIlllIlllII)
        for (IlIlIlIIlIlIIIlII, _IIIIIlIIIlIIIIIII, IIlllIIIllIIIllII) in os.walk(
            IlIlIIIllIIllIlll, topdown=_IIIIlIlIIIlIlIlll
        )
        for IIIIIlIIlllIlllII in IIlllIIIllIIIllII
        if IIIIIlIIlllIlllII.endswith(_IlIIllIllIllIIIIl)
        and _IlIllIlIIIlllIIIl not in IIIIIlIIlllIlllII
    ]
    return {
        _IIllIIIlIIlllIlII: sorted(IIIllIIIlIIllIIIl),
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }, {
        _IIllIIIlIIlllIlII: sorted(IlIIIllIlIIlIIlll),
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }


def IllIIllIllIlIIIll():
    IlIllIIlllllIllII = [
        os.path.join(IllIIlIIIllIllIII, IIIIIlIllIIlIlllI)
        for IIIIIlIllIIlIlllI in os.listdir(
            os.path.join(IlIIIllllllIllIll, _IIlIIIIIlIlIIIllI)
        )
    ]
    IIllIllIlllIIIIII = [
        os.path.join(IllIIIIllIIIIlIll, IlIlIlIIlIIlllIlI)
        for IlIlIlIIlIIlllIlI in os.listdir(
            os.path.join(IlIIIllllllIllIll, _IlIIlllllIlllIlIl)
        )
    ]
    return {
        _IIllIIIlIIlllIlII: sorted(IIllIllIlllIIIIII),
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }, {
        _IIllIIIlIIlllIlII: sorted(IlIllIIlllllIllII),
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }


IIIllIllIllIIIllI = {
    _IllIIIIIIlIIIIlll: 32000,
    _IIIllIIIlIIllIlII: 40000,
    _IlIlllllIIIIIlIlI: 48000,
}


def IlllIIIIlIIIIlIIl(IIIIIlIlIIIIllllI, IIIIIlIlIlIlIlllI):
    while IIIIIlIlIlIlIlllI.poll() is _IllIIIllIIIllllIl:
        time.sleep(0.5)
    IIIIIlIlIIIIllllI[0] = _IlllIIIlllIllIllI


def IIlIlIIIlIlllIIll(IlIIlIllIIllIlllI, IIIIlllllIlIIllII):
    while not all(
        IlIIllIIllIlIIlII.poll() is not _IllIIIllIIIllllIl
        for IlIIllIIllIlIIlII in IIIIlllllIlIIllII
    ):
        time.sleep(0.5)
    IlIIlIllIIllIlllI[0] = _IlllIIIlllIllIllI


def IlIllllllllIlllIl(IlIlIIlIIIlIIIlll, IlllIIllllIIlIlII, IllIIllIllllIIlII):
    global IlllIIIllIIIlIllI, IIIIllIIIIIlIlllI, IlIIIIlllIIIllllI
    IlllIIIllIIIlIllI = IlIlIIlIIIlIIIlll
    IIIIllIIIIIlIlllI = IlllIIllllIIlIlII
    IlIIIIlllIIIllllI = IllIIllIllllIIlII
    rvc_globals.DoFormant = IlIlIIlIIIlIIIlll
    rvc_globals.Quefrency = IlllIIllllIIlIlII
    rvc_globals.Timbre = IllIIllIllllIIlII
    IIllIIllIllIIllll = {
        _IIIllIllllIlIllIl: IlllIIIllIIIlIllI,
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }
    return (
        {_IlllIIllllIIIIlIl: IlllIIIllIIIlIllI, _IIllIIllllllIllII: _IlllllIlIIlIlIlII},
    ) + (IIllIIllIllIIllll,) * 6


def IllIIIllIlIllllll(IIllIIIIlIIIllIll, IIIIIIllllllllIll):
    global IIIIllIIIIIlIlllI, IlIIIIlllIIIllllI, IlllIIIllIIIlIllI
    IIIIllIIIIIlIlllI = IIllIIIIlIIIllIll
    IlIIIIlllIIIllllI = IIIIIIllllllllIll
    IlllIIIllIIIlIllI = _IlllIIIlllIllIllI
    rvc_globals.DoFormant = _IlllIIIlllIllIllI
    rvc_globals.Quefrency = IIllIIIIlIIIllIll
    rvc_globals.Timbre = IIIIIIllllllllIll
    return {
        _IlllIIllllIIIIlIl: IIIIllIIIIIlIlllI,
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }, {_IlllIIllllIIIIlIl: IlIIIIlllIIIllllI, _IIllIIllllllIllII: _IlllllIlIIlIlIlII}


def IIIIIIIIIIIIlIlll(IlIIIIlIlIIlIlIlI, IIlIllllIlllIIIIl, IIlllIlIlIllIlllI):
    if IlIIIIlIlIIlIlIlI:
        with open(IlIIIIlIlIIlIlIlI, _IlIIIIlllIlIIllll) as IllIllIIllIIIllII:
            IlllIIIIIIlIlIIII = IllIllIIllIIIllII.readlines()
            IIlIllllIlllIIIIl, IIlllIlIlIllIlllI = (
                IlllIIIIIIlIlIIII[0].strip(),
                IlllIIIIIIlIlIIII[1],
            )
        IllIIIllIlIllllll(IIlIllllIlllIIIIl, IIlllIlIlIllIlllI)
    else:
        IIlIllllIlllIIIIl, IIlllIlIlIllIlllI = IllllIIlIlIlIlIll(
            IlIIIIlIlIIlIlIlI, IIlIllllIlllIIIIl, IIlllIlIlIllIlllI
        )
    return (
        {
            _IIllIIIlIIlllIlII: IIIllllIIlIlllIlI(),
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
        {_IlllIIllllIIIIlIl: IIlIllllIlllIIIIl, _IIllIIllllllIllII: _IlllllIlIIlIlIlII},
        {_IlllIIllllIIIIlIl: IIlllIlIlIllIlllI, _IIllIIllllllIllII: _IlllllIlIIlIlIlII},
    )


def IllIIllIllIllIlll(
    IIlIIllIllllllIII, IIlIlIIllIlIIIIII, IlIlIlIllllllIIIl, IlllllIllllIIlIII
):
    IlIlIlIllllllIIIl = IIIllIllIllIIIllI[IlIlIlIllllllIIIl]
    IIlIlllIllIllIIlI = os.path.join(
        IlIIIllllllIllIll, _IIIlIlIlIIIIllllI, IIlIlIIllIlIIIIII
    )
    IIlllllIIIIlIIIII = os.path.join(IIlIlllIllIllIIlI, "preprocess.log")
    os.makedirs(IIlIlllIllIllIIlI, exist_ok=_IlllIIIlllIllIllI)
    with open(IIlllllIIIIlIIIII, "w") as IlllIIIIIIIlllIlI:
        0
    IIIlIlIlIIlIIIIII = f"{IlIllIlIlIIlIlIll.python_cmd} trainset_preprocess_pipeline_print.py {IIlIIllIllllllIII} {IlIIIIlllllIllIIl(IlIlIlIllllllIIIl)} {IlIIIIlllllIllIIl(IlllllIllllIIlIII)} {IIlIlllIllIllIIlI} {IlIIIIlllllIllIIl(IlIllIlIlIIlIlIll.noparallel)}"
    print(IIIlIlIlIIlIIIIII)
    IlIllIlIlIIIlIIlI = Popen(IIIlIlIlIIlIIIIII, shell=_IlllIIIlllIllIllI)
    IlIllIIIlllIIllll = [_IIIIlIlIIIlIlIlll]
    threading.Thread(
        target=IlllIIIIlIIIIlIIl, args=(IlIllIIIlllIIllll, IlIllIlIlIIIlIIlI)
    ).start()
    while not IlIllIIIlllIIllll[0]:
        with open(IIlllllIIIIlIIIII, _IlIIIIlllIlIIllll) as IlllIIIIIIIlllIlI:
            yield IlllIIIIIIIlllIlI.read()
        time.sleep(1)
    with open(IIlllllIIIIlIIIII, _IlIIIIlllIlIIllll) as IlllIIIIIIIlllIlI:
        IlIIlllIIIIllIlIl = IlllIIIIIIIlllIlI.read()
    print(IlIIlllIIIIllIlIl)
    yield IlIIlllIIIIllIlIl


def IllllIllIllIIIllI(
    IIlIllIllIIlIllll,
    IIllIIIllIIlIIlII,
    IlIlIIlllIIIIIllI,
    IllIllllIIIlllIll,
    IlIllIllllIIIIlll,
    IIIlIlIlIIlIIIIIl,
    IllIlIllIlIllllIl,
):
    IIlIllIllIIlIllll = IIlIllIllIIlIllll.split("-")
    IllIIlIIIlIIlIIII = f"{IlIIIllllllIllIll}/logs/{IlIllIllllIIIIlll}"
    IlIIllIlIIIIIIlII = f"{IllIIlIIIlIIlIIII}/extract_fl_feature.log"
    os.makedirs(IllIIlIIIlIIlIIII, exist_ok=_IlllIIIlllIllIllI)
    with open(IlIIllIlIIIIIIlII, "w") as IIIIlIlllIllllllI:
        0
    if IllIllllIIIlllIll:
        IlIIllIlIlllIllIl = f"{IlIllIlIlIIlIlIll.python_cmd} extract_fl_print.py {IllIIlIIIlIIlIIII} {IlIIIIlllllIllIIl(IIllIIIllIIlIIlII)} {IlIIIIlllllIllIIl(IlIlIIlllIIIIIllI)} {IlIIIIlllllIllIIl(IllIlIllIlIllllIl)}"
        print(IlIIllIlIlllIllIl)
        IIIlIllIlllIlllll = Popen(
            IlIIllIlIlllIllIl, shell=_IlllIIIlllIllIllI, cwd=IlIIIllllllIllIll
        )
        IIIIIlIllIlIlIlII = [_IIIIlIlIIIlIlIlll]
        threading.Thread(
            target=IlllIIIIlIIIIlIIl, args=(IIIIIlIllIlIlIlII, IIIlIllIlllIlllll)
        ).start()
        while not IIIIIlIllIlIlIlII[0]:
            with open(IlIIllIlIIIIIIlII, _IlIIIIlllIlIIllll) as IIIIlIlllIllllllI:
                yield IIIIlIlllIllllllI.read()
            time.sleep(1)
    IlIIllllllIIIIllI = len(IIlIllIllIIlIllll)
    IIlIlIllllIlIlIll = []
    for IIIllIIIIIIlIllll, IIlIlllIIlIIlIIlI in enumerate(IIlIllIllIIlIllll):
        IlIIllIlIlllIllIl = f"{IlIllIlIlIIlIlIll.python_cmd} extract_feature_print.py {IlIIIIlllllIllIIl(IlIllIlIlIIlIlIll.device)} {IlIIIIlllllIllIIl(IlIIllllllIIIIllI)} {IlIIIIlllllIllIIl(IIIllIIIIIIlIllll)} {IlIIIIlllllIllIIl(IIlIlllIIlIIlIIlI)} {IllIIlIIIlIIlIIII} {IlIIIIlllllIllIIl(IIIlIlIlIIlIIIIIl)}"
        print(IlIIllIlIlllIllIl)
        IIIlIllIlllIlllll = Popen(
            IlIIllIlIlllIllIl, shell=_IlllIIIlllIllIllI, cwd=IlIIIllllllIllIll
        )
        IIlIlIllllIlIlIll.append(IIIlIllIlllIlllll)
    IIIIIlIllIlIlIlII = [_IIIIlIlIIIlIlIlll]
    threading.Thread(
        target=IIlIlIIIlIlllIIll, args=(IIIIIlIllIlIlIlII, IIlIlIllllIlIlIll)
    ).start()
    while not IIIIIlIllIlIlIlII[0]:
        with open(IlIIllIlIIIIIIlII, _IlIIIIlllIlIIllll) as IIIIlIlllIllllllI:
            yield IIIIlIlllIllllllI.read()
        time.sleep(1)
    with open(IlIIllIlIIIIIIlII, _IlIIIIlllIlIIllll) as IIIIlIlllIllllllI:
        IIlllIIIIIlllIIlI = IIIIlIlllIllllllI.read()
    print(IIlllIIIIIlllIIlI)
    yield IIlllIIIIIlllIIlI


def IllllllIIIIlIIIIl(IIIIIIIlIlllIIllI, IIlllllIIIlIIIIIl, IlIlllIlllIIllllI):
    IlIlIIIlllIIlllll = (
        "" if IlIlllIlllIIllllI == _IlIIIIIIlIlIllIIl else _IIlllIIllIIlllIll
    )
    IIIllIIIllIlIlIII = _IllIIIlIllllIIllI if IIlllllIIIlIIIIIl else ""
    IIllIlIlIlIlIIIll = {_IlIIIIIIllllIIlll: "", _IIIlIIIlIIIIlllll: ""}
    for IllIIllllllIlIlIl in IIllIlIlIlIlIIIll:
        IIllllIIlIIllllII = f"/kaggle/input/ax-rmf/pretrained{IlIlIIIlllIIlllll}/{IIIllIIIllIlIlIII}{IllIIllllllIlIlIl}{IIIIIIIlIlllIIllI}.pth"
        if os.access(IIllllIIlIIllllII, os.F_OK):
            IIllIlIlIlIlIIIll[IllIIllllllIlIlIl] = IIllllIIlIIllllII
        else:
            print(f"{IIllllIIlIIllllII} doesn't exist, will not use pretrained model.")
    return IIllIlIlIlIlIIIll[_IlIIIIIIllllIIlll], IIllIlIlIlIlIIIll[_IIIlIIIlIIIIlllll]


def IlllIIIlIIIllllII(IIIIlIIlllIIIllll, IlllIlIlllIIIlllI, IlllIlIlIIIIIlllI):
    IIIlIlIllIIlIllIl = (
        "" if IlllIlIlIIIIIlllI == _IlIIIIIIlIlIllIIl else _IIlllIIllIIlllIll
    )
    IIIIlIIlllIIIllll = (
        _IIIllIIIlIIllIlII
        if IIIIlIIlllIIIllll == _IllIIIIIIlIIIIlll
        and IlllIlIlIIIIIlllI == _IlIIIIIIlIlIllIIl
        else IIIIlIIlllIIIllll
    )
    IlIIIIllllIlIllII = (
        {
            _IIllIIIlIIlllIlII: [_IIIllIIIlIIllIlII, _IlIlllllIIIIIlIlI],
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            _IlllIIllllIIIIlIl: IIIIlIIlllIIIllll,
        }
        if IlllIlIlIIIIIlllI == _IlIIIIIIlIlIllIIl
        else {
            _IIllIIIlIIlllIlII: [
                _IIIllIIIlIIllIlII,
                _IlIlllllIIIIIlIlI,
                _IllIIIIIIlIIIIlll,
            ],
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            _IlllIIllllIIIIlIl: IIIIlIIlllIIIllll,
        }
    )
    IIIlIIlIlllIIIIII = _IllIIIlIllllIIllI if IlllIlIlllIIIlllI else ""
    IIlIIlIlIIllIllll = {_IlIIIIIIllllIIlll: "", _IIIlIIIlIIIIlllll: ""}
    for IlllIlIlIlllIIIll in IIlIIlIlIIllIllll:
        IlllIIlIlIIIlIlll = f"/kaggle/input/ax-rmf/pretrained{IIIlIlIllIIlIllIl}/{IIIlIIlIlllIIIIII}{IlllIlIlIlllIIIll}{IIIIlIIlllIIIllll}.pth"
        if os.access(IlllIIlIlIIIlIlll, os.F_OK):
            IIlIIlIlIIllIllll[IlllIlIlIlllIIIll] = IlllIIlIlIIIlIlll
        else:
            print(f"{IlllIIlIlIIIlIlll} doesn't exist, will not use pretrained model.")
    return (
        IIlIIlIlIIllIllll[_IlIIIIIIllllIIlll],
        IIlIIlIlIIllIllll[_IIIlIIIlIIIIlllll],
        IlIIIIllllIlIllII,
    )


def IIlllIIIlIlIllIll(IlIlIIIIlIlIllIIl, IIIllIlllllIIIlll, IllIllIIlIlIIlIII):
    IIlllIIIIIIIIIIlI = (
        "" if IllIllIIlIlIIlIII == _IlIIIIIIlIlIllIIl else _IIlllIIllIIlllIll
    )
    IIIlIllIlIIlIIIII = "/kaggle/input/ax-rmf/pretrained%s/f0%s%s.pth"
    IIIIlIIlIIlIIlIll = {_IlIIIIIIllllIIlll: "", _IIIlIIIlIIIIlllll: ""}
    for IlIllIlIlIllIIlIl in IIIIlIIlIIlIIlIll:
        IIIlIIIIIIIIllIll = IIIlIllIlIIlIIIII % (
            IIlllIIIIIIIIIIlI,
            IlIllIlIlIllIIlIl,
            IIIllIlllllIIIlll,
        )
        if os.access(IIIlIIIIIIIIllIll, os.F_OK):
            IIIIlIIlIIlIIlIll[IlIllIlIlIllIIlIl] = IIIlIIIIIIIIllIll
        else:
            print(IIIlIIIIIIIIllIll, "doesn't exist, will not use pretrained model")
    return (
        {_IIIllIllllIlIllIl: IlIlIIIIlIlIllIIl, _IIllIIllllllIllII: _IlllllIlIIlIlIlII},
        IIIIlIIlIIlIIlIll[_IlIIIIIIllllIIlll],
        IIIIlIIlIIlIIlIll[_IIIlIIIlIIIIlllll],
        {_IIIllIllllIlIllIl: IlIlIIIIlIlIllIIl, _IIllIIllllllIllII: _IlllllIlIIlIlIlII},
    )


global IIllllIIlIlIlIlll


def IIllIIllIlIIlllII(IIIIllIIIIIIlIIII, IIIIllIlIllllIIII):
    IIlllIIIlllllllIl = 1
    IlllIlIlIIlIIIlII = os.path.join(IIIIllIIIIIIlIIII, "1_16k_wavs")
    if os.path.isdir(IlllIlIlIIlIIIlII):
        IllIIlIIIllIlllIl = len(glob1(IlllIlIlIIlIIIlII, "*.wav"))
        if IllIIlIIIllIlllIl > 0:
            IIlllIIIlllllllIl = IlIlllIIIlIlIIlll.ceil(
                IllIIlIIIllIlllIl / IIIIllIlIllllIIII
            )
            if IIlllIIIlllllllIl > 1:
                IIlllIIIlllllllIl += 1
    return IIlllIIIlllllllIl


global IIIIllIllIllIIIll, IlIlIIllIllIIIIII


def IllIIlllIlllIIIIl(
    IlIlllIIlllIIIlll,
    IIlIlIIIIIlIllIlI,
    IllIIlIlIlIIIlIII,
    IIlIIlIIIIIIlllIl,
    IlllllllIIIIlIllI,
    IIllIlIIllllIllll,
    IllIllIIIIlIllIll,
    IIIIIlIIlIIIllIIl,
    IIllIllllIlllIIll,
    IIlIIlIlIIlllIlIl,
    IllIIIllIlllllIlI,
    IlIIIIlllIlIIlllI,
    IllllIlIlIlllIIIl,
    IllllllIlIlIlIlIl,
):
    with open(_IlIllIlllIIIIIIIl, "w+") as IlIlIIIIIIIllllll:
        IlIlIIIIIIIllllll.write("False")
    IllllllIllIIlIlIl = os.path.join(
        IlIIIllllllIllIll, _IIIlIlIlIIIIllllI, IlIlllIIlllIIIlll
    )
    os.makedirs(IllllllIllIIlIlIl, exist_ok=_IlllIIIlllIllIllI)
    IIIIlllllIlIlIlll = os.path.join(IllllllIllIIlIlIl, "0_gt_wavs")
    IIllllIIllllIllll = "256" if IllllllIlIlIlIlIl == _IlIIIIIIlIlIllIIl else "768"
    IIIllIlIlIllllIII = os.path.join(IllllllIllIIlIlIl, f"3_feature{IIllllIIllllIllll}")
    IllIIlIlllllIlllI = IIllIIllIlIIlllII(IllllllIllIIlIlIl, IllIllIIIIlIllIll)
    IIIllIIIlIIllIIlI = [IIIIlllllIlIlIlll, IIIllIlIlIllllIII]
    if IllIIlIlIlIIIlIII:
        IlIIllIlIIIlIllll = f"{IllllllIllIIlIlIl}/2a_f0"
        IllIIllIllIIlIlll = f"{IllllllIllIIlIlIl}/2b-f0nsf"
        IIIllIIIlIIllIIlI.extend([IlIIllIlIIIlIllll, IllIIllIllIIlIlll])
    IlIIIlIIIlIlIIIlI = set(
        IIIIlIIlllIIlIlII.split(_IIIIlllIlIIIlllll)[0]
        for IlllIlIIIIIIIIIII in IIIllIIIlIIllIIlI
        for IIIIlIIlllIIlIlII in os.listdir(IlllIlIIIIIIIIIII)
    )

    def IIlllIlIlIIlIlllI(IIIlIllIlllllIlll):
        IIlllIllIIIIlIIIl = [IIIIlllllIlIlIlll, IIIllIlIlIllllIII]
        if IllIIlIlIlIIIlIII:
            IIlllIllIIIIlIIIl.extend([IlIIllIlIIIlIllll, IllIIllIllIIlIlll])
        return "|".join(
            [
                IIIllIllIlllIIlII.replace("\\", "\\\\")
                + "/"
                + IIIlIllIlllllIlll
                + (
                    ".wav.npy"
                    if IIIllIllIlllIIlII in [IlIIllIlIIIlIllll, IllIIllIllIIlIlll]
                    else ".wav"
                    if IIIllIllIlllIIlII == IIIIlllllIlIlIlll
                    else ".npy"
                )
                for IIIllIllIlllIIlII in IIlllIllIIIIlIIIl
            ]
        )

    IIlIIlIIIIllllIll = [
        f"{IIlllIlIlIIlIlllI(IlIIllIIlIlIlIlII)}|{IIlIIlIIIIIIlllIl}"
        for IlIIllIIlIlIlIlII in IlIIIlIIIlIlIIIlI
    ]
    IIIlIIlIIIIIlllII = f"{IlIIIllllllIllIll}/logs/mute"
    for _IIlIlIlIllIIllIII in range(2):
        IlIlllIllIIIIIIII = f"{IIIlIIlIIIIIlllII}/0_gt_wavs/mute{IIlIlIIIIIlIllIlI}.wav|{mute_dir}/3_feature{IIllllIIllllIllll}/mute.npy"
        if IllIIlIlIlIIIlIII:
            IlIlllIllIIIIIIII += f"|{IIIlIIlIIIIIlllII}/2a_f0/mute.wav.npy|{mute_dir}/2b-f0nsf/mute.wav.npy"
        IIlIIlIIIIllllIll.append(IlIlllIllIIIIIIII + f"|{IIlIIlIIIIIIlllIl}")
    shuffle(IIlIIlIIIIllllIll)
    with open(f"{IllllllIllIIlIlIl}/filelist.txt", "w") as IlIIlIIIIlIllIlII:
        IlIIlIIIIlIllIlII.write(_IIlIlIIllllIlIIlI.join(IIlIIlIIIIllllIll))
    print("write filelist done")
    print("use gpus:", IllIIIllIlllllIlI)
    if IIllIllllIlllIIll == "":
        print("no pretrained Generator")
    if IIlIIlIlIIlllIlIl == "":
        print("no pretrained Discriminator")
    IIlllIlIlIIlIllIl = f"-pg {IIllIllllIlllIIll}" if IIllIllllIlllIIll else ""
    IllIlIIllllIlllIl = f"-pd {IIlIIlIlIIlllIlIl}" if IIlIIlIlIIlllIlIl else ""
    IIIllIIIlllIIIIII = f"{IlIllIlIlIIlIlIll.python_cmd} train_nsf_sim_cache_sid_load_pretrain.py -e {IlIlllIIlllIIIlll} -sr {IIlIlIIIIIlIllIlI} -f0 {int(IllIIlIlIlIIIlIII)} -bs {IllIllIIIIlIllIll} -g {IllIIIllIlllllIlI if IllIIIllIlllllIlI is not _IllIIIllIIIllllIl else''} -te {IIllIlIIllllIllll} -se {IlllllllIIIIlIllI} {IIlllIlIlIIlIllIl} {IllIlIIllllIlllIl} -l {int(IIIIIlIIlIIIllIIl)} -c {int(IlIIIIlllIlIIlllI)} -sw {int(IllllIlIlIlllIIIl)} -v {IllllllIlIlIlIlIl} -li {IllIIlIlllllIlllI}"
    print(IIIllIIIlllIIIIII)
    global IIIIllIIlIllIIllI
    IIIIllIIlIllIIllI = Popen(
        IIIllIIIlllIIIIII, shell=_IlllIIIlllIllIllI, cwd=IlIIIllllllIllIll
    )
    global IIIIllIllIllIIIll
    IIIIllIllIllIIIll = IIIIllIIlIllIIllI.pid
    IIIIllIIlIllIIllI.wait()
    return (
        IIIllIIllIIIIIIlI("Training is done, check train.log"),
        {
            _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
        {
            _IIIllIllllIlIllIl: _IlllIIIlllIllIllI,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        },
    )


def IIIIllIllllllIllI(IlIIIIIIIIllIIllI, IIIlIIlIIIIIIllII):
    IlIlIlllIIllllIII = os.path.join(
        IlIIIllllllIllIll, _IIIlIlIlIIIIllllI, IlIIIIIIIIllIIllI
    )
    os.makedirs(IlIlIlllIIllllIII, exist_ok=_IlllIIIlllIllIllI)
    IlIIllllIIIIIlIIl = "256" if IIIlIIlIIIIIIllII == _IlIIIIIIlIlIllIIl else "768"
    IlIlllllIIlllIlII = os.path.join(IlIlIlllIIllllIII, f"3_feature{IlIIllllIIIIIlIIl}")
    if not os.path.exists(IlIlllllIIlllIlII) or len(os.listdir(IlIlllllIIlllIlII)) == 0:
        return "请先进行特征提取!"
    IlIIIlIIllIlIlIIl = [
        IIlIlllllIlIlIIlI.load(os.path.join(IlIlllllIIlllIlII, IIllIIlIlllIlIllI))
        for IIllIIlIlllIlIllI in sorted(os.listdir(IlIlllllIIlllIlII))
    ]
    IIIlIIIllllIIIlII = IIlIlllllIlIlIIlI.concatenate(IlIIIlIIllIlIlIIl, 0)
    IIlIlllllIlIlIIlI.random.shuffle(IIIlIIIllllIIIlII)
    IlllllllIIllllIlI = []
    if IIIlIIIllllIIIlII.shape[0] > 2 * 10**5:
        IlllllllIIllllIlI.append(
            "Trying doing kmeans %s shape to 10k centers." % IIIlIIIllllIIIlII.shape[0]
        )
        yield _IIlIlIIllllIlIIlI.join(IlllllllIIllllIlI)
        try:
            IIIlIIIllllIIIlII = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=_IlllIIIlllIllIllI,
                    batch_size=256 * IlIllIlIlIIlIlIll.n_cpu,
                    compute_labels=_IIIIlIlIIIlIlIlll,
                    init="random",
                )
                .fit(IIIlIIIllllIIIlII)
                .cluster_centers_
            )
        except Exception as IllIIlllIIIllIIlI:
            IlllllllIIllllIlI.append(str(IllIIlllIIIllIIlI))
            yield _IIlIlIIllllIlIIlI.join(IlllllllIIllllIlI)
    IIlIlllllIlIlIIlI.save(
        os.path.join(IlIlIlllIIllllIII, "total_fea.npy"), IIIlIIIllllIIIlII
    )
    IlIlIlIllIIlIIIIl = min(
        int(16 * IIlIlllllIlIlIIlI.sqrt(IIIlIIIllllIIIlII.shape[0])),
        IIIlIIIllllIIIlII.shape[0] // 39,
    )
    IlllllllIIllllIlI.append("%s,%s" % (IIIlIIIllllIIIlII.shape, IlIlIlIllIIlIIIIl))
    yield _IIlIlIIllllIlIIlI.join(IlllllllIIllllIlI)
    IIIlllllllIIIlIIl = faiss.index_factory(
        int(IlIIllllIIIIIlIIl), f"IVF{IlIlIlIllIIlIIIIl},Flat"
    )
    IllIIIllIIIIIIIll = faiss.extract_index_ivf(IIIlllllllIIIlIIl)
    IllIIIllIIIIIIIll.nprobe = 1
    IIIlllllllIIIlIIl.train(IIIlIIIllllIIIlII)
    IIIlIIIlIIllIlllI = f"{IlIlIlllIIllllIII}/trained_IVF{IlIlIlIllIIlIIIIl}_Flat_nprobe_{IllIIIllIIIIIIIll.nprobe}_{IlIIIIIIIIllIIllI}_{IIIlIIlIIIIIIllII}.index"
    faiss.write_index(IIIlllllllIIIlIIl, IIIlIIIlIIllIlllI)
    IlllllllIIllllIlI.append("adding")
    yield _IIlIlIIllllIlIIlI.join(IlllllllIIllllIlI)
    IlIllIIlIlIlIIIlI = 8192
    for IlIIlllIIlIlllIlI in range(0, IIIlIIIllllIIIlII.shape[0], IlIllIIlIlIlIIIlI):
        IIIlllllllIIIlIIl.add(
            IIIlIIIllllIIIlII[IlIIlllIIlIlllIlI : IlIIlllIIlIlllIlI + IlIllIIlIlIlIIIlI]
        )
    IIIlIIIlIIllIlllI = f"{IlIlIlllIIllllIII}/added_IVF{IlIlIlIllIIlIIIIl}_Flat_nprobe_{IllIIIllIIIIIIIll.nprobe}_{IlIIIIIIIIllIIllI}_{IIIlIIlIIIIIIllII}.index"
    faiss.write_index(IIIlllllllIIIlIIl, IIIlIIIlIIllIlllI)
    IlllllllIIllllIlI.append(
        f"Successful Index Construction，added_IVF{IlIlIlIllIIlIIIIl}_Flat_nprobe_{IllIIIllIIIIIIIll.nprobe}_{IlIIIIIIIIllIIllI}_{IIIlIIlIIIIIIllII}.index"
    )
    yield _IIlIlIIllllIlIIlI.join(IlllllllIIllllIlI)


def IIIIIIlIIIlIlIIIl(IIlIllIlIIIIIIIIl):
    IlIIlIIIlIIllIIIl = os.path.join(os.path.dirname(IIlIllIlIIIIIIIIl), "train.log")
    if not os.path.exists(IlIIlIIIlIIllIIIl):
        return (
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
        )
    try:
        with open(IlIIlIIIlIIllIIIl, _IlIIIIlllIlIIllll) as IIlIllIIIIIIIllIl:
            IlIllIllIIlllIIII = next(IIlIllIIIIIIIllIl).strip()
            IlIIlIlIlllIlIlIl = eval(IlIllIllIIlllIIII.split("\t")[-1])
            IlIlllIlIIlIlllll, IIIlllllIlIIllIlI = IlIIlIlIlllIlIlIl.get(
                _IIIIIlIlIlIIIlIll
            ), IlIIlIlIlllIlIlIl.get("if_f0")
            IlIllIIlIIIlllIlI = (
                _IIIllllllIllIIIII
                if IlIIlIlIlllIlIlIl.get(_IllIllIllIllllIIl) == _IIIllllllIllIIIII
                else _IlIIIIIIlIlIllIIl
            )
            return IlIlllIlIIlIlllll, str(IIIlllllIlIIllIlI), IlIllIIlIIIlllIlI
    except Exception as IllIlllIlllIlIllI:
        print(
            f"Exception occurred: {str(IllIlllIlllIlIllI)}, Traceback: {traceback.format_exc()}"
        )
        return (
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
            {_IIllIIllllllIllII: _IlllllIlIIlIlIlII},
        )


def IlIllIlIlllIIllII(IlIlllIllllllIlII, IIIlIllllIllIIllI):
    IlllllIIIlIllIIlI = "rnd"
    IlIIIIllIIlIlIIII = "pitchf"
    IIlIllIlIIllllIIl = "pitch"
    IIlllIlIllllIllIl = "phone"
    IIlIIllllIIllllIl = IIIIIIlIIIIIIIIII.device(_IIlllIIIIIlllIIlI)
    IIlIIlIIIlIlIlllI = IIIIIIlIIIIIIIIII.load(
        IlIlllIllllllIlII, map_location=IIlIIllllIIllllIl
    )
    IIIIIIIIIllllIllI = (
        256
        if IIlIIlIIIlIlIlllI.get(_IllIllIllIllllIIl, _IlIIIIIIlIlIllIIl)
        == _IlIIIIIIlIlIllIIl
        else 768
    )
    IIIIllIlIIllllllI = {
        IIlllIlIllllIllIl: IIIIIIlIIIIIIIIII.rand(1, 200, IIIIIIIIIllllIllI),
        "phone_lengths": IIIIIIlIIIIIIIIII.LongTensor([200]),
        IIlIllIlIIllllIIl: IIIIIIlIIIIIIIIII.randint(5, 255, (1, 200)),
        IlIIIIllIIlIlIIII: IIIIIIlIIIIIIIIII.rand(1, 200),
        "ds": IIIIIIlIIIIIIIIII.zeros(1).long(),
        IlllllIIIlIllIIlI: IIIIIIlIIIIIIIIII.rand(1, 192, 200),
    }
    IIlIIlIIIlIlIlllI[_IIlIIlIIlIlIIIlII][-3] = IIlIIlIIIlIlIlllI[_IIIlllIlIlIIIlIll][
        _IllIIIIlIIllIIlIl
    ].shape[0]
    IllIlllllIlllIIII = SynthesizerTrnMsNSFsidM(
        *IIlIIlIIIlIlIlllI[_IIlIIlIIlIlIIIlII],
        is_half=_IIIIlIlIIIlIlIlll,
        version=IIlIIlIIIlIlIlllI.get(_IllIllIllIllllIIl, _IlIIIIIIlIlIllIIl),
    )
    IllIlllllIlllIIII.load_state_dict(
        IIlIIlIIIlIlIlllI[_IIIlllIlIlIIIlIll], strict=_IIIIlIlIIIlIlIlll
    )
    IllIlllllIlllIIII = IllIlllllIlllIIII.to(IIlIIllllIIllllIl)
    IIlIlIllIllllllll = {
        IIlllIlIllllIllIl: [1],
        IIlIllIlIIllllIIl: [1],
        IlIIIIllIIlIlIIII: [1],
        IlllllIIIlIllIIlI: [2],
    }
    IIIIIIlIIIIIIIIII.onnx.export(
        IllIlllllIlllIIII,
        tuple(
            IllIIIIlIIlIlllll.to(IIlIIllllIIllllIl)
            for IllIIIIlIIlIlllll in IIIIllIlIIllllllI.values()
        ),
        IIIlIllllIllIIllI,
        dynamic_axes=IIlIlIllIllllllll,
        do_constant_folding=_IIIIlIlIIIlIlIlll,
        opset_version=13,
        verbose=_IIIIlIlIIIlIlIlll,
        input_names=list(IIIIllIlIIllllllI.keys()),
        output_names=["audio"],
    )
    return "Finished"


import scipy.io.wavfile as wavfile

IIIIIlIIlIlIIllll = _IlIIIlIIlIlIIlIII


def IlIIIllIIIIlllIIl(IllIllIlIllIIllII):
    IllllIIlIIIIIlIIl = '(?:(?<=\\s)|^)"(.*?)"(?=\\s|$)|(\\S+)'
    IIllIlIllIlIllIII = IIIIlIllIlIlIlllI.findall(IllllIIlIIIIIlIIl, IllIllIlIllIIllII)
    IIllIlIllIlIllIII = [
        IlIIlIIlIlIIIIllI[0] if IlIIlIIlIlIIIIllI[0] else IlIIlIIlIlIIIIllI[1]
        for IlIIlIIlIlIIIIllI in IIllIlIllIlIllIII
    ]
    return IIllIlIllIlIllIII


IlIIlllIIlllIlIIl = lambda IIIllllllllllIlll: all(
    IlIllllllIIIIlIll is not _IllIIIllIIIllllIl
    for IlIllllllIIIIlIll in IIIllllllllllIlll
)


def IIIlIIlIIlIIIIllI(IlIlIlIIIIIIIlllI):
    (
        IlIllllIIlllIlIIl,
        IllIlllIllIIlIlIl,
        IlIIIllIlIlllIIlI,
        IlllllIIIlIlIIlII,
        IlllIIIlIIIIllIII,
        IIlIllIIIIIIIllII,
        IIlIIlllIlllllIlI,
        IIIlllIlIllIlIIll,
        IIlIlIlIIlIllIIII,
        IIllIIlIIIlIIlllI,
        IlllIllIlIIlllIII,
        IlllIIIIlIIllllII,
        IIllllIllIIIIllII,
        _IlIllIIllIlIllIII,
        IlllIIlIllIIIIlIl,
        IllIlllIIlIlIIIIl,
        IIlllllIlllIlIIll,
    ) = IlIIIllIIIIlllIIl(IlIlIlIIIIIIIlllI)[:17]
    IlllIIIlIIIIllIII, IIIlllIlIllIlIIll, IIlIlIlIIlIllIIII, IIllIIlIIIlIIlllI = map(
        int,
        [IlllIIIlIIIIllIII, IIIlllIlIllIlIIll, IIlIlIlIIlIllIIII, IIllIIlIIIlIIlllI],
    )
    IIlIllIIIIIIIllII, IlllIllIlIIlllIII, IlllIIIIlIIllllII, IIllllIllIIIIllII = map(
        float,
        [IIlIllIIIIIIIllII, IlllIllIlIIlllIII, IlllIIIIlIIllllII, IIllllIllIIIIllII],
    )
    if IIlllllIlllIlIIll.lower() == "false":
        IlIIIllIlIIIIIIII = _IlIlllIlIlIlIlIll
        IlIIIIllIIlIlIIlI = _IlIlllIlIlIlIlIll
    else:
        IlIIIllIlIIIIIIII, IlIIIIllIIlIlIIlI = map(
            float, IlIIIllIIIIlllIIl(IlIlIlIIIIIIIlllI)[17:19]
        )
    rvc_globals.DoFormant = IIlllllIlllIlIIll.lower() == "true"
    rvc_globals.Quefrency = IlIIIllIlIIIIIIII
    rvc_globals.Timbre = IlIIIIllIIlIlIIlI
    IIlIlIIllIIlIIIlI = "Infer-CLI:"
    IlIlIllIIIIIIIIlI = f"audio-others/{IlIIIllIlIlllIIlI}"
    print(f"{IIlIlIIllIIlIIIlI} Starting the inference...")
    IIIIlIIIlIllIIIll = IIIlllIllllIIIIll(
        IlIllllIIlllIlIIl, IIllllIllIIIIllII, IIllllIllIIIIllII
    )
    print(IIIIlIIIlIllIIIll)
    print(f"{IIlIlIIllIIlIIIlI} Performing inference...")
    IIIllIlIllIlIIllI = IlIIlIIlIIlllIIII(
        IlllIIIlIIIIllIII,
        IllIlllIllIIlIlIl,
        IllIlllIllIIlIlIl,
        IIlIllIIIIIIIllII,
        _IllIIIllIIIllllIl,
        IIlIIlllIlllllIlI,
        IlllllIIIlIlIIlII,
        IlllllIIIlIlIIlII,
        IlllIIIIlIIllllII,
        IIlIlIlIIlIllIIII,
        IIllIIlIIIlIIlllI,
        IlllIllIlIIlllIII,
        IIllllIllIIIIllII,
        IIIlllIlIllIlIIll,
        f0_min=IlllIIlIllIIIIlIl,
        note_min=_IllIIIllIIIllllIl,
        f0_max=IllIlllIIlIlIIIIl,
        note_max=_IllIIIllIIIllllIl,
        fl_autotune=_IIIIlIlIIIlIlIlll,
    )
    if "Success." in IIIllIlIllIlIIllI[0]:
        print(
            f"{IIlIlIIllIIlIIIlI} Inference succeeded. Writing to {IlIlIllIIIIIIIIlI}..."
        )
        wavfile.write(
            IlIlIllIIIIIIIIlI, IIIllIlIllIlIIllI[1][0], IIIllIlIllIlIIllI[1][1]
        )
        print(f"{IIlIlIIllIIlIIIlI} Finished! Saved output to {IlIlIllIIIIIIIIlI}")
    else:
        print(
            f"{IIlIlIIllIIlIIIlI} Inference failed. Here's the traceback: {IIIllIlIllIlIIllI[0]}"
        )


def IIlIIlIlIIllIlIIl(IIIllllIIIllllllI):
    print("Pre-process: Starting...")
    IlIIlllIIlllIlIIl(
        IllIIllIllIllIlll(
            *IlIIIllIIIIlllIIl(IIIllllIIIllllllI)[:3],
            int(IlIIIllIIIIlllIIl(IIIllllIIIllllllI)[3]),
        )
    )
    print("Pre-process: Finished")


def IIIlIlIlIlIIlIllI(IllIlIIllIIlIlllI):
    (
        IlIIlIllIllIlIlII,
        IlIlIllIIlIIlIlll,
        IIlIlllIIIIlIlIll,
        IIlIlIllIlIIllIII,
        IlIIIlIlIIIIlIlll,
        IIllllIlIIlIlIIIl,
        IlllllIIIIIIIlIII,
    ) = IlIIIllIIIIlllIIl(IllIlIIllIIlIlllI)
    IIlIlllIIIIlIlIll = int(IIlIlllIIIIlIlIll)
    IIlIlIllIlIIllIII = bool(int(IIlIlIllIlIIllIII))
    IIllllIlIIlIlIIIl = int(IIllllIlIIlIlIIIl)
    print(
        f"Extract Feature Has Pitch: {IIlIlIllIlIIllIII}Extract Feature Version: {IlllllIIIIIIIlIII}Feature Extraction: Starting..."
    )
    IlllllllllllIlIIl = IllllIllIllIIIllI(
        IlIlIllIIlIIlIlll,
        IIlIlllIIIIlIlIll,
        IlIIIlIlIIIIlIlll,
        IIlIlIllIlIIllIII,
        IlIIlIllIllIlIlII,
        IlllllIIIIIIIlIII,
        IIllllIlIIlIlIIIl,
    )
    IlIIlllIIlllIlIIl(IlllllllllllIlIIl)
    print("Feature Extraction: Finished")


def IIIllIlllIIIlIlIl(IlIllIlIIlIIIIIll):
    IlIllIlIIlIIIIIll = IlIIIllIIIIlllIIl(IlIllIlIIlIIIIIll)
    IllIIIIlIIIlllIlI = IlIllIlIIlIIIIIll[0]
    IlIlllIIIlIlIIIll = IlIllIlIIlIIIIIll[1]
    IIlIIlIIlIIIIllIl = [
        bool(int(IIIIIlIlllIIlllll)) for IIIIIlIlllIIlllll in IlIllIlIIlIIIIIll[2:11]
    ]
    IIIIIlIllIllIllII = IlIllIlIIlIIIIIll[11]
    IIIIllllIlIlIIlII = (
        "/kaggle/input/ax-rmf/pretrained/"
        if IIIIIlIllIllIllII == _IlIIIIIIlIlIllIIl
        else "/kaggle/input/ax-rmf/pretrained_v2/"
    )
    IIllIllIlIIIIllll = f"{IIIIllllIlIlIIlII}f0G{IlIlllIIIlIlIIIll}.pth"
    IllIIIllIlllllIll = f"{IIIIllllIlIlIIlII}f0D{IlIlllIIIlIlIIIll}.pth"
    print("Train-CLI: Training...")
    IllIIlllIlllIIIIl(
        IllIIIIlIIIlllIlI,
        IlIlllIIIlIlIIIll,
        *IIlIIlIIlIIIIllIl,
        IIllIllIlIIIIllll,
        IllIIIllIlllllIll,
        IIIIIlIllIllIllII,
    )


def IIIIlIlIIIIIIIIII(IIllIIlIIIlIlIIlI):
    IlIIIlllllllIllIl = "Train Feature Index-CLI"
    print(f"{IlIIIlllllllIllIl}: Training... Please wait")
    IlIIlllIIlllIlIIl(IIIIllIllllllIllI(*IlIIIllIIIIlllIIl(IIllIIlIIIlIlIIlI)))
    print(f"{IlIIIlllllllIllIl}: Done!")


def IllIIIIlIllIIllIl(IlIIIIlIIIIIlllII):
    IIIIlIIlllIlIllll = extract_small_model(*IlIIIllIIIIlllIIl(IlIIIIlIIIIIlllII))
    print(
        "Extract Small Model: Success!"
        if IIIIlIIlllIlIllll == "Success."
        else f"{IIIIlIIlllIlIllll}\nExtract Small Model: Failed!"
    )


def IllllIIlIlIlIlIll(IIlIIIlIIlIIlIIIl, IllIlIIlllIlIIIll, IIlIIlIlllIIIlIlI):
    if IIlIIIlIIlIIlIIIl:
        try:
            with open(IIlIIIlIIlIIlIIIl, _IlIIIIlllIlIIllll) as IlIIIlIIIlIllIlII:
                IIIlIlllIIIlllIll = IlIIIlIIIlIllIlII.read().splitlines()
            IllIlIIlllIlIIIll, IIlIIlIlllIIIlIlI = (
                IIIlIlllIIIlllIll[0],
                IIIlIlllIIIlllIll[1],
            )
            IllIIIllIlIllllll(IllIlIIlllIlIIIll, IIlIIlIlllIIIlIlI)
        except IndexError:
            print("Error: File does not have enough lines to read 'qfer' and 'tmbr'")
        except FileNotFoundError:
            print("Error: File does not exist")
        except Exception as IllllIIlllIIIIllI:
            print("An unexpected error occurred", IllllIIlllIIIIllI)
    return {
        _IlllIIllllIIIIlIl: IllIlIIlllIlIIIll,
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }, {_IlllIIllllIIIIlIl: IIlIIlIlllIIIlIlI, _IIllIIllllllIllII: _IlllllIlIIlIlIlII}


def IIllIlIlIIIIIlIII():
    IllIlIlIIIlIIlIII = {
        _IlIIIlIIlIlIIlIII: "\n    go home            : Takes you back to home with a navigation list.\n    go infer           : Takes you to inference command execution.\n    go pre-process     : Takes you to training step.1) pre-process command execution.\n    go extract-feature : Takes you to training step.2) extract-feature command execution.\n    go train           : Takes you to training step.3) being or continue training command execution.\n    go train-feature   : Takes you to the train feature index command execution.\n    go extract-model   : Takes you to the extract small model command execution.",
        _IlIlIlIlIIlllIlII: "\n    arg 1) model name with .pth in ./weights: mi-test.pth\n    arg 2) source audio path: myFolder\\MySource.wav\n    arg 3) output file name to be placed in './audio-others': MyTest.wav\n    arg 4) feature index file path: logs/mi-test/added_IVF3l42_Flat_nprobe_1.index\n    arg 5) speaker id: 0\n    arg 6) transposition: 0\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)\n    arg 8) crepe hop length: 160\n    arg 9) harvest median filter radius: 3 (0-7)\n    arg 10) post resample rate: 0\n    arg 11) mix volume envelope: 1\n    arg 12) feature index ratio: 0.78 (0-1)\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2",
        _IllIIlllIIIllIIIl: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Number of CPU threads to use: 8 \n\nExample: mi-test mydataset 40k 24",
        _IlIlllIlllIIlllIl: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 3) Number of CPU threads to use: 8\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)\n    arg 6) Crepe hop length: 128\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n\nExample: mi-test 0 24 1 harvest 128 v2",
        _IIIllIllIIlllIlll: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Sample rate: 40k (32k, 40k, 48k)\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 4) speaker id: 0\n    arg 5) Save epoch iteration: 50\n    arg 6) Total epochs: 10000\n    arg 7) Batch size: 8\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2",
        _IllIIIlIIIIIIlIIl: "\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test v2",
        _IIllllIllIIllllIl: '\n    arg 1) Model Path: logs/mi-test/G_168000.pth\n    arg 2) Model save name: MyModel\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) Model information: "My Model"\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2',
    }
    print(IllIlIlIIIlIIlIII.get(IIIIIlIIlIlIIllll, "Invalid page"))


def IIlIlIlIIIlllIlIl(IlllIIlllIIlIIlll):
    global IIIIIlIIlIlIIllll
    IIIIIlIIlIlIIllll = IlllIIlllIIlIIlll
    return 0


def IIIlIllIIlllIIIll(IlIIlllIlllIIIIIl):
    IlIIlIlIIIIlIIIIl = {
        "go home": _IlIIIlIIlIlIIlIII,
        "go infer": _IlIlIlIlIIlllIlII,
        "go pre-process": _IllIIlllIIIllIIIl,
        "go extract-feature": _IlIlllIlllIIlllIl,
        "go train": _IIIllIllIIlllIlll,
        "go train-feature": _IllIIIlIIIIIIlIIl,
        "go extract-model": _IIllllIllIIllllIl,
    }
    IlIIlllIIIIIIIIII = {
        _IlIlIlIlIIlllIlII: IIIlIIlIIlIIIIllI,
        _IllIIlllIIIllIIIl: IIlIIlIlIIllIlIIl,
        _IlIlllIlllIIlllIl: IIIlIlIlIlIIlIllI,
        _IIIllIllIIlllIlll: IIIllIlllIIIlIlIl,
        _IllIIIlIIIIIIlIIl: IIIIlIlIIIIIIIIII,
        _IIllllIllIIllllIl: IllIIIIlIllIIllIl,
    }
    if IlIIlllIlllIIIIIl in IlIIlIlIIIIlIIIIl:
        return IIlIlIlIIIlllIlIl(IlIIlIlIIIIlIIIIl[IlIIlllIlllIIIIIl])
    if IlIIlllIlllIIIIIl[:3] == "go ":
        print(f"page '{IlIIlllIlllIIIIIl[3:]}' does not exist!")
        return 0
    if IIIIIlIIlIlIIllll in IlIIlllIIIIIIIIII:
        IlIIlllIIIIIIIIII[IIIIIlIIlIlIIllll](IlIIlllIlllIIIIIl)


def IlIlllIllIIlllIll():
    while _IlllIIIlllIllIllI:
        print(f"\nYou are currently in '{IIIIIlIIlIlIIllll}':")
        IIllIlIlIIIIIlIII()
        print(f"{IIIIIlIIlIlIIllll}: ", end="")
        try:
            IIIlIllIIlllIIIll(input())
        except Exception as IlIlIlIIIIIIIIllI:
            print(f"An error occurred: {traceback.format_exc()}")


if IlIllIlIlIIlIlIll.is_cli:
    print(
        "\n\nMangio-RVC-Fork v2 CLI App!\nWelcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n"
    )
    IlIlllIllIIlllIll()
"\ndef get_presets():\n    data = None\n    with open('../inference-presets.json', 'r') as file:\n        data = json.load(file)\n    preset_names = []\n    for preset in data['presets']:\n        preset_names.append(preset['name'])\n    \n    return preset_names\n"


def IllllIIIIlIlllIll(IIllIIIllIlIlIIll):
    IlIlllIIIIIllIlll = IIllIIIllIlIlIIll != _IlIIllllIIIlIlIlI
    if rvc_globals.NotesIrHertz:
        return (
            {
                _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: IlIlllIIIIIllIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: IlIlllIIIIIllIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
        )
    else:
        return (
            {
                _IIIllIllllIlIllIl: IlIlllIIIIIllIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: IlIlllIIIIIllIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
            {
                _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
                _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
            },
        )


def IIIlIllIIllIIIlII(IIllIlllIIIlIIIIl):
    IIllIllIllIIlIlll = IIIIlIllIlIlIlllI.sub("\\.pth|\\.onnx$", "", IIllIlllIIIlIIIIl)
    IIllIIIIllIIlIIII = os.path.split(IIllIllIllIIlIlll)[-1]
    if IIIIlIllIlIlIlllI.match(".+_e\\d+_s\\d+$", IIllIIIIllIIlIIII):
        IlllllIIlIIlIIlIl = IIllIIIIllIIlIIII.rsplit("_", 2)[0]
    else:
        IlllllIIlIIlIIlIl = IIllIIIIllIIlIIII
    IlIIIIIlIlIllIlll = os.path.join(IlIlIIIllIIllIlll, IlllllIIlIIlIIlIl)
    IlllIIlIIIIIIlIII = [IlIIIIIlIlIllIlll] if os.path.exists(IlIIIIIlIlIllIlll) else []
    IlllIIlIIIIIIlIII.append(IlIlIIIllIIllIlll)
    IIllIIIIllIlIIlll = []
    for IlIIIlIIIlllIIIIl in IlllIIlIIIIIIlIII:
        for IllIIlllIIIIIlIlI in os.listdir(IlIIIlIIIlllIIIIl):
            if (
                IllIIlllIIIIIlIlI.endswith(_IlIIllIllIllIIIIl)
                and _IlIllIlIIIlllIIIl not in IllIIlllIIIIIlIlI
            ):
                IIIIIIIIllIIIllII = any(
                    IIlIlIIlllIIlIIlI.lower() in IllIIlllIIIIIlIlI.lower()
                    for IIlIlIIlllIIlIIlI in [IIllIIIIllIIlIIII, IlllllIIlIIlIIlIl]
                )
                IIIlIlllllIllIlll = IlIIIlIIIlllIIIIl == IlIIIIIlIlIllIlll
                if IIIIIIIIllIIIllII or IIIlIlllllIllIlll:
                    IIIllIlllIlIIIlIl = os.path.join(
                        IlIIIlIIIlllIIIIl, IllIIlllIIIIIlIlI
                    )
                    if IIIllIlllIlIIIlIl in IllIlIIlIlIIIIIlI:
                        IIllIIIIllIlIIlll.append(
                            (
                                IIIllIlllIlIIIlIl,
                                os.path.getsize(IIIllIlllIlIIIlIl),
                                _IlIlIIIlllIllIlll not in IllIIlllIIIIIlIlI,
                            )
                        )
    if IIllIIIIllIlIIlll:
        IIllIIIIllIlIIlll.sort(
            key=lambda IllIllIlIIlIIlIIl: (-IllIllIlIIlIIlIIl[2], -IllIllIlIIlIIlIIl[1])
        )
        IIIllllIIIlIIllIl = IIllIIIIllIlIIlll[0][0]
        return IIIllllIIIlIIllIl, IIIllllIIIlIIllIl
    return "", ""


def IIIIllIIIIIlIllll(IlIIlllIlIIlIllll):
    if IlIIlllIlIIlIllll:
        try:
            with open(_IlIllIlllIIIIIIIl, "w+") as IIIIIlIIIIlIlIllI:
                IIIIIlIIIIlIlIllI.write("True")
            os.kill(IIIIllIllIllIIIll, SIGTERM)
        except Exception as IlIIIlIlIIIIlIIIl:
            print(f"Couldn't click due to {IlIIIlIlIIIIlIIIl}")
        return {
            _IIIllIllllIlIllIl: _IlllIIIlllIllIllI,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }, {
            _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
        }
    return {
        _IIIllIllllIlIllIl: _IIIIlIlIIIlIlIlll,
        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
    }, {_IIIllIllllIlIllIl: _IlllIIIlllIllIllI, _IIllIIllllllIllII: _IlllllIlIIlIlIlII}


IIllIlIIIIlIlllIl = "weights/"


def IIIllIIIlIIIlllII(IIIIIlIlIIlIlIlll):
    IIIIlIIIIIlIIIlll = {
        "C": -9,
        "C#": -8,
        _IIIlIIIlIIIIlllll: -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        _IlIIIIIIllllIIlll: -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    IlllIllIlIlIIlllI, IllIlIllIlIlIIlII = IIIIIlIlIIlIlIlll[:-1], int(
        IIIIIlIlIIlIlIlll[-1]
    )
    IIIIlIIIIIlIlIIll = IIIIlIIIIIlIIIlll[IlllIllIlIlIIlllI]
    IlllIIlIIIIIlIllI = 12 * (IllIlIllIlIlIIlII - 4) + IIIIlIIIIIlIlIIll
    IllIIlIllIIlllIll = 44e1 * (2.0 ** (_IlIlllIlIlIlIlIll / 12)) ** IlllIIlIIIIIlIllI
    return IllIIlIllIIlllIll


def IIllIlllIIllIllIl(IIIlIIlIllIIllllI):
    if IIIlIIlIllIIllllI is _IllIIIllIIIllllIl:
        0
    else:
        IlIIIllIllllllIIl = IIIlIIlIllIIllllI
        IIIIlllIIlIlllIll = (
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        )
        IIlIIIllIlIIIlIlI = "./audios/" + IIIIlllIIlIlllIll
        shutil.move(IlIIIllIllllllIIl, IIlIIIllIlIIIlIlI)
        return IIIIlllIIlIlllIll


def IllllIIlllIIlIlII(IllIIlIlIIIllIIll):
    if IllIIlIlIIIllIIll is _IllIIIllIIIllllIl:
        0
    else:
        IlIIllllllllIlIIl = IllIIlIlIIIllIIll.name
        IlIIlIIIIIIllllIl = os.path.join(
            _IIlIIIIIlIlIIIllI, os.path.basename(IlIIllllllllIlIIl)
        )
        if os.path.exists(IlIIlIIIIIIllllIl):
            os.remove(IlIIlIIIIIIllllIl)
            print(_IIlIIlllIllIIIIlI)
        shutil.move(IlIIllllllllIlIIl, IlIIlIIIIIIllllIl)


def IllIIllIIlIIIlIll(IlIIIllIIlllIllII):
    IIIIlIlIllIlIlIII = IlIIIllIIlllIllII.name
    IIlIllllIIlIllIIl = os.path.join(
        _IIlIIIIIlIlIIIllI, os.path.basename(IIIIlIlIllIlIlIII)
    )
    if os.path.exists(IIlIllllIIlIllIIl):
        os.remove(IIlIllllIIlIllIIl)
        print(_IIlIIlllIllIIIIlI)
    shutil.move(IIIIlIlIllIlIlIII, IIlIllllIIlIllIIl)
    return IIlIllllIIlIllIIl


from gtts import gTTS
import edge_tts, asyncio


def IIlllIIIlIIllllll(
    IIlIIllIIIlIIlIII,
    IlllIIllIlIIlIIIl,
    IIIlIlIIIlIlllIII,
    IlIlIllIIIlllIIll,
    IIIllIlIlIlllIlII,
    IIlIIIIlllIIIlIIl,
    IIIlllllllIIIlIlI,
    IlIIlIIIllIlIIlll,
    IlllIlIlllIIIIlIl,
    IlIllIlllIIlIIllI,
    IIIlllllIlIIllIll,
    IIlIllIIIllIllIIl,
    IIIIIIIllIIlIIIlI,
    IIlllIIlIIlIlIlII,
    IIlllllIIllllIlll,
):
    global IIllIllIlIlIlIlII, IIllIIIllIIllllll, IIIIIlIIIlIllllII, IIIlIlllIIIIlllIl, IllIIIIIlllllIlIl, IIIIlIIlIIlllIIIl
    if IlllIIllIlIIlIIIl is _IllIIIllIIIllllIl:
        return _IIIIllIIllIllIIII, _IllIIIllIIIllllIl
    IIIlIlIIIlIlllIII = int(IIIlIlIIIlIlllIII)
    try:
        IIllIIllllllIIIIl = load_audio(IlllIIllIlIIlIIIl, 16000)
        IlIlIlIlIllllIlll = IIlIlllllIlIlIIlI.abs(IIllIIllllllIIIIl).max() / 0.95
        if IlIlIlIlIllllIlll > 1:
            IIllIIllllllIIIIl /= IlIlIlIlIllllIlll
        IIIlIlllIlIlIIIIl = [0, 0, 0]
        if not IIIlIlllIIIIlllIl:
            IlIIlIllIIIIIIlIl()
        IlIIIlIIllIlllllI = IIIIlIIlIIlllIIIl.get(_IllIIIlIllllIIllI, 1)
        IIlIIIIlllIIIlIIl = (
            IIlIIIIlllIIIlIIl.strip(_IlIlIIIlllIllIlll)
            .strip(_IIlllllllllIllIll)
            .strip(_IIlIlIIllllIlIIlI)
            .strip(_IIlllllllllIllIll)
            .strip(_IlIlIIIlllIllIlll)
            .replace(_IlIllIlIIIlllIIIl, "added")
            if IIlIIIIlllIIIlIIl != ""
            else IIIlllllllIIIlIlI
        )
        IIIIllIllIlIlIlll = IIIIIlIIIlIllllII.pipeline(
            IIIlIlllIIIIlllIl,
            IIllIIIllIIllllll,
            IIlIIllIIIlIIlIII,
            IIllIIllllllIIIIl,
            IlllIIllIlIIlIIIl,
            IIIlIlllIlIlIIIIl,
            IIIlIlIIIlIlllIII,
            IIIllIlIlIlllIlII,
            IIlIIIIlllIIIlIIl,
            IlIIlIIIllIlIIlll,
            IlIIIlIIllIlllllI,
            IlllIlIlllIIIIlIl,
            IIllIllIlIlIlIlII,
            IlIllIlllIIlIIllI,
            IIIlllllIlIIllIll,
            IllIIIIIlllllIlIl,
            IIlIllIIIllIllIIl,
            IIIIIIIllIIlIIIlI,
            IIlllIIlIIlIlIlII,
            IIlllllIIllllIlll,
            f0_file=IlIlIllIIIlllIIll,
        )
        if IIllIllIlIlIlIlII != IlIllIlllIIlIIllI >= 16000:
            IIllIllIlIlIlIlII = IlIllIlllIIlIIllI
        IIlllIlllllllllIl = (
            _IIIlIllIIIlllIllI % IIlIIIIlllIIIlIIl
            if os.path.exists(IIlIIIIlllIIIlIIl)
            else _IlIIlIIllllIIIlIl
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            IIlllIlllllllllIl,
            IIIlIlllIlIlIIIIl[0],
            IIIlIlllIlIlIIIIl[1],
            IIIlIlllIlIlIIIIl[2],
        ), (IIllIllIlIlIlIlII, IIIIllIllIlIlIlll)
    except:
        IIllIlllIlllIIlIl = traceback.format_exc()
        print(IIllIlllIlllIIlIl)
        return IIllIlllIlllIIlIl, (_IllIIIllIIIllllIl, _IllIIIllIIIllllIl)


def IlllIllIllIIIIlIl(
    _IIIIIIIllIIIlIlIl,
    IIIIIlIIIIlIIlIlI,
    IllIIlllIIIIlIlIl="",
    IllIlIIllIIIlllll=0,
    IIllIIIIllllIIIlI=_IIIlIIllIlIlIllII,
    IIIlIIllIIIIlIIll=float(0.66),
    IllIIIIIllIIlIlll=float(64),
    IlIlIIIlllIllIlIl=_IIIIlIlIIIlIlIlll,
    IIIIlIIllIIIIlIIl=_IIIIlIlIIIlIlIlll,
    IIlIIlIllllIIIIII="",
    IIllIllllllIIIIII="",
):
    IIIlllIllllIIIIll(
        sid=IllIIlllIIIIlIlIl, to_return_protectl=0.33, to_return_protect1=0.33
    )
    for _IIIlllIlIllIlIllI in _IIIIIIIllIIIlIlIl:
        IIIIlIlIIIIlIIIIl = (
            "audio2/" + IIIIIlIIIIlIIlIlI[_IIIlllIlIllIlIllI]
            if _IIIlllIlIllIlIllI != _IlIIIllIlIIlIIIlI
            else IIIIIlIIIIlIIlIlI[0]
        )
        try:
            print(IIIIIlIIIIlIIlIlI[_IIIlllIlIllIlIllI], IllIIlllIIIIlIlIl)
        except:
            pass
        IIlllIlIIllllllll, (IllllllIllIllIlll, IIlIlIIIIIIlIIIll) = IIlllIIIlIIllllll(
            sid=0,
            input_audio_path=IIIIlIlIIIIlIIIIl,
            f0_up_key=IllIlIIllIIIlllll,
            f0_file=_IllIIIllIIIllllIl,
            f0_method=IIllIIIIllllIIIlI,
            file_index=IIlIIlIllllIIIIII,
            file_index2=IIllIllllllIIIIII,
            index_rate=IIIlIIllIIIIlIIll,
            filter_radius=int(3),
            resample_sr=int(0),
            rms_mix_rate=float(0.25),
            protect=float(0.33),
            crepe_hop_length=IllIIIIIllIIlIlll,
            fl_autotune=IlIlIIIlllIllIlIl,
            rmvpe_onnx=IIIIlIIllIIIIlIIl,
        )
        sf.write(
            file=IIIIlIlIIIIlIIIIl, samplerate=IllllllIllIllIlll, data=IIlIlIIIIIIlIIIll
        )


def IIIlIIIlIllIIIIlI(IlllIllIlllIlllII, IIIIllIIlIllIlIIl):
    try:
        return IlllIllIlllIlllII.to(IIIIllIIlIllIlIIl)
    except Exception as IIlllIIIIIllIlIII:
        print(IIlllIIIIIllIlIII)
        return IlllIllIlllIlllII


def __bark__(IIlIIlllIIIlllIll, IllIlllIIlIlIIIII):
    IlIIllIllllIIlIII = "tts"
    IllIIlllIIIIIllll = "suno/bark"
    os.makedirs(
        os.path.join(IlIIIllllllIllIll, IlIIllIllllIIlIII), exist_ok=_IlllIIIlllIllIllI
    )
    from transformers import AutoProcessor, BarkModel

    IIIIlIllllIIIlIlI = (
        "cuda:0" if IIIIIIlIIIIIIIIII.cuda.is_available() else _IIlllIIIIIlllIIlI
    )
    IlIlIllllIIIllllI = (
        IIIIIIlIIIIIIIIII.float32
        if _IIlllIIIIIlllIIlI in IIIIlIllllIIIlIlI
        else IIIIIIlIIIIIIIIII.float16
    )
    IlIIllIlIIlIIlllI = AutoProcessor.from_pretrained(
        IllIIlllIIIIIllll,
        cache_dir=os.path.join(IlIIIllllllIllIll, IlIIllIllllIIlIII, IllIIlllIIIIIllll),
        torch_dtype=IlIlIllllIIIllllI,
    )
    IIlIlIIlIllIllllI = BarkModel.from_pretrained(
        IllIIlllIIIIIllll,
        cache_dir=os.path.join(IlIIIllllllIllIll, IlIIllIllllIIlIII, IllIIlllIIIIIllll),
        torch_dtype=IlIlIllllIIIllllI,
    ).to(IIIIlIllllIIIlIlI)
    IlIIIIIIlIlIlllII = IlIIllIlIIlIIlllI(
        text=[IIlIIlllIIIlllIll], return_tensors="pt", voice_preset=IllIlllIIlIlIIIII
    )
    IllIIIIIIIllIIIII = {
        IlIIIlIIllIIlIIll: IIIlIIIlIllIIIIlI(IIIlIIlIIIIlIIlII, IIIIlIllllIIIlIlI)
        if hasattr(IIIlIIlIIIIlIIlII, "to")
        else IIIlIIlIIIIlIIlII
        for (IlIIIlIIllIIlIIll, IIIlIIlIIIIlIIlII) in IlIIIIIIlIlIlllII.items()
    }
    IIIlIlllllIlIIlIl = IIlIlIIlIllIllllI.generate(
        **IllIIIIIIIllIIIII, do_sample=_IlllIIIlllIllIllI
    )
    IllIIIIIIllIIllII = IIlIlIIlIllIllllI.generation_config.sample_rate
    IllIllIIlIlIIIlll = IIIlIlllllIlIIlIl.cpu().numpy().squeeze()
    return IllIllIIlIlIIIlll, IllIIIIIIllIIllII


def IIIIllIIIIIIIIIII(
    IllIIlIlIllIIlllI,
    IIIIIIllllIIIlIIl,
    IlllIlllIIllllIll,
    IIlIlIlIlIIlIIlII,
    IIlIIIIIIIlllIllI,
    IllIlIlIllIIIlIIl,
    IllIlIlIIlIllIIII,
    IlIIlIIIIllllllII,
    IlIIlIlIIIIIIlllI,
    IIllIIIlllIlIIIII,
):
    IIllIIllIllIlIIIl = "converted_bark.wav"
    IlIlIIIlIIllIlIlI = "bark_out.wav"
    IIIIIlIIlllIIllll = "converted_tts.wav"
    if IIIIIIllllIIIlIIl == _IllIIIllIIIllllIl:
        return
    IIllIIllIllIlIlIl = os.path.join(
        IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IIIIIlIIlllIIllll
    )
    IIIlllIlIlIllIIII = (
        _IlllIIIlllIllIllI
        if IllIlIlIllIIIlIIl == _IIIlIlIIllllIIIlI
        else _IIIIlIlIIIlIlIlll
    )
    if "SET_LIMIT" == os.getenv("DEMO"):
        if len(IllIIlIlIllIIlllI) > 60:
            IllIIlIlIllIIlllI = IllIIlIlIllIIlllI[:60]
            print("DEMO; limit to 60 characters")
    IlIlIIIllllIllllI = IIIIIIllllIIIlIIl[:2]
    if IIllIIIlllIlIIIII == _IIlIIllIlIIIIlIll:
        try:
            asyncio.run(
                edge_tts.Communicate(
                    IllIIlIlIllIIlllI, "-".join(IIIIIIllllIIIlIIl.split("-")[:-1])
                ).save(IIllIIllIllIlIlIl)
            )
        except:
            try:
                IlllllIlIIIIIIIIl = gTTS(IllIIlIlIllIIlllI, lang=IlIlIIIllllIllllI)
                IlllllIlIIIIIIIIl.save(IIllIIllIllIlIlIl)
                IlllllIlIIIIIIIIl.save
                print(
                    f"No audio was received. Please change the tts voice for {IIIIIIllllIIIlIIl}. USING gTTS."
                )
            except:
                IlllllIlIIIIIIIIl = gTTS("a", lang=IlIlIIIllllIllllI)
                IlllllIlIIIIIIIIl.save(IIllIIllIllIlIlIl)
                print("Error: Audio will be replaced.")
        os.system("cp audio-outputs/converted_tts.wav audio-outputs/real_tts.wav")
        IlllIllIllIIIIlIl(
            [_IlIIIllIlIIlIIIlI],
            ["audio-outputs/converted_tts.wav"],
            model_voice_path=IlllIlllIIllllIll,
            transpose=IIlIIIIIIIlllIllI,
            f0method=IllIlIlIllIIIlIIl,
            index_rate_=IllIlIlIIlIllIIII,
            crepe_hop_length_=IlIIlIIIIllllllII,
            fl_autotune=IlIIlIlIIIIIIlllI,
            rmvpe_onnx=IIIlllIlIlIllIIII,
            file_index="",
            file_index2=IIlIlIlIlIIlIIlII,
        )
        return os.path.join(
            IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IIIIIlIIlllIIllll
        ), os.path.join(IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, "real_tts.wav")
    elif IIllIIIlllIlIIIII == _IIIlllIlllIIIIlII:
        try:
            IIIlllIllllIIIIll(
                sid=IlllIlllIIllllIll, to_return_protectl=0.33, to_return_protect1=0.33
            )
            IllIIlIIllIIIIlIl = IllIIlIlIllIIlllI.replace(
                _IIlIlIIllllIlIIlI, _IlIlIIIlllIllIlll
            ).strip()
            IlIIllllIIIIlIIll = sent_tokenize(IllIIlIIllIIIIlIl)
            print(IlIIllllIIIIlIIll)
            IllIllIIlIIIIlllI = IIlIlllllIlIlIIlI.zeros(int(0.25 * SAMPLE_RATE))
            IllIlIlIlIlIllIII = []
            IlllllIIlIlIIIlIl = os.path.join(
                IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IlIlIIIlIIllIlIlI
            )
            for IIIlIIlIlllIlIlll in IlIIllllIIIIlIIll:
                IIIllllIIIlIIllII, _IIlIIllIIlIlIIIll = __bark__(
                    IIIlIIlIlllIlIlll, IIIIIIllllIIIlIIl.split("-")[0]
                )
                IllIlIlIlIlIllIII += [IIIllllIIIlIIllII, IllIllIIlIIIIlllI.copy()]
            sf.write(
                file=IlllllIIlIlIIIlIl,
                samplerate=SAMPLE_RATE,
                data=IIlIlllllIlIlIIlI.concatenate(IllIlIlIlIlIllIII),
            )
            IIIllIIlIIIlIIIIl, (
                IlllIIIIlllIIIlIl,
                IlIlIlIlIlIIIlIlI,
            ) = IIlllIIIlIIllllll(
                sid=0,
                input_audio_path=os.path.join(
                    IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IlIlIIIlIIllIlIlI
                ),
                f0_up_key=IIlIIIIIIIlllIllI,
                f0_file=_IllIIIllIIIllllIl,
                f0_method=IllIlIlIllIIIlIIl,
                file_index="",
                file_index2=IIlIlIlIlIIlIIlII,
                index_rate=IllIlIlIIlIllIIII,
                filter_radius=int(3),
                resample_sr=int(0),
                rms_mix_rate=float(0.25),
                protect=float(0.33),
                crepe_hop_length=IlIIlIIIIllllllII,
                fl_autotune=IlIIlIlIIIIIIlllI,
                rmvpe_onnx=IIIlllIlIlIllIIII,
            )
            wavfile.write(
                os.path.join(IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IIllIIllIllIlIIIl),
                rate=IlllIIIIlllIIIlIl,
                data=IlIlIlIlIlIIIlIlI,
            )
            return (
                os.path.join(IlIIIllllllIllIll, _IIlIIIlIIIllIIIll, IIllIIllIllIlIIIl),
                IlllllIIlIlIIIlIl,
            )
        except Exception as IlIlIllllllIlIlll:
            print(f"{IlIlIllllllIlIlll}")
            return _IllIIIllIIIllllIl, _IllIIIllIIIllllIl


def IlllIlllIIlllIIII(IlIIIllIllIlllIlI=IllIlIlIIllllIlIl.themes.Soft()):
    IlIIllIlllIlIllIl = "Model information to be placed:"
    IIIIIllIIllIlIIll = "Model architecture version:"
    IIIIIllIIIIIlIlIl = "Name:"
    IllIlIllllIIlIIlI = "-0.5"
    IlIIIlIIlIlIlIIII = "Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:"
    IIlIllllIllllIIIl = "You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder."
    IlIlllIlllIllIllI = (
        "Export audio (click on the three dots in the lower right corner to download)"
    )
    IlIIllIIllIlIIIll = "Default value is 1.0"
    IIlIlIIIIllllllII = "Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:"
    IIlllIllIllllIllI = "Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:"
    IIlIlIIIIIIIIIIIl = "Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:"
    IIlIIlIIIlIIIllll = "Feature search database file path:"
    IlllIlllIIIlIIIlI = "Max pitch:"
    IIlIIIllllllIIllI = "Min pitch:"
    IIIIlIlllIIlllIII = "If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness."
    IIlIlIIlIlIlIllIl = "Mangio-Crepe Hop Length (Only applies to mangio-crepe): Hop length refers to the time it takes for the speaker to jump to a dramatic pitch. Lower hop lengths take more time to infer but are more pitch accurate."
    IIIlIIllIllllllIl = "Enable autotune"
    IIIlIlIlIIlIllIIl = "crepe-tiny"
    IIlIIIllllIIIlIII = "Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):"
    IllIIlIlIlIIllIIl = "Search feature ratio:"
    IlllIIllIlIIIIIIl = "Auto-detect index path and select from the dropdown:"
    IlIllIllllIlIlllI = "filepath"
    IllllIlIllIIlllIl = "Drag your audio here:"
    IlIllIlIIIIIlIlII = "Refresh"
    IlIIIIlIIIIlIIIll = "Path to Model:"
    IIIllIIIIIIIlIIll = "Model information to be placed"
    IIlllIllIIIIIIllI = "Name for saving"
    IIIIllllIIIlllIlI = "Whether the model has pitch guidance."
    IlIIlIIIIllIIllIl = "Target sample rate:"
    IlIlIllIlIIIIlIII = "multiple"
    IlIlIIlllIIIIIIIl = "opt"
    IIlllllIllllIllII = "crepe"
    IlIllIIIIIlllllII = "dio"
    IIIllIlIllIllllII = "harvest"
    IlIIlllIllIIlIIII = "Select the pitch extraction algorithm:"
    IllIlllIlIIlIIllI = "Advanced Settings"
    IlIlIIllllIlIIIII = "Convert"
    IllIlIlllIlIIllII = "rmvpe+"
    IIIIlIlIlllIlIIll = "Path to model"
    IIIIlIllIllllllll = "mangio-crepe-tiny"
    IIIlllIlIllllIllI = "mangio-crepe"
    IlllIIIlllllIIIll = "Output information:"
    IIlIIIIlIlIIlIlll = "primary"
    IlllIlIlIllllIlII = IIlIIIlIIIlIIlIII[0] if IIlIIIlIIIlIIlIII else ""
    with IllIlIlIIllllIlIl.Blocks(
        theme="JohnSmith9982/small_and_pretty", title="AX-RVC"
    ) as IIlllIIllIlIlllll:
        IllIlIlIIllllIlIl.HTML("<h1> 🍏 AX-RVC (Mangio-RVC-Fork) </h1>")
        with IllIlIlIIllllIlIl.Tabs():
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Model Inference")):
                with IllIlIlIIllllIlIl.Row():
                    IIIlIIIlIlIIIIIIl = IllIlIlIIllllIlIl.Dropdown(
                        label=IIIllIIllIIIIIIlI("Inferencing voice:"),
                        choices=sorted(IIlIIIlIIIlIIlIII),
                        value=IlllIlIlIllllIlII,
                    )
                    IIlIlIlllllllIllI = IllIlIlIIllllIlIl.Button(
                        IIIllIIllIIIIIIlI(IlIllIlIIIIIlIlII), variant=IIlIIIIlIlIIlIlll
                    )
                    IIlIlIllIllIlIllI = IllIlIlIIllllIlIl.Button(
                        IIIllIIllIIIIIIlI("Unload voice to save GPU memory"),
                        variant=IIlIIIIlIlIIlIlll,
                    )
                    IIlIlIllIllIlIllI.click(
                        fn=lambda: {
                            _IlllIIllllIIIIlIl: "",
                            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                        },
                        inputs=[],
                        outputs=[IIIlIIIlIlIIIIIIl],
                    )
                with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Single")):
                    with IllIlIlIIllllIlIl.Row():
                        IllllIlIIlIIlIlll = IllIlIlIIllllIlIl.Slider(
                            minimum=0,
                            maximum=2333,
                            step=1,
                            label=IIIllIIllIIIIIIlI("Select Speaker/Singer ID:"),
                            value=0,
                            visible=_IIIIlIlIIIlIlIlll,
                            interactive=_IlllIIIlllIllIllI,
                        )
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Row():
                            with IllIlIlIIllllIlIl.Column():
                                IllIlIIIlllIIlllI = IllIlIlIIllllIlIl.File(
                                    label=IIIllIIllIIIIIIlI(IllllIlIllIIlllIl)
                                )
                                IlllIlIllllIIIIlI = IllIlIlIIllllIlIl.Audio(
                                    source="microphone",
                                    label=IIIllIIllIIIIIIlI("Or record an audio:"),
                                    type=IlIllIllllIlIlllI,
                                )
                                IlIIIIIlllllIlIIl = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(
                                        "Manual path to the audio file to be processed"
                                    ),
                                    value=os.path.join(
                                        IlIIIllllllIllIll,
                                        _IIlIIIIIlIlIIIllI,
                                        "someguy.mp3",
                                    ),
                                    visible=_IIIIlIlIIIlIlIlll,
                                )
                                IllIlIIIIlllIIlII = IllIlIlIIllllIlIl.Dropdown(
                                    label=IIIllIIllIIIIIIlI(
                                        "Auto detect audio path and select from the dropdown:"
                                    ),
                                    choices=sorted(IIIlIIllllIIIIIII),
                                    value="",
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IllIlIIIIlllIIlII.select(
                                    fn=lambda: "",
                                    inputs=[],
                                    outputs=[IlIIIIIlllllIlIIl],
                                )
                                IlIIIIIlllllIlIIl.input(
                                    fn=lambda: "",
                                    inputs=[],
                                    outputs=[IllIlIIIIlllIIlII],
                                )
                                IllIlIIIlllIIlllI.upload(
                                    fn=IllIIllIIlIIIlIll,
                                    inputs=[IllIlIIIlllIIlllI],
                                    outputs=[IlIIIIIlllllIlIIl],
                                )
                                IllIlIIIlllIIlllI.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IllIlIIIIlllIIlII],
                                )
                                IlllIlIllllIIIIlI.change(
                                    fn=IIllIlllIIllIllIl,
                                    inputs=[IlllIlIllllIIIIlI],
                                    outputs=[IlIIIIIlllllIlIIl],
                                )
                                IlllIlIllllIIIIlI.change(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IllIlIIIIlllIIlII],
                                )
                            IlIIllIlllllIllll, _IlllIIlIIllllllll = IIIlIllIIllIIIlII(
                                IIIlIIIlIlIIIIIIl.value
                            )
                            with IllIlIlIIllllIlIl.Column():
                                IlIIIllIIIlIIIlIl = IllIlIlIIllllIlIl.Dropdown(
                                    label=IIIllIIllIIIIIIlI(IlllIIllIlIIIIIIl),
                                    choices=IIlllIlIIIlIIlIll(),
                                    value=IlIIllIlllllIllll,
                                    interactive=_IlllIIIlllIllIllI,
                                    allow_custom_value=_IlllIIIlllIllIllI,
                                )
                                IIlIlIllIIIlllllI = IllIlIlIIllllIlIl.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=IIIllIIllIIIIIIlI(IllIIlIlIlIIllIIl),
                                    value=0.75,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIlIlIlllllllIllI.click(
                                    fn=IlllIIIllllIlllIl,
                                    inputs=[],
                                    outputs=[
                                        IIIlIIIlIlIIIIIIl,
                                        IlIIIllIIIlIIIlIl,
                                        IllIlIIIIlllIIlII,
                                    ],
                                )
                                with IllIlIlIIllllIlIl.Column():
                                    IIlIIlIlllIllllIl = IllIlIlIIllllIlIl.Number(
                                        label=IIIllIIllIIIIIIlI(IIlIIIllllIIIlIII),
                                        value=0,
                                    )
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI(IllIlllIlIIlIIllI),
                            open=_IIIIlIlIIIlIlIlll,
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                with IllIlIlIIllllIlIl.Column():
                                    IIIlIIllIIllIlllI = IllIlIlIIllllIlIl.Radio(
                                        label=IIIllIIllIIIIIIlI(IlIIlllIllIIlIIII),
                                        choices=[
                                            _IIIlIIllIlIlIllII,
                                            IIIllIlIllIllllII,
                                            IlIllIIIIIlllllII,
                                            IIlllllIllllIllII,
                                            IIIlIlIlIIlIllIIl,
                                            IIIlllIlIllllIllI,
                                            IIIIlIllIllllllll,
                                            _IlIIllllIIIlIlIlI,
                                            _IIIlIlIIllllIIIlI,
                                            IllIlIlllIlIIllII,
                                        ],
                                        value=IllIlIlllIlIIllII,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlllIllIIllIIIlIl = IllIlIlIIllllIlIl.Checkbox(
                                        label=IIIlIIllIllllllIl,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlIIlIIlllllIIIll = IllIlIlIIllllIlIl.Checkbox(
                                        value=bool(IlllIIIllIIIlIllI),
                                        label=IIIllIIllIIIIIIlI(
                                            "Formant shift inference audio"
                                        ),
                                        info=IIIllIIllIIIIIIlI(
                                            "Used for male to female and vice-versa conversions"
                                        ),
                                        interactive=_IlllIIIlllIllIllI,
                                        visible=_IlllIIIlllIllIllI,
                                    )
                                    IlIIllIIlIllllIII = IllIlIlIIllllIlIl.Slider(
                                        minimum=1,
                                        maximum=512,
                                        step=1,
                                        label=IIIllIIllIIIIIIlI(IIlIlIIlIlIlIllIl),
                                        value=120,
                                        interactive=_IlllIIIlllIllIllI,
                                        visible=_IIIIlIlIIIlIlIlll,
                                    )
                                    IlIIIIIIllllIIlll = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=IIIllIIllIIIIIIlI(IIIIlIlllIIlllIII),
                                        value=3,
                                        step=1,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIlIllIIIIIIIIIl = IllIlIlIIllllIlIl.Slider(
                                        label=IIIllIIllIIIIIIlI(IIlIIIllllllIIllI),
                                        info=IIIllIIllIIIIIIlI(
                                            "Specify minimal pitch for inference [HZ]"
                                        ),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=50,
                                        maximum=16000,
                                        interactive=_IlllIIIlllIllIllI,
                                        visible=not rvc_globals.NotesIrHertz
                                        and IIIlIIllIIllIlllI.value
                                        != _IlIIllllIIIlIlIlI,
                                    )
                                    IIIlIIlIIIllIlIll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IIlIIIllllllIIllI),
                                        info=IIIllIIllIIIIIIlI(
                                            "Specify minimal pitch for inference [NOTE][OCTAVE]"
                                        ),
                                        placeholder="C5",
                                        visible=rvc_globals.NotesIrHertz
                                        and IIIlIIllIIllIlllI.value
                                        != _IlIIllllIIIlIlIlI,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIlIIlIIllllIlIll = IllIlIlIIllllIlIl.Slider(
                                        label=IIIllIIllIIIIIIlI(IlllIlllIIIlIIIlI),
                                        info=IIIllIIllIIIIIIlI(
                                            "Specify max pitch for inference [HZ]"
                                        ),
                                        step=0.1,
                                        minimum=1,
                                        scale=0,
                                        value=1100,
                                        maximum=16000,
                                        interactive=_IlllIIIlllIllIllI,
                                        visible=not rvc_globals.NotesIrHertz
                                        and IIIlIIllIIllIlllI.value
                                        != _IlIIllllIIIlIlIlI,
                                    )
                                    IIlIlIIIlIlIlIlII = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIlllIIIlIIIlI),
                                        info=IIIllIIllIIIIIIlI(
                                            "Specify max pitch for inference [NOTE][OCTAVE]"
                                        ),
                                        placeholder="C6",
                                        visible=rvc_globals.NotesIrHertz
                                        and IIIlIIllIIllIlllI.value
                                        != _IlIIllllIIIlIlIlI,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlIllllIlIlIlIlll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IIlIIlIIIlIIIllll),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                IIIlIIllIIllIlllI.change(
                                    fn=lambda IlllIlIIIIIlIlllI: {
                                        _IIIllIllllIlIllIl: IlllIlIIIIIlIlllI
                                        in [IIIlllIlIllllIllI, IIIIlIllIllllllll],
                                        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                                    },
                                    inputs=[IIIlIIllIIllIlllI],
                                    outputs=[IlIIllIIlIllllIII],
                                )
                                IIIlIIllIIllIlllI.change(
                                    fn=IllllIIIIlIlllIll,
                                    inputs=[IIIlIIllIIllIlllI],
                                    outputs=[
                                        IIIlIllIIIIIIIIIl,
                                        IIIlIIlIIIllIlIll,
                                        IIlIIlIIllllIlIll,
                                        IIlIlIIIlIlIlIlII,
                                    ],
                                )
                                with IllIlIlIIllllIlIl.Column():
                                    IllIIlllIlIIIlIII = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=IIIllIIllIIIIIIlI(IIlIlIIIIIIIIIIIl),
                                        value=0,
                                        step=1,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllIIIlIIlllIIlII = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IIIllIIllIIIIIIlI(IIlllIllIllllIllI),
                                        value=0.25,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIIlIlllllIIllII = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=IIIllIIllIIIIIIlI(IIlIlIIIIllllllII),
                                        value=0.33,
                                        step=0.01,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllIlIIllIllllIIl = IllIlIlIIllllIlIl.File(
                                        label=IIIllIIllIIIIIIlI(
                                            "F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:"
                                        )
                                    )
                                    IIIllIIlIIllllIII = IllIlIlIIllllIlIl.Dropdown(
                                        value="",
                                        choices=IIIllllIIlIlllIlI(),
                                        label=IIIllIIllIIIIIIlI(
                                            "Browse presets for formanting"
                                        ),
                                        info=IIIllIIllIIIIIIlI(
                                            "Presets are located in formantshiftcfg/ folder"
                                        ),
                                        visible=bool(IlllIIIllIIIlIllI),
                                    )
                                    IIIllIIIlIIllIlII = IllIlIlIIllllIlIl.Button(
                                        value="🔄",
                                        visible=bool(IlllIIIllIIIlIllI),
                                        variant=IIlIIIIlIlIIlIlll,
                                    )
                                    IllIllIIlIlIlIIlI = IllIlIlIIllllIlIl.Slider(
                                        value=IIIIllIIIIIlIlllI,
                                        info=IIIllIIllIIIIIIlI(IlIIllIIllIlIIIll),
                                        label=IIIllIIllIIIIIIlI(
                                            "Quefrency for formant shifting"
                                        ),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(IlllIIIllIIIlIllI),
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIIIIIIllIlIllll = IllIlIlIIllllIlIl.Slider(
                                        value=IlIIIIlllIIIllllI,
                                        info=IIIllIIllIIIIIIlI(IlIIllIIllIlIIIll),
                                        label=IIIllIIllIIIIIIlI(
                                            "Timbre for formant shifting"
                                        ),
                                        minimum=0.0,
                                        maximum=16.0,
                                        step=0.1,
                                        visible=bool(IlllIIIllIIIlIllI),
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlIlIIIIlllIlIllI = IllIlIlIIllllIlIl.Button(
                                        IIIllIIllIIIIIIlI("Apply"),
                                        variant=IIlIIIIlIlIIlIlll,
                                        visible=bool(IlllIIIllIIIlIllI),
                                    )
                                IIIllIIlIIllllIII.change(
                                    fn=IllllIIlIlIlIlIll,
                                    inputs=[
                                        IIIllIIlIIllllIII,
                                        IllIllIIlIlIlIIlI,
                                        IIIIIIIIllIlIllll,
                                    ],
                                    outputs=[IllIllIIlIlIlIIlI, IIIIIIIIllIlIllll],
                                )
                                IlIIlIIlllllIIIll.change(
                                    fn=IlIllllllllIlllIl,
                                    inputs=[
                                        IlIIlIIlllllIIIll,
                                        IllIllIIlIlIlIIlI,
                                        IIIIIIIIllIlIllll,
                                    ],
                                    outputs=[
                                        IlIIlIIlllllIIIll,
                                        IllIllIIlIlIlIIlI,
                                        IIIIIIIIllIlIllll,
                                        IlIlIIIIlllIlIllI,
                                        IIIllIIlIIllllIII,
                                        IIIllIIIlIIllIlII,
                                    ],
                                )
                                IlIlIIIIlllIlIllI.click(
                                    fn=IllIIIllIlIllllll,
                                    inputs=[IllIllIIlIlIlIIlI, IIIIIIIIllIlIllll],
                                    outputs=[IllIllIIlIlIlIIlI, IIIIIIIIllIlIllll],
                                )
                                IIIllIIIlIIllIlII.click(
                                    fn=IIIIIIIIIIIIlIlll,
                                    inputs=[
                                        IIIllIIlIIllllIII,
                                        IllIllIIlIlIlIIlI,
                                        IIIIIIIIllIlIllll,
                                    ],
                                    outputs=[
                                        IIIllIIlIIllllIII,
                                        IllIllIIlIlIlIIlI,
                                        IIIIIIIIllIlIllll,
                                    ],
                                )
                    with IllIlIlIIllllIlIl.Row():
                        IIIlIIllIIlllIlII = IllIlIlIIllllIlIl.Textbox(
                            label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll)
                        )
                        IIlIIIIIlIIlIlllI = IllIlIlIIllllIlIl.Audio(
                            label=IIIllIIllIIIIIIlI(IlIlllIlllIllIllI)
                        )
                    IllllIIlIIlIlllIl = IllIlIlIIllllIlIl.Button(
                        IIIllIIllIIIIIIlI(IlIlIIllllIlIIIII), variant=IIlIIIIlIlIIlIlll
                    ).style(full_width=_IlllIIIlllIllIllI)
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Row():
                            IllllIIlIIlIlllIl.click(
                                IlIIlIIlIIlllIIII,
                                [
                                    IllllIlIIlIIlIlll,
                                    IlIIIIIlllllIlIIl,
                                    IllIlIIIIlllIIlII,
                                    IIlIIlIlllIllllIl,
                                    IllIlIIllIllllIIl,
                                    IIIlIIllIIllIlllI,
                                    IlIllllIlIlIlIlll,
                                    IlIIIllIIIlIIIlIl,
                                    IIlIlIllIIIlllllI,
                                    IlIIIIIIllllIIlll,
                                    IllIIlllIlIIIlIII,
                                    IllIIIlIIlllIIlII,
                                    IIIIlIlllllIIllII,
                                    IlIIllIIlIllllIII,
                                    IIIlIllIIIIIIIIIl,
                                    IIIlIIlIIIllIlIll,
                                    IIlIIlIIllllIlIll,
                                    IIlIlIIIlIlIlIlII,
                                    IlllIllIIllIIIlIl,
                                ],
                                [IIIlIIllIIlllIlII, IIlIIIIIlIIlIlllI],
                            )
                with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Batch")):
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Row():
                            with IllIlIlIIllllIlIl.Column():
                                IIIllIlIlIIlIIIIl = IllIlIlIIllllIlIl.Number(
                                    label=IIIllIIllIIIIIIlI(IIlIIIllllIIIlIII), value=0
                                )
                                IlllIlIlllIIIIIII = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI("Specify output folder:"),
                                    value=IlIlIIlllIIIIIIIl,
                                )
                                IIIIIlIIIlIllIIll = IllIlIlIIllllIlIl.Dropdown(
                                    label=IIIllIIllIIIIIIlI(IlllIIllIlIIIIIIl),
                                    choices=IIlllIlIIIlIIlIll(),
                                    value=IlIIllIlllllIllll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIIlIIIlIlIIIIIIl.select(
                                    fn=IIIlIllIIllIIIlII,
                                    inputs=[IIIlIIIlIlIIIIIIl],
                                    outputs=[IlIIIllIIIlIIIlIl, IIIIIlIIIlIllIIll],
                                )
                                IIlIlIlllllllIllI.click(
                                    fn=lambda: IlllIIIllllIlllIl()[1],
                                    inputs=[],
                                    outputs=IIIIIlIIIlIllIIll,
                                )
                                IIIIlIlIlIIllIllI = IllIlIlIIllllIlIl.Slider(
                                    minimum=0,
                                    maximum=1,
                                    label=IIIllIIllIIIIIIlI(IllIIlIlIlIIllIIl),
                                    value=0.75,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                            with IllIlIlIIllllIlIl.Column():
                                IllllIllIIlllllll = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(
                                        "Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):"
                                    ),
                                    value=os.path.join(
                                        IlIIIllllllIllIll, _IIlIIIIIlIlIIIllI
                                    ),
                                    lines=2,
                                )
                                IllIIlllIlIlllIlI = IllIlIlIIllllIlIl.File(
                                    file_count=IlIlIllIlIIIIlIII,
                                    label=IIIllIIllIIIIIIlI(IIlIllllIllllIIIl),
                                )
                    with IllIlIlIIllllIlIl.Row():
                        with IllIlIlIIllllIlIl.Column():
                            IllIlIlllIIlIllIl = IllIlIlIIllllIlIl.Checkbox(
                                value=_IIIIlIlIIIlIlIlll,
                                label=IIIllIIllIIIIIIlI(IllIlllIlIIlIIllI),
                                interactive=_IlllIIIlllIllIllI,
                            )
                            with IllIlIlIIllllIlIl.Row(
                                visible=_IIIIlIlIIIlIlIlll
                            ) as IIlIlllIllIIIIlll:
                                with IllIlIlIIllllIlIl.Row(
                                    label=IIIllIIllIIIIIIlI(IllIlllIlIIlIIllI),
                                    open=_IIIIlIlIIIlIlIlll,
                                ):
                                    with IllIlIlIIllllIlIl.Column():
                                        IllIllllIlllIIllI = IllIlIlIIllllIlIl.Textbox(
                                            label=IIIllIIllIIIIIIlI(IIlIIlIIIlIIIllll),
                                            value="",
                                            interactive=_IlllIIIlllIllIllI,
                                        )
                                        IllIlIIIlllIIllll = IllIlIlIIllllIlIl.Radio(
                                            label=IIIllIIllIIIIIIlI(IlIIlllIllIIlIIII),
                                            choices=[
                                                _IIIlIIllIlIlIllII,
                                                IIIllIlIllIllllII,
                                                IlIllIIIIIlllllII,
                                                IIlllllIllllIllII,
                                                IIIlIlIlIIlIllIIl,
                                                IIIlllIlIllllIllI,
                                                IIIIlIllIllllllll,
                                                _IlIIllllIIIlIlIlI,
                                                _IIIlIlIIllllIIIlI,
                                                IllIlIlllIlIIllII,
                                            ],
                                            value=IllIlIlllIlIIllII,
                                            interactive=_IlllIIIlllIllIllI,
                                        )
                                        IlllIllIIllIIIlIl = IllIlIlIIllllIlIl.Checkbox(
                                            label=IIIlIIllIllllllIl,
                                            interactive=_IlllIIIlllIllIllI,
                                        )
                                        IIllllIlIIllIllll = IllIlIlIIllllIlIl.Radio(
                                            label=IIIllIIllIIIIIIlI(
                                                "Export file format"
                                            ),
                                            choices=[
                                                _IIlIlIlIIllIIIIll,
                                                _IlIIlllIlIllIIlll,
                                                _IIIlIllllIIlIIIIl,
                                                _IlIIllIllllIlIIII,
                                            ],
                                            value=_IIlIlIlIIllIIIIll,
                                            interactive=_IlllIIIlllIllIllI,
                                        )
                                with IllIlIlIIllllIlIl.Column():
                                    IIlllllIlIIllllIl = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=48000,
                                        label=IIIllIIllIIIIIIlI(IIlIlIIIIIIIIIIIl),
                                        value=0,
                                        step=1,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlIIIlIIIlIllIllI = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IIIllIIllIIIIIIlI(IIlllIllIllllIllI),
                                        value=1,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIIlIIllIIllIlll = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=0.5,
                                        label=IIIllIIllIIIIIIlI(IIlIlIIIIllllllII),
                                        value=0.33,
                                        step=0.01,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIllllIIIlIIIlII = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=7,
                                        label=IIIllIIllIIIIIIlI(IIIIlIlllIIlllIII),
                                        value=3,
                                        step=1,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                            IIllIlIIllIlIlIlI = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll)
                            )
                            IIlIIlIlIlIlIIlll = IllIlIlIIllllIlIl.Button(
                                IIIllIIllIIIIIIlI(IlIlIIllllIlIIIII),
                                variant=IIlIIIIlIlIIlIlll,
                            )
                            IIlIIlIlIlIlIIlll.click(
                                IlllIllIIIIlllllI,
                                [
                                    IllllIlIIlIIlIlll,
                                    IllllIllIIlllllll,
                                    IlllIlIlllIIIIIII,
                                    IllIIlllIlIlllIlI,
                                    IIIllIlIlIIlIIIIl,
                                    IllIlIIIlllIIllll,
                                    IllIllllIlllIIllI,
                                    IIIIIlIIIlIllIIll,
                                    IIIIlIlIlIIllIllI,
                                    IIIllllIIIlIIIlII,
                                    IIlllllIlIIllllIl,
                                    IlIIIlIIIlIllIllI,
                                    IIIIlIIllIIllIlll,
                                    IIllllIlIIllIllll,
                                    IlIIllIIlIllllIII,
                                    IIIlIllIIIIIIIIIl
                                    if not rvc_globals.NotesIrHertz
                                    else IIIlIIlIIIllIlIll,
                                    IIlIIlIIllllIlIll
                                    if not rvc_globals.NotesIrHertz
                                    else IIlIlIIIlIlIlIlII,
                                    IlllIllIIllIIIlIl,
                                ],
                                [IIllIlIIllIlIlIlI],
                            )
                    IIIlIIIlIlIIIIIIl.change(
                        fn=IIIlllIllllIIIIll,
                        inputs=[
                            IIIlIIIlIlIIIIIIl,
                            IIIIlIlllllIIllII,
                            IIIIlIIllIIllIlll,
                        ],
                        outputs=[
                            IllllIlIIlIIlIlll,
                            IIIIlIlllllIIllII,
                            IIIIlIIllIIllIlll,
                        ],
                    )
                    (
                        IllllIlIIlIIlIlll,
                        IIIIlIlllllIIllII,
                        IIIIlIIllIIllIlll,
                    ) = IIIlllIllllIIIIll(
                        IIIlIIIlIlIIIIIIl.value, IIIIlIlllllIIllII, IIIIlIIllIIllIlll
                    )

                    def IlIlIIlllIlIIIIIl(IIIlIIlIlIlllllIl):
                        return {
                            _IIIllIllllIlIllIl: IIIlIIlIlIlllllIl,
                            _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                        }

                    IllIlIlllIIlIllIl.change(
                        fn=IlIlIIlllIlIIIIIl,
                        inputs=[IllIlIlllIIlIllIl],
                        outputs=[IIlIlllIllIIIIlll],
                    )
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Train")):
                with IllIlIlIIllllIlIl.Accordion(
                    label=IIIllIIllIIIIIIlI("Step 1: Processing data")
                ):
                    with IllIlIlIIllllIlIl.Row():
                        IlllIllllIlIlIIll = IllIlIlIIllllIlIl.Textbox(
                            label=IIIllIIllIIIIIIlI("Enter the model name:"),
                            value=IIIllIIllIIIIIIlI("Model_Name"),
                        )
                        IlllIIllllIllIllI = IllIlIlIIllllIlIl.Radio(
                            label=IIIllIIllIIIIIIlI(IlIIlIIIIllIIllIl),
                            choices=[
                                _IIIllIIIlIIllIlII,
                                _IlIlllllIIIIIlIlI,
                                _IllIIIIIIlIIIIlll,
                            ],
                            value=_IIIllIIIlIIllIlII,
                            interactive=_IlllIIIlllIllIllI,
                        )
                        IIllIlIIlIIIIlllI = IllIlIlIIllllIlIl.Checkbox(
                            label=IIIllIIllIIIIIIlI(IIIIllllIIIlllIlI),
                            value=_IlllIIIlllIllIllI,
                            interactive=_IlllIIIlllIllIllI,
                        )
                        IIIllIIIlllIllIII = IllIlIlIIllllIlIl.Radio(
                            label=IIIllIIllIIIIIIlI("Version:"),
                            choices=[_IlIIIIIIlIlIllIIl, _IIIllllllIllIIIII],
                            value=_IIIllllllIllIIIII,
                            interactive=_IlllIIIlllIllIllI,
                            visible=_IlllIIIlllIllIllI,
                        )
                        IlIIIIlIlIIlIlllI = IllIlIlIIllllIlIl.Slider(
                            minimum=0,
                            maximum=IlIllIlIlIIlIlIll.n_cpu,
                            step=1,
                            label=IIIllIIllIIIIIIlI("Number of CPU processes:"),
                            value=int(
                                IIlIlllllIlIlIIlI.ceil(IlIllIlIlIIlIlIll.n_cpu / 1.5)
                            ),
                            interactive=_IlllIIIlllIllIllI,
                        )
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Accordion(
                        label=IIIllIIllIIIIIIlI("Step 2: Skipping pitch extraction")
                    ):
                        with IllIlIlIIllllIlIl.Row():
                            with IllIlIlIIllllIlIl.Column():
                                IlIlIIIllIIIllIIl = IllIlIlIIllllIlIl.Dropdown(
                                    choices=sorted(IlIIIIIIIIllIIlIl),
                                    label=IIIllIIllIIIIIIlI("Select your dataset:"),
                                    value=IlIIIllIlIlIllllI(),
                                )
                                IIIlllIIlllIlIlll = IllIlIlIIllllIlIl.Button(
                                    IIIllIIllIIIIIIlI("Update list"),
                                    variant=IIlIIIIlIlIIlIlll,
                                )
                            IllIIIllIIIIIlllI = IllIlIlIIllllIlIl.Slider(
                                minimum=0,
                                maximum=4,
                                step=1,
                                label=IIIllIIllIIIIIIlI("Specify the model ID:"),
                                value=0,
                                interactive=_IlllIIIlllIllIllI,
                            )
                            IIIlllIIlllIlIlll.click(
                                easy_infer.update_dataset_list,
                                [IllIIIllIIIIIlllI],
                                IlIlIIIllIIIllIIl,
                            )
                            IIlIIlIlIlIlIIlll = IllIlIlIIllllIlIl.Button(
                                IIIllIIllIIIIIIlI("Process data"),
                                variant=IIlIIIIlIlIIlIlll,
                            )
                            IIlIllIlIIIIIIIII = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll), value=""
                            )
                            IIlIIlIlIlIlIIlll.click(
                                IllIIllIllIllIlll,
                                [
                                    IlIlIIIllIIIllIIl,
                                    IlllIllllIlIlIIll,
                                    IlllIIllllIllIllI,
                                    IlIIIIlIlIIlIlllI,
                                ],
                                [IIlIllIlIIIIIIIII],
                            )
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Accordion(
                        label=IIIllIIllIIIIIIlI("Step 3: Extracting features")
                    ):
                        with IllIlIlIIllllIlIl.Row():
                            with IllIlIlIIllllIlIl.Column():
                                IlIlIlIlIIIIlIIIl = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(IlIIIlIIlIlIlIIII),
                                    value=IIIlIlllllIllIlII,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI("GPU Information:"),
                                    value=IIlIIlllIlIlIIIll,
                                )
                            with IllIlIlIIllllIlIl.Column():
                                IIIIlIIIIIllIllII = IllIlIlIIllllIlIl.Radio(
                                    label=IIIllIIllIIIIIIlI(IlIIlllIllIIlIIII),
                                    choices=[
                                        _IIIlIIllIlIlIllII,
                                        IIIllIlIllIllllII,
                                        IlIllIIIIIlllllII,
                                        IIlllllIllllIllII,
                                        IIIlllIlIllllIllI,
                                        _IlIIllllIIIlIlIlI,
                                    ],
                                    value=_IlIIllllIIIlIlIlI,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IlllIIlIIlIlIIlIl = IllIlIlIIllllIlIl.Slider(
                                    minimum=1,
                                    maximum=512,
                                    step=1,
                                    label=IIIllIIllIIIIIIlI(IIlIlIIlIlIlIllIl),
                                    value=64,
                                    interactive=_IlllIIIlllIllIllI,
                                    visible=_IIIIlIlIIIlIlIlll,
                                )
                                IIIIlIIIIIllIllII.change(
                                    fn=lambda IIIIIllIlllIIlIlI: {
                                        _IIIllIllllIlIllIl: IIIIIllIlllIIlIlI
                                        in [IIIlllIlIllllIllI, IIIIlIllIllllllll],
                                        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                                    },
                                    inputs=[IIIIlIIIIIllIllII],
                                    outputs=[IlllIIlIIlIlIIlIl],
                                )
                            IlllIlIIIllIIIIll = IllIlIlIIllllIlIl.Button(
                                IIIllIIllIIIIIIlI("Feature extraction"),
                                variant=IIlIIIIlIlIIlIlll,
                            )
                            IlIlIlIlIIllIllII = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                value="",
                                max_lines=8,
                                interactive=_IIIIlIlIIIlIlIlll,
                            )
                            IlllIlIIIllIIIIll.click(
                                IllllIllIllIIIllI,
                                [
                                    IlIlIlIlIIIIlIIIl,
                                    IlIIIIlIlIIlIlllI,
                                    IIIIlIIIIIllIllII,
                                    IIllIlIIlIIIIlllI,
                                    IlllIllllIlIlIIll,
                                    IIIllIIIlllIllIII,
                                    IlllIIlIIlIlIIlIl,
                                ],
                                [IlIlIlIlIIllIllII],
                            )
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Row():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI("Step 4: Model training started")
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                IIlIlIlllIllIIIII = IllIlIlIIllllIlIl.Slider(
                                    minimum=1,
                                    maximum=100,
                                    step=1,
                                    label=IIIllIIllIIIIIIlI("Save frequency:"),
                                    value=10,
                                    interactive=_IlllIIIlllIllIllI,
                                    visible=_IlllIIIlllIllIllI,
                                )
                                IIIlIllllIllIIIIl = IllIlIlIIllllIlIl.Slider(
                                    minimum=1,
                                    maximum=10000,
                                    step=2,
                                    label=IIIllIIllIIIIIIlI("Training epochs:"),
                                    value=750,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIllIlIlllllllIIl = IllIlIlIIllllIlIl.Slider(
                                    minimum=1,
                                    maximum=50,
                                    step=1,
                                    label=IIIllIIllIIIIIIlI("Batch size per GPU:"),
                                    value=IIlIllIllIllIIIIl,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                            with IllIlIlIIllllIlIl.Row():
                                IIlIIlIlIlIIlIlIl = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI(
                                        "Whether to save only the latest .ckpt file to save hard drive space"
                                    ),
                                    value=_IlllIIIlllIllIllI,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIlIlllIIIlllIIll = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI(
                                        "Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training"
                                    ),
                                    value=_IIIIlIlIIIlIlIlll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IllIIllIIllIlllII = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI(
                                        "Save a small final model to the 'weights' folder at each save point"
                                    ),
                                    value=_IlllIIIlllIllIllI,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                            with IllIlIlIIllllIlIl.Row():
                                IIlIllIIlIllIlIII = IllIlIlIIllllIlIl.Textbox(
                                    lines=4,
                                    label=IIIllIIllIIIIIIlI(
                                        "Load pre-trained base model G path:"
                                    ),
                                    value="/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth",
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIIIlIIlIllIlIlIl = IllIlIlIIllllIlIl.Textbox(
                                    lines=4,
                                    label=IIIllIIllIIIIIIlI(
                                        "Load pre-trained base model D path:"
                                    ),
                                    value="/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth",
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIlIIIllllIlllIIl = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(IlIIIlIIlIlIlIIII),
                                    value=IIIlIlllllIllIlII,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IlllIIllllIllIllI.change(
                                    IllllllIIIIlIIIIl,
                                    [
                                        IlllIIllllIllIllI,
                                        IIllIlIIlIIIIlllI,
                                        IIIllIIIlllIllIII,
                                    ],
                                    [IIlIllIIlIllIlIII, IIIIlIIlIllIlIlIl],
                                )
                                IIIllIIIlllIllIII.change(
                                    IlllIIIlIIIllllII,
                                    [
                                        IlllIIllllIllIllI,
                                        IIllIlIIlIIIIlllI,
                                        IIIllIIIlllIllIII,
                                    ],
                                    [
                                        IIlIllIIlIllIlIII,
                                        IIIIlIIlIllIlIlIl,
                                        IlllIIllllIllIllI,
                                    ],
                                )
                                IIllIlIIlIIIIlllI.change(
                                    fn=IIlllIIIlIlIllIll,
                                    inputs=[
                                        IIllIlIIlIIIIlllI,
                                        IlllIIllllIllIllI,
                                        IIIllIIIlllIllIII,
                                    ],
                                    outputs=[
                                        IIIIlIIIIIllIllII,
                                        IIlIllIIlIllIlIII,
                                        IIIIlIIlIllIlIlIl,
                                    ],
                                )
                                IIllIlIIlIIIIlllI.change(
                                    fn=lambda IIlIllIlllIlIIIlI: {
                                        _IIIllIllllIlIllIl: IIlIllIlllIlIIIlI
                                        in [IIIlllIlIllllIllI, IIIIlIllIllllllll],
                                        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                                    },
                                    inputs=[IIIIlIIIIIllIllII],
                                    outputs=[IlllIIlIIlIlIIlIl],
                                )
                                IIIIIlIlIIIllllIl = IllIlIlIIllllIlIl.Button(
                                    IIIllIIllIIIIIIlI("Stop training"),
                                    variant=IIlIIIIlIlIIlIlll,
                                    visible=_IIIIlIlIIIlIlIlll,
                                )
                                IIlIllIlllllIIIlI = IllIlIlIIllllIlIl.Button(
                                    IIIllIIllIIIIIIlI("Train model"),
                                    variant=IIlIIIIlIlIIlIlll,
                                    visible=_IlllIIIlllIllIllI,
                                )
                                IIlIllIlllllIIIlI.click(
                                    fn=IIIIllIIIIIlIllll,
                                    inputs=[
                                        IllIlIlIIllllIlIl.Number(
                                            value=0, visible=_IIIIlIlIIIlIlIlll
                                        )
                                    ],
                                    outputs=[IIlIllIlllllIIIlI, IIIIIlIlIIIllllIl],
                                )
                                IIIIIlIlIIIllllIl.click(
                                    fn=IIIIllIIIIIlIllll,
                                    inputs=[
                                        IllIlIlIIllllIlIl.Number(
                                            value=1, visible=_IIIIlIlIIIlIlIlll
                                        )
                                    ],
                                    outputs=[IIlIllIlllllIIIlI, IIIIIlIlIIIllllIl],
                                )
                                with IllIlIlIIllllIlIl.Column():
                                    IIlIlIIlIllIIIllI = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                        value="",
                                        max_lines=4,
                                    )
                                    IllllIllIlIIIIIll = IllIlIlIIllllIlIl.Dropdown(
                                        label=IIIllIIllIIIIIIlI("Save type"),
                                        choices=[
                                            IIIllIIllIIIIIIlI("Save all"),
                                            IIIllIIllIIIIIIlI("Save D and G"),
                                            IIIllIIllIIIIIIlI("Save voice"),
                                        ],
                                        value=IIIllIIllIIIIIIlI("Choose the method"),
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllIlIIIlIlllIlIl = IllIlIlIIllllIlIl.Button(
                                        IIIllIIllIIIIIIlI("Train feature index"),
                                        variant=IIlIIIIlIlIIlIlll,
                                    )
                                    IIIllIIlllIIIIlII = IllIlIlIIllllIlIl.Button(
                                        IIIllIIllIIIIIIlI("Save model"),
                                        variant=IIlIIIIlIlIIlIlll,
                                    )
                                IllIIllIIllIlllII.change(
                                    fn=lambda IlIllIlIllIllllII: {
                                        _IIIllIllllIlIllIl: IlIllIlIllIllllII,
                                        _IIllIIllllllIllII: _IlllllIlIIlIlIlII,
                                    },
                                    inputs=[IllIIllIIllIlllII],
                                    outputs=[IIlIlIlllIllIIIII],
                                )
                            IIlIllIlllllIIIlI.click(
                                IllIIlllIlllIIIIl,
                                [
                                    IlllIllllIlIlIIll,
                                    IlllIIllllIllIllI,
                                    IIllIlIIlIIIIlllI,
                                    IllIIIllIIIIIlllI,
                                    IIlIlIlllIllIIIII,
                                    IIIlIllllIllIIIIl,
                                    IIllIlIlllllllIIl,
                                    IIlIIlIlIlIIlIlIl,
                                    IIlIllIIlIllIlIII,
                                    IIIIlIIlIllIlIlIl,
                                    IIlIIIllllIlllIIl,
                                    IIlIlllIIIlllIIll,
                                    IllIIllIIllIlllII,
                                    IIIllIIIlllIllIII,
                                ],
                                [
                                    IIlIlIIlIllIIIllI,
                                    IIIIIlIlIIIllllIl,
                                    IIlIllIlllllIIIlI,
                                ],
                            )
                            IllIlIIIlIlllIlIl.click(
                                IIIIllIllllllIllI,
                                [IlllIllllIlIlIIll, IIIllIIIlllIllIII],
                                IIlIlIIlIllIIIllI,
                            )
                            IIIllIIlllIIIIlII.click(
                                easy_infer.save_model,
                                [IlllIllllIlIlIIll, IllllIllIlIIIIIll],
                                IIlIlIIlIllIIIllI,
                            )
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Row():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI(
                                "Step 5: Export lowest points on a graph of the model"
                            )
                        ):
                            IlIIIIllIlllIIIIl = IllIlIlIIllllIlIl.Textbox(
                                visible=_IIIIlIlIIIlIlIlll
                            )
                            IIIIIIIIllIIIIIll = IllIlIlIIllllIlIl.Textbox(
                                visible=_IIIIlIlIIIlIlIlll
                            )
                            IIlIIIlIIlIIlllll = IllIlIlIIllllIlIl.Textbox(
                                visible=_IIIIlIlIIIlIlIlll, value=IIllIlIIIIlIlllIl
                            )
                            with IllIlIlIIllllIlIl.Row():
                                IIIlIllllIllIllIl = IllIlIlIIllllIlIl.Slider(
                                    minimum=1,
                                    maximum=25,
                                    label=IIIllIIllIIIIIIlI(
                                        "How many lowest points to save:"
                                    ),
                                    value=3,
                                    step=1,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IlllllllIlIIllllI = IllIlIlIIllllIlIl.Button(
                                    value=IIIllIIllIIIIIIlI(
                                        "Export lowest points of a model"
                                    ),
                                    variant=IIlIIIIlIlIIlIlll,
                                )
                                IlIlIIIllllIIIIIl = IllIlIlIIllllIlIl.File(
                                    file_count=IlIlIllIlIIIIlIII,
                                    label=IIIllIIllIIIIIIlI("Output models:"),
                                    interactive=_IIIIlIlIIIlIlIlll,
                                )
                            with IllIlIlIIllllIlIl.Row():
                                IllIlIlIlIlllllll = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                    value="",
                                    max_lines=10,
                                )
                                IIIIllIllIlIIIlII = IllIlIlIIllllIlIl.Dataframe(
                                    label=IIIllIIllIIIIIIlI(
                                        "Stats of selected models:"
                                    ),
                                    datatype="number",
                                    type="pandas",
                                )
                            IlllllllIlIIllllI.click(
                                lambda IIlIIlIllIlIIlIIl: os.path.join(
                                    _IIIlIlIlIIIIllllI, IIlIIlIllIlIIlIIl, "lowestvals"
                                ),
                                inputs=[IlllIllllIlIlIIll],
                                outputs=[IlIIIIllIlllIIIIl],
                            )
                            IlllllllIlIIllllI.click(
                                fn=IIlllllllIIIIIlIl.main,
                                inputs=[
                                    IlllIllllIlIlIIll,
                                    IIlIlIlllIllIIIII,
                                    IIIlIllllIllIllIl,
                                ],
                                outputs=[IIIIIIIIllIIIIIll],
                            )
                            IIIIIIIIllIIIIIll.change(
                                fn=IIlllllllIIIIIlIl.selectweights,
                                inputs=[
                                    IlllIllllIlIlIIll,
                                    IIIIIIIIllIIIIIll,
                                    IIlIIIlIIlIIlllll,
                                    IlIIIIllIlllIIIIl,
                                ],
                                outputs=[
                                    IllIlIlIlIlllllll,
                                    IlIlIIIllllIIIIIl,
                                    IIIIllIllIlIIIlII,
                                ],
                            )
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("UVR5")):
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Row():
                        with IllIlIlIIllllIlIl.Column():
                            IIlIlllIlIlIIIlII = IllIlIlIIllllIlIl.Radio(
                                label=IIIllIIllIIIIIIlI("Model Architecture:"),
                                choices=[_IlIIIIIllllIlIlII, _IIIIlIIllIIlllIll],
                                value=_IlIIIIIllllIlIlII,
                                interactive=_IlllIIIlllIllIllI,
                            )
                            IIIIllllIlllllIlI = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(
                                    "Enter the path of the audio folder to be processed:"
                                ),
                                value=os.path.join(
                                    IlIIIllllllIllIll, _IIlIIIIIlIlIIIllI
                                ),
                            )
                            IlIIlIIIIIlIlIIlI = IllIlIlIIllllIlIl.File(
                                file_count=IlIlIllIlIIIIlIII,
                                label=IIIllIIllIIIIIIlI(IIlIllllIllllIIIl),
                            )
                        with IllIlIlIIllllIlIl.Column():
                            IlIIlIIlIIlllllIl = IllIlIlIIllllIlIl.Dropdown(
                                label=IIIllIIllIIIIIIlI("Model:"),
                                choices=IllllIIlIIllIIlll,
                            )
                            IIlllllIlIIIIIlII = IllIlIlIIllllIlIl.Slider(
                                minimum=0,
                                maximum=20,
                                step=1,
                                label="Vocal Extraction Aggressive",
                                value=10,
                                interactive=_IlllIIIlllIllIllI,
                                visible=_IIIIlIlIIIlIlIlll,
                            )
                            IIlllIIlIIIIlllII = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(
                                    "Specify the output folder for vocals:"
                                ),
                                value=IlIlIIlllIIIIIIIl,
                            )
                            IlllIIIIlllIlllIl = IllIlIlIIllllIlIl.Textbox(
                                label=IIIllIIllIIIIIIlI(
                                    "Specify the output folder for accompaniment:"
                                ),
                                value=IlIlIIlllIIIIIIIl,
                            )
                            IllIIIlllllIIIlll = IllIlIlIIllllIlIl.Radio(
                                label=IIIllIIllIIIIIIlI("Export file format:"),
                                choices=[
                                    _IIlIlIlIIllIIIIll,
                                    _IlIIlllIlIllIIlll,
                                    _IIIlIllllIIlIIIIl,
                                    _IlIIllIllllIlIIII,
                                ],
                                value=_IlIIlllIlIllIIlll,
                                interactive=_IlllIIIlllIllIllI,
                            )
                        IIlIlllIlIlIIIlII.change(
                            fn=IIlllIIllllllllll,
                            inputs=IIlIlllIlIlIIIlII,
                            outputs=IlIIlIIlIIlllllIl,
                        )
                        IlllIlIIIllIIIIll = IllIlIlIIllllIlIl.Button(
                            IIIllIIllIIIIIIlI(IlIlIIllllIlIIIII),
                            variant=IIlIIIIlIlIIlIlll,
                        )
                        IlllIlIIIIlIlIlII = IllIlIlIIllllIlIl.Textbox(
                            label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll)
                        )
                        IlllIlIIIllIIIIll.click(
                            IIIIllIllIllllIIl,
                            [
                                IlIIlIIlIIlllllIl,
                                IIIIllllIlllllIlI,
                                IIlllIIlIIIIlllII,
                                IlIIlIIIIIlIlIIlI,
                                IlllIIIIlllIlllIl,
                                IIlllllIlIIIIIlII,
                                IllIIIlllllIIIlll,
                                IIlIlllIlIlIIIlII,
                            ],
                            [IlllIlIIIIlIlIlII],
                        )
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("TTS")):
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Column():
                        IIllllIlIlIIlIIII = IllIlIlIIllllIlIl.Textbox(
                            label=IIIllIIllIIIIIIlI("Text:"),
                            placeholder=IIIllIIllIIIIIIlI(
                                "Enter the text you want to convert to voice..."
                            ),
                            lines=6,
                        )
                with IllIlIlIIllllIlIl.Group():
                    with IllIlIlIIllllIlIl.Row():
                        with IllIlIlIIllllIlIl.Column():
                            IllIIIlIlIIIllIIl = [_IIlIIllIlIIIIlIll, _IIIlllIlllIIIIlII]
                            IIIllllIlIllIIIIl = IllIlIlIIllllIlIl.Dropdown(
                                IllIIIlIlIIIllIIl,
                                value=_IIlIIllIlIIIIlIll,
                                label=IIIllIIllIIIIIIlI("TTS Method:"),
                                visible=_IlllIIIlllIllIllI,
                            )
                            IIIIlIIIIlIlIlIlI = IllIlIlIIllllIlIl.Dropdown(
                                IlIIIlllIIIlIIIII,
                                label=IIIllIIllIIIIIIlI("TTS Model:"),
                                visible=_IlllIIIlllIllIllI,
                            )
                            IIIllllIlIllIIIIl.change(
                                fn=IlIlIlllIlllIIIlI,
                                inputs=IIIllllIlIllIIIIl,
                                outputs=IIIIlIIIIlIlIlIlI,
                            )
                        with IllIlIlIIllllIlIl.Column():
                            IlllIlIllIIIIIIIl = IllIlIlIIllllIlIl.Dropdown(
                                label=IIIllIIllIIIIIIlI("RVC Model:"),
                                choices=sorted(IIlIIIlIIIlIIlIII),
                                value=IlllIlIlIllllIlII,
                            )
                            IlIIllIlllllIllll, _IlllIIlIIllllllll = IIIlIllIIllIIIlII(
                                IlllIlIllIIIIIIIl.value
                            )
                            IIlllIllIlIIlIllI = IllIlIlIIllllIlIl.Dropdown(
                                label=IIIllIIllIIIIIIlI("Select the .index file:"),
                                choices=IIlllIlIIIlIIlIll(),
                                value=IlIIllIlllllIllll,
                                interactive=_IlllIIIlllIllIllI,
                                allow_custom_value=_IlllIIIlllIllIllI,
                            )
                with IllIlIlIIllllIlIl.Row():
                    IllIIIIlIlIlIlIlI = IllIlIlIIllllIlIl.Button(
                        IIIllIIllIIIIIIlI(IlIllIlIIIIIlIlII), variant=IIlIIIIlIlIIlIlll
                    )
                    IllIIIIlIlIlIlIlI.click(
                        fn=IlIllIllIIIIllIII,
                        inputs=[],
                        outputs=[IlllIlIllIIIIIIIl, IIlllIllIlIIlIllI],
                    )
                with IllIlIlIIllllIlIl.Row():
                    IIIIIIIIlIIlIIlIl = IllIlIlIIllllIlIl.Audio(
                        label=IIIllIIllIIIIIIlI("Audio TTS:")
                    )
                    IIlIllIIllllIlIII = IllIlIlIIllllIlIl.Audio(
                        label=IIIllIIllIIIIIIlI("Audio RVC:")
                    )
                with IllIlIlIIllllIlIl.Row():
                    IIIlIIIIllIIIIllI = IllIlIlIIllllIlIl.Button(
                        IIIllIIllIIIIIIlI(IlIlIIllllIlIIIII), variant=IIlIIIIlIlIIlIlll
                    )
                IIIlIIIIllIIIIllI.click(
                    IIIIllIIIIIIIIIII,
                    inputs=[
                        IIllllIlIlIIlIIII,
                        IIIIlIIIIlIlIlIlI,
                        IlllIlIllIIIIIIIl,
                        IIlllIllIlIIlIllI,
                        IIlIIlIlllIllllIl,
                        IIIIlIIIIIllIllII,
                        IIlIlIllIIIlllllI,
                        IlIIllIIlIllllIII,
                        IlllIllIIllIIIlIl,
                        IIIllllIlIllIIIIl,
                    ],
                    outputs=[IIlIllIIllllIlIII, IIIIIIIIlIIlIIlIl],
                )
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Resources")):
                easy_infer.download_model()
                easy_infer.download_backup()
                easy_infer.download_dataset(IlIlIIIllIIIllIIl)
                easy_infer.download_audio()
                easy_infer.youtube_separator()
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Extra")):
                IllIlIlIIllllIlIl.Markdown(
                    value=IIIllIIllIIIIIIlI(
                        "This section contains some extra utilities that often may be in experimental phases"
                    )
                )
                with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Merge Audios")):
                    with IllIlIlIIllllIlIl.Group():
                        IllIlIlIIllllIlIl.Markdown(
                            value="## "
                            + IIIllIIllIIIIIIlI(
                                "Merge your generated audios with the instrumental"
                            )
                        )
                        IllIlIlIIllllIlIl.Markdown(
                            value="",
                            scale=IllIlIllllIIlIIlI,
                            visible=_IlllIIIlllIllIllI,
                        )
                        IllIlIlIIllllIlIl.Markdown(
                            value="",
                            scale=IllIlIllllIIlIIlI,
                            visible=_IlllIIIlllIllIllI,
                        )
                        with IllIlIlIIllllIlIl.Row():
                            with IllIlIlIIllllIlIl.Column():
                                IllIlIIIlllIIlllI = IllIlIlIIllllIlIl.File(
                                    label=IIIllIIllIIIIIIlI(IllllIlIllIIlllIl)
                                )
                                IllIlIlIIllllIlIl.Markdown(
                                    value=IIIllIIllIIIIIIlI(
                                        "### Instrumental settings:"
                                    )
                                )
                                IllIlIIIIlllIIlII = IllIlIlIIllllIlIl.Dropdown(
                                    label=IIIllIIllIIIIIIlI(
                                        "Choose your instrumental:"
                                    ),
                                    choices=sorted(IIllIlllIIllIIllI),
                                    value="",
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IllIIIIIIIIIlllIl = IllIlIlIIllllIlIl.Slider(
                                    minimum=0,
                                    maximum=10,
                                    label=IIIllIIllIIIIIIlI(
                                        "Volume of the instrumental audio:"
                                    ),
                                    value=_IlIlllIlIlIlIlIll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IllIlIlIIllllIlIl.Markdown(
                                    value=IIIllIIllIIIIIIlI("### Audio settings:")
                                )
                                IIlllIllIlIIIlIII = IllIlIlIIllllIlIl.Dropdown(
                                    label=IIIllIIllIIIIIIlI(
                                        "Select the generated audio"
                                    ),
                                    choices=sorted(IIIlIIllllIIIIIII),
                                    value="",
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                with IllIlIlIIllllIlIl.Row():
                                    IlIllIlIlIIllIIll = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=10,
                                        label=IIIllIIllIIIIIIlI(
                                            "Volume of the generated audio:"
                                        ),
                                        value=_IlIlllIlIlIlIlIll,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                IllIlIlIIllllIlIl.Markdown(
                                    value=IIIllIIllIIIIIIlI("### Add the effects:")
                                )
                                IIlllIIIlllIIIIII = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI("Reverb"),
                                    value=_IIIIlIlIIIlIlIlll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IlIIlIIIllIlIIlIl = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI("Compressor"),
                                    value=_IIIIlIlIIIlIlIlll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IlIlllIIIlllIlllI = IllIlIlIIllllIlIl.Checkbox(
                                    label=IIIllIIllIIIIIIlI("Noise Gate"),
                                    value=_IIIIlIlIIIlIlIlll,
                                    interactive=_IlllIIIlllIllIllI,
                                )
                                IIIlIIlIIIIlIIllI = IllIlIlIIllllIlIl.Button(
                                    IIIllIIllIIIIIIlI("Merge"),
                                    variant=IIlIIIIlIlIIlIlll,
                                ).style(full_width=_IlllIIIlllIllIllI)
                                IIIlIIllIIlllIlII = IllIlIlIIllllIlIl.Textbox(
                                    label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll)
                                )
                                IIlIIIIIlIIlIlllI = IllIlIlIIllllIlIl.Audio(
                                    label=IIIllIIllIIIIIIlI(IlIlllIlllIllIllI),
                                    type=IlIllIllllIlIlllI,
                                )
                                IllIlIIIlllIIlllI.upload(
                                    fn=IllIIllIIlIIIlIll,
                                    inputs=[IllIlIIIlllIIlllI],
                                    outputs=[IllIlIIIIlllIIlII],
                                )
                                IllIlIIIlllIIlllI.upload(
                                    fn=easy_infer.change_choices2,
                                    inputs=[],
                                    outputs=[IllIlIIIIlllIIlII],
                                )
                                IIlIlIlllllllIllI.click(
                                    fn=lambda: IllIIllIllIlIIIll(),
                                    inputs=[],
                                    outputs=[IllIlIIIIlllIIlII, IIlllIllIlIIIlIII],
                                )
                                IIIlIIlIIIIlIIllI.click(
                                    fn=IlllIIlIIIIlllllI,
                                    inputs=[
                                        IllIlIIIIlllIIlII,
                                        IIlllIllIlIIIlIII,
                                        IllIIIIIIIIIlllIl,
                                        IlIllIlIlIIllIIll,
                                        IIlllIIIlllIIIIII,
                                        IlIIlIIIllIlIIlIl,
                                        IlIlllIIIlllIlllI,
                                    ],
                                    outputs=[IIIlIIllIIlllIlII, IIlIIIIIlIIlIlllI],
                                )
                with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Processing")):
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI(
                                "Model fusion, can be used to test timbre fusion"
                            )
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                with IllIlIlIIllllIlIl.Column():
                                    IIIIIllIIllIIlIIl = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IIIIIllIIIIIlIlIl),
                                        value="",
                                        max_lines=1,
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIlllIllIIIIIIllI
                                        ),
                                    )
                                    IlllIlIllIIllIIII = IllIlIlIIllllIlIl.Slider(
                                        minimum=0,
                                        maximum=1,
                                        label=IIIllIIllIIIIIIlI("Weight for Model A:"),
                                        value=0.5,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIlIIIIllIIlllIl = IllIlIlIIllllIlIl.Checkbox(
                                        label=IIIllIIllIIIIIIlI(IIIIllllIIIlllIlI),
                                        value=_IlllIIIlllIllIllI,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIIIIIlIIlIlllllI = IllIlIlIIllllIlIl.Radio(
                                        label=IIIllIIllIIIIIIlI(IIIIIllIIllIlIIll),
                                        choices=[
                                            _IlIIIIIIlIlIllIIl,
                                            _IIIllllllIllIIIII,
                                        ],
                                        value=_IIIllllllIllIIIII,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IIlIIIllIIIlIIllI = IllIlIlIIllllIlIl.Radio(
                                        label=IIIllIIllIIIIIIlI(IlIIlIIIIllIIllIl),
                                        choices=[
                                            _IIIllIIIlIIllIlII,
                                            _IlIlllllIIIIIlIlI,
                                        ],
                                        value=_IIIllIIIlIIllIlII,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                with IllIlIlIIllllIlIl.Column():
                                    IlIIlIIllllIIlIlI = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI("Path to Model A:"),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIIlIlIlllIlIIll
                                        ),
                                    )
                                    IIllIlIlIlIIIlIll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI("Path to Model B:"),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIIlIlIlllIlIIll
                                        ),
                                    )
                                    IIlllllIIlIlIlIll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlIIllIlllIlIllIl),
                                        value="",
                                        max_lines=8,
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIllIIIIIIIlIIll
                                        ),
                                    )
                                    IIllllllllllllllI = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                        value="",
                                        max_lines=8,
                                    )
                            IllIIllIIIlIIlIlI = IllIlIlIIllllIlIl.Button(
                                IIIllIIllIIIIIIlI("Fusion"), variant=IIlIIIIlIlIIlIlll
                            )
                            IllIIllIIIlIIlIlI.click(
                                merge,
                                [
                                    IlIIlIIllllIIlIlI,
                                    IIllIlIlIlIIIlIll,
                                    IlllIlIllIIllIIII,
                                    IIlIIIllIIIlIIllI,
                                    IIIlIIIIllIIlllIl,
                                    IIlllllIIlIlIlIll,
                                    IIIIIllIIllIIlIIl,
                                    IIIIIIlIIlIlllllI,
                                ],
                                IIllllllllllllllI,
                            )
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI("Modify model information")
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                with IllIlIlIIllllIlIl.Column():
                                    IIlllllIIllIlIIlI = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlIIIIlIIIIlIIIll),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIIlIlIlllIlIIll
                                        ),
                                    )
                                    IlllllIIIIllIIlll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(
                                            "Model information to be modified:"
                                        ),
                                        value="",
                                        max_lines=8,
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIllIIIIIIIlIIll
                                        ),
                                    )
                                with IllIlIlIIllllIlIl.Column():
                                    IllIlIlIllIIIIlll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI("Save file name:"),
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIlllIllIIIIIIllI
                                        ),
                                        value="",
                                        max_lines=8,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllIlIIlllIlIIlII = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                        value="",
                                        max_lines=8,
                                    )
                            IIIllIIlllIIIIlII = IllIlIlIIllllIlIl.Button(
                                IIIllIIllIIIIIIlI("Modify"), variant=IIlIIIIlIlIIlIlll
                            )
                            IIIllIIlllIIIIlII.click(
                                change_info,
                                [
                                    IIlllllIIllIlIIlI,
                                    IlllllIIIIllIIlll,
                                    IllIlIlIllIIIIlll,
                                ],
                                IllIlIIlllIlIIlII,
                            )
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI("View model information")
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                with IllIlIlIIllllIlIl.Column():
                                    IIIllllIlIIlllIll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlIIIIlIIIIlIIIll),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIIlIlIlllIlIIll
                                        ),
                                    )
                                    IIIIllIllllIIIlIl = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                        value="",
                                        max_lines=8,
                                    )
                                    IIlIlIIlIlIIllIll = IllIlIlIIllllIlIl.Button(
                                        IIIllIIllIIIIIIlI("View"),
                                        variant=IIlIIIIlIlIIlIlll,
                                    )
                            IIlIlIIlIlIIllIll.click(
                                show_info, [IIIllllIlIIlllIll], IIIIllIllllIIIlIl
                            )
                    with IllIlIlIIllllIlIl.Group():
                        with IllIlIlIIllllIlIl.Accordion(
                            label=IIIllIIllIIIIIIlI("Model extraction")
                        ):
                            with IllIlIlIIllllIlIl.Row():
                                with IllIlIlIIllllIlIl.Column():
                                    IIllIllIlIIIlllIl = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IIIIIllIIIIIlIlIl),
                                        value="",
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIlllIllIIIIIIllI
                                        ),
                                    )
                                    IllIlIIlIlIIIllII = IllIlIlIIllllIlIl.Checkbox(
                                        label=IIIllIIllIIIIIIlI(IIIIllllIIIlllIlI),
                                        value=_IlllIIIlllIllIllI,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllIllIIIlIIIIIII = IllIlIlIIllllIlIl.Radio(
                                        label=IIIllIIllIIIIIIlI(IIIIIllIIllIlIIll),
                                        choices=[
                                            _IlIIIIIIlIlIllIIl,
                                            _IIIllllllIllIIIII,
                                        ],
                                        value=_IIIllllllIllIIIII,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IllllllIIlIIlllII = IllIlIlIIllllIlIl.Radio(
                                        label=IIIllIIllIIIIIIlI(IlIIlIIIIllIIllIl),
                                        choices=[
                                            _IllIIIIIIlIIIIlll,
                                            _IIIllIIIlIIllIlII,
                                            _IlIlllllIIIIIlIlI,
                                        ],
                                        value=_IIIllIIIlIIllIlII,
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                with IllIlIlIIllllIlIl.Column():
                                    IIIIIlIlIllIllIll = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlIIIIlIIIIlIIIll),
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIIlIlIlllIlIIll
                                        ),
                                        interactive=_IlllIIIlllIllIllI,
                                    )
                                    IlIlIIIIlllllllIl = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlIIllIlllIlIllIl),
                                        value="",
                                        max_lines=8,
                                        interactive=_IlllIIIlllIllIllI,
                                        placeholder=IIIllIIllIIIIIIlI(
                                            IIIllIIIIIIIlIIll
                                        ),
                                    )
                                    IllIIllIIllIllllI = IllIlIlIIllllIlIl.Textbox(
                                        label=IIIllIIllIIIIIIlI(IlllIIIlllllIIIll),
                                        value="",
                                        max_lines=8,
                                    )
                            with IllIlIlIIllllIlIl.Row():
                                IlllllIIlIIIlllII = IllIlIlIIllllIlIl.Button(
                                    IIIllIIllIIIIIIlI("Extract"),
                                    variant=IIlIIIIlIlIIlIlll,
                                )
                                IIIIIlIlIllIllIll.change(
                                    IIIIIIlIIIlIlIIIl,
                                    [IIIIIlIlIllIllIll],
                                    [
                                        IllllllIIlIIlllII,
                                        IllIlIIlIlIIIllII,
                                        IllIllIIIlIIIIIII,
                                    ],
                                )
                            IlllllIIlIIIlllII.click(
                                extract_small_model,
                                [
                                    IIIIIlIlIllIllIll,
                                    IIllIllIlIIIlllIl,
                                    IllllllIIlIIlllII,
                                    IllIlIIlIlIIIllII,
                                    IlIlIIIIlllllllIl,
                                    IllIllIIIlIIIIIII,
                                ],
                                IllIIllIIllIllllI,
                            )
            with IllIlIlIIllllIlIl.TabItem(IIIllIIllIIIIIIlI("Settings")):
                with IllIlIlIIllllIlIl.Row():
                    IllIlIlIIllllIlIl.Markdown(
                        value=IIIllIIllIIIIIIlI("Pitch settings")
                    )
                    IIIllIIlIIIIlIIll = IllIlIlIIllllIlIl.Checkbox(
                        label=IIIllIIllIIIIIIlI(
                            "Whether to use note names instead of their hertz value. E.G. [C5, D6] instead of [523.25, 1174.66]Hz"
                        ),
                        value=rvc_globals.NotesIrHertz,
                        interactive=_IlllIIIlllIllIllI,
                    )
            IIIllIIlIIIIlIIll.change(
                fn=lambda IIIllllIlIIlIlIlI: rvc_globals.__setattr__(
                    "NotesIrHertz", IIIllllIlIIlIlIlI
                ),
                inputs=[IIIllIIlIIIIlIIll],
                outputs=[],
            )
            IIIllIIlIIIIlIIll.change(
                fn=IllllIIIIlIlllIll,
                inputs=[IIIlIIllIIllIlllI],
                outputs=[
                    IIIlIllIIIIIIIIIl,
                    IIIlIIlIIIllIlIll,
                    IIlIIlIIllllIlIll,
                    IIlIlIIIlIlIlIlII,
                ],
            )
        return IIlllIIllIlIlllll


def IllIIlIIIIlIlllll(IllIIIIlIlIIllIll):
    IllIllIlIlIIIIlll = "0.0.0.0"
    IllIlIlllIlllllII = IlIllIlIlIIlIlIll.iscolab or IlIllIlIlIIlIlIll.paperspace
    IlIIIllIllIlIIIlI = 511
    IllllIlllllIlIlIl = 1022
    if IlIllIlIlIIlIlIll.iscolab or IlIllIlIlIIlIlIll.paperspace:
        IllIIIIlIlIIllIll.queue(
            concurrency_count=IlIIIllIllIlIIIlI, max_size=IllllIlllllIlIlIl
        ).launch(
            server_name=IllIllIlIlIIIIlll,
            inbrowser=not IlIllIlIlIIlIlIll.noautoopen,
            server_port=IlIllIlIlIIlIlIll.listen_port,
            quiet=_IlllIIIlllIllIllI,
            favicon_path="./images/icon.png",
            share=_IIIIlIlIIIlIlIlll,
        )
    else:
        IllIIIIlIlIIllIll.queue(
            concurrency_count=IlIIIllIllIlIIIlI, max_size=IllllIlllllIlIlIl
        ).launch(
            server_name=IllIllIlIlIIIIlll,
            inbrowser=not IlIllIlIlIIlIlIll.noautoopen,
            server_port=IlIllIlIlIIlIlIll.listen_port,
            quiet=_IlllIIIlllIllIllI,
            favicon_path=".\\images\\icon.png",
            share=IllIlIlllIlllllII,
        )


if __name__ == "__main__":
    if os.name == "nt":
        print(
            IIIllIIllIIIIIIlI(
                "Any ConnectionResetErrors post-conversion are irrelevant and purely visual; they can be ignored.\n"
            )
        )
    IlIllllIllIIlllIl = IlllIlllIIlllIIII(UTheme=IlIllIlIlIIlIlIll.grtheme)
    IllIIlIIIIlIlllll(IlIllllIllIIlllIl)
