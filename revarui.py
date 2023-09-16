_IIlllIllIIIIlIIIl ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IIIIlIIIllIIIlllI ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IlIIIllIllIIIIlll ='Default value is 1.0'
_IllIIlIllIIIIIlll ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIIlIlllIIIIlllIl ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IIIllIIIIlIIlIllI ='后处理重采样至最终采样率，0为不进行重采样'
_IlIIIllIIllIlIIIl ='特征检索库文件路径,为空则使用下拉的选择结果'
_IIIIIllIlllIllIlI ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IIIIIlIIlllIIIIII ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU'
_IIIlIIlIlllllllII ='变调(整数, 半音数量, 升八度12降八度-12)'
_IIlllllllIIllIllI ='mangio-crepe-tiny'
_IIlIIllllIIIIIlll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlllIIllIlIIllllI ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIIIlIllllIIllll ='IVF%s,Flat'
_IlIllIIlIIlIIlIIl ='%s/total_fea.npy'
_IIllllIlllllIIlII ='Trying doing kmeans %s shape to 10k centers.'
_IIIlllIlIIllIlIIl ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IllllIIllIlllIlIl ='write filelist done'
_IIIIllIlIlIIllllI ='%s/filelist.txt'
_IllIlIlllIllIllIl ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IIIIIlllIIllIIlII ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IlIIIlllIlIIIIlll ='%s/%s.wav|%s/%s.npy|%s'
_IIIllIllIlIlllIIl ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IIlIlllIIIIllllII ='%s/2b-f0nsf'
_IllllIIIIlllIlllI ='%s/0_gt_wavs'
_IIlIIlIIlIlIIlIIl ='emb_g.weight'
_IIIIIlIlIlIllIllI ='clean_empty_cache'
_IIIIIlIIIlIllIIlI ='sample_rate'
_IIIllIIlllllIlIII ='Whether the model has pitch guidance.'
_IIIlIIlIllIlIIlll ='目标采样率'
_IIllIlIIlIIlIllll ='crepe_hop_length'
_IIlllIIIIlIlllIII ='crepe'
_IIlIlllIlIIIlllII ='harvest'
_IIlIlIIlIIllIllII ='mangio-crepe'
_IllIIllIIIIlllIIl ='presets'
_IIlIllIIlllllIlIl ='EXTRACT-MODEL'
_IIllIIlIIllIIlIIl ='TRAIN-FEATURE'
_IIIIIlIllIllIIIll ='TRAIN'
_IllllllIlIIlIIlll ='EXTRACT-FEATURE'
_IlIlllllIllIIIIlI ='PRE-PROCESS'
_IIlIIlIllIlIIllll ='INFER'
_IIIlIlIIllIIlIlll ='%s/3_feature768'
_IIllIIIIllIIlIlll ='%s/3_feature256'
_IlllIIllIIlIIllll ='_v2'
_IlIlllIlllllIIlII ='cpu'
_IllIIIlIIlIIllllI ='mp3'
_IlIllIlllIIlIIlll ='wav'
_IllIIllllllllIlIl ='csvdb/stop.csv'
_IIIIIlIIllIIllllI ='../inference-presets.json'
_IlIlllIlllIllIIlI ='-pd %s'
_IIIlllIlIlIIlllIl ='-pg %s'
_IIIIIIIlllIIIlllI ="doesn't exist, will not use pretrained model"
_IIIllllIIlIIllIII ='32k'
_IIIIIIlllIIlIIllI ='/audios/'
_IlIIIlIlIIlIlIIIl ='weight'
_IlIIIlIIllIlllIII ='trained'
_IIIllIllllIIIIIlI ='%s/logs/%s'
_IIllIIllIIIllIlll ='48k'
_IIlIIlIIllllIIIIl ='flac'
_IlIIllllIlIIlIlll ='f0'
_IIIIIIllIIlIlIIII ='.index'
_IllllIIIIllIIIlII ='.pth'
_IIlllllIIllIIllIl ='rmvpe'
_IllllIlIllllIlIII ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IllllIlIlllIIIlIl ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IlIlIIlIllIllIlll ='choices'
_IIIlIIIllIIIIllII ='version'
_IIlIIIIIIIIllllII ='%s/%s'
_IlIIlllIIIllIIIll ='./logs/'
_IlIlllIlIllIIlIll ='w+'
_IllIIllllIIlIIIll ='formanting'
_IllIIIIlIllIllIlI ='csvdb/formanting.csv'
_IIlIllIIlIIIlllIl ='v2'
_IIlIIllIIllIlIIII ='w'
_IIllIIIIIlIlIllll ='输出信息'
_IllIlIIllIllllIll ='40k'
_IIlllIlIIlIlIllII ='\\\\'
_IlIIIIlIllIlllIlI ='"'
_IIIIIIlIIIIlIlIlI =' '
_IIlIllllIlllIIIII ='config'
_IIllIllIlIIllIllI ='/'
_IIIllIIIlIllIlIIl ='value'
_IIIIlIIIIlIlIllIl ='.'
_IlllIllIllIIlIIll ='primary'
_IIllIlIlIIlllIIlI =None 
_IIlIlIlIIIlIIIIII ='r'
_IllllIlIIIIllllll ='v1'
_IlllllllIIllllIII ='\\'
_IIlIIlIIllIlIIllI ='\n'
_IIIIllIlIlllllIIl ='visible'
_IIllIlIIIIIlIllll =False 
_IlIIlIIlllIlIlIlI ='update'
_IlllllllIIIIlIlII ='__type__'
_IIIlIlIIIlllIllII =True 
import os ,shutil ,sys ,json ,math ,signal 
IllIIIIlIIIlIIIll =os .getcwd ()
sys .path .append (IllIIIIlIIIlIIIll )
import traceback ,pdb ,warnings ,numpy as np ,torch ,re 
os .environ ['IPENBLAS_NUM_THREADS']='1'
os .environ ['no_proxy']='localhost, 127.0.0.1, ::1'
import logging ,threading 
from random import shuffle 
from subprocess import Popen 
from time import sleep 
import faiss ,ffmpeg ,gradio as gr ,soundfile as sf 
from config import Config 
from fairseq import checkpoint_utils 
from i18n import I18nAuto 
from lib .infer_pack .models import SynthesizerTrnMs256NSFsid ,SynthesizerTrnMs256NSFsid_nono ,SynthesizerTrnMs768NSFsid ,SynthesizerTrnMs768NSFsid_nono 
from lib .infer_pack .models_onnx import SynthesizerTrnMsNSFsidM 
from infer_uvr5 import _audio_pre_ ,_audio_pre_new 
from MDXNet import MDXNetDereverb 
from my_utils import load_audio ,CSVutil 
from train .process_ckpt import change_info ,extract_small_model ,merge ,show_info 
from vc_infer_pipeline import VC 
from sklearn .cluster import MiniBatchKMeans 
IllIIIIlIlIlIlIIl =os .path .join (IllIIIIlIIIlIIIll ,'TEMP')
shutil .rmtree (IllIIIIlIlIlIlIIl ,ignore_errors =_IIIlIlIIIlllIllII )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IllIIIIlIIIlIIIll ,ignore_errors =_IIIlIlIIIlllIllII )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IllIIIIlIIIlIIIll ,ignore_errors =_IIIlIlIIIlllIllII )
os .makedirs (IllIIIIlIlIlIlIIl ,exist_ok =_IIIlIlIIIlllIllII )
os .makedirs (os .path .join (IllIIIIlIIIlIIIll ,'logs'),exist_ok =_IIIlIlIIIlllIllII )
os .makedirs (os .path .join (IllIIIIlIIIlIIIll ,'audios'),exist_ok =_IIIlIlIIIlllIllII )
os .makedirs (os .path .join (IllIIIIlIIIlIIIll ,'datasets'),exist_ok =_IIIlIlIIIlllIllII )
os .makedirs (os .path .join (IllIIIIlIIIlIIIll ,'weights'),exist_ok =_IIIlIlIIIlllIllII )
os .environ ['TEMP']=IllIIIIlIlIlIlIIl 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
logging .getLogger ('numba').setLevel (logging .WARNING )
import csv 
if not os .path .isdir ('csvdb/'):os .makedirs ('csvdb');IlllIIllllllIIIIl ,IlIIIllIIllIIIlIl =open (_IllIIIIlIllIllIlI ,_IIlIIllIIllIlIIII ),open (_IllIIllllllllIlIl ,_IIlIIllIIllIlIIII );IlllIIllllllIIIIl .close ();IlIIIllIIllIIIlIl .close ()
global IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI 
try :IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI =CSVutil (_IllIIIIlIllIllIlI ,_IIlIlIlIIIlIIIIII ,_IllIIllllIIlIIIll );IIIIllIlIIllllIIl =(lambda IllIllIIllIIlllll :_IIIlIlIIIlllIllII if IllIllIIllIIlllll .lower ()=='true'else _IIllIlIIIIIlIllll if IllIllIIllIIlllll .lower ()=='false'else IllIllIIllIIlllll )(IIIIllIlIIllllIIl )
except (ValueError ,TypeError ,IndexError ):IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI =_IIllIlIIIIIlIllll ,1. ,1. ;CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI )
IllIIllIIIIlIIlIl =Config ()
IlIlllIIllllIlIII =I18nAuto ()
IlIlllIIllllIlIII .print ()
IIlIIIIIIlIIllIlI =torch .cuda .device_count ()
IlIllllllIlllllIl =[]
IllIIlIllIIIlllll =[]
IlIlIlIIIIIIIIIIl =_IIllIlIIIIIlIllll 
IIIIIlIIlIlIlIIll =0 
if torch .cuda .is_available ()or IIlIIIIIIlIIllIlI !=0 :
	for IlIIIIIlIlIllIIIl in range (IIlIIIIIIlIIllIlI ):
		IIIIlIllIllllllIl =torch .cuda .get_device_name (IlIIIIIlIlIllIIIl )
		if any (IlllllIlIIlIllIlI in IIIIlIllIllllllIl .upper ()for IlllllIlIIlIllIlI in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IlIlIlIIIIIIIIIIl =_IIIlIlIIIlllIllII ;IlIllllllIlllllIl .append ('%s\t%s'%(IlIIIIIlIlIllIIIl ,IIIIlIllIllllllIl ));IllIIlIllIIIlllll .append (int (torch .cuda .get_device_properties (IlIIIIIlIlIllIIIl ).total_memory /1024 /1024 /1024 +.4 ))
if IlIlIlIIIIIIIIIIl and len (IlIllllllIlllllIl )>0 :IIlIlIllIlIllllII =_IIlIIlIIllIlIIllI .join (IlIllllllIlllllIl );IIIllIIlIIIlllllI =min (IllIIlIllIIIlllll )//2 
else :IIlIlIllIlIllllII =IlIlllIIllllIlIII ('很遗憾您这没有能用的显卡来支持您训练');IIIllIIlIIIlllllI =1 
IllllIIlllIlllIII ='-'.join ([IIIIIIIlllIlIllIl [0 ]for IIIIIIIlllIlIllIl in IlIllllllIlllllIl ])
IIlllIllIIIllIIIl =_IIllIlIlIIlllIIlI 
def IIIlIlllllIlIllII ():
	global IIlllIllIIIllIIIl ;IllIIIIIIlIIIIlll ,_IlIlllllIIlIIIlIl ,_IlIlllllIIlIIIlIl =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IIlllIllIIIllIIIl =IllIIIIIIlIIIIlll [0 ];IIlllIllIIIllIIIl =IIlllIllIIIllIIIl .to (IllIIllIIIIlIIlIl .device )
	if IllIIllIIIIlIIlIl .is_half :IIlllIllIIIllIIIl =IIlllIllIIIllIIIl .half ()
	else :IIlllIllIIIllIIIl =IIlllIllIIIllIIIl .float ()
	IIlllIllIIIllIIIl .eval ()
IlIllIIIIIlIlIIlI ='weights'
IIIlIllIIlIIllIII ='uvr5_weights'
IlIIIIIIIIlIlIIlI =_IlIIlllIIIllIIIll 
IlIIIllllllIlIIIl ='audios'
IlIlIIllIlllllIll =[]
for IlIllIIlIIIIlllIl in os .listdir (IlIllIIIIIlIlIIlI ):
	if IlIllIIlIIIIlllIl .endswith (_IllllIIIIllIIIlII ):IlIlIIllIlllllIll .append (IlIllIIlIIIIlllIl )
IllIlIllIIIlIllIl =[]
global IIlIlIlIIIIIIllIl 
IIlIlIlIIIIIIllIl =[]
IllIIIlllIlllIlIl =[]
for (IIIllIlllIlIlIIIl ,IIlIIllIIIIllllIl ,IlllllIIlllllIllI )in os .walk (IlIIIIIIIIlIlIIlI ,topdown =_IIllIlIIIIIlIllll ):
	for IlIllIIlIIIIlllIl in IlllllIIlllllIllI :
		if IlIllIIlIIIIlllIl .endswith (_IIIIIIllIIlIlIIII )and _IlIIIlIIllIlllIII not in IlIllIIlIIIIlllIl :IllIlIllIIIlIllIl .append ('%s\\%s'%(IIIllIlllIlIlIIIl ,IlIllIIlIIIIlllIl ))
for (IIIllIlllIlIlIIIl ,IIlIIllIIIIllllIl ,IlllllIIlllllIllI )in os .walk (IlIIIllllllIlIIIl ,topdown =_IIllIlIIIIIlIllll ):
	for IlIllIIlIIIIlllIl in IlllllIIlllllIllI :IllIIIlllIlllIlIl .append (_IIlIIIIIIIIllllII %(IIIllIlllIlIlIIIl ,IlIllIIlIIIIlllIl ))
IllIlllIlllIIlIIl =[]
for IlIllIIlIIIIlllIl in os .listdir (IIIlIllIIlIIllIII ):
	if IlIllIIlIIIIlllIl .endswith (_IllllIIIIllIIIlII )or 'onnx'in IlIllIIlIIIIlllIl :IllIlllIlllIIlIIl .append (IlIllIIlIIIIlllIl .replace (_IllllIIIIllIIIlII ,''))
def IIIIIlIIIIIllllIl ():
	if len (IlIlIIllIlllllIll )>0 :return sorted (IlIlIIllIlllllIll )[0 ]
	else :return ''
def IIlIIlllllIIlIllI ():
	if IIIIIlIIIIIllllIl ()!='':
		IIIIlllIlIlllllII =sorted (IlIlIIllIlllllIll )[0 ].split (_IIIIlIIIIlIlIllIl )[0 ];IIllllIIIlIIlIllI =_IlIIlllIIIllIIIll +IIIIlllIlIlllllII 
		if os .path .exists (IIllllIIIlIIlIllI ):
			for IIlIIlIllIllIIIIl in os .listdir (IIllllIIIlIIlIllI ):
				if IIlIIlIllIllIIIIl .endswith (_IIIIIIllIIlIlIIII ):return os .path .join (IIllllIIIlIIlIllI ,IIlIIlIllIllIIIIl ).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )
			return ''
		else :return ''
def IllIllIlIllIIllII ():
	for (IlllllIIllIlIlIIl ,IIIIlIlIlIIllIllI ,IIlIlllIIIllIllII )in os .walk (_IlIIlllIIIllIIIll ):
		for IIIlllIIlllllIlll in IIlIlllIIIllIllII :
			if IIIlllIIlllllIlll .endswith (_IIIIIIllIIlIlIIII )and _IlIIIlIIllIlllIII not in IIIlllIIlllllIlll :IIlIlIlIIIIIIllIl .append (os .path .join (IlllllIIllIlIlIIl ,IIIlllIIlllllIlll ).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI ))
	if len (IIlIlIlIIIIIIllIl )>0 :return IIlIlIlIIIIIIllIl 
	else :return ''
IIlIIlIIlIlIlIIlI =[]
def IlllIlIlIllllIIIl ():
	IlIIllIIlIIIlllll =[]
	for (IIlIllIIIlllIllII ,IlllllllIlIIllIll ,IlIllIlllIIlIlIIl )in os .walk ('./formantshiftcfg/'):
		for IIllIIllIllIIlllI in IlIllIlllIIlIlIIl :
			if IIllIIllIllIIlllI .endswith ('.txt'):IlIIllIIlIIIlllll .append (os .path .join (IIlIllIIIlllIllII ,IIllIIllIllIIlllI ).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI ))
	if len (IlIIllIIlIIIlllll )>0 :return IlIIllIIlIIIlllll 
	else :return ''
def IIIllIllIIIIIlIII (IlllIllIllIllllII ,IllIlIIllIIlIIIll ,IIIlllIlIlIIIIIIl ,IlIlIIllIIIIllIll ,IllIIIIlIIIlIIIII ,IIIIIlIlIIIIIlllI ,IlIlIlIIIllIIlllI ,IIIIIIIlIlIIIllIl ,IllIlIlIlllIIllIl ,IIllIlIlIIlIIIlIl ,IllIlllIllIIllIlI ,IlIIIIllIlIllllII ,IIlllIIIIIIllIlII ,IlIIIllllIIIlllIl ):
	global IllIIlllIIIlIIlIl ,IIIIllIIIllIlllII ,IIllIIllIIlIllIlI ,IIlllIllIIIllIIIl ,IIlIllIIIlIllllll 
	if IllIlIIllIIlIIIll is _IIllIlIlIIlllIIlI or IllIlIIllIIlIIIll is _IIllIlIlIIlllIIlI :return 'You need to upload an audio',_IIllIlIlIIlllIIlI 
	IlIlIIllIIIIllIll =int (IlIlIIllIIIIllIll )
	try :
		if IllIlIIllIIlIIIll =='':IlIIlIlllIIlIIIll =load_audio (IIIlllIlIlIIIIIIl ,16000 ,IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI )
		else :IlIIlIlllIIlIIIll =load_audio (IllIlIIllIIlIIIll ,16000 ,IIIIllIlIIllllIIl ,IIllIIIIllIlIlllI ,IIIIIIIlIIIIIlllI )
		IlIIIIIlIlllIIlll =np .abs (IlIIlIlllIIlIIIll ).max ()/.95 
		if IlIIIIIlIlllIIlll >1 :IlIIlIlllIIlIIIll /=IlIIIIIlIlllIIlll 
		IlIIIllIIIIlIIIlI =[0 ,0 ,0 ]
		if not IIlllIllIIIllIIIl :IIIlIlllllIlIllII ()
		IlIIIIlllIllIlllI =IlllIIIlIIIIIlllI .get (_IlIIllllIlIIlIlll ,1 );IlIlIlIIIllIIlllI =IlIlIlIIIllIIlllI .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI ).replace (_IlIIIlIIllIlllIII ,'added')if IlIlIlIIIllIIlllI !=''else IIIIIIIlIlIIIllIl ;IIllIIIllIllIIIII =IIllIIllIIlIllIlI .pipeline (IIlllIllIIIllIIIl ,IIIIllIIIllIlllII ,IlllIllIllIllllII ,IlIIlIlllIIlIIIll ,IIIlllIlIlIIIIIIl ,IlIIIllIIIIlIIIlI ,IlIlIIllIIIIllIll ,IIIIIlIlIIIIIlllI ,IlIlIlIIIllIIlllI ,IllIlIlIlllIIllIl ,IlIIIIlllIllIlllI ,IIllIlIlIIlIIIlIl ,IllIIlllIIIlIIlIl ,IllIlllIllIIllIlI ,IlIIIIllIlIllllII ,IIlIllIIIlIllllll ,IIlllIIIIIIllIlII ,IlIIIllllIIIlllIl ,f0_file =IllIIIIlIIIlIIIII )
		if IllIIlllIIIlIIlIl !=IllIlllIllIIllIlI >=16000 :IllIIlllIIIlIIlIl =IllIlllIllIIllIlI 
		IIIllllllIIllIIlI ='Using index:%s.'%IlIlIlIIIllIIlllI if os .path .exists (IlIlIlIIIllIIlllI )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IIIllllllIIllIIlI ,IlIIIllIIIIlIIIlI [0 ],IlIIIllIIIIlIIIlI [1 ],IlIIIllIIIIlIIIlI [2 ]),(IllIIlllIIIlIIlIl ,IIllIIIllIllIIIII )
	except :IIlIlIIlIIlIIIIIl =traceback .format_exc ();print (IIlIlIIlIIlIIIIIl );return IIlIlIIlIIlIIIIIl ,(_IIllIlIlIIlllIIlI ,_IIllIlIlIIlllIIlI )
def IlIlIIllIIlllllII (IlllIIllIllIIllll ,IIIllllIllllIlllI ,IllllIIIIlIIlIlIl ,IlIIlIIlIIIlllIll ,IIIlllIlllllIllIl ,IIIIlIIIIIIlllIII ,IlIllllIlIIIIlIll ,IlIlllIIIlIlllIII ,IIllIIIIIlllIIlIl ,IlIIIllIllIlllIlI ,IlIllIlIlllIIIIlI ,IIlIIllIIlIlIIIlI ,IlIlIlIllllIlIlII ,IlIlIlllIllllIIlI ,IIllIIIlIIIIllIll ):
	try :
		IIIllllIllllIlllI =IIIllllIllllIlllI .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI );IllllIIIIlIIlIlIl =IllllIIIIlIIlIlIl .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI );os .makedirs (IllllIIIIlIIlIlIl ,exist_ok =_IIIlIlIIIlllIllII )
		try :
			if IIIllllIllllIlllI !='':IlIIlIIlIIIlllIll =[os .path .join (IIIllllIllllIlllI ,IlIlIIllIlIIIIIIl )for IlIlIIllIlIIIIIIl in os .listdir (IIIllllIllllIlllI )]
			else :IlIIlIIlIIIlllIll =[IIlIlIIIIIIIlIlIl .name for IIlIlIIIIIIIlIlIl in IlIIlIIlIIIlllIll ]
		except :traceback .print_exc ();IlIIlIIlIIIlllIll =[IlIIIllIlIlIlIIlI .name for IlIIIllIlIlIlIIlI in IlIIlIIlIIIlllIll ]
		IIlIllIIIlIlllIII =[]
		for IIllIlIlIlIllIlll in IlIIlIIlIIIlllIll :
			IIllIllIIIlIlIlll ,IIllIlIIIIIIllIlI =IIIllIllIIIIIlIII (IlllIIllIllIIllll ,IIllIlIlIlIllIlll ,_IIllIlIlIIlllIIlI ,IIIlllIlllllIllIl ,_IIllIlIlIIlllIIlI ,IIIIlIIIIIIlllIII ,IlIllllIlIIIIlIll ,IlIlllIIIlIlllIII ,IIllIIIIIlllIIlIl ,IlIIIllIllIlllIlI ,IlIllIlIlllIIIIlI ,IIlIIllIIlIlIIIlI ,IlIlIlIllllIlIlII ,IIllIIIlIIIIllIll )
			if 'Success'in IIllIllIIIlIlIlll :
				try :
					IllllIIIlIllllIlI ,IIllIIllIIIllIIII =IIllIlIIIIIIllIlI 
					if IlIlIlllIllllIIlI in [_IlIllIlllIIlIIlll ,_IIlIIlIIllllIIIIl ,_IllIIIlIIlIIllllI ,'ogg','aac']:sf .write ('%s/%s.%s'%(IllllIIIIlIIlIlIl ,os .path .basename (IIllIlIlIlIllIlll ),IlIlIlllIllllIIlI ),IIllIIllIIIllIIII ,IllllIIIlIllllIlI )
					else :
						IIllIlIlIlIllIlll ='%s/%s.wav'%(IllllIIIIlIIlIlIl ,os .path .basename (IIllIlIlIlIllIlll ));sf .write (IIllIlIlIlIllIlll ,IIllIIllIIIllIIII ,IllllIIIlIllllIlI )
						if os .path .exists (IIllIlIlIlIllIlll ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IIllIlIlIlIllIlll ,IIllIlIlIlIllIlll [:-4 ]+'.%s'%IlIlIlllIllllIIlI ))
				except :IIllIllIIIlIlIlll +=traceback .format_exc ()
			IIlIllIIIlIlllIII .append ('%s->%s'%(os .path .basename (IIllIlIlIlIllIlll ),IIllIllIIIlIlIlll ));yield _IIlIIlIIllIlIIllI .join (IIlIllIIIlIlllIII )
		yield _IIlIIlIIllIlIIllI .join (IIlIllIIIlIlllIII )
	except :yield traceback .format_exc ()
def IIIIIIIIlIlIlllll (IlIIIllIlIIllIlIl ,IIIIIllIIllllIlll ,IllIlllllllllIIII ,IlIIIlIllIIlIlIII ,IIIIlIIllIIIlIIIl ,IIllIlllllIllIllI ,IllIlIIIIIlIIllII ):
	IllIIIlllllIIllll ='streams';IlIlIIlIIllIIIIII ='onnx_dereverb_By_FoxJoy';IllIIlIIIIIllIllI =[]
	try :
		IIIIIllIIllllIlll =IIIIIllIIllllIlll .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI );IllIlllllllllIIII =IllIlllllllllIIII .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI );IIIIlIIllIIIlIIIl =IIIIlIIllIIIlIIIl .strip (_IIIIIIlIIIIlIlIlI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIlIIlIIllIlIIllI ).strip (_IlIIIIlIllIlllIlI ).strip (_IIIIIIlIIIIlIlIlI )
		if IlIIIllIlIIllIlIl ==IlIlIIlIIllIIIIII :IlIllllIIIIIIIIll =MDXNetDereverb (15 )
		else :IIlIllIlIlllIllII =_audio_pre_ if 'DeEcho'not in IlIIIllIlIIllIlIl else _audio_pre_new ;IlIllllIIIIIIIIll =IIlIllIlIlllIllII (agg =int (IIllIlllllIllIllI ),model_path =os .path .join (IIIlIllIIlIIllIII ,IlIIIllIlIIllIlIl +_IllllIIIIllIIIlII ),device =IllIIllIIIIlIIlIl .device ,is_half =IllIIllIIIIlIIlIl .is_half )
		if IIIIIllIIllllIlll !='':IlIIIlIllIIlIlIII =[os .path .join (IIIIIllIIllllIlll ,IIlIIIIIIIlIIlIlI )for IIlIIIIIIIlIIlIlI in os .listdir (IIIIIllIIllllIlll )]
		else :IlIIIlIllIIlIlIII =[IIIlIlIIIlllIlIII .name for IIIlIlIIIlllIlIII in IlIIIlIllIIlIlIII ]
		for IIIllllIIIlIlIIll in IlIIIlIllIIlIlIII :
			IIlllIIIIlIlllIlI =os .path .join (IIIIIllIIllllIlll ,IIIllllIIIlIlIIll );IllllIllIlllIllll =1 ;IIIIlIllllIllIIIl =0 
			try :
				IIlIllIlllllIllIl =ffmpeg .probe (IIlllIIIIlIlllIlI ,cmd ='ffprobe')
				if IIlIllIlllllIllIl [IllIIIlllllIIllll ][0 ]['channels']==2 and IIlIllIlllllIllIl [IllIIIlllllIIllll ][0 ][_IIIIIlIIIlIllIIlI ]=='44100':IllllIllIlllIllll =0 ;IlIllllIIIIIIIIll ._path_audio_ (IIlllIIIIlIlllIlI ,IIIIlIIllIIIlIIIl ,IllIlllllllllIIII ,IllIlIIIIIlIIllII );IIIIlIllllIllIIIl =1 
			except :IllllIllIlllIllll =1 ;traceback .print_exc ()
			if IllllIllIlllIllll ==1 :IllIIlIIIIIIllIll ='%s/%s.reformatted.wav'%(IllIIIIlIlIlIlIIl ,os .path .basename (IIlllIIIIlIlllIlI ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIlllIIIIlIlllIlI ,IllIIlIIIIIIllIll ));IIlllIIIIlIlllIlI =IllIIlIIIIIIllIll 
			try :
				if IIIIlIllllIllIIIl ==0 :IlIllllIIIIIIIIll ._path_audio_ (IIlllIIIIlIlllIlI ,IIIIlIIllIIIlIIIl ,IllIlllllllllIIII ,IllIlIIIIIlIIllII )
				IllIIlIIIIIllIllI .append ('%s->Success'%os .path .basename (IIlllIIIIlIlllIlI ));yield _IIlIIlIIllIlIIllI .join (IllIIlIIIIIllIllI )
			except :IllIIlIIIIIllIllI .append ('%s->%s'%(os .path .basename (IIlllIIIIlIlllIlI ),traceback .format_exc ()));yield _IIlIIlIIllIlIIllI .join (IllIIlIIIIIllIllI )
	except :IllIIlIIIIIllIllI .append (traceback .format_exc ());yield _IIlIIlIIllIlIIllI .join (IllIIlIIIIIllIllI )
	finally :
		try :
			if IlIIIllIlIIllIlIl ==IlIlIIlIIllIIIIII :del IlIllllIIIIIIIIll .pred .model ;del IlIllllIIIIIIIIll .pred .model_ 
			else :del IlIllllIIIIIIIIll .model ;del IlIllllIIIIIIIIll 
		except :traceback .print_exc ()
		print (_IIIIIlIlIlIllIllI )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IIlIIlIIllIlIIllI .join (IllIIlIIIIIllIllI )
def IlIlllIllIIIlIIll (IlIIIIllIIIIIlIII ,IlIlllIlIIIIlIIII ,IIllllllIllIlIIlI ):
	global IIllIlIlIIIIlIlll ,IllIIlllIIIlIIlIl ,IIIIllIIIllIlllII ,IIllIIllIIlIllIlI ,IlllIIIlIIIIIlllI ,IIlIllIIIlIllllll 
	if IlIIIIllIIIIIlIII ==''or IlIIIIllIIIIIlIII ==[]:
		global IIlllIllIIIllIIIl 
		if IIlllIllIIIllIIIl is not _IIllIlIlIIlllIIlI :
			print (_IIIIIlIlIlIllIllI );del IIIIllIIIllIlllII ,IIllIlIlIIIIlIlll ,IIllIIllIIlIllIlI ,IIlllIllIIIllIIIl ,IllIIlllIIIlIIlIl ;IIlllIllIIIllIIIl =IIIIllIIIllIlllII =IIllIlIlIIIIlIlll =IIllIIllIIlIllIlI =IIlllIllIIIllIIIl =IllIIlllIIIlIIlIl =_IIllIlIlIIlllIIlI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IIlIlIllIllIlIllI =IlllIIIlIIIIIlllI .get (_IlIIllllIlIIlIlll ,1 );IIlIllIIIlIllllll =IlllIIIlIIIIIlllI .get (_IIIlIIIllIIIIllII ,_IllllIlIIIIllllll )
			if IIlIllIIIlIllllll ==_IllllIlIIIIllllll :
				if IIlIlIllIllIlIllI ==1 :IIIIllIIIllIlllII =SynthesizerTrnMs256NSFsid (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ],is_half =IllIIllIIIIlIIlIl .is_half )
				else :IIIIllIIIllIlllII =SynthesizerTrnMs256NSFsid_nono (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ])
			elif IIlIllIIIlIllllll ==_IIlIllIIlIIIlllIl :
				if IIlIlIllIllIlIllI ==1 :IIIIllIIIllIlllII =SynthesizerTrnMs768NSFsid (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ],is_half =IllIIllIIIIlIIlIl .is_half )
				else :IIIIllIIIllIlllII =SynthesizerTrnMs768NSFsid_nono (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ])
			del IIIIllIIIllIlllII ,IlllIIIlIIIIIlllI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IlllIIIlIIIIIlllI =_IIllIlIlIIlllIIlI 
		return {_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	IIIIIllIIIIlllIII =_IIlIIIIIIIIllllII %(IlIllIIIIIlIlIIlI ,IlIIIIllIIIIIlIII );print ('loading %s'%IIIIIllIIIIlllIII );IlllIIIlIIIIIlllI =torch .load (IIIIIllIIIIlllIII ,map_location =_IlIlllIlllllIIlII );IllIIlllIIIlIIlIl =IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ][-1 ];IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ][-3 ]=IlllIIIlIIIIIlllI [_IlIIIlIlIIlIlIIIl ][_IIlIIlIIlIlIIlIIl ].shape [0 ];IIlIlIllIllIlIllI =IlllIIIlIIIIIlllI .get (_IlIIllllIlIIlIlll ,1 )
	if IIlIlIllIllIlIllI ==0 :IlIlllIlIIIIlIIII =IIllllllIllIlIIlI ={_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IIIllIIIlIllIlIIl :.5 ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	else :IlIlllIlIIIIlIIII ={_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IIIllIIIlIllIlIIl :IlIlllIlIIIIlIIII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI };IIllllllIllIlIIlI ={_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IIIllIIIlIllIlIIl :IIllllllIllIlIIlI ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	IIlIllIIIlIllllll =IlllIIIlIIIIIlllI .get (_IIIlIIIllIIIIllII ,_IllllIlIIIIllllll )
	if IIlIllIIIlIllllll ==_IllllIlIIIIllllll :
		if IIlIlIllIllIlIllI ==1 :IIIIllIIIllIlllII =SynthesizerTrnMs256NSFsid (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ],is_half =IllIIllIIIIlIIlIl .is_half )
		else :IIIIllIIIllIlllII =SynthesizerTrnMs256NSFsid_nono (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ])
	elif IIlIllIIIlIllllll ==_IIlIllIIlIIIlllIl :
		if IIlIlIllIllIlIllI ==1 :IIIIllIIIllIlllII =SynthesizerTrnMs768NSFsid (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ],is_half =IllIIllIIIIlIIlIl .is_half )
		else :IIIIllIIIllIlllII =SynthesizerTrnMs768NSFsid_nono (*IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ])
	del IIIIllIIIllIlllII .enc_q ;print (IIIIllIIIllIlllII .load_state_dict (IlllIIIlIIIIIlllI [_IlIIIlIlIIlIlIIIl ],strict =_IIllIlIIIIIlIllll ));IIIIllIIIllIlllII .eval ().to (IllIIllIIIIlIIlIl .device )
	if IllIIllIIIIlIIlIl .is_half :IIIIllIIIllIlllII =IIIIllIIIllIlllII .half ()
	else :IIIIllIIIllIlllII =IIIIllIIIllIlllII .float ()
	IIllIIllIIlIllIlI =VC (IllIIlllIIIlIIlIl ,IllIIllIIIIlIIlIl );IIllIlIlIIIIlIlll =IlllIIIlIIIIIlllI [_IIlIllllIlllIIIII ][-3 ];return {_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,'maximum':IIllIlIlIIIIlIlll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },IlIlllIlIIIIlIIII ,IIllllllIllIlIIlI 
def IlIIIlIIIIlIlIIlI ():
	IllIlllllllIlllII =[]
	for IIIIlllIIlIlIlIll in os .listdir (IlIllIIIIIlIlIIlI ):
		if IIIIlllIIlIlIlIll .endswith (_IllllIIIIllIIIlII ):IllIlllllllIlllII .append (IIIIlllIIlIlIlIll )
	IlIIIIIIlIlllllIl =[];IllIIlIIIIIlllIll =[];IlIllIllIllIllllI =os .path .abspath (os .getcwd ())+_IIIIIIlllIIlIIllI 
	for (IlIlIIIllIIllIlIl ,IIIIIIIIIIlIIIlII ,IlIIlIlIIlIIllIll )in os .walk (IlIIIIIIIIlIlIIlI ,topdown =_IIllIlIIIIIlIllll ):
		for IIIIlllIIlIlIlIll in IlIIlIlIIlIIllIll :
			if IIIIlllIIlIlIlIll .endswith (_IIIIIIllIIlIlIIII )and _IlIIIlIIllIlllIII not in IIIIlllIIlIlIlIll :IlIIIIIIlIlllllIl .append (_IIlIIIIIIIIllllII %(IlIlIIIllIIllIlIl ,IIIIlllIIlIlIlIll ))
	for IIIlIIllIIllIlllI in os .listdir (IlIllIllIllIllllI ):IllIIlIIIIIlllIll .append (_IIlIIIIIIIIllllII %(IlIIIllllllIlIIIl ,IIIlIIllIIllIlllI ))
	return {_IlIlIIlIllIllIlll :sorted (IllIlllllllIlllII ),_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlIlIIlIllIllIlll :sorted (IlIIIIIIlIlllllIl ),_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlIlIIlIllIllIlll :sorted (IllIIlIIIIIlllIll ),_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IlIlllllIIllllIlI ():return {_IIIllIIIlIllIlIIl :'',_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
IIlIIIllllIIIIIll ={_IIIllllIIlIIllIII :32000 ,_IllIlIIllIllllIll :40000 ,_IIllIIllIIIllIlll :48000 }
def IllIIlIlllIIllIlI (IlIIIIllllllllIlI ,IllIIIIIlIllIIlll ):
	while 1 :
		if IllIIIIIlIllIIlll .poll ()is _IIllIlIlIIlllIIlI :sleep (.5 )
		else :break 
	IlIIIIllllllllIlI [0 ]=_IIIlIlIIIlllIllII 
def IIlIIIlllllIlIlIl (IIlllIIIIlIlIllIl ,IIIIIIllIIllIIIIl ):
	while 1 :
		IlIIIlIllllIIIllI =1 
		for IIIIlllllllIlIllI in IIIIIIllIIllIIIIl :
			if IIIIlllllllIlIllI .poll ()is _IIllIlIlIIlllIIlI :IlIIIlIllllIIIllI =0 ;sleep (.5 );break 
		if IlIIIlIllllIIIllI ==1 :break 
	IIlllIIIIlIlIllIl [0 ]=_IIIlIlIIIlllIllII 
def IlllllIllIIIlIlll (IIlIlllIIIIlIIIll ,IlllllIIIIIIIIIII ,IIlIllIIlIIlIIIlI ,IlllllllIIIllIlIl ,IIIlIIlIlllllIlIl ,IIIlIIIllIIlIlIll ):
	if IIlIlllIIIIlIIIll :IIlIlllIlIIIIIlII =_IIIlIlIIIlllIllII ;CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IIlIlllIlIIIIIlII ,IlllllIIIIIIIIIII ,IIlIllIIlIIlIIIlI );return {_IIIllIIIlIllIlIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	else :IIlIlllIlIIIIIlII =_IIllIlIIIIIlIllll ;CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IIlIlllIlIIIIIlII ,IlllllIIIIIIIIIII ,IIlIllIIlIIlIIIlI );return {_IIIllIIIlIllIlIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IlIlIIlIIIlIIlllI (IIIlIllllIlIlIIII ,IIIIIIlIllIIIIlll ):IlllIlIIllllIllll =IIIlIllllIlIlIIII ;IlIIIIlIIIIIlIllI =IIIIIIlIllIIIIlll ;IIllIlIIlllIlllll =_IIIlIlIIIlllIllII ;CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IIllIlIIlllIlllll ,IIIlIllllIlIlIIII ,IIIIIIlIllIIIIlll );return {_IIIllIIIlIllIlIIl :IlllIlIIllllIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIllIIIlIllIlIIl :IlIIIIlIIIIIlIllI ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IIIIIlIIlIllllIIl (IIIIlIlIlllIlIIII ,IIlIllIIIIllIIlll ,IIlIIlllIIIIIlIIl ):
	IIlIllIIIIllIIlll ,IIlIIlllIIIIIlIIl =IIIIlIlIIllIIlIll (IIIIlIlIlllIlIIII ,IIlIllIIIIllIIlll ,IIlIIlllIIIIIlIIl )
	if str (IIIIlIlIlllIlIIII )!='':
		with open (str (IIIIlIlIlllIlIIII ),_IIlIlIlIIIlIIIIII )as IIIlIlllIlIIIllIl :IIIIIIlllIlllIIIl =IIIlIlllIlIIIllIl .readlines ();IIlIllIIIIllIIlll ,IIlIIlllIIIIIlIIl =IIIIIIlllIlllIIIl [0 ].split (_IIlIIlIIllIlIIllI )[0 ],IIIIIIlllIlllIIIl [1 ];IlIlIIlIIIlIIlllI (IIlIllIIIIllIIlll ,IIlIIlllIIIIIlIIl )
	else :0 
	return {_IlIlIIlIllIllIlll :IlllIlIlIllllIIIl (),_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIllIIIlIllIlIIl :IIlIllIIIIllIIlll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIllIIIlIllIlIIl :IIlIIlllIIIIIlIIl ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IIlIlIllIllIlIIII (IlllllIIIllllIIll ,IIlIlIIllllIlIIIl ,IIIlIllIIllIlIlIl ,IlIIIIllIIIlllllI ):
	IIIlllIllIllIIIII ='%s/logs/%s/preprocess.log';IIIlIllIIllIlIlIl =IIlIIIllllIIIIIll [IIIlIllIIllIlIlIl ];os .makedirs (_IIIllIllllIIIIIlI %(IllIIIIlIIIlIIIll ,IIlIlIIllllIlIIIl ),exist_ok =_IIIlIlIIIlllIllII );IlIIlIlIIlIllllII =open (IIIlllIllIllIIIII %(IllIIIIlIIIlIIIll ,IIlIlIIllllIlIIIl ),_IIlIIllIIllIlIIII );IlIIlIlIIlIllllII .close ();IllIIlIllIlIIIIlI =IllIIllIIIIlIIlIl .python_cmd +' trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s '%(IlllllIIIllllIIll ,IIIlIllIIllIlIlIl ,IlIIIIllIIIlllllI ,IllIIIIlIIIlIIIll ,IIlIlIIllllIlIIIl )+str (IllIIllIIIIlIIlIl .noparallel );print (IllIIlIllIlIIIIlI );IIIlIIIlIllIIllIl =Popen (IllIIlIllIlIIIIlI ,shell =_IIIlIlIIIlllIllII );IIIlIIlIlIIllIIIl =[_IIllIlIIIIIlIllll ];threading .Thread (target =IllIIlIlllIIllIlI ,args =(IIIlIIlIlIIllIIIl ,IIIlIIIlIllIIllIl )).start ()
	while 1 :
		with open (IIIlllIllIllIIIII %(IllIIIIlIIIlIIIll ,IIlIlIIllllIlIIIl ),_IIlIlIlIIIlIIIIII )as IlIIlIlIIlIllllII :yield IlIIlIlIIlIllllII .read ()
		sleep (1 )
		if IIIlIIlIlIIllIIIl [0 ]:break 
	with open (IIIlllIllIllIIIII %(IllIIIIlIIIlIIIll ,IIlIlIIllllIlIIIl ),_IIlIlIlIIIlIIIIII )as IlIIlIlIIlIllllII :IlIIIIllIlIIlIIll =IlIIlIlIIlIllllII .read ()
	print (IlIIIIllIlIIlIIll );yield IlIIIIllIlIIlIIll 
def IIIllIllllllllIlI (IlIIlIIlllllIlllI ,IIIIIIlIIllIlIlIl ,IlllllIllIIIIIIlI ,IllIllIlIIIllIIlI ,IlIlIlIlIlIllIllI ,IllIIIllIlIIlIIlI ,IIlIIlIIlllIllllI ):
	IIIIIIlIlIIlllIIl ='%s/logs/%s/extract_fl_feature.log';IlIIlIIlllllIlllI =IlIIlIIlllllIlllI .split ('-');os .makedirs (_IIIllIllllIIIIIlI %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),exist_ok =_IIIlIlIIIlllIllII );IllIlIllllIIlllIl =open (IIIIIIlIlIIlllIIl %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),_IIlIIllIIllIlIIII );IllIlIllllIIlllIl .close ()
	if IllIllIlIIIllIIlI :
		IIIIllIIIllIIIIlI =IllIIllIIIIlIIlIl .python_cmd +' extract_fl_print.py %s/logs/%s %s %s %s'%(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ,IIIIIIlIIllIlIlIl ,IlllllIllIIIIIIlI ,IIlIIlIIlllIllllI );print (IIIIllIIIllIIIIlI );IlIllIIllIllllllI =Popen (IIIIllIIIllIIIIlI ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );IlIIllIlIIIIIlIlI =[_IIllIlIIIIIlIllll ];threading .Thread (target =IllIIlIlllIIllIlI ,args =(IlIIllIlIIIIIlIlI ,IlIllIIllIllllllI )).start ()
		while 1 :
			with open (IIIIIIlIlIIlllIIl %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),_IIlIlIlIIIlIIIIII )as IllIlIllllIIlllIl :yield IllIlIllllIIlllIl .read ()
			sleep (1 )
			if IlIIllIlIIIIIlIlI [0 ]:break 
		with open (IIIIIIlIlIIlllIIl %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),_IIlIlIlIIIlIIIIII )as IllIlIllllIIlllIl :IlIllIIIIlIlIlIIl =IllIlIllllIIlllIl .read ()
		print (IlIllIIIIlIlIlIIl );yield IlIllIIIIlIlIlIIl 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IIllIlIIIllIlIIIl =len (IlIIlIIlllllIlllI );IlllIllIIIllIIIII =[]
	for (IIllllIllIlIIIlll ,IlIIlIlllIIIlIlII )in enumerate (IlIIlIIlllllIlllI ):IIIIllIIIllIIIIlI =IllIIllIIIIlIIlIl .python_cmd +' extract_feature_print.py %s %s %s %s %s/logs/%s %s'%(IllIIllIIIIlIIlIl .device ,IIllIlIIIllIlIIIl ,IIllllIllIlIIIlll ,IlIIlIlllIIIlIlII ,IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ,IllIIIllIlIIlIIlI );print (IIIIllIIIllIIIIlI );IlIllIIllIllllllI =Popen (IIIIllIIIllIIIIlI ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );IlllIllIIIllIIIII .append (IlIllIIllIllllllI )
	IlIIllIlIIIIIlIlI =[_IIllIlIIIIIlIllll ];threading .Thread (target =IIlIIIlllllIlIlIl ,args =(IlIIllIlIIIIIlIlI ,IlllIllIIIllIIIII )).start ()
	while 1 :
		with open (IIIIIIlIlIIlllIIl %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),_IIlIlIlIIIlIIIIII )as IllIlIllllIIlllIl :yield IllIlIllllIIlllIl .read ()
		sleep (1 )
		if IlIIllIlIIIIIlIlI [0 ]:break 
	with open (IIIIIIlIlIIlllIIl %(IllIIIIlIIIlIIIll ,IlIlIlIlIlIllIllI ),_IIlIlIlIIIlIIIIII )as IllIlIllllIIlllIl :IlIllIIIIlIlIlIIl =IllIlIllllIIlllIl .read ()
	print (IlIllIIIIlIlIlIIl );yield IlIllIIIIlIlIlIIl 
def IlllIIIlIIlIIIIlI (IIlIIlIIlIIIIlllI ,IlllllIIIlIIIIlIl ,IlllIIlllIIIIllIl ):
	IlllIIIIllIlIlIIl =''if IlllIIlllIIIIllIl ==_IllllIlIIIIllllll else _IlllIIllIIlIIllll ;IIllIIlIlIllIIlIl =_IlIIllllIlIIlIlll if IlllllIIIlIIIIlIl else '';IIIllIlIIIlIllIll =os .access (_IllllIlIlllIIIlIl %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI ),os .F_OK );IIIIIlllllIIlllll =os .access (_IllllIlIllllIlIII %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI ),os .F_OK )
	if not IIIllIlIIIlIllIll :print (_IllllIlIlllIIIlIl %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI ),_IIIIIIIlllIIIlllI )
	if not IIIIIlllllIIlllll :print (_IllllIlIllllIlIII %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI ),_IIIIIIIlllIIIlllI )
	return _IllllIlIlllIIIlIl %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI )if IIIllIlIIIlIllIll else '',_IllllIlIllllIlIII %(IlllIIIIllIlIlIIl ,IIllIIlIlIllIIlIl ,IIlIIlIIlIIIIlllI )if IIIIIlllllIIlllll else ''
def IIIlIlllllllIIlII (IIlIlllIlIIlIlIlI ,IIlIlIlIIllllllII ,IIlIllIlIIIlllIII ):
	IIIIIllllIlIIIIll =''if IIlIllIlIIIlllIII ==_IllllIlIIIIllllll else _IlllIIllIIlIIllll 
	if IIlIlllIlIIlIlIlI ==_IIIllllIIlIIllIII and IIlIllIlIIIlllIII ==_IllllIlIIIIllllll :IIlIlllIlIIlIlIlI =_IllIlIIllIllllIll 
	IIlIlIIlIlIIlllll ={_IlIlIIlIllIllIlll :[_IllIlIIllIllllIll ,_IIllIIllIIIllIlll ],_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI ,_IIIllIIIlIllIlIIl :IIlIlllIlIIlIlIlI }if IIlIllIlIIIlllIII ==_IllllIlIIIIllllll else {_IlIlIIlIllIllIlll :[_IllIlIIllIllllIll ,_IIllIIllIIIllIlll ,_IIIllllIIlIIllIII ],_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI ,_IIIllIIIlIllIlIIl :IIlIlllIlIIlIlIlI };IIIIllIlllIIlIlII =_IlIIllllIlIIlIlll if IIlIlIlIIllllllII else '';IIIIlIIlIIlIIIIll =os .access (_IllllIlIlllIIIlIl %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI ),os .F_OK );IIIlIlIlllIIlIIIl =os .access (_IllllIlIllllIlIII %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI ),os .F_OK )
	if not IIIIlIIlIIlIIIIll :print (_IllllIlIlllIIIlIl %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI ),_IIIIIIIlllIIIlllI )
	if not IIIlIlIlllIIlIIIl :print (_IllllIlIllllIlIII %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI ),_IIIIIIIlllIIIlllI )
	return _IllllIlIlllIIIlIl %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI )if IIIIlIIlIIlIIIIll else '',_IllllIlIllllIlIII %(IIIIIllllIlIIIIll ,IIIIllIlllIIlIlII ,IIlIlllIlIIlIlIlI )if IIIlIlIlllIIlIIIl else '',IIlIlIIlIlIIlllll 
def IlIIllllIIlIIIIII (IIIlIlIIIIlIIlIll ,IlllIIlIIIIlIlllI ,IIIlIIIlIIllllIII ,IIlllIIIllIllllIl ,IIlIIllIIllllllll ,IlIIllIlllIIIlIlI ,IlIllIlIllIIIIIIl ,IllIlIIlllIIIlIll ,IlllIIIlIlIlIIlII ):
	IlIIllIllIlllIlII ='not exist, will not use pretrained model';IIIIIlllllllIIlll ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IIIlIlIllIlIIlllI ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IlIlIIIIllIlIIIll =''if IIIlIIIlIIllllIII ==_IllllIlIIIIllllll else _IlllIIllIIlIIllll ;IIlIlIllIlIlIlIII =os .access (IIIlIlIllIlIIlllI %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI ),os .F_OK );IIlllIllIIlIlIIlI =os .access (IIIIIlllllllIIlll %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI ),os .F_OK )
	if not IIlIlIllIlIlIlIII :print (IIIlIlIllIlIIlllI %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI ),IlIIllIllIlllIlII )
	if not IIlllIllIIlIlIIlI :print (IIIIIlllllllIIlll %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI ),IlIIllIllIlllIlII )
	if IIIlIlIIIIlIIlIll :return {_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },IIIlIlIllIlIIlllI %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI )if IIlIlIllIlIlIlIII else '',IIIIIlllllllIIlll %(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI )if IIlllIllIIlIlIIlI else '',{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	return {_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI )if IIlIlIllIlIlIlIII else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IlIlIIIIllIlIIIll ,IlllIIlIIIIlIlllI )if IIlllIllIIlIlIIlI else '',{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
global IllIlIIIIlIIlIlII 
def IllIlIlIlIIllllll (IllllllIIlIllIllI ,IIIllIlIIllIIIlII ):
	IIIlIIlIllIIIIllI =1 ;IIlllIlIIIIlIllII =os .path .join (IllllllIIlIllIllI ,'1_16k_wavs')
	if os .path .exists (IIlllIlIIIIlIllII )and os .path .isdir (IIlllIlIIIIlIllII ):
		IllllIIlllllIIIll =[IIIllIllllIIIIlIl for IIIllIllllIIIIlIl in os .listdir (IIlllIlIIIIlIllII )if IIIllIllllIIIIlIl .endswith ('.wav')]
		if IllllIIlllllIIIll :
			IlIIIIIlIIIIllIIl =len (IllllIIlllllIIIll );IIIlIIlIllIIIIllI =math .ceil (IlIIIIIlIIIIllIIl /IIIllIlIIllIIIlII )
			if IIIlIIlIllIIIIllI >1 :IIIlIIlIllIIIIllI +=1 
	return IIIlIIlIllIIIIllI 
def IllIIlIllIIllIIII (IllIlIlllIIllIllI ,IlIlllllIlIIIIIll ,IIIIlllIIllIIlIlI ,IlIIllIIIlIlIlIll ,IllIlIlIIllIllIII ,IIIllIIIllllIIIII ,IllIIIlIllllllIlI ,IlllIllIllIlIlIlI ,IlllIIIllllIIIIIl ,IIIlIIIlIIIllIllI ,IlIIIIlIlllIIllIl ,IIIIlIIlIIIIIllIl ,IIIllIlllIllIllII ,IllIlIIIIllllIlIl ):
	IIIllllIllIllIIll ='\x08';CSVutil (_IllIIllllllllIlIl ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,_IIllIlIIIIIlIllll );IIllIIIllIIlllllI =_IIIllIllllIIIIIlI %(IllIIIIlIIIlIIIll ,IllIlIlllIIllIllI );os .makedirs (IIllIIIllIIlllllI ,exist_ok =_IIIlIlIIIlllIllII );IIIllllIIIIllIlII =_IllllIIIIlllIlllI %IIllIIIllIIlllllI ;IlIIlIIIIIlllIIIl =_IIllIIIIllIIlIlll %IIllIIIllIIlllllI if IllIlIIIIllllIlIl ==_IllllIlIIIIllllll else _IIIlIlIIllIIlIlll %IIllIIIllIIlllllI ;IlIlIlllIlIllIIlI =IllIlIlIlIIllllll (IIllIIIllIIlllllI ,IllIIIlIllllllIlI )
	if IIIIlllIIllIIlIlI :IIIllllIIlIIIIIII ='%s/2a_f0'%IIllIIIllIIlllllI ;IIlllIIllllIlIlIl =_IIlIlllIIIIllllII %IIllIIIllIIlllllI ;IlIlIIIIIIlIIIlll =set ([IlIIlllIIlllIllII .split (_IIIIlIIIIlIlIllIl )[0 ]for IlIIlllIIlllIllII in os .listdir (IIIllllIIIIllIlII )])&set ([IlllIIllIlIIIlIIl .split (_IIIIlIIIIlIlIllIl )[0 ]for IlllIIllIlIIIlIIl in os .listdir (IlIIlIIIIIlllIIIl )])&set ([IIlIIlIIIllllIIlI .split (_IIIIlIIIIlIlIllIl )[0 ]for IIlIIlIIIllllIIlI in os .listdir (IIIllllIIlIIIIIII )])&set ([IlIIIIIIIlIIlllII .split (_IIIIlIIIIlIlIllIl )[0 ]for IlIIIIIIIlIIlllII in os .listdir (IIlllIIllllIlIlIl )])
	else :IlIlIIIIIIlIIIlll =set ([IllIllIIlIIIlllIl .split (_IIIIlIIIIlIlIllIl )[0 ]for IllIllIIlIIIlllIl in os .listdir (IIIllllIIIIllIlII )])&set ([IlllIlIllllllIlll .split (_IIIIlIIIIlIlIllIl )[0 ]for IlllIlIllllllIlll in os .listdir (IlIIlIIIIIlllIIIl )])
	IIlIIIlIIlIlIIIII =[]
	for IIllIIlIllIIllllI in IlIlIIIIIIlIIIlll :
		if IIIIlllIIllIIlIlI :IIlIIIlIIlIlIIIII .append (_IIIllIllIlIlllIIl %(IIIllllIIIIllIlII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IlIIlIIIIIlllIIIl .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IIIllllIIlIIIIIII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IIlllIIllllIlIlIl .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IlIIllIIIlIlIlIll ))
		else :IIlIIIlIIlIlIIIII .append (_IlIIIlllIlIIIIlll %(IIIllllIIIIllIlII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IlIIlIIIIIlllIIIl .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIllIIlIllIIllllI ,IlIIllIIIlIlIlIll ))
	IIlIlIlllllIllllI =256 if IllIlIIIIllllIlIl ==_IllllIlIIIIllllll else 768 
	if IIIIlllIIllIIlIlI :
		for _IllllIIIlllIllIIl in range (2 ):IIlIIIlIIlIlIIIII .append (_IIIIIlllIIllIIlII %(IllIIIIlIIIlIIIll ,IlIlllllIlIIIIIll ,IllIIIIlIIIlIIIll ,IIlIlIlllllIllllI ,IllIIIIlIIIlIIIll ,IllIIIIlIIIlIIIll ,IlIIllIIIlIlIlIll ))
	else :
		for _IllllIIIlllIllIIl in range (2 ):IIlIIIlIIlIlIIIII .append (_IllIlIlllIllIllIl %(IllIIIIlIIIlIIIll ,IlIlllllIlIIIIIll ,IllIIIIlIIIlIIIll ,IIlIlIlllllIllllI ,IlIIllIIIlIlIlIll ))
	shuffle (IIlIIIlIIlIlIIIII )
	with open (_IIIIllIlIlIIllllI %IIllIIIllIIlllllI ,_IIlIIllIIllIlIIII )as IlIlIlIIIlIIIlIII :IlIlIlIIIlIIIlIII .write (_IIlIIlIIllIlIIllI .join (IIlIIIlIIlIlIIIII ))
	print (_IllllIIllIlllIlIl );print ('use gpus:',IlIIIIlIlllIIllIl )
	if IlllIIIllllIIIIIl =='':print ('no pretrained Generator')
	if IIIlIIIlIIIllIllI =='':print ('no pretrained Discriminator')
	if IlIIIIlIlllIIllIl :IlIlIIIlIlIlIlllI =IllIIllIIIIlIIlIl .python_cmd +' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s'%(IllIlIlllIIllIllI ,IlIlllllIlIIIIIll ,1 if IIIIlllIIllIIlIlI else 0 ,IllIIIlIllllllIlI ,IlIIIIlIlllIIllIl ,IIIllIIIllllIIIII ,IllIlIlIIllIllIII ,_IIIlllIlIlIIlllIl %IlllIIIllllIIIIIl if IlllIIIllllIIIIIl !=''else '',_IlIlllIlllIllIIlI %IIIlIIIlIIIllIllI if IIIlIIIlIIIllIllI !=''else '',1 if IlllIllIllIlIlIlI ==_IIIlIlIIIlllIllII else 0 ,1 if IIIIlIIlIIIIIllIl ==_IIIlIlIIIlllIllII else 0 ,1 if IIIllIlllIllIllII ==_IIIlIlIIIlllIllII else 0 ,IllIlIIIIllllIlIl ,IlIlIlllIlIllIIlI )
	else :IlIlIIIlIlIlIlllI =IllIIllIIIIlIIlIl .python_cmd +' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s'%(IllIlIlllIIllIllI ,IlIlllllIlIIIIIll ,1 if IIIIlllIIllIIlIlI else 0 ,IllIIIlIllllllIlI ,IIIllIIIllllIIIII ,IllIlIlIIllIllIII ,_IIIlllIlIlIIlllIl %IlllIIIllllIIIIIl if IlllIIIllllIIIIIl !=''else IIIllllIllIllIIll ,_IlIlllIlllIllIIlI %IIIlIIIlIIIllIllI if IIIlIIIlIIIllIllI !=''else IIIllllIllIllIIll ,1 if IlllIllIllIlIlIlI ==_IIIlIlIIIlllIllII else 0 ,1 if IIIIlIIlIIIIIllIl ==_IIIlIlIIIlllIllII else 0 ,1 if IIIllIlllIllIllII ==_IIIlIlIIIlllIllII else 0 ,IllIlIIIIllllIlIl ,IlIlIlllIlIllIIlI )
	print (IlIlIIIlIlIlIlllI );global IIIllIIllllIIllIl ;IIIllIIllllIIllIl =Popen (IlIlIIIlIlIlIlllI ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );global IlllIllIIllllIIIl ;IlllIllIIllllIIIl =IIIllIIllllIIllIl .pid ;IIIllIIllllIIllIl .wait ();return _IIIlllIlIIllIlIIl ,{_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IIlIlIIIlllIlIllI (IIlIllIIlllllIIll ,IIIllIlIllIlIlllI ):
	IlIIIllIIlIIllIIl =_IIIllIllllIIIIIlI %(IllIIIIlIIIlIIIll ,IIlIllIIlllllIIll );os .makedirs (IlIIIllIIlIIllIIl ,exist_ok =_IIIlIlIIIlllIllII );IlIIlllllllIIIlII =_IIllIIIIllIIlIlll %IlIIIllIIlIIllIIl if IIIllIlIllIlIlllI ==_IllllIlIIIIllllll else _IIIlIlIIllIIlIlll %IlIIIllIIlIIllIIl 
	if not os .path .exists (IlIIlllllllIIIlII ):return '请先进行特征提取!'
	IlllIlIlIllIIlIII =list (os .listdir (IlIIlllllllIIIlII ))
	if len (IlllIlIlIllIIlIII )==0 :return '请先进行特征提取！'
	IIlIlIIllIlIIIlIl =[];IlIllIIlIIlIIIIIl =[]
	for IIlllIlIIIIIlIllI in sorted (IlllIlIlIllIIlIII ):IlIIllIIIlIllllll =np .load (_IIlIIIIIIIIllllII %(IlIIlllllllIIIlII ,IIlllIlIIIIIlIllI ));IlIllIIlIIlIIIIIl .append (IlIIllIIIlIllllll )
	IllIIllllIIIlIIlI =np .concatenate (IlIllIIlIIlIIIIIl ,0 );IIIlIIlIIIlllllII =np .arange (IllIIllllIIIlIIlI .shape [0 ]);np .random .shuffle (IIIlIIlIIIlllllII );IllIIllllIIIlIIlI =IllIIllllIIIlIIlI [IIIlIIlIIIlllllII ]
	if IllIIllllIIIlIIlI .shape [0 ]>2e5 :
		IIlIlIIllIlIIIlIl .append (_IIllllIlllllIIlII %IllIIllllIIIlIIlI .shape [0 ]);yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl )
		try :IllIIllllIIIlIIlI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIIlIlIIIlllIllII ,batch_size =256 *IllIIllIIIIlIIlIl .n_cpu ,compute_labels =_IIllIlIIIIIlIllll ,init ='random').fit (IllIIllllIIIlIIlI ).cluster_centers_ 
		except :IllIllIIlIlIIlIll =traceback .format_exc ();print (IllIllIIlIlIIlIll );IIlIlIIllIlIIIlIl .append (IllIllIIlIlIIlIll );yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl )
	np .save (_IlIllIIlIIlIIlIIl %IlIIIllIIlIIllIIl ,IllIIllllIIIlIIlI );IlIIllIIIlllIIIlI =min (int (16 *np .sqrt (IllIIllllIIIlIIlI .shape [0 ])),IllIIllllIIIlIIlI .shape [0 ]//39 );IIlIlIIllIlIIIlIl .append ('%s,%s'%(IllIIllllIIIlIIlI .shape ,IlIIllIIIlllIIIlI ));yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl );IIIllllllIIIlIlII =faiss .index_factory (256 if IIIllIlIllIlIlllI ==_IllllIlIIIIllllll else 768 ,_IlIIIlIllllIIllll %IlIIllIIIlllIIIlI );IIlIlIIllIlIIIlIl .append ('training');yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl );IllIllIlIlIlIlIll =faiss .extract_index_ivf (IIIllllllIIIlIlII );IllIllIlIlIlIlIll .nprobe =1 ;IIIllllllIIIlIlII .train (IllIIllllIIIlIIlI );faiss .write_index (IIIllllllIIIlIlII ,_IlllIIllIlIIllllI %(IlIIIllIIlIIllIIl ,IlIIllIIIlllIIIlI ,IllIllIlIlIlIlIll .nprobe ,IIlIllIIlllllIIll ,IIIllIlIllIlIlllI ));IIlIlIIllIlIIIlIl .append ('adding');yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl );IlIIIIIIllIIIlIll =8192 
	for IlIlIlIIllIIIlIll in range (0 ,IllIIllllIIIlIIlI .shape [0 ],IlIIIIIIllIIIlIll ):IIIllllllIIIlIlII .add (IllIIllllIIIlIIlI [IlIlIlIIllIIIlIll :IlIlIlIIllIIIlIll +IlIIIIIIllIIIlIll ])
	faiss .write_index (IIIllllllIIIlIlII ,_IIlIIllllIIIIIlll %(IlIIIllIIlIIllIIl ,IlIIllIIIlllIIIlI ,IllIllIlIlIlIlIll .nprobe ,IIlIllIIlllllIIll ,IIIllIlIllIlIlllI ));IIlIlIIllIlIIIlIl .append ('Successful Index Construction，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIllIIIlllIIIlI ,IllIllIlIlIlIlIll .nprobe ,IIlIllIIlllllIIll ,IIIllIlIllIlIlllI ));yield _IIlIIlIIllIlIIllI .join (IIlIlIIllIlIIIlIl )
def IlIlllIlIlIIllIlI (IlIIllIlIIllIIlII ,IIlIIIlIIlllllllI ,IlIIlIIlllIIIllll ,IIlIIllIlllIIlllI ,IlIIIIllllllllIll ,IIlIIIlllIIIIllII ,IlIIlIIIIIIIIlIll ,IlIIIIlIIIlllIllI ,IlIIlIIIlIllIIIII ,IlllllIIlIIlllllI ,IIIlIllIIIIIlIIIl ,IIIllIlIIIllIIIIl ,IlIIIllIIIlIIlIll ,IIIIlllllIlIlIlIl ,IllIIlIIlIlIllIll ,IllllllIlIIIIlIlI ,IIIIllIIIlllllIII ,IlIIIlIIIlIllIlIl ):
	IlIIllIlllIIIIIlI =[]
	def IIIlIlllIlllIIIll (IlllIllIllIllllIl ):IlIIllIlllIIIIIlI .append (IlllIllIllIllllIl );return _IIlIIlIIllIlIIllI .join (IlIIllIlllIIIIIlI )
	IllllIlIllIIlllll =_IIIllIllllIIIIIlI %(IllIIIIlIIIlIIIll ,IlIIllIlIIllIIlII );IIllllIllIIlIlllI ='%s/preprocess.log'%IllllIlIllIIlllll ;IlllIIIIIlIIllIll ='%s/extract_fl_feature.log'%IllllIlIllIIlllll ;IlllllIlllIIIIIII =_IllllIIIIlllIlllI %IllllIlIllIIlllll ;IlIllllIlIllIlllI =_IIllIIIIllIIlIlll %IllllIlIllIIlllll if IIIIllIIIlllllIII ==_IllllIlIIIIllllll else _IIIlIlIIllIIlIlll %IllllIlIllIIlllll ;os .makedirs (IllllIlIllIIlllll ,exist_ok =_IIIlIlIIIlllIllII );open (IIllllIllIIlIlllI ,_IIlIIllIIllIlIIII ).close ();IIIlIlIlIlIlIllIl =IllIIllIIIIlIIlIl .python_cmd +' trainset_preprocess_pipeline_print.py %s %s %s %s '%(IIlIIllIlllIIlllI ,IIlIIIllllIIIIIll [IIlIIIlIIlllllllI ],IIlIIIlllIIIIllII ,IllllIlIllIIlllll )+str (IllIIllIIIIlIIlIl .noparallel );yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII ('step1:正在处理数据'));yield IIIlIlllIlllIIIll (IIIlIlIlIlIlIllIl );IIllIIlIlIllIlIll =Popen (IIIlIlIlIlIlIllIl ,shell =_IIIlIlIIIlllIllII );IIllIIlIlIllIlIll .wait ()
	with open (IIllllIllIIlIlllI ,_IIlIlIlIIIlIIIIII )as IIIlIlIlllIIIIllI :print (IIIlIlIlllIIIIllI .read ())
	open (IlllIIIIIlIIllIll ,_IIlIIllIIllIlIIII )
	if IlIIlIIlllIIIllll :
		yield IIIlIlllIlllIIIll ('step2a:正在提取音高');IIIlIlIlIlIlIllIl =IllIIllIIIIlIIlIl .python_cmd +' extract_fl_print.py %s %s %s %s'%(IllllIlIllIIlllll ,IIlIIIlllIIIIllII ,IlIIlIIIIIIIIlIll ,IlIIIlIIIlIllIlIl );yield IIIlIlllIlllIIIll (IIIlIlIlIlIlIllIl );IIllIIlIlIllIlIll =Popen (IIIlIlIlIlIlIllIl ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );IIllIIlIlIllIlIll .wait ()
		with open (IlllIIIIIlIIllIll ,_IIlIlIlIIIlIIIIII )as IIIlIlIlllIIIIllI :print (IIIlIlIlllIIIIllI .read ())
	else :yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII ('step2a:无需提取音高'))
	yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII ('step2b:正在提取特征'));IIlIllIIIlIllIIIl =IIIIlllllIlIlIlIl .split ('-');IIlllllllIIIlllll =len (IIlIllIIIlIllIIIl );IIIIIIlllIIllIIlI =[]
	for (IIllIIlIlIIllIIII ,IIllllIIlIIlIllll )in enumerate (IIlIllIIIlIllIIIl ):IIIlIlIlIlIlIllIl =IllIIllIIIIlIIlIl .python_cmd +' extract_feature_print.py %s %s %s %s %s %s'%(IllIIllIIIIlIIlIl .device ,IIlllllllIIIlllll ,IIllIIlIlIIllIIII ,IIllllIIlIIlIllll ,IllllIlIllIIlllll ,IIIIllIIIlllllIII );yield IIIlIlllIlllIIIll (IIIlIlIlIlIlIllIl );IIllIIlIlIllIlIll =Popen (IIIlIlIlIlIlIllIl ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );IIIIIIlllIIllIIlI .append (IIllIIlIlIllIlIll )
	for IIllIIlIlIllIlIll in IIIIIIlllIIllIIlI :IIllIIlIlIllIlIll .wait ()
	with open (IlllIIIIIlIIllIll ,_IIlIlIlIIIlIIIIII )as IIIlIlIlllIIIIllI :print (IIIlIlIlllIIIIllI .read ())
	yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII ('step3a:正在训练模型'))
	if IlIIlIIlllIIIllll :IlIIllIIIIIIIIIII ='%s/2a_f0'%IllllIlIllIIlllll ;IIlllIIllllIlIIII =_IIlIlllIIIIllllII %IllllIlIllIIlllll ;IllIIIlIlIIlIllIl =set ([IllIIIllllIIlIlII .split (_IIIIlIIIIlIlIllIl )[0 ]for IllIIIllllIIlIlII in os .listdir (IlllllIlllIIIIIII )])&set ([IlIlIlIIlIIllIlll .split (_IIIIlIIIIlIlIllIl )[0 ]for IlIlIlIIlIIllIlll in os .listdir (IlIllllIlIllIlllI )])&set ([IlIlllllllllllllI .split (_IIIIlIIIIlIlIllIl )[0 ]for IlIlllllllllllllI in os .listdir (IlIIllIIIIIIIIIII )])&set ([IlIlllIIllIIIIIll .split (_IIIIlIIIIlIlIllIl )[0 ]for IlIlllIIllIIIIIll in os .listdir (IIlllIIllllIlIIII )])
	else :IllIIIlIlIIlIllIl =set ([IllIllIIlIlIlllIl .split (_IIIIlIIIIlIlIllIl )[0 ]for IllIllIIlIlIlllIl in os .listdir (IlllllIlllIIIIIII )])&set ([IIlllllllIllllIlI .split (_IIIIlIIIIlIlIllIl )[0 ]for IIlllllllIllllIlI in os .listdir (IlIllllIlIllIlllI )])
	IlIIllllIIlIlIIll =[]
	for IIIIIIIIllIIllllI in IllIIIlIlIIlIllIl :
		if IlIIlIIlllIIIllll :IlIIllllIIlIlIIll .append (_IIIllIllIlIlllIIl %(IlllllIlllIIIIIII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IlIllllIlIllIlllI .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IlIIllIIIIIIIIIII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IIlllIIllllIlIIII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IlIIIIllllllllIll ))
		else :IlIIllllIIlIlIIll .append (_IlIIIlllIlIIIIlll %(IlllllIlllIIIIIII .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IlIllllIlIllIlllI .replace (_IlllllllIIllllIII ,_IIlllIlIIlIlIllII ),IIIIIIIIllIIllllI ,IlIIIIllllllllIll ))
	IIIIIlllIllIIIllI =256 if IIIIllIIIlllllIII ==_IllllIlIIIIllllll else 768 
	if IlIIlIIlllIIIllll :
		for _IIIlIlIIlIIIIllII in range (2 ):IlIIllllIIlIlIIll .append (_IIIIIlllIIllIIlII %(IllIIIIlIIIlIIIll ,IIlIIIlIIlllllllI ,IllIIIIlIIIlIIIll ,IIIIIlllIllIIIllI ,IllIIIIlIIIlIIIll ,IllIIIIlIIIlIIIll ,IlIIIIllllllllIll ))
	else :
		for _IIIlIlIIlIIIIllII in range (2 ):IlIIllllIIlIlIIll .append (_IllIlIlllIllIllIl %(IllIIIIlIIIlIIIll ,IIlIIIlIIlllllllI ,IllIIIIlIIIlIIIll ,IIIIIlllIllIIIllI ,IlIIIIllllllllIll ))
	shuffle (IlIIllllIIlIlIIll )
	with open (_IIIIllIlIlIIllllI %IllllIlIllIIlllll ,_IIlIIllIIllIlIIII )as IIIlIlIlllIIIIllI :IIIlIlIlllIIIIllI .write (_IIlIIlIIllIlIIllI .join (IlIIllllIIlIlIIll ))
	yield IIIlIlllIlllIIIll (_IllllIIllIlllIlIl )
	if IIIIlllllIlIlIlIl :IIIlIlIlIlIlIllIl =IllIIllIIIIlIIlIl .python_cmd +' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(IlIIllIlIIllIIlII ,IIlIIIlIIlllllllI ,1 if IlIIlIIlllIIIllll else 0 ,IlllllIIlIIlllllI ,IIIIlllllIlIlIlIl ,IlIIlIIIlIllIIIII ,IlIIIIlIIIlllIllI ,_IIIlllIlIlIIlllIl %IIIllIlIIIllIIIIl if IIIllIlIIIllIIIIl !=''else '',_IlIlllIlllIllIIlI %IlIIIllIIIlIIlIll if IlIIIllIIIlIIlIll !=''else '',1 if IIIlIllIIIIIlIIIl ==_IIIlIlIIIlllIllII else 0 ,1 if IllIIlIIlIlIllIll ==_IIIlIlIIIlllIllII else 0 ,1 if IllllllIlIIIIlIlI ==_IIIlIlIIIlllIllII else 0 ,IIIIllIIIlllllIII )
	else :IIIlIlIlIlIlIllIl =IllIIllIIIIlIIlIl .python_cmd +' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(IlIIllIlIIllIIlII ,IIlIIIlIIlllllllI ,1 if IlIIlIIlllIIIllll else 0 ,IlllllIIlIIlllllI ,IlIIlIIIlIllIIIII ,IlIIIIlIIIlllIllI ,_IIIlllIlIlIIlllIl %IIIllIlIIIllIIIIl if IIIllIlIIIllIIIIl !=''else '',_IlIlllIlllIllIIlI %IlIIIllIIIlIIlIll if IlIIIllIIIlIIlIll !=''else '',1 if IIIlIllIIIIIlIIIl ==_IIIlIlIIIlllIllII else 0 ,1 if IllIIlIIlIlIllIll ==_IIIlIlIIIlllIllII else 0 ,1 if IllllllIlIIIIlIlI ==_IIIlIlIIIlllIllII else 0 ,IIIIllIIIlllllIII )
	yield IIIlIlllIlllIIIll (IIIlIlIlIlIlIllIl );IIllIIlIlIllIlIll =Popen (IIIlIlIlIlIlIllIl ,shell =_IIIlIlIIIlllIllII ,cwd =IllIIIIlIIIlIIIll );IIllIIlIlIllIlIll .wait ();yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII (_IIIlllIlIIllIlIIl ));IIIIlIIIlIllllIIl =[];IIlIlIlllIlIlIIlI =list (os .listdir (IlIllllIlIllIlllI ))
	for IIIIIIIIllIIllllI in sorted (IIlIlIlllIlIlIIlI ):IIlllIIllIlIllIIl =np .load (_IIlIIIIIIIIllllII %(IlIllllIlIllIlllI ,IIIIIIIIllIIllllI ));IIIIlIIIlIllllIIl .append (IIlllIIllIlIllIIl )
	IlIlllIlIlIlllIll =np .concatenate (IIIIlIIIlIllllIIl ,0 );IlIlIlIIIIlIIIllI =np .arange (IlIlllIlIlIlllIll .shape [0 ]);np .random .shuffle (IlIlIlIIIIlIIIllI );IlIlllIlIlIlllIll =IlIlllIlIlIlllIll [IlIlIlIIIIlIIIllI ]
	if IlIlllIlIlIlllIll .shape [0 ]>2e5 :
		IIlIIIIIlIIllIlll =_IIllllIlllllIIlII %IlIlllIlIlIlllIll .shape [0 ];print (IIlIIIIIlIIllIlll );yield IIIlIlllIlllIIIll (IIlIIIIIlIIllIlll )
		try :IlIlllIlIlIlllIll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIIlIlIIIlllIllII ,batch_size =256 *IllIIllIIIIlIIlIl .n_cpu ,compute_labels =_IIllIlIIIIIlIllll ,init ='random').fit (IlIlllIlIlIlllIll ).cluster_centers_ 
		except :IIlIIIIIlIIllIlll =traceback .format_exc ();print (IIlIIIIIlIIllIlll );yield IIIlIlllIlllIIIll (IIlIIIIIlIIllIlll )
	np .save (_IlIllIIlIIlIIlIIl %IllllIlIllIIlllll ,IlIlllIlIlIlllIll );IlIllIlIIllIlllll =min (int (16 *np .sqrt (IlIlllIlIlIlllIll .shape [0 ])),IlIlllIlIlIlllIll .shape [0 ]//39 );yield IIIlIlllIlllIIIll ('%s,%s'%(IlIlllIlIlIlllIll .shape ,IlIllIlIIllIlllll ));IllIIlIIIlIIlllII =faiss .index_factory (256 if IIIIllIIIlllllIII ==_IllllIlIIIIllllll else 768 ,_IlIIIlIllllIIllll %IlIllIlIIllIlllll );yield IIIlIlllIlllIIIll ('training index');IllIllIlIIlIlllIl =faiss .extract_index_ivf (IllIIlIIIlIIlllII );IllIllIlIIlIlllIl .nprobe =1 ;IllIIlIIIlIIlllII .train (IlIlllIlIlIlllIll );faiss .write_index (IllIIlIIIlIIlllII ,_IlllIIllIlIIllllI %(IllllIlIllIIlllll ,IlIllIlIIllIlllll ,IllIllIlIIlIlllIl .nprobe ,IlIIllIlIIllIIlII ,IIIIllIIIlllllIII ));yield IIIlIlllIlllIIIll ('adding index');IIIIlllIIlIIlIlII =8192 
	for IllIIlIIllIIlIIII in range (0 ,IlIlllIlIlIlllIll .shape [0 ],IIIIlllIIlIIlIlII ):IllIIlIIIlIIlllII .add (IlIlllIlIlIlllIll [IllIIlIIllIIlIIII :IllIIlIIllIIlIIII +IIIIlllIIlIIlIlII ])
	faiss .write_index (IllIIlIIIlIIlllII ,_IIlIIllllIIIIIlll %(IllllIlIllIIlllll ,IlIllIlIIllIlllll ,IllIllIlIIlIlllIl .nprobe ,IlIIllIlIIllIIlII ,IIIIllIIIlllllIII ));yield IIIlIlllIlllIIIll ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIllIlIIllIlllll ,IllIllIlIIlIlllIl .nprobe ,IlIIllIlIIllIIlII ,IIIIllIIIlllllIII ));yield IIIlIlllIlllIIIll (IlIlllIIllllIlIII ('全流程结束！'))
def IIIIIIIIIllIIIIII (IlIlllIIlIIlIllll ):
	IlllIllIIIIllllIl ='train.log'
	if not os .path .exists (IlIlllIIlIIlIllll .replace (os .path .basename (IlIlllIIlIIlIllll ),IlllIllIIIIllllIl )):return {_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
	try :
		with open (IlIlllIIlIIlIllll .replace (os .path .basename (IlIlllIIlIIlIllll ),IlllIllIIIIllllIl ),_IIlIlIlIIIlIIIIII )as IlllllIIIlIllllll :IIlIlllIIlIlllIll =eval (IlllllIIIlIllllll .read ().strip (_IIlIIlIIllIlIIllI ).split (_IIlIIlIIllIlIIllI )[0 ].split ('\t')[-1 ]);IIIIlIlllllllllIl ,IlllllllIlIIIIlIl =IIlIlllIIlIlllIll [_IIIIIlIIIlIllIIlI ],IIlIlllIIlIlllIll ['if_f0'];IIIlllIllIlIlllll =_IIlIllIIlIIIlllIl if _IIIlIIIllIIIIllII in IIlIlllIIlIlllIll and IIlIlllIIlIlllIll [_IIIlIIIllIIIIllII ]==_IIlIllIIlIIIlllIl else _IllllIlIIIIllllll ;return IIIIlIlllllllllIl ,str (IlllllllIlIIIIlIl ),IIIlllIllIlIlllll 
	except :traceback .print_exc ();return {_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IlIIlIlIlllIlIlII (IIIIlIIlIlIIIlIIl ,IllIIlllIlIlIIIII ):IlIlIIllIllIlIlIl ='rnd';IIllllIIIllIlllIl ='pitchf';IIllIIllIIllllIlI ='pitch';IlIlIlIlIlllllIlI ='phone';IlIlIIIIllIIllllI =torch .load (IIIIlIIlIlIIIlIIl ,map_location =_IlIlllIlllllIIlII );IlIlIIIIllIIllllI [_IIlIllllIlllIIIII ][-3 ]=IlIlIIIIllIIllllI [_IlIIIlIlIIlIlIIIl ][_IIlIIlIIlIlIIlIIl ].shape [0 ];IIllllIllIllIllIl =256 if IlIlIIIIllIIllllI .get (_IIIlIIIllIIIIllII ,_IllllIlIIIIllllll )==_IllllIlIIIIllllll else 768 ;IIllllIlllIlIlIII =torch .rand (1 ,200 ,IIllllIllIllIllIl );IIlllIlllIlllIlll =torch .tensor ([200 ]).long ();IIIlIlllllllIlllI =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IlIlIllIIlIlIlIlI =torch .rand (1 ,200 );IlIlIIIlIlIlllIII =torch .LongTensor ([0 ]);IllllIIlIIIlIlIll =torch .rand (1 ,192 ,200 );IIIIIlIlIIIllIllI =_IlIlllIlllllIIlII ;IIIlllIIIIIIllIII =SynthesizerTrnMsNSFsidM (*IlIlIIIIllIIllllI [_IIlIllllIlllIIIII ],is_half =_IIllIlIIIIIlIllll ,version =IlIlIIIIllIIllllI .get (_IIIlIIIllIIIIllII ,_IllllIlIIIIllllll ));IIIlllIIIIIIllIII .load_state_dict (IlIlIIIIllIIllllI [_IlIIIlIlIIlIlIIIl ],strict =_IIllIlIIIIIlIllll );IIlIlIlIlllllIlll =[IlIlIlIlIlllllIlI ,'phone_lengths',IIllIIllIIllllIlI ,IIllllIIIllIlllIl ,'ds',IlIlIIllIllIlIlIl ];IIlllIllIlIlllIIl =['audio'];torch .onnx .export (IIIlllIIIIIIllIII ,(IIllllIlllIlIlIII .to (IIIIIlIlIIIllIllI ),IIlllIlllIlllIlll .to (IIIIIlIlIIIllIllI ),IIIlIlllllllIlllI .to (IIIIIlIlIIIllIllI ),IlIlIllIIlIlIlIlI .to (IIIIIlIlIIIllIllI ),IlIlIIIlIlIlllIII .to (IIIIIlIlIIIllIllI ),IllllIIlIIIlIlIll .to (IIIIIlIlIIIllIllI )),IllIIlllIlIlIIIII ,dynamic_axes ={IlIlIlIlIlllllIlI :[1 ],IIllIIllIIllllIlI :[1 ],IIllllIIIllIlllIl :[1 ],IlIlIIllIllIlIlIl :[2 ]},do_constant_folding =_IIllIlIIIIIlIllll ,opset_version =13 ,verbose =_IIllIlIIIIIlIllll ,input_names =IIlIlIlIlllllIlll ,output_names =IIlllIllIlIlllIIl );return 'Finished'
import re as regex ,scipy .io .wavfile as wavfile 
IlIllIIlllIIlIIlI ='HOME'
def IlIlllIIIllIIllll (IlllIllIIlIIIllll ):IlIIIIIllllIlllIl ='(?:(?<=\\s)|^)"(.*?)"(?=\\s|$)|(\\S+)';IlIlllIIIIIIlIIII =regex .findall (IlIIIIIllllIlllIl ,IlllIllIIlIIIllll );IlIlllIIIIIIlIIII =[IllIlIlIllllllIll [0 ]if IllIlIlIllllllIll [0 ]else IllIlIlIllllllIll [1 ]for IllIlIlIllllllIll in IlIlllIIIIIIlIIII ];return IlIlllIIIIIIlIIII 
def IlIlIllIlllIIlIII (IIllllIlllIlllIII ):
	for _IlIIlllIIllllIIIl in IIllllIlllIlllIII :0 
def IllllllIllllllIlI (IIlIlllllllIllIII ):
	IlIIIllIllIIlIIll ='audio-outputs';IIlIlllllllIllIII =IlIlllIIIllIIllll (IIlIlllllllIllIII );IlIlIlIIIIIlIllll =IIlIlllllllIllIII [0 ];IIlIlIIIlIIIlllIl =IIlIlllllllIllIII [1 ];IIlllIlIIllIlllII =IIlIlllllllIllIII [2 ];IlIlIllIlIlIIlIll =IIlIlllllllIllIII [3 ];IIllIllIlIIlIIlII =_IIllIlIlIIlllIIlI ;IIIllllIlIIIIIlll =int (IIlIlllllllIllIII [4 ]);IIIIIlIllllIIlIII =float (IIlIlllllllIllIII [5 ]);IIIllIlIIlIlIIlII =IIlIlllllllIllIII [6 ];IlIlIIlIlIIlIlIlI =int (IIlIlllllllIllIII [7 ]);IIllIIlIIlllIIIlI =int (IIlIlllllllIllIII [8 ]);IIIlllIIIIIlIIIII =int (IIlIlllllllIllIII [9 ]);IIlllIIIIlIIIIIlI =float (IIlIlllllllIllIII [10 ]);IIlIllIIlIIIlIlll =float (IIlIlllllllIllIII [11 ]);IIllIllIllIlIIIlI =float (IIlIlllllllIllIII [12 ]);IIlllIlIIllIIlIlI =.5 
	if IIlIlllllllIllIII [14 ]=='False'or IIlIlllllllIllIII [14 ]=='false':IllllIlllIIlIllll =_IIllIlIIIIIlIllll ;IlIIIlIlIlllIlllI =.0 ;IIIlIlllIIIllIIlI =.0 ;CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IllllIlllIIlIllll ,IlIIIlIlIlllIlllI ,IIIlIlllIIIllIIlI )
	else :IllllIlllIIlIllll =_IIIlIlIIIlllIllII ;IlIIIlIlIlllIlllI =float (IIlIlllllllIllIII [15 ]);IIIlIlllIIIllIIlI =float (IIlIlllllllIllIII [16 ]);CSVutil (_IllIIIIlIllIllIlI ,_IlIlllIlIllIIlIll ,_IllIIllllIIlIIIll ,IllllIlllIIlIllll ,IlIIIlIlIlllIlllI ,IIIlIlllIIIllIIlI )
	print ('Mangio-RVC-Fork Infer-CLI: Starting the inference...');IIlIllllIIlIIIIlI =IlIlllIllIIIlIIll (IlIlIlIIIIIlIllll ,IIllIllIllIlIIIlI ,IIlllIlIIllIIlIlI );print (IIlIllllIIlIIIIlI );print ('Mangio-RVC-Fork Infer-CLI: Performing inference...');IlIIllIIlllIlIIll =IIIllIllIIIIIlIII (IIIllllIlIIIIIlll ,IIlIlIIIlIIIlllIl ,IIlIlIIIlIIIlllIl ,IIIIIlIllllIIlIII ,IIllIllIlIIlIIlII ,IIIllIlIIlIlIIlII ,IlIlIllIlIlIIlIll ,IlIlIllIlIlIIlIll ,IIlIllIIlIIIlIlll ,IIllIIlIIlllIIIlI ,IIIlllIIIIIlIIIII ,IIlllIIIIlIIIIIlI ,IIllIllIllIlIIIlI ,IlIlIIlIlIIlIlIlI )
	if 'Success.'in IlIIllIIlllIlIIll [0 ]:print ('Mangio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s...'%(IlIIIllIllIIlIIll ,IIlllIlIIllIlllII ));wavfile .write (_IIlIIIIIIIIllllII %(IlIIIllIllIIlIIll ,IIlllIlIIllIlllII ),IlIIllIIlllIlIIll [1 ][0 ],IlIIllIIlllIlIIll [1 ][1 ]);print ('Mangio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s'%(IlIIIllIllIIlIIll ,IIlllIlIIllIlllII ))
	else :print ("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ");print (IlIIllIIlllIlIIll [0 ])
def IlIIlIIlIlIllIIIl (IIIlIIllllllIIlII ):IIIlIIllllllIIlII =IlIlllIIIllIIllll (IIIlIIllllllIIlII );IIlIlllIllIIIIllI =IIIlIIllllllIIlII [0 ];IIllIIIllIllIlllI =IIIlIIllllllIIlII [1 ];IIIllllIIIlllIlll =IIIlIIllllllIIlII [2 ];IlIllIlIlllIIlIlI =int (IIIlIIllllllIIlII [3 ]);print ('Mangio-RVC-Fork Pre-process: Starting...');IIllIIIIIlIIlIIII =IIlIlIllIllIlIIII (IIllIIIllIllIlllI ,IIlIlllIllIIIIllI ,IIIllllIIIlllIlll ,IlIllIlIlllIIlIlI );IlIlIllIlllIIlIII (IIllIIIIIlIIlIIII );print ('Mangio-RVC-Fork Pre-process: Finished')
def IIllIIllIIlIlIIlI (IIIIlllIlIIIllIll ):IIIIlllIlIIIllIll =IlIlllIIIllIIllll (IIIIlllIlIIIllIll );IlllIIIIllllIIIIl =IIIIlllIlIIIllIll [0 ];IIIlllIIlIIlllllI =IIIIlllIlIIIllIll [1 ];IlllllIIlllIllIII =int (IIIIlllIlIIIllIll [2 ]);IIlllIIIIIIlllIIl =_IIIlIlIIIlllIllII if int (IIIIlllIlIIIllIll [3 ])==1 else _IIllIlIIIIIlIllll ;IllllIlIllIIIllII =IIIIlllIlIIIllIll [4 ];IllIlIllIIIIIllll =int (IIIIlllIlIIIllIll [5 ]);IlIllIlIllllllllI =IIIIlllIlIIIllIll [6 ];print ('Mangio-RVC-CLI: Extract Feature Has Pitch: '+str (IIlllIIIIIIlllIIl ));print ('Mangio-RVC-CLI: Extract Feature Version: '+str (IlIllIlIllllllllI ));print ('Mangio-RVC-Fork Feature Extraction: Starting...');IlllIllIIlllIllII =IIIllIllllllllIlI (IIIlllIIlIIlllllI ,IlllllIIlllIllIII ,IllllIlIllIIIllII ,IIlllIIIIIIlllIIl ,IlllIIIIllllIIIIl ,IlIllIlIllllllllI ,IllIlIllIIIIIllll );IlIlIllIlllIIlIII (IlllIllIIlllIllII );print ('Mangio-RVC-Fork Feature Extraction: Finished')
def IIlIlIlIIlIllIIIl (IIllIIlIlIlIlIllI ):IIllIIlIlIlIlIllI =IlIlllIIIllIIllll (IIllIIlIlIlIlIllI );IIIlIIllIlIIIlIII =IIllIIlIlIlIlIllI [0 ];IIIlllllIlIllIIlI =IIllIIlIlIlIlIllI [1 ];IIlIIllIllIIlIIIl =_IIIlIlIIIlllIllII if int (IIllIIlIlIlIlIllI [2 ])==1 else _IIllIlIIIIIlIllll ;IllllIIllIIIlIlIl =int (IIllIIlIlIlIlIllI [3 ]);IIIIIlllIlllllIlI =int (IIllIIlIlIlIlIllI [4 ]);IIlllIlllIlIlIIll =int (IIllIIlIlIlIlIllI [5 ]);IIIIlIIIIIllIIIIl =int (IIllIIlIlIlIlIllI [6 ]);IIllIllIlllIllIll =IIllIIlIlIlIlIllI [7 ];IIlllIIlIllllllII =_IIIlIlIIIlllIllII if int (IIllIIlIlIlIlIllI [8 ])==1 else _IIllIlIIIIIlIllll ;IIIIlIIIIlllllllI =_IIIlIlIIIlllIllII if int (IIllIIlIlIlIlIllI [9 ])==1 else _IIllIlIIIIIlIllll ;IllIIllllllIIIllI =_IIIlIlIIIlllIllII if int (IIllIIlIlIlIlIllI [10 ])==1 else _IIllIlIIIIIlIllll ;IIIIlllllllIIlIlI =IIllIIlIlIlIlIllI [11 ];IllIllllIIlllIIll ='/kaggle/input/ax-rmf/pretrained/'if IIIIlllllllIIlIlI ==_IllllIlIIIIllllll else '/kaggle/input/ax-rmf/pretrained_v2/';IlllIIIlIIIIlIlIl ='%sf0G%s.pth'%(IllIllllIIlllIIll ,IIIlllllIlIllIIlI );IIlIIIllllllIIllI ='%sf0D%s.pth'%(IllIllllIIlllIIll ,IIIlllllIlIllIIlI );print ('Mangio-RVC-Fork Train-CLI: Training...');IllIIlIllIIllIIII (IIIlIIllIlIIIlIII ,IIIlllllIlIllIIlI ,IIlIIllIllIIlIIIl ,IllllIIllIIIlIlIl ,IIIIIlllIlllllIlI ,IIlllIlllIlIlIIll ,IIIIlIIIIIllIIIIl ,IIlllIIlIllllllII ,IlllIIIlIIIIlIlIl ,IIlIIIllllllIIllI ,IIllIllIlllIllIll ,IIIIlIIIIlllllllI ,IllIIllllllIIIllI ,IIIIlllllllIIlIlI )
def IlllIlIllIIlIlIll (IlIIlIlIIllllllII ):IlIIlIlIIllllllII =IlIlllIIIllIIllll (IlIIlIlIIllllllII );IllIIIlIlllIIllIl =IlIIlIlIIllllllII [0 ];IlIllllllIIIIIIlI =IlIIlIlIIllllllII [1 ];print ('Mangio-RVC-Fork Train Feature Index-CLI: Training... Please wait');IIlIIIIIlIIlIIIIl =IIlIlIIIlllIlIllI (IllIIIlIlllIIllIl ,IlIllllllIIIIIIlI );IlIlIllIlllIIlIII (IIlIIIIIlIIlIIIIl );print ('Mangio-RVC-Fork Train Feature Index-CLI: Done!')
def IIllllIlIllIIlIII (IlIlIlIIIIlIIlIll ):
	IlIlIlIIIIlIIlIll =IlIlllIIIllIIllll (IlIlIlIIIIlIIlIll );IlIIlIIIIIlIIIlII =IlIlIlIIIIlIIlIll [0 ];IlllIIIlIIlIIllIl =IlIlIlIIIIlIIlIll [1 ];IIlIIlllllllIIIIl =IlIlIlIIIIlIIlIll [2 ];IlllIllIIllllIlll =IlIlIlIIIIlIIlIll [3 ];IlIIllIlIlIllllll =IlIlIlIIIIlIIlIll [4 ];IIlllllIIllIllIlI =IlIlIlIIIIlIIlIll [5 ];IIIIIIlllllIIIIIl =extract_small_model (IlIIlIIIIIlIIIlII ,IlllIIIlIIlIIllIl ,IIlIIlllllllIIIIl ,IlllIllIIllllIlll ,IlIIllIlIlIllllll ,IIlllllIIllIllIlI )
	if IIIIIIlllllIIIIIl =='Success.':print ('Mangio-RVC-Fork Extract Small Model: Success!')
	else :print (str (IIIIIIlllllIIIIIl ));print ('Mangio-RVC-Fork Extract Small Model: Failed!')
def IIIIlIlIIllIIlIll (IIllIIIlIIIlllIll ,IllIIIIIIllIIllII ,IllllIIllIIllIIll ):
	if str (IIllIIIlIIIlllIll )!='':
		with open (str (IIllIIIlIIIlllIll ),_IIlIlIlIIIlIIIIII )as IllllIIIllIIIllII :IIIIlIllIIIlllllI =IllllIIIllIIIllII .readlines ();IllIIIIIIllIIllII ,IllllIIllIIllIIll =IIIIlIllIIIlllllI [0 ].split (_IIlIIlIIllIlIIllI )[0 ],IIIIlIllIIIlllllI [1 ];IlIlIIlIIIlIIlllI (IllIIIIIIllIIllII ,IllllIIllIIllIIll )
	else :0 
	return {_IIIllIIIlIllIlIIl :IllIIIIIIllIIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIllIIIlIllIlIIl :IllllIIllIIllIIll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IllllIIIIlIIlIIII ():
	if IlIllIIlllIIlIIlI =='HOME':print ('\n    go home            : Takes you back to home with a navigation list.\n    go infer           : Takes you to inference command execution.\n    go pre-process     : Takes you to training step.1) pre-process command execution.\n    go extract-feature : Takes you to training step.2) extract-feature command execution.\n    go train           : Takes you to training step.3) being or continue training command execution.\n    go train-feature   : Takes you to the train feature index command execution.\n    go extract-model   : Takes you to the extract small model command execution.')
	elif IlIllIIlllIIlIIlI ==_IIlIIlIllIlIIllll :print ("\n    arg 1) model name with .pth in ./weights: mi-test.pth\n    arg 2) source audio path: myFolder\\MySource.wav\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav\n    arg 4) feature index file path: logs/mi-test/added_IVF3l42_Flat_nprobe_1.index\n    arg 5) speaker id: 0\n    arg 6) transposition: 0\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)\n    arg 8) crepe hop length: 160\n    arg 9) harvest median filter radius: 3 (0-7)\n    arg 10) post resample rate: 0\n    arg 11) mix volume envelope: 1\n    arg 12) feature index ratio: 0.78 (0-1)\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2")
	elif IlIllIIlllIIlIIlI ==_IlIlllllIllIIIIlI :print ('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Number of CPU threads to use: 8 \n\nExample: mi-test mydataset 40k 24')
	elif IlIllIIlllIIlIIlI ==_IllllllIlIIlIIlll :print ('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 3) Number of CPU threads to use: 8\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)\n    arg 6) Crepe hop length: 128\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n\nExample: mi-test 0 24 1 harvest 128 v2')
	elif IlIllIIlllIIlIIlI ==_IIIIIlIllIllIIIll :print ('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Sample rate: 40k (32k, 40k, 48k)\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 4) speaker id: 0\n    arg 5) Save epoch iteration: 50\n    arg 6) Total epochs: 10000\n    arg 7) Batch size: 8\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2')
	elif IlIllIIlllIIlIIlI ==_IIllIIlIIllIIlIIl :print ('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test v2')
	elif IlIllIIlllIIlIIlI ==_IIlIllIIlllllIlIl :print ('\n    arg 1) Model Path: logs/mi-test/G_168000.pth\n    arg 2) Model save name: MyModel\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) Model information: "My Model"\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2')
def IlllIlIllIllllIII (IIlIIlIlllIIIlIlI ):global IlIllIIlllIIlIIlI ;IlIllIIlllIIlIIlI =IIlIIlIlllIIIlIlI ;return 0 
def IIIIlIIIIlIlIIIIl (IllIIIlllIIllIlII ):
	if IllIIIlllIIllIlII =='go home':return IlllIlIllIllllIII ('HOME')
	elif IllIIIlllIIllIlII =='go infer':return IlllIlIllIllllIII (_IIlIIlIllIlIIllll )
	elif IllIIIlllIIllIlII =='go pre-process':return IlllIlIllIllllIII (_IlIlllllIllIIIIlI )
	elif IllIIIlllIIllIlII =='go extract-feature':return IlllIlIllIllllIII (_IllllllIlIIlIIlll )
	elif IllIIIlllIIllIlII =='go train':return IlllIlIllIllllIII (_IIIIIlIllIllIIIll )
	elif IllIIIlllIIllIlII =='go train-feature':return IlllIlIllIllllIII (_IIllIIlIIllIIlIIl )
	elif IllIIIlllIIllIlII =='go extract-model':return IlllIlIllIllllIII (_IIlIllIIlllllIlIl )
	elif IllIIIlllIIllIlII [:3 ]=='go ':print ("page '%s' does not exist!"%IllIIIlllIIllIlII [3 :]);return 0 
	if IlIllIIlllIIlIIlI ==_IIlIIlIllIlIIllll :IllllllIllllllIlI (IllIIIlllIIllIlII )
	elif IlIllIIlllIIlIIlI ==_IlIlllllIllIIIIlI :IlIIlIIlIlIllIIIl (IllIIIlllIIllIlII )
	elif IlIllIIlllIIlIIlI ==_IllllllIlIIlIIlll :IIllIIllIIlIlIIlI (IllIIIlllIIllIlII )
	elif IlIllIIlllIIlIIlI ==_IIIIIlIllIllIIIll :IIlIlIlIIlIllIIIl (IllIIIlllIIllIlII )
	elif IlIllIIlllIIlIIlI ==_IIllIIlIIllIIlIIl :IlllIlIllIIlIlIll (IllIIIlllIIllIlII )
	elif IlIllIIlllIIlIIlI ==_IIlIllIIlllllIlIl :IIllllIlIllIIlIII (IllIIIlllIIllIlII )
def IlIIlIlIlllIlIIII ():
	while _IIIlIlIIIlllIllII :
		print ("\nYou are currently in '%s':"%IlIllIIlllIIlIIlI );IllllIIIIlIIlIIII ();IIlIlIlllIlIlllII =input ('%s: '%IlIllIIlllIIlIIlI )
		try :IIIIlIIIIlIlIIIIl (IIlIlIlllIlIlllII )
		except :print (traceback .format_exc ())
if IllIIllIIIIlIIlIl .is_cli :print ('\n\nMangio-RVC-Fork v2 CLI App!\n');print ('Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n');IlIIlIlIlllIlIIII ()
def IlIlIIIIllIIlllll ():
	IIllIIlIllllllIlI =_IIllIlIlIIlllIIlI 
	with open (_IIIIIlIIllIIllllI ,_IIlIlIlIIIlIIIIII )as IlllllIIllIIlIIlI :IIllIIlIllllllIlI =json .load (IlllllIIllIIlIIlI )
	IIlIIIIllIllIIIII =[]
	for IIllIIllIlllIllII in IIllIIlIllllllIlI [_IllIIllIIIIlllIIl ]:IIlIIIIllIllIIIII .append (IIllIIllIlllIllII ['name'])
	return IIlIIIIllIllIIIII 
def IllIIllIIlIIIllll (IlllIIIlIIllllIlI ):return {_IIIIllIlIlllllIIl :IlllIIIlIIllllIlI ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IlIIlIIlllIIIIlll (IIIlIIIIllIllIIIl ):
	IlIlIlIllIIlIlIlI =_IIllIlIIIIIlIllll ;IIIllIIlIlllllIII =IIIlIIIIllIllIIIl .split (_IIIIlIIIIlIlIllIl )[0 ].split ('_')[0 ];IIlIIlIlIIlIIllll =_IlIIlllIIIllIIIll +IIIllIIlIlllllIII 
	if os .path .exists (IIlIIlIlIIlIIllll ):
		for IlllllIlllIIIIIlI in os .listdir (IIlIIlIlIIlIIllll .replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )):
			if IlllllIlllIIIIIlI .endswith (_IIIIIIllIIlIlIIII ):
				for IllllIlllIIIIIIII in range (len (IIlIlIlIIIIIIllIl )):
					if IIlIlIlIIIIIIllIl [IllllIlllIIIIIIII ]==os .path .join (_IlIIlllIIIllIIIll +IIIllIIlIlllllIII ,IlllllIlllIIIIIlI ).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI ):break 
					elif IIlIlIlIIIIIIllIl [IllllIlllIIIIIIII ]==os .path .join (_IlIIlllIIIllIIIll +IIIllIIlIlllllIII .lower (),IlllllIlllIIIIIlI ).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI ):IIlIIlIlIIlIIllll =_IlIIlllIIIllIIIll +IIIllIIlIlllllIII .lower ();break 
				IIlIIIIIlIIIIlIIl =os .path .join (IIlIIlIlIIlIIllll .replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI ),IlllllIlllIIIIIlI .replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI );return IIlIIIIIlIIIIlIIl ,IIlIIIIIlIIIIlIIl 
	else :return '',''
def IllIIIIIlIIllIllI (IllIIIlllllIlIIII ):
	if int (IllIIIlllllIlIIII )==1 :
		CSVutil (_IllIIllllllllIlIl ,_IlIlllIlIllIIlIll ,'stop','True')
		try :os .kill (IlllIllIIllllIIIl ,signal .SIGTERM )
		except Exception as IIlIIIIIIIllIllll :print (f"Couldn't click due to {IIlIIIIIIIllIllll}");pass 
	else :0 
	return {_IIIIllIlIlllllIIl :_IIllIlIIIIIlIllll ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI },{_IIIIllIlIlllllIIl :_IIIlIlIIIlllIllII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
def IllIIIlIllIIIIIII (IlIIlIIIllIlIlllI ):IIIIIlIlIllIIlIII =_IIIlIlIIIlllIllII if IlIIlIIIllIlIlllI ==_IIlIlIIlIIllIllII or IlIIlIIIllIlIlllI ==_IIlllllllIIllIllI else _IIllIlIIIIIlIllll ;return {_IIIIllIlIlllllIIl :IIIIIlIlIllIIlIII ,_IlllllllIIIIlIlII :_IlIIlIIlllIlIlIlI }
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC 🍏')as IlllIlIIlIIIIIIlI :
	gr .HTML ('<h1> AX RVC ( Mangio-RVC-Fork ) 🍏 </h1>')
	with gr .Tabs ():
		with gr .TabItem (IlIlllIIllllIlIII ('模型推理')):
			with gr .Row ():IlIIIIIlllIlIlIIl =gr .Dropdown (label =IlIlllIIllllIlIII ('推理音色'),choices =sorted (IlIlIIllIlllllIll ),value ='');IlIIIllIlIlllllII =gr .Button (IlIlllIIllllIlIII ('Refresh voice list, index path and audio files'),variant =_IlllIllIllIIlIIll );IlllIlllIlIIllIlI =gr .Button (IlIlllIIllllIlIII ('卸载音色省显存'),variant =_IlllIllIllIIlIIll );IlIIllIIlIlIIllII =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IlIlllIIllllIlIII ('请选择说话人id'),value =0 ,visible =_IIllIlIIIIIlIllll ,interactive =_IIIlIlIIIlllIllII );IlllIlllIlIIllIlI .click (fn =IlIlllllIIllllIlI ,inputs =[],outputs =[IlIIIIIlllIlIlIIl ])
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IIlIlIIIlIIllIlIl =gr .Number (label =IlIlllIIllllIlIII (_IIIlIIlIlllllllII ),value =0 );IllIIlIlIIIIIllIl =gr .Textbox (label =IlIlllIIllllIlIII ("Add audio's name to the path to the audio file to be processed (default is the correct format example) Remove the path to use an audio from the dropdown list:"),value =os .path .abspath (os .getcwd ()).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )+_IIIIIIlllIIlIIllI +'audio.wav');IlllIIIIlIIllllII =gr .Dropdown (label =IlIlllIIllllIlIII ('Auto detect audio path and select from the dropdown:'),choices =sorted (IllIIIlllIlllIlIl ),value ='',interactive =_IIIlIlIIIlllIllII );IlllIIIIlIIllllII .change (fn =lambda :'',inputs =[],outputs =[IllIIlIlIIIIIllIl ]);IIlIIlIIllIlllIll =gr .Radio (label =IlIlllIIllllIlIII (_IIIIIlIIlllIIIIII ),choices =['pm',_IIlIlllIlIIIlllII ,'dio',_IIlllIIIIlIlllIII ,'crepe-tiny',_IIlIlIIlIIllIllII ,_IIlllllllIIllIllI ,_IIlllllIIllIIllIl ],value =_IIlllllIIllIIllIl ,interactive =_IIIlIlIIIlllIllII );IlllIIllIlIIlIlIl =gr .Slider (minimum =1 ,maximum =512 ,step =1 ,label =IlIlllIIllllIlIII (_IIllIlIIlIIlIllll ),value =120 ,interactive =_IIIlIlIIIlllIllII ,visible =_IIllIlIIIIIlIllll );IIlIIlIIllIlllIll .change (fn =IllIIIlIllIIIIIII ,inputs =[IIlIIlIIllIlllIll ],outputs =[IlllIIllIlIIlIlIl ]);IIllIIIIIlIlllIIl =gr .Slider (minimum =0 ,maximum =7 ,label =IlIlllIIllllIlIII (_IIIIIllIlllIllIlI ),value =3 ,step =1 ,interactive =_IIIlIlIIIlllIllII )
					with gr .Column ():IIlIlIllIIIlIIllI =gr .Textbox (label =IlIlllIIllllIlIII (_IlIIIllIIllIlIIIl ),value ='',interactive =_IIIlIlIIIlllIllII );IlIllIIIIlIlIlIII =gr .Dropdown (label ="3. Path to your added.index file (if it didn't automatically find it.)",choices =IllIllIlIllIIllII (),value =IIlIIlllllIIlIllI (),interactive =_IIIlIlIIIlllIllII ,allow_custom_value =_IIIlIlIIIlllIllII );IlIIIllIlIlllllII .click (fn =IlIIIlIIIIlIlIIlI ,inputs =[],outputs =[IlIIIIIlllIlIlIIl ,IlIllIIIIlIlIlIII ,IlllIIIIlIIllllII ]);IIIIIIIIllIIllIII =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIIllllIlIII ('检索特征占比'),value =.75 ,interactive =_IIIlIlIIIlllIllII )
					with gr .Column ():IlIllIllIllIllIII =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIlllIIllllIlIII (_IIIllIIIIlIIlIllI ),value =0 ,step =1 ,interactive =_IIIlIlIIIlllIllII );IIlIIlIIIIIIIlIlI =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIIllllIlIII (_IIIlIlllIIIIlllIl ),value =.25 ,interactive =_IIIlIlIIIlllIllII );IIlllIlIIlIllIllI =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIlllIIllllIlIII (_IllIIlIllIIIIIlll ),value =.33 ,step =.01 ,interactive =_IIIlIlIIIlllIllII );IIlIllIllIlIlIlII =gr .Checkbox (value =bool (IIIIllIlIIllllIIl ),label ='[EXPERIMENTAL] Formant shift inference audio',info ='Used for male to female and vice-versa conversions',interactive =_IIIlIlIIIlllIllII ,visible =_IIIlIlIIIlllIllII );IIIIIIllIlIIIIllI =gr .Dropdown (value ='',choices =IlllIlIlIllllIIIl (),label ='browse presets for formanting',visible =bool (IIIIllIlIIllllIIl ));IIlIlllIllIlIIlIl =gr .Button (value ='🔄',visible =bool (IIIIllIlIIllllIIl ),variant =_IlllIllIllIIlIIll );IIllIIIllIlIlllIl =gr .Slider (value =IIllIIIIllIlIlllI ,info =_IlIIIllIllIIIIlll ,label ='Quefrency for formant shifting',minimum =.0 ,maximum =16. ,step =.1 ,visible =bool (IIIIllIlIIllllIIl ),interactive =_IIIlIlIIIlllIllII );IllIIIIIllllIIIII =gr .Slider (value =IIIIIIIlIIIIIlllI ,info =_IlIIIllIllIIIIlll ,label ='Timbre for formant shifting',minimum =.0 ,maximum =16. ,step =.1 ,visible =bool (IIIIllIlIIllllIIl ),interactive =_IIIlIlIIIlllIllII );IIIIIIllIlIIIIllI .change (fn =IIIIlIlIIllIIlIll ,inputs =[IIIIIIllIlIIIIllI ,IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ],outputs =[IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ]);IIIIIIlIIlIIlIlII =gr .Button ('Apply',variant =_IlllIllIllIIlIIll ,visible =bool (IIIIllIlIIllllIIl ));IIlIllIllIlIlIlII .change (fn =IlllllIllIIIlIlll ,inputs =[IIlIllIllIlIlIlII ,IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ,IIIIIIlIIlIIlIlII ,IIIIIIllIlIIIIllI ,IIlIlllIllIlIIlIl ],outputs =[IIlIllIllIlIlIlII ,IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ,IIIIIIlIIlIIlIlII ,IIIIIIllIlIIIIllI ,IIlIlllIllIlIIlIl ]);IIIIIIlIIlIIlIlII .click (fn =IlIlIIlIIIlIIlllI ,inputs =[IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ],outputs =[IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ]);IIlIlllIllIlIIlIl .click (fn =IIIIIlIIlIllllIIl ,inputs =[IIIIIIllIlIIIIllI ,IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ],outputs =[IIIIIIllIlIIIIllI ,IIllIIIllIlIlllIl ,IllIIIIIllllIIIII ])
					IIIllIllIllIIIIIl =gr .File (label =IlIlllIIllllIlIII ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IIlIlIllIlIIIlIII =gr .Button (IlIlllIIllllIlIII ('转换'),variant =_IlllIllIllIIlIIll )
					with gr .Row ():IlllIIIlIIIlIlIII =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ));IIlIIllllllIIIIll =gr .Audio (label =IlIlllIIllllIlIII ('输出音频(右下角三个点,点了可以下载)'))
					IIlIlIllIlIIIlIII .click (IIIllIllIIIIIlIII ,[IlIIllIIlIlIIllII ,IllIIlIlIIIIIllIl ,IlllIIIIlIIllllII ,IIlIlIIIlIIllIlIl ,IIIllIllIllIIIIIl ,IIlIIlIIllIlllIll ,IIlIlIllIIIlIIllI ,IlIllIIIIlIlIlIII ,IIIIIIIIllIIllIII ,IIllIIIIIlIlllIIl ,IlIllIllIllIllIII ,IIlIIlIIIIIIIlIlI ,IIlllIlIIlIllIllI ,IlllIIllIlIIlIlIl ],[IlllIIIlIIIlIlIII ,IIlIIllllllIIIIll ])
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IIlIIIlIIIIlIIlIl =gr .Number (label =IlIlllIIllllIlIII (_IIIlIIlIlllllllII ),value =0 );IlIIllIIlllIIlIII =gr .Textbox (label =IlIlllIIllllIlIII ('指定输出文件夹'),value ='opt');IllllIIIIllIIllII =gr .Radio (label =IlIlllIIllllIlIII (_IIIIIlIIlllIIIIII ),choices =['pm',_IIlIlllIlIIIlllII ,_IIlllIIIIlIlllIII ,_IIlllllIIllIIllIl ],value =_IIlllllIIllIIllIl ,interactive =_IIIlIlIIIlllIllII );IlllllllIIIIlIlIl =gr .Slider (minimum =0 ,maximum =7 ,label =IlIlllIIllllIlIII (_IIIIIllIlllIllIlI ),value =3 ,step =1 ,interactive =_IIIlIlIIIlllIllII )
					with gr .Column ():IIlIIlIllIIlIllll =gr .Textbox (label =IlIlllIIllllIlIII (_IlIIIllIIllIlIIIl ),value ='',interactive =_IIIlIlIIIlllIllII );IllIIIlllIIIlIIIl =gr .Dropdown (label =IlIlllIIllllIlIII ('自动检测index路径,下拉式选择(dropdown)'),choices =IllIllIlIllIIllII (),value =IIlIIlllllIIlIllI (),interactive =_IIIlIlIIIlllIllII );IlIIIIIlllIlIlIIl .select (fn =IlIIlIIlllIIIIlll ,inputs =[IlIIIIIlllIlIlIIl ],outputs =[IlIllIIIIlIlIlIII ,IllIIIlllIIIlIIIl ]);IlIIIllIlIlllllII .click (fn =lambda :IlIIIlIIIIlIlIIlI ()[1 ],inputs =[],outputs =IllIIIlllIIIlIIIl );IIIlIlIIllIIllIlI =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIIllllIlIII ('检索特征占比'),value =1 ,interactive =_IIIlIlIIIlllIllII )
					with gr .Column ():IllllIIllIlllllII =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIlllIIllllIlIII (_IIIllIIIIlIIlIllI ),value =0 ,step =1 ,interactive =_IIIlIlIIIlllIllII );IIIllllIIlIllIlll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIIllllIlIII (_IIIlIlllIIIIlllIl ),value =1 ,interactive =_IIIlIlIIIlllIllII );IIIIlIIlIlIIlIlll =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIlllIIllllIlIII (_IllIIlIllIIIIIlll ),value =.33 ,step =.01 ,interactive =_IIIlIlIIIlllIllII )
					with gr .Column ():IlllllIlIllIIIlll =gr .Textbox (label =IlIlllIIllllIlIII ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value =os .path .abspath (os .getcwd ()).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )+_IIIIIIlllIIlIIllI );IIIlllIlIIllllIll =gr .File (file_count ='multiple',label =IlIlllIIllllIlIII (_IIIIlIIIllIIIlllI ))
					with gr .Row ():IlIIlllIlllIllIII =gr .Radio (label =IlIlllIIllllIlIII ('导出文件格式'),choices =[_IlIllIlllIIlIIlll ,_IIlIIlIIllllIIIIl ,_IllIIIlIIlIIllllI ,'m4a'],value =_IIlIIlIIllllIIIIl ,interactive =_IIIlIlIIIlllIllII );IlIIIlllIIllllIII =gr .Button (IlIlllIIllllIlIII ('转换'),variant =_IlllIllIllIIlIIll );IlIIllIIIllIIIIll =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ))
					IlIIIlllIIllllIII .click (IlIlIIllIIlllllII ,[IlIIllIIlIlIIllII ,IlllllIlIllIIIlll ,IlIIllIIlllIIlIII ,IIIlllIlIIllllIll ,IIlIIIlIIIIlIIlIl ,IllllIIIIllIIllII ,IIlIIlIllIIlIllll ,IllIIIlllIIIlIIIl ,IIIlIlIIllIIllIlI ,IlllllllIIIIlIlIl ,IllllIIllIlllllII ,IIIllllIIlIllIlll ,IIIIlIIlIlIIlIlll ,IlIIlllIlllIllIII ,IlllIIllIlIIlIlIl ],[IlIIllIIIllIIIIll ])
			IlIIIIIlllIlIlIIl .change (fn =IlIlllIllIIIlIIll ,inputs =[IlIIIIIlllIlIlIIl ,IIlllIlIIlIllIllI ,IIIIlIIlIlIIlIlll ],outputs =[IlIIllIIlIlIIllII ,IIlllIlIIlIllIllI ,IIIIlIIlIlIIlIlll ])
		with gr .TabItem (IlIlllIIllllIlIII ('伴奏人声分离&去混响&去回声')):
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IIIIlllIlllIllIIl =gr .Textbox (label =IlIlllIIllllIlIII ('输入待处理音频文件夹路径'),value =os .getcwd ().replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )+_IIIIIIlllIIlIIllI );IlIllIIIIIlIIlIIl =gr .File (file_count ='multiple',label =IlIlllIIllllIlIII (_IIIIlIIIllIIIlllI ))
					with gr .Column ():IlllIlllllIlllIII =gr .Dropdown (label =IlIlllIIllllIlIII ('模型'),choices =IllIlllIlllIIlIIl );IIlllIIlIIlIIIllI =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IIIlIlIIIlllIllII ,visible =_IIllIlIIIIIlIllll );IllIllllIlIIlIllI =gr .Textbox (label =IlIlllIIllllIlIII ('指定输出主人声文件夹'),value ='opt');IlIlIIlIllIIIlIlI =gr .Textbox (label =IlIlllIIllllIlIII ('指定输出非主人声文件夹'),value ='opt');IIIIIlIIllIIlIlll =gr .Radio (label =IlIlllIIllllIlIII ('导出文件格式'),choices =[_IlIllIlllIIlIIlll ,_IIlIIlIIllllIIIIl ,_IllIIIlIIlIIllllI ,'m4a'],value =_IIlIIlIIllllIIIIl ,interactive =_IIIlIlIIIlllIllII )
					IIIlIIIIllIIllllI =gr .Button (IlIlllIIllllIlIII ('转换'),variant =_IlllIllIllIIlIIll );IllIIIlIIIlIllIII =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ));IIIlIIIIllIIllllI .click (IIIIIIIIlIlIlllll ,[IlllIlllllIlllIII ,IIIIlllIlllIllIIl ,IllIllllIlIIlIllI ,IlIllIIIIIlIIlIIl ,IlIlIIlIllIIIlIlI ,IIlllIIlIIlIIIllI ,IIIIIlIIllIIlIlll ],[IllIIIlIIIlIllIII ])
		with gr .TabItem (IlIlllIIllllIlIII ('训练')):
			gr .Markdown (value =IlIlllIIllllIlIII ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIIIlIlllIIIllIll =gr .Textbox (label =IlIlllIIllllIlIII ('输入实验名'),value ='mi-test');IlIIlIlllllIIIlII =gr .Radio (label =IlIlllIIllllIlIII (_IIIlIIlIllIlIIlll ),choices =[_IllIlIIllIllllIll ],value =_IllIlIIllIllllIll ,interactive =_IIIlIlIIIlllIllII );IllIIlIlIllIllIll =gr .Checkbox (label =_IIIllIIlllllIlIII ,value =_IIIlIlIIIlllIllII ,interactive =_IIIlIlIIIlllIllII );IlIIlIIIIlIIllIIl =gr .Radio (label =IlIlllIIllllIlIII ('版本'),choices =[_IIlIllIIlIIIlllIl ],value =_IIlIllIIlIIIlllIl ,interactive =_IIIlIlIIIlllIllII ,visible =_IIIlIlIIIlllIllII );IIlIllIllIlllIIII =gr .Slider (minimum =0 ,maximum =IllIIllIIIIlIIlIl .n_cpu ,step =1 ,label =IlIlllIIllllIlIII ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IllIIllIIIIlIIlIl .n_cpu /1.5 )),interactive =_IIIlIlIIIlllIllII )
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IlIIlIIIIIIIIIIll =gr .Textbox (label =IlIlllIIllllIlIII ('输入训练文件夹路径'),value =os .path .abspath (os .getcwd ())+'\\datasets\\');IlIIIlIIllllIlIll =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IlIlllIIllllIlIII ('请指定说话人id'),value =0 ,interactive =_IIIlIlIIIlllIllII );IlIIIlllIIllllIII =gr .Button (IlIlllIIllllIlIII ('处理数据'),variant =_IlllIllIllIIlIIll );IIlIIllllIlIIIIll =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='');IlIIIlllIIllllIII .click (IIlIlIllIllIlIIII ,[IlIIlIIIIIIIIIIll ,IIIIlIlllIIIllIll ,IlIIlIlllllIIIlII ,IIlIllIllIlllIIII ],[IIlIIllllIlIIIIll ])
			with gr .Group ():
				IIIIlIllIllllIIlI =gr .Markdown (value =IlIlllIIllllIlIII ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IIIIIlIlIIllIIlll =gr .Textbox (label =IlIlllIIllllIlIII (_IIlllIllIIIIlIIIl ),value =IllllIIlllIlllIII ,interactive =_IIIlIlIIIlllIllII );IlIllIIlllIllIIII =gr .Textbox (label =IlIlllIIllllIlIII ('显卡信息'),value =IIlIlIllIlIllllII )
					with gr .Column ():IIlIllIIllllIlIll =gr .Radio (label =IlIlllIIllllIlIII ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =['pm',_IIlIlllIlIIIlllII ,'dio',_IIlllIIIIlIlllIII ,_IIlIlIIlIIllIllII ,_IIlllllIIllIIllIl ],value =_IIlllllIIllIIllIl ,interactive =_IIIlIlIIIlllIllII );IlIlIIIllllIllIII =gr .Slider (minimum =1 ,maximum =512 ,step =1 ,label =IlIlllIIllllIlIII (_IIllIlIIlIIlIllll ),value =64 ,interactive =_IIIlIlIIIlllIllII ,visible =_IIllIlIIIIIlIllll );IIlIllIIllllIlIll .change (fn =IllIIIlIllIIIIIII ,inputs =[IIlIllIIllllIlIll ],outputs =[IlIlIIIllllIllIII ])
					IIIlIIIIllIIllllI =gr .Button (IlIlllIIllllIlIII ('特征提取'),variant =_IlllIllIllIIlIIll );IIIIlIllIIlIllIll =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =8 ,interactive =_IIllIlIIIIIlIllll );IIIlIIIIllIIllllI .click (IIIllIllllllllIlI ,[IIIIIlIlIIllIIlll ,IIlIllIllIlllIIII ,IIlIllIIllllIlIll ,IllIIlIlIllIllIll ,IIIIlIlllIIIllIll ,IlIIlIIIIlIIllIIl ,IlIlIIIllllIllIII ],[IIIIlIllIIlIllIll ])
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IIIlIIIlIlIlIllll =gr .Slider (minimum =1 ,maximum =50 ,step =1 ,label =IlIlllIIllllIlIII ('保存频率save_every_epoch'),value =5 ,interactive =_IIIlIlIIIlllIllII ,visible =_IIIlIlIIIlllIllII );IllIIIllIllIIIllI =gr .Slider (minimum =1 ,maximum =10000 ,step =1 ,label =IlIlllIIllllIlIII ('总训练轮数total_epoch'),value =20 ,interactive =_IIIlIlIIIlllIllII );IIllIlllIIIIIllIl =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IlIlllIIllllIlIII ('每张显卡的batch_size'),value =IIIllIIlIIIlllllI ,interactive =_IIIlIlIIIlllIllII );IIIIllIIIllIlIIlI =gr .Checkbox (label ='Whether to save only the latest .ckpt file to save hard drive space',value =_IIIlIlIIIlllIllII ,interactive =_IIIlIlIIIlllIllII );IIllIlllIIIlIlIII =gr .Checkbox (label ='Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement',value =_IIllIlIIIIIlIllll ,interactive =_IIIlIlIIIlllIllII );IIIIllIlIIlllIlIl =gr .Checkbox (label ="Save a small final model to the 'weights' folder at each save point",value =_IIIlIlIIIlllIllII ,interactive =_IIIlIlIIIlllIllII )
				with gr .Row ():IIIIIIIllIlIlIlII =gr .Textbox (lines =2 ,label =IlIlllIIllllIlIII ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IIIlIlIIIlllIllII );IllllIIlllIIllIII =gr .Textbox (lines =2 ,label =IlIlllIIllllIlIII ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IIIlIlIIIlllIllII );IlIIlIlllllIIIlII .change (IlllIIIlIIlIIIIlI ,[IlIIlIlllllIIIlII ,IllIIlIlIllIllIll ,IlIIlIIIIlIIllIIl ],[IIIIIIIllIlIlIlII ,IllllIIlllIIllIII ]);IlIIlIIIIlIIllIIl .change (IIIlIlllllllIIlII ,[IlIIlIlllllIIIlII ,IllIIlIlIllIllIll ,IlIIlIIIIlIIllIIl ],[IIIIIIIllIlIlIlII ,IllllIIlllIIllIII ,IlIIlIlllllIIIlII ]);IllIIlIlIllIllIll .change (fn =IlIIllllIIlIIIIII ,inputs =[IllIIlIlIllIllIll ,IlIIlIlllllIIIlII ,IlIIlIIIIlIIllIIl ,IIIIlIllIllllIIlI ,IIIIIlIlIIllIIlll ,IlIllIIlllIllIIII ,IlIlIIIllllIllIII ,IIIlIIIIllIIllllI ,IIIIlIllIIlIllIll ],outputs =[IIlIllIIllllIlIll ,IIIIIIIllIlIlIlII ,IllllIIlllIIllIII ,IIIIlIllIllllIIlI ,IIIIIlIlIIllIIlll ,IlIllIIlllIllIIII ,IlIlIIIllllIllIII ,IIIlIIIIllIIllllI ,IIIIlIllIIlIllIll ]);IllIIlIlIllIllIll .change (fn =IllIIIlIllIIIIIII ,inputs =[IIlIllIIllllIlIll ],outputs =[IlIlIIIllllIllIII ]);IIlIlIllIllIIlIlI =gr .Textbox (label =IlIlllIIllllIlIII (_IIlllIllIIIIlIIIl ),value =IllllIIlllIlllIII ,interactive =_IIIlIlIIIlllIllII );IllllIIlIIIIIIIlI =gr .Button ('Stop Training',variant =_IlllIllIllIIlIIll ,visible =_IIllIlIIIIIlIllll );IIIIlllIllIIlIllI =gr .Button (IlIlllIIllllIlIII ('训练模型'),variant =_IlllIllIllIIlIIll ,visible =_IIIlIlIIIlllIllII );IIIIlllIllIIlIllI .click (fn =IllIIIIIlIIllIllI ,inputs =[gr .Number (value =0 ,visible =_IIllIlIIIIIlIllll )],outputs =[IIIIlllIllIIlIllI ,IllllIIlIIIIIIIlI ]);IllllIIlIIIIIIIlI .click (fn =IllIIIIIlIIllIllI ,inputs =[gr .Number (value =1 ,visible =_IIllIlIIIIIlIllll )],outputs =[IllllIIlIIIIIIIlI ,IIIIlllIllIIlIllI ]);IIlIllIllIIlIIIIl =gr .Button (IlIlllIIllllIlIII ('训练特征索引'),variant =_IlllIllIllIIlIIll );IIlIllIIIlIlIIIll =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =10 );IIIIllIlIIlllIlIl .change (fn =IllIIllIIlIIIllll ,inputs =[IIIIllIlIIlllIlIl ],outputs =[IIIlIIIlIlIlIllll ]);IIIIlllIllIIlIllI .click (IllIIlIllIIllIIII ,[IIIIlIlllIIIllIll ,IlIIlIlllllIIIlII ,IllIIlIlIllIllIll ,IlIIIlIIllllIlIll ,IIIlIIIlIlIlIllll ,IllIIIllIllIIIllI ,IIllIlllIIIIIllIl ,IIIIllIIIllIlIIlI ,IIIIIIIllIlIlIlII ,IllllIIlllIIllIII ,IIlIlIllIllIIlIlI ,IIllIlllIIIlIlIII ,IIIIllIlIIlllIlIl ,IlIIlIIIIlIIllIIl ],[IIlIllIIIlIlIIIll ,IllllIIlIIIIIIIlI ,IIIIlllIllIIlIllI ]);IIlIllIllIIlIIIIl .click (IIlIlIIIlllIlIllI ,[IIIIlIlllIIIllIll ,IlIIlIIIIlIIllIIl ],IIlIllIIIlIlIIIll )
		with gr .TabItem (IlIlllIIllllIlIII ('ckpt处理')):
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('模型融合, 可用于测试音色融合'))
				with gr .Row ():IIIlIIllIlIllllll =gr .Textbox (label =IlIlllIIllllIlIII ('A模型路径'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='Path to your model A.');IllIlIllIIllllIll =gr .Textbox (label =IlIlllIIllllIlIII ('B模型路径'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='Path to your model B.');IllllIlIlllIllIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIIllllIlIII ('A模型权重'),value =.5 ,interactive =_IIIlIlIIIlllIllII )
				with gr .Row ():IlIllIIllIllIlIlI =gr .Radio (label =IlIlllIIllllIlIII (_IIIlIIlIllIlIIlll ),choices =[_IllIlIIllIllllIll ,_IIllIIllIIIllIlll ],value =_IllIlIIllIllllIll ,interactive =_IIIlIlIIIlllIllII );IlllllIlllIIIIIIl =gr .Checkbox (label =_IIIllIIlllllIlIII ,value =_IIIlIlIIIlllIllII ,interactive =_IIIlIlIIIlllIllII );IllllIllllIlIIIll =gr .Textbox (label =IlIlllIIllllIlIII ('要置入的模型信息'),value ='',max_lines =8 ,interactive =_IIIlIlIIIlllIllII ,placeholder ='Model information to be placed.');IlIlIIIlIIIIllllI =gr .Textbox (label =IlIlllIIllllIlIII ('保存的模型名不带后缀'),value ='',placeholder ='Name for saving.',max_lines =1 ,interactive =_IIIlIlIIIlllIllII );IlIIlIllIllIllIII =gr .Radio (label =IlIlllIIllllIlIII ('模型版本型号'),choices =[_IllllIlIIIIllllll ,_IIlIllIIlIIIlllIl ],value =_IllllIlIIIIllllll ,interactive =_IIIlIlIIIlllIllII )
				with gr .Row ():IIIIlllllllIlIIII =gr .Button (IlIlllIIllllIlIII ('融合'),variant =_IlllIllIllIIlIIll );IIlIllllllIIIllII =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =8 )
				IIIIlllllllIlIIII .click (merge ,[IIIlIIllIlIllllll ,IllIlIllIIllllIll ,IllllIlIlllIllIIl ,IlIllIIllIllIlIlI ,IlllllIlllIIIIIIl ,IllllIllllIlIIIll ,IlIlIIIlIIIIllllI ,IlIIlIllIllIllIII ],IIlIllllllIIIllII )
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('修改模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr .Row ():IllllIllllIIllIll =gr .Textbox (label =IlIlllIIllllIlIII ('模型路径'),placeholder ='Path to your Model.',value ='',interactive =_IIIlIlIIIlllIllII );IIlIIllIllIIIIlll =gr .Textbox (label =IlIlllIIllllIlIII ('要改的模型信息'),value ='',max_lines =8 ,interactive =_IIIlIlIIIlllIllII ,placeholder ='Model information to be changed.');IIIllIllIlIIllIll =gr .Textbox (label =IlIlllIIllllIlIII ('保存的文件名, 默认空为和源文件同名'),placeholder ='Either leave empty or put in the Name of the Model to be saved.',value ='',max_lines =8 ,interactive =_IIIlIlIIIlllIllII )
				with gr .Row ():IllIIIIlIIlIlIIIl =gr .Button (IlIlllIIllllIlIII ('修改'),variant =_IlllIllIllIIlIIll );IllIlllIIlIlllIII =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =8 )
				IllIIIIlIIlIlIIIl .click (change_info ,[IllllIllllIIllIll ,IIlIIllIllIIIIlll ,IIIllIllIlIIllIll ],IllIlllIIlIlllIII )
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('查看模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr .Row ():IIllIIllllIIlllII =gr .Textbox (label =IlIlllIIllllIlIII ('模型路径'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='Model path here.');IlllllIllIIlIIIlI =gr .Button (IlIlllIIllllIlIII ('查看'),variant =_IlllIllIllIIlIIll );IIllIlIIlIIlIllll =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =8 )
				IlllllIllIIlIIIlI .click (show_info ,[IIllIIllllIIlllII ],IIllIlIIlIIlIllll )
			with gr .Group ():
				gr .Markdown (value =IlIlllIIllllIlIII ('模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况'))
				with gr .Row ():IlIlllllIIlIIIlIl =gr .Textbox (lines =3 ,label =IlIlllIIllllIlIII ('模型路径'),value =os .path .abspath (os .getcwd ()).replace (_IlllllllIIllllIII ,_IIllIllIlIIllIllI )+'/logs/[YIUR_MIDEL]/G_23333.pth',interactive =_IIIlIlIIIlllIllII );IIIIlIIIIIlIIIllI =gr .Textbox (label =IlIlllIIllllIlIII ('保存名'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='Your filename here.');IllIlIIIIlIIIllII =gr .Radio (label =IlIlllIIllllIlIII (_IIIlIIlIllIlIIlll ),choices =[_IIIllllIIlIIllIII ,_IllIlIIllIllllIll ,_IIllIIllIIIllIlll ],value =_IllIlIIllIllllIll ,interactive =_IIIlIlIIIlllIllII );IllIlIIllllIlIlII =gr .Checkbox (label =_IIIllIIlllllIlIII ,value =_IIIlIlIIIlllIllII ,interactive =_IIIlIlIIIlllIllII );IIlIIIIllIIIllIII =gr .Radio (label =IlIlllIIllllIlIII ('模型版本型号'),choices =[_IllllIlIIIIllllll ,_IIlIllIIlIIIlllIl ],value =_IIlIllIIlIIIlllIl ,interactive =_IIIlIlIIIlllIllII );IlIIIIIIlIIIlIllI =gr .Textbox (label =IlIlllIIllllIlIII ('要置入的模型信息'),value ='',max_lines =8 ,interactive =_IIIlIlIIIlllIllII ,placeholder ='Model info here.');IllllIIIlIIIllllI =gr .Button (IlIlllIIllllIlIII ('提取'),variant =_IlllIllIllIIlIIll );IllIIIIIlIIllllII =gr .Textbox (label =IlIlllIIllllIlIII (_IIllIIIIIlIlIllll ),value ='',max_lines =8 );IlIlllllIIlIIIlIl .change (IIIIIIIIIllIIIIII ,[IlIlllllIIlIIIlIl ],[IllIlIIIIlIIIllII ,IllIlIIllllIlIlII ,IIlIIIIllIIIllIII ])
				IllllIIIlIIIllllI .click (extract_small_model ,[IlIlllllIIlIIIlIl ,IIIIlIIIIIlIIIllI ,IllIlIIIIlIIIllII ,IllIlIIllllIlIlII ,IlIIIIIIlIIIlIllI ,IIlIIIIllIIIllIII ],IllIIIIIlIIllllII )
		with gr .TabItem (IlIlllIIllllIlIII ('Onnx导出')):
			with gr .Row ():IIllllIllIIllIllI =gr .Textbox (label =IlIlllIIllllIlIII ('RVC模型路径'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='RVC model path.')
			with gr .Row ():IIlIIIIIlIIIIIlll =gr .Textbox (label =IlIlllIIllllIlIII ('Onnx输出路径'),value ='',interactive =_IIIlIlIIIlllIllII ,placeholder ='Onnx model output path.')
			with gr .Row ():IlllIIllIlIIIIlIl =gr .Label (label ='info')
			with gr .Row ():IlIIllIlIlllIIlIl =gr .Button (IlIlllIIllllIlIII ('导出Onnx模型'),variant =_IlllIllIllIIlIIll )
			IlIIllIlIlllIIlIl .click (IlIIlIlIlllIlIlII ,[IIllllIllIIllIllI ,IIlIIIIIlIIIIIlll ],IlllIIllIlIIIIlIl )
		IlIIIlllIlIlllIll =IlIlllIIllllIlIII ('常见问题解答')
		with gr .TabItem (IlIIIlllIlIlllIll ):
			try :
				if IlIIIlllIlIlllIll =='常见问题解答':
					with open ('docs/faq.md',_IIlIlIlIIIlIIIIII ,encoding ='utf8')as IIIIlllIIIIlllllI :IIllIIlllIlIIIIII =IIIIlllIIIIlllllI .read ()
				else :
					with open ('docs/faq_en.md',_IIlIlIlIIIlIIIIII ,encoding ='utf8')as IIIIlllIIIIlllllI :IIllIIlllIlIIIIII =IIIIlllIIIIlllllI .read ()
				gr .Markdown (value =IIllIIlllIlIIIIII )
			except :gr .Markdown (value =traceback .format_exc ())
	def IlIllIIIIIllIIlll (IlIlIIlIIllIlIIlI ,IllIIllIlIlIIIlll ,IlIIIIlIllIIllIlI ,IIllIIIllIlllIlll ,IlIIlllllIlIlllll ,IlIIIllIIIIIlIlII ,IlllllIIIIlIlIIIl ,IIlIIIlIIIIIIlIII ,IlIlIIlIIlIlllIll ,IIllllIIIIIllIIII ,IlIIIlllIIIIlllIl ,IIllIlIlllIlIIlll ,IIIIIlIIIlllllIll ,IIlIlIIIIlIllIlll ,IIIIIllIlllllIlIl ):
		IlIlIIlIllIllIIll =_IIllIlIlIIlllIIlI 
		with open (_IIIIIlIIllIIllllI ,_IIlIlIlIIIlIIIIII )as IllllllIIIllIlIII :IlIlIIlIllIllIIll =json .load (IllllllIIIllIlIII )
		IlIIllIllIlIIIllI ={'name':IlIlIIlIIllIlIIlI ,'model':IllIIllIlIlIIIlll ,'transpose':IlIIIIlIllIIllIlI ,'audio_file':IIllIIIllIlllIlll ,'auto_audio_file':IlIIlllllIlIlllll ,'f0_method':IlIIIllIIIIIlIlII ,_IIllIlIIlIIlIllll :IlllllIIIIlIlIIIl ,'median_filtering':IIlIIIlIIIIIIlIII ,'feature_path':IlIlIIlIIlIlllIll ,'auto_feature_path':IIllllIIIIIllIIII ,'search_feature_ratio':IlIIIlllIIIIlllIl ,'resample':IIllIlIlllIlIIlll ,'volume_envelope':IIIIIlIIIlllllIll ,'protect_voiceless':IIlIlIIIIlIllIlll ,'fl_file_path':IIIIIllIlllllIlIl };IlIlIIlIllIllIIll [_IllIIllIIIIlllIIl ].append (IlIIllIllIlIIIllI )
		with open (_IIIIIlIIllIIllllI ,_IIlIIllIIllIlIIII )as IllllllIIIllIlIII :json .dump (IlIlIIlIllIllIIll ,IllllllIIIllIlIII );IllllllIIIllIlIII .flush ()
		print ('Saved Preset %s into inference-presets.json!'%IlIlIIlIIllIlIIlI )
	def IlllIllllIIIlIlII (IIlllIIlIllIIllIl ):
		print ('Changed Preset to %s!'%IIlllIIlIllIIllIl );IllIlIlllllIlIlIl =_IIllIlIlIIlllIIlI 
		with open (_IIIIIlIIllIIllllI ,_IIlIlIlIIIlIIIIII )as IIllllllllIIIlllI :IllIlIlllllIlIlIl =json .load (IIllllllllIIIlllI )
		print ('Searching for '+IIlllIIlIllIIllIl );IllllIllIllllIIIl =_IIllIlIlIIlllIIlI 
		for IIllIlIIIllIlIlIl in IllIlIlllllIlIlIl [_IllIIllIIIIlllIIl ]:
			if IIllIlIIIllIlIlIl ['name']==IIlllIIlIllIIllIl :print ('Found a preset');IllllIllIllllIIIl =IIllIlIIIllIlIlIl 
		return ()
	if IllIIllIIIIlIIlIl .iscolab or IllIIllIIIIlIIlIl .paperspace :IlllIlIIlIIIIIIlI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IllIIllIIIIlIIlIl .listen_port ,share =_IIllIlIIIIIlIllll )
	else :IlllIlIIlIIIIIIlI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IllIIllIIIIlIIlIl .noautoopen ,server_port =IllIIllIIIIlIIlIl .listen_port ,quiet =_IIllIlIIIIIlIllll ,share =_IIllIlIIIIIlIllll )