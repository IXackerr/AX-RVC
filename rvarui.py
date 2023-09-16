_IlllIllIIlIllIlll ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IIlllIllIllllIIll ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IIllIllllIlllIllI ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIlIlIlIIlIlIlIIl ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IlIlllllIIllIlIII ='后处理重采样至最终采样率，0为不进行重采样'
_IIllIlIIIIllIlIlI ='自动检测index路径,下拉式选择(dropdown)'
_IlIlIIIIIlIlIllIl ='特征检索库文件路径,为空则使用下拉的选择结果'
_IlIllIlIlllIlllIl ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IlIlIlIllIlIIIllI ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IlIIIIIlIIllIIIIl ='变调(整数, 半音数量, 升八度12降八度-12)'
_IIIIIllIlIlIIlIll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IIIIllIIlIllIlllI ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIIlIllIIllIIIIl ='IVF%s,Flat'
_IlIlIIllllIIIllll ='%s/total_fea.npy'
_IlIIIIIIIIlIlIllI ='Trying doing kmeans %s shape to 10k centers.'
_IIllIIllllllIIIII ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IllllIlIllIllIIIl =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IllllIIIlIlIlIllI =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIIIIIIllllIIIllI ='write filelist done'
_IIlIIIIlIIlllIIlI ='%s/filelist.txt'
_IlIIIIIIIIIIIIIlI ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IIlIIIIIIllIlIlIl ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IllllIIlIlIlllIlI ='%s/%s.wav|%s/%s.npy|%s'
_IIIIlllIlIIlIIlII ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IllIlIIllllIIllIl ='%s/2b-f0nsf'
_IIIlIlllIlIlIIIll ='%s/0_gt_wavs'
_IIIIllIlllIllllll ='emb_g.weight'
_IlIlllIIIlIlIlIIl ='clean_empty_cache'
_IIllIllIlllIlIlIl ='sample_rate'
_IlIlIIlIIlIIIIlll ='%s->%s'
_IlllIlllIllllllIl ='.index'
_IlIllIIlllIlIlIII ='weights'
_IllIllIIlllIlIllI ='opt'
_IlllIIllIIlllllll ='rmvpe'
_IllIlIIIIlIIIIlII ='harvest'
_IIlIIIlIIIIlllIll ='%s/3_feature768'
_IIlIlIlIIlIlIlIll ='%s/3_feature256'
_IIlIlIIlIllllIlIl ='_v2'
_IIIIllIIIIIlIlIll ='48k'
_IIIlIIllIIllIIIll ='32k'
_IIlIIIIIllIIlllII ='cpu'
_IllIlllIIIlIIIIIl ='wav'
_IlIIlIIlIllllIIII ='trained'
_IllIllIllIlIlllIl ='logs'
_IlIIllIIIlIlllllI ='-pd %s'
_IIlIIIllIllIllIIl ='-pg %s'
_IIlllIllIllllIIlI ='choices'
_IlIlIlIIIIIlIlIll ='weight'
_IIlIIllIIIIlIIIll ='pm'
_IllIlIIIlIIllIlII ='rmvpe_gpu'
_IlllIlIIlIIIllIIl ='%s/logs/%s'
_IlllIIIlllIlIIIII ='flac'
_IlIIlIlIIlIllIlIl ='f0'
_IllIIIllIIIllllII ='%s/%s'
_IlIlllIIlIIlIIIlI ='.pth'
_IIlllIIIIlIlIIIIl ='输出信息'
_IIlllIllIlllIllll ='not exist, will not use pretrained model'
_IllIIIIIlIIIIIlII ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IllIllllIIIlIlllI ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IIlIlIIIllIIlIIll ='40k'
_IlIllIIIIllllllIl ='value'
_IIIlllIIIIIIlIlIl ='v2'
_IIIllllllllIIlIII ='version'
_IIllIIIlIIIIIlllI ='visible'
_IIlIlIlIlIlllIlll ='primary'
_IIllllIIIlIllIIIl =None 
_IlIIIllllllIIlIlI ='\\\\'
_IIIIllllllllIIlII ='\\'
_IlIlIIlIIIIlIlllI ='"'
_IllIllIIlllIIIIll =' '
_IlIllIIllIIllIllI ='config'
_IlIlllIIIIlllIIII ='.'
_IIIlIlIIIIllIlIlI ='r'
_IIllIllIIllIlIIIl ='是'
_IIIlIlllIIlllIIII ='update'
_IIIIIlIIlllIlllII ='__type__'
_IIIlIIIllIIIIlIlI ='v1'
_IIllIIlIIlllIllll ='\n'
_IllIlIllIlIlIllII =False 
_IIlIIIIllIlIIlIII =True 
import os ,shutil ,sys 
IIlllllllIIlIIIII =os .getcwd ()
sys .path .append (IIlllllllIIlIIIII )
import traceback ,pdb ,warnings ,numpy as np ,torch 
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
from lib .audio import load_audio 
from lib .train .process_ckpt import change_info ,extract_small_model ,merge ,show_info 
from vc_infer_pipeline import VC 
from sklearn .cluster import MiniBatchKMeans 
logging .getLogger ('numba').setLevel (logging .WARNING )
IIlllllllIIlIIIII =os .getcwd ()
IllllIlIllIIlIIIl =os .path .join (IIlllllllIIlIIIII ,'TEMP')
shutil .rmtree (IllllIlIllIIlIIIl ,ignore_errors =_IIlIIIIllIlIIlIII )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IIlllllllIIlIIIII ,ignore_errors =_IIlIIIIllIlIIlIII )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IIlllllllIIlIIIII ,ignore_errors =_IIlIIIIllIlIIlIII )
os .makedirs (IllllIlIllIIlIIIl ,exist_ok =_IIlIIIIllIlIIlIII )
os .makedirs (os .path .join (IIlllllllIIlIIIII ,_IllIllIllIlIlllIl ),exist_ok =_IIlIIIIllIlIIlIII )
os .makedirs (os .path .join (IIlllllllIIlIIIII ,_IlIllIIlllIlIlIII ),exist_ok =_IIlIIIIllIlIIlIII )
os .environ ['TEMP']=IllllIlIllIIlIIIl 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IIlIIIIllIIlIlllI =Config ()
IllIIIIIIlIllIIlI =I18nAuto ()
IllIIIIIIlIllIIlI .print ()
IlIIIIllllIlllIll =torch .cuda .device_count ()
IIlllllIlIllllllI =[]
IIllllllIIlllIIII =[]
IIIllllIIllIIIIIl =_IllIlIllIlIlIllII 
if torch .cuda .is_available ()or IlIIIIllllIlllIll !=0 :
	for IIIIlllIIllIlIIIl in range (IlIIIIllllIlllIll ):
		IlIlIIlllIIIlIIlI =torch .cuda .get_device_name (IIIIlllIIllIlIIIl )
		if any (IllIIlllIllIlIIII in IlIlIIlllIIIlIIlI .upper ()for IllIIlllIllIlIIII in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IIIllllIIllIIIIIl =_IIlIIIIllIlIIlIII ;IIlllllIlIllllllI .append ('%s\t%s'%(IIIIlllIIllIlIIIl ,IlIlIIlllIIIlIIlI ));IIllllllIIlllIIII .append (int (torch .cuda .get_device_properties (IIIIlllIIllIlIIIl ).total_memory /1024 /1024 /1024 +.4 ))
if IIIllllIIllIIIIIl and len (IIlllllIlIllllllI )>0 :IlIlIIlIIIlIIlIll =_IIllIIlIIlllIllll .join (IIlllllIlIllllllI );IllIIlllIIIlllllI =min (IIllllllIIlllIIII )//2 
else :IlIlIIlIIIlIIlIll =IllIIIIIIlIllIIlI ('很遗憾您这没有能用的显卡来支持您训练');IllIIlllIIIlllllI =1 
IIlIllllIllIlIIIl ='-'.join ([IIIIlIlIlIlIlIIIl [0 ]for IIIIlIlIlIlIlIIIl in IIlllllIlIllllllI ])
class IIIIIIlllIIlIlIll (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IIlIIIlIIIllIIlII ,**IllllIllllIlIllll ):super ().__init__ (variant ='tool',**IllllIllllIlIllll )
	def get_block_name (IlIlIlllIllIllIll ):return 'button'
IllIlIIIlIIlIIIll =_IIllllIIIlIllIIIl 
def IllIIIlllIIllIllI ():
	global IllIlIIIlIIlIIIll ;IllIIlIIlIIlllllI ,_IllIIlIIIlIIIlIIl ,_IllIIlIIIlIIIlIIl =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IllIlIIIlIIlIIIll =IllIIlIIlIIlllllI [0 ];IllIlIIIlIIlIIIll =IllIlIIIlIIlIIIll .to (IIlIIIIllIIlIlllI .device )
	if IIlIIIIllIIlIlllI .is_half :IllIlIIIlIIlIIIll =IllIlIIIlIIlIIIll .half ()
	else :IllIlIIIlIIlIIIll =IllIlIIIlIIlIIIll .float ()
	IllIlIIIlIIlIIIll .eval ()
IIIllIlIllIlllIIl =_IlIllIIlllIlIlIII 
IlllllIIIIIlIllll ='uvr5_weights'
IIIllllIlllllIllI =_IllIllIllIlIlllIl 
IlIIlllllIIlIllII =[]
for IIllllllIlllIllII in os .listdir (IIIllIlIllIlllIIl ):
	if IIllllllIlllIllII .endswith (_IlIlllIIlIIlIIIlI ):IlIIlllllIIlIllII .append (IIllllllIlllIllII )
IIllIIIIllIllIIlI =[]
for (IllIlIlIlIIIllIII ,IllIIllIlIIIIIIIl ,IIIIlIlIllIIllIll )in os .walk (IIIllllIlllllIllI ,topdown =_IllIlIllIlIlIllII ):
	for IIllllllIlllIllII in IIIIlIlIllIIllIll :
		if IIllllllIlllIllII .endswith (_IlllIlllIllllllIl )and _IlIIlIIlIllllIIII not in IIllllllIlllIllII :IIllIIIIllIllIIlI .append (_IllIIIllIIIllllII %(IllIlIlIlIIIllIII ,IIllllllIlllIllII ))
IlIlIIIIlIllIIlll =[]
for IIllllllIlllIllII in os .listdir (IlllllIIIIIlIllll ):
	if IIllllllIlllIllII .endswith (_IlIlllIIlIIlIIIlI )or 'onnx'in IIllllllIlllIllII :IlIlIIIIlIllIIlll .append (IIllllllIlllIllII .replace (_IlIlllIIlIIlIIIlI ,''))
IIIIllllllIIllIIl =_IIllllIIIlIllIIIl 
def IIllIlIlIllIllIll (IlllIIlllIIIlIllI ,IlIIIIIlIIIllIIIl ,IIlIllIIIIlIlIlII ,IlIIIlIlIIlIIlIII ,IIllIlIIIllllIlIl ,IIlllIllIlIlllIIl ,IIlIIIlIIlIIllllI ,IIlllIIIlIIlIIIIl ,IlIlIllllllIlIlll ,IllIIlllIlIIlIIIl ,IIllllIIIllIlIlIl ,IIlIIlllIlIlIIIll ):
	global IlIlIlIIlIlllIlIl ,IIlIIIIlIIIIlllll ,IllIllIIIIlIlIIlI ,IllIlIIIlIIlIIIll ,IIlIllllIIllIIlIl ,IIIIllllllIIllIIl 
	if IlIIIIIlIIIllIIIl is _IIllllIIIlIllIIIl :return 'You need to upload an audio',_IIllllIIIlIllIIIl 
	IIlIllIIIIlIlIlII =int (IIlIllIIIIlIlIlII )
	try :
		IIIlIlIlIlllIllIl =load_audio (IlIIIIIlIIIllIIIl ,16000 );IIllIIIIllllllIlI =np .abs (IIIlIlIlIlllIllIl ).max ()/.95 
		if IIllIIIIllllllIlI >1 :IIIlIlIlIlllIllIl /=IIllIIIIllllllIlI 
		IllIIIIIIlIIIIIll =[0 ,0 ,0 ]
		if not IllIlIIIlIIlIIIll :IllIIIlllIIllIllI ()
		IIIIIIlIllIllIIIl =IIIIllllllIIllIIl .get (_IlIIlIlIIlIllIlIl ,1 );IIlllIllIlIlllIIl =IIlllIllIlIlllIIl .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll ).replace (_IlIIlIIlIllllIIII ,'added')if IIlllIllIlIlllIIl !=''else IIlIIIlIIlIIllllI ;IIlllIlIIIIllIIII =IllIllIIIIlIlIIlI .pipeline (IllIlIIIlIIlIIIll ,IIlIIIIlIIIIlllll ,IlllIIlllIIIlIllI ,IIIlIlIlIlllIllIl ,IlIIIIIlIIIllIIIl ,IllIIIIIIlIIIIIll ,IIlIllIIIIlIlIlII ,IIllIlIIIllllIlIl ,IIlllIllIlIlllIIl ,IIlllIIIlIIlIIIIl ,IIIIIIlIllIllIIIl ,IlIlIllllllIlIlll ,IlIlIlIIlIlllIlIl ,IllIIlllIlIIlIIIl ,IIllllIIIllIlIlIl ,IIlIllllIIllIIlIl ,IIlIIlllIlIlIIIll ,f0_file =IlIIIlIlIIlIIlIII )
		if IlIlIlIIlIlllIlIl !=IllIIlllIlIIlIIIl >=16000 :IlIlIlIIlIlllIlIl =IllIIlllIlIIlIIIl 
		IIlIIIIIlllllIIlI ='Using index:%s.'%IIlllIllIlIlllIIl if os .path .exists (IIlllIllIlIlllIIl )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IIlIIIIIlllllIIlI ,IllIIIIIIlIIIIIll [0 ],IllIIIIIIlIIIIIll [1 ],IllIIIIIIlIIIIIll [2 ]),(IlIlIlIIlIlllIlIl ,IIlllIlIIIIllIIII )
	except :IlllIllllllIllIlI =traceback .format_exc ();print (IlllIllllllIllIlI );return IlllIllllllIllIlI ,(_IIllllIIIlIllIIIl ,_IIllllIIIlIllIIIl )
def IlIlllIIIIlIIlllI (IllIIlIlIllIlIlIl ,IlIIlIIIIllIIIlll ,IllIIlIIIllllIIlI ,IllIllIIlIIlIlIll ,IIIIlIlllIlIIIlll ,IIIllIllllIlIllIl ,IlIllIIllIIIlIIlI ,IlIlIIlllllIlIIIl ,IIIllIIIIIlIIIlll ,IllIIllIllIIlIlll ,IllIlllIlIIlllIlI ,IIIIlIIlIIIIIIlII ,IIlIllllIllllIlll ,IIllIIlllllIlIIII ):
	try :
		IlIIlIIIIllIIIlll =IlIIlIIIIllIIIlll .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll );IllIIlIIIllllIIlI =IllIIlIIIllllIIlI .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll );os .makedirs (IllIIlIIIllllIIlI ,exist_ok =_IIlIIIIllIlIIlIII )
		try :
			if IlIIlIIIIllIIIlll !='':IllIllIIlIIlIlIll =[os .path .join (IlIIlIIIIllIIIlll ,IIIIIlllllIlIllIl )for IIIIIlllllIlIllIl in os .listdir (IlIIlIIIIllIIIlll )]
			else :IllIllIIlIIlIlIll =[IllIIlIIllIIllIlI .name for IllIIlIIllIIllIlI in IllIllIIlIIlIlIll ]
		except :traceback .print_exc ();IllIllIIlIIlIlIll =[IIIllIIIllIlllIll .name for IIIllIIIllIlllIll in IllIllIIlIIlIlIll ]
		IIlllIIlIlIIlIlll =[]
		for IIllllllIllllIIIl in IllIllIIlIIlIlIll :
			IIllIlllIIIllIIII ,IIIlIIIlIIIIIIIII =IIllIlIlIllIllIll (IllIIlIlIllIlIlIl ,IIllllllIllllIIIl ,IIIIlIlllIlIIIlll ,_IIllllIIIlIllIIIl ,IIIllIllllIlIllIl ,IlIllIIllIIIlIIlI ,IlIlIIlllllIlIIIl ,IIIllIIIIIlIIIlll ,IllIIllIllIIlIlll ,IllIlllIlIIlllIlI ,IIIIlIIlIIIIIIlII ,IIlIllllIllllIlll )
			if 'Success'in IIllIlllIIIllIIII :
				try :
					IllIlIlllIlIIIlll ,IlllIlllIIlIlllll =IIIlIIIlIIIIIIIII 
					if IIllIIlllllIlIIII in [_IllIlllIIIlIIIIIl ,_IlllIIIlllIlIIIII ]:sf .write ('%s/%s.%s'%(IllIIlIIIllllIIlI ,os .path .basename (IIllllllIllllIIIl ),IIllIIlllllIlIIII ),IlllIlllIIlIlllll ,IllIlIlllIlIIIlll )
					else :
						IIllllllIllllIIIl ='%s/%s.wav'%(IllIIlIIIllllIIlI ,os .path .basename (IIllllllIllllIIIl ));sf .write (IIllllllIllllIIIl ,IlllIlllIIlIlllll ,IllIlIlllIlIIIlll )
						if os .path .exists (IIllllllIllllIIIl ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IIllllllIllllIIIl ,IIllllllIllllIIIl [:-4 ]+'.%s'%IIllIIlllllIlIIII ))
				except :IIllIlllIIIllIIII +=traceback .format_exc ()
			IIlllIIlIlIIlIlll .append (_IlIlIIlIIlIIIIlll %(os .path .basename (IIllllllIllllIIIl ),IIllIlllIIIllIIII ));yield _IIllIIlIIlllIllll .join (IIlllIIlIlIIlIlll )
		yield _IIllIIlIIlllIllll .join (IIlllIIlIlIIlIlll )
	except :yield traceback .format_exc ()
def IIIlllIIIIIlIIIII (IllIIIIIIIlllIIll ,IIIlIlllllIlIIlII ,IIllIIIIIlIIIlIII ,IlIlIlIIIlIIlllll ,IllIIllllIlIlIlII ,IIIlIlIlIIlIIIIll ,IlIIllIlllIIIlllI ):
	IlIllIllllllllllI ='streams';IIIllIllIIIllIIll ='onnx_dereverb_By_FoxJoy';IIlIIllIlIIlllIII =[]
	try :
		IIIlIlllllIlIIlII =IIIlIlllllIlIIlII .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll );IIllIIIIIlIIIlIII =IIllIIIIIlIIIlIII .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll );IllIIllllIlIlIlII =IllIIllllIlIlIlII .strip (_IllIllIIlllIIIIll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IIllIIlIIlllIllll ).strip (_IlIlIIlIIIIlIlllI ).strip (_IllIllIIlllIIIIll )
		if IllIIIIIIIlllIIll ==IIIllIllIIIllIIll :from MDXNet import MDXNetDereverb ;IllIlIlIllIIllIlI =MDXNetDereverb (15 )
		else :IIIlIIIIIIllIlllI =_audio_pre_ if 'DeEcho'not in IllIIIIIIIlllIIll else _audio_pre_new ;IllIlIlIllIIllIlI =IIIlIIIIIIllIlllI (agg =int (IIIlIlIlIIlIIIIll ),model_path =os .path .join (IlllllIIIIIlIllll ,IllIIIIIIIlllIIll +_IlIlllIIlIIlIIIlI ),device =IIlIIIIllIIlIlllI .device ,is_half =IIlIIIIllIIlIlllI .is_half )
		if IIIlIlllllIlIIlII !='':IlIlIlIIIlIIlllll =[os .path .join (IIIlIlllllIlIIlII ,IlIlIIllIIllllIll )for IlIlIIllIIllllIll in os .listdir (IIIlIlllllIlIIlII )]
		else :IlIlIlIIIlIIlllll =[IllllIlIlllllIIlI .name for IllllIlIlllllIIlI in IlIlIlIIIlIIlllll ]
		for IIlllllIlIlIlIlll in IlIlIlIIIlIIlllll :
			IIIlIllllllIIlIIl =os .path .join (IIIlIlllllIlIIlII ,IIlllllIlIlIlIlll );IlIlllIIIIlIllIIl =1 ;IIIIIIlIIlllllllI =0 
			try :
				IllIlIIIIIIIlIIlI =ffmpeg .probe (IIIlIllllllIIlIIl ,cmd ='ffprobe')
				if IllIlIIIIIIIlIIlI [IlIllIllllllllllI ][0 ]['channels']==2 and IllIlIIIIIIIlIIlI [IlIllIllllllllllI ][0 ][_IIllIllIlllIlIlIl ]=='44100':IlIlllIIIIlIllIIl =0 ;IllIlIlIllIIllIlI ._path_audio_ (IIIlIllllllIIlIIl ,IllIIllllIlIlIlII ,IIllIIIIIlIIIlIII ,IlIIllIlllIIIlllI );IIIIIIlIIlllllllI =1 
			except :IlIlllIIIIlIllIIl =1 ;traceback .print_exc ()
			if IlIlllIIIIlIllIIl ==1 :IIlIIlIlIIIIlIlII ='%s/%s.reformatted.wav'%(IllllIlIllIIlIIIl ,os .path .basename (IIIlIllllllIIlIIl ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIIlIllllllIIlIIl ,IIlIIlIlIIIIlIlII ));IIIlIllllllIIlIIl =IIlIIlIlIIIIlIlII 
			try :
				if IIIIIIlIIlllllllI ==0 :IllIlIlIllIIllIlI ._path_audio_ (IIIlIllllllIIlIIl ,IllIIllllIlIlIlII ,IIllIIIIIlIIIlIII ,IlIIllIlllIIIlllI )
				IIlIIllIlIIlllIII .append ('%s->Success'%os .path .basename (IIIlIllllllIIlIIl ));yield _IIllIIlIIlllIllll .join (IIlIIllIlIIlllIII )
			except :IIlIIllIlIIlllIII .append (_IlIlIIlIIlIIIIlll %(os .path .basename (IIIlIllllllIIlIIl ),traceback .format_exc ()));yield _IIllIIlIIlllIllll .join (IIlIIllIlIIlllIII )
	except :IIlIIllIlIIlllIII .append (traceback .format_exc ());yield _IIllIIlIIlllIllll .join (IIlIIllIlIIlllIII )
	finally :
		try :
			if IllIIIIIIIlllIIll ==IIIllIllIIIllIIll :del IllIlIlIllIIllIlI .pred .model ;del IllIlIlIllIIllIlI .pred .model_ 
			else :del IllIlIlIllIIllIlI .model ;del IllIlIlIllIIllIlI 
		except :traceback .print_exc ()
		print (_IlIlllIIIlIlIlIIl )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IIllIIlIIlllIllll .join (IIlIIllIlIIlllIII )
def IIIlIlIIIllllIlIl (IIlllIllIIlIlIlII ):
	IlllIIlIlIIIlllII ='';IIIlllllIlIlllIll =os .path .join (_IllIllIllIlIlllIl ,IIlllIllIIlIlIlII .split (_IlIlllIIIIlllIIII )[0 ],'')
	for IIlIIlIllIlIllIII in IIllIIIIllIllIIlI :
		if IIIlllllIlIlllIll in IIlIIlIllIlIllIII :IlllIIlIlIIIlllII =IIlIIlIllIlIllIII ;break 
	return IlllIIlIlIIIlllII 
def IIlIlIlllIllIIlII (IIlIIlIlIIlIIllIl ,IIllIllllIlllIIlI ,IIlllIIlIlIlllIlI ):
	global IIllllllllIlIIIlI ,IlIlIlIIlIlllIlIl ,IIlIIIIlIIIIlllll ,IllIllIIIIlIlIIlI ,IIIIllllllIIllIIl ,IIlIllllIIllIIlIl 
	if IIlIIlIlIIlIIllIl ==''or IIlIIlIlIIlIIllIl ==[]:
		global IllIlIIIlIIlIIIll 
		if IllIlIIIlIIlIIIll is not _IIllllIIIlIllIIIl :
			print (_IlIlllIIIlIlIlIIl );del IIlIIIIlIIIIlllll ,IIllllllllIlIIIlI ,IllIllIIIIlIlIIlI ,IllIlIIIlIIlIIIll ,IlIlIlIIlIlllIlIl ;IllIlIIIlIIlIIIll =IIlIIIIlIIIIlllll =IIllllllllIlIIIlI =IllIllIIIIlIlIIlI =IllIlIIIlIIlIIIll =IlIlIlIIlIlllIlIl =_IIllllIIIlIllIIIl 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IllIlllIlllIIllll =IIIIllllllIIllIIl .get (_IlIIlIlIIlIllIlIl ,1 );IIlIllllIIllIIlIl =IIIIllllllIIllIIl .get (_IIIllllllllIIlIII ,_IIIlIIIllIIIIlIlI )
			if IIlIllllIIllIIlIl ==_IIIlIIIllIIIIlIlI :
				if IllIlllIlllIIllll ==1 :IIlIIIIlIIIIlllll =SynthesizerTrnMs256NSFsid (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ],is_half =IIlIIIIllIIlIlllI .is_half )
				else :IIlIIIIlIIIIlllll =SynthesizerTrnMs256NSFsid_nono (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ])
			elif IIlIllllIIllIIlIl ==_IIIlllIIIIIIlIlIl :
				if IllIlllIlllIIllll ==1 :IIlIIIIlIIIIlllll =SynthesizerTrnMs768NSFsid (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ],is_half =IIlIIIIllIIlIlllI .is_half )
				else :IIlIIIIlIIIIlllll =SynthesizerTrnMs768NSFsid_nono (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ])
			del IIlIIIIlIIIIlllll ,IIIIllllllIIllIIl 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IIllIIIlIIIIIlllI :_IllIlIllIlIlIllII ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
	IlIIIlllllIIlIlIl =_IllIIIllIIIllllII %(IIIllIlIllIlllIIl ,IIlIIlIlIIlIIllIl );print ('loading %s'%IlIIIlllllIIlIlIl );IIIIllllllIIllIIl =torch .load (IlIIIlllllIIlIlIl ,map_location =_IIlIIIIIllIIlllII );IlIlIlIIlIlllIlIl =IIIIllllllIIllIIl [_IlIllIIllIIllIllI ][-1 ];IIIIllllllIIllIIl [_IlIllIIllIIllIllI ][-3 ]=IIIIllllllIIllIIl [_IlIlIlIIIIIlIlIll ][_IIIIllIlllIllllll ].shape [0 ];IllIlllIlllIIllll =IIIIllllllIIllIIl .get (_IlIIlIlIIlIllIlIl ,1 )
	if IllIlllIlllIIllll ==0 :IIllIllllIlllIIlI =IIlllIIlIlIlllIlI ={_IIllIIIlIIIIIlllI :_IllIlIllIlIlIllII ,_IlIllIIIIllllllIl :.5 ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
	else :IIllIllllIlllIIlI ={_IIllIIIlIIIIIlllI :_IIlIIIIllIlIIlIII ,_IlIllIIIIllllllIl :IIllIllllIlllIIlI ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII };IIlllIIlIlIlllIlI ={_IIllIIIlIIIIIlllI :_IIlIIIIllIlIIlIII ,_IlIllIIIIllllllIl :IIlllIIlIlIlllIlI ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
	IIlIllllIIllIIlIl =IIIIllllllIIllIIl .get (_IIIllllllllIIlIII ,_IIIlIIIllIIIIlIlI )
	if IIlIllllIIllIIlIl ==_IIIlIIIllIIIIlIlI :
		if IllIlllIlllIIllll ==1 :IIlIIIIlIIIIlllll =SynthesizerTrnMs256NSFsid (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ],is_half =IIlIIIIllIIlIlllI .is_half )
		else :IIlIIIIlIIIIlllll =SynthesizerTrnMs256NSFsid_nono (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ])
	elif IIlIllllIIllIIlIl ==_IIIlllIIIIIIlIlIl :
		if IllIlllIlllIIllll ==1 :IIlIIIIlIIIIlllll =SynthesizerTrnMs768NSFsid (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ],is_half =IIlIIIIllIIlIlllI .is_half )
		else :IIlIIIIlIIIIlllll =SynthesizerTrnMs768NSFsid_nono (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ])
	del IIlIIIIlIIIIlllll .enc_q ;print (IIlIIIIlIIIIlllll .load_state_dict (IIIIllllllIIllIIl [_IlIlIlIIIIIlIlIll ],strict =_IllIlIllIlIlIllII ));IIlIIIIlIIIIlllll .eval ().to (IIlIIIIllIIlIlllI .device )
	if IIlIIIIllIIlIlllI .is_half :IIlIIIIlIIIIlllll =IIlIIIIlIIIIlllll .half ()
	else :IIlIIIIlIIIIlllll =IIlIIIIlIIIIlllll .float ()
	IllIllIIIIlIlIIlI =VC (IlIlIlIIlIlllIlIl ,IIlIIIIllIIlIlllI );IIllllllllIlIIIlI =IIIIllllllIIllIIl [_IlIllIIllIIllIllI ][-3 ];return {_IIllIIIlIIIIIlllI :_IIlIIIIllIlIIlIII ,'maximum':IIllllllllIlIIIlI ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },IIllIllllIlllIIlI ,IIlllIIlIlIlllIlI ,IIIlIlIIIllllIlIl (IIlIIlIlIIlIIllIl )
def IlIIIlllIllllIllI ():
	IIlIllIllIlIlIIlI =[]
	for IlIIlllIllIlIIlIl in os .listdir (IIIllIlIllIlllIIl ):
		if IlIIlllIllIlIIlIl .endswith (_IlIlllIIlIIlIIIlI ):IIlIllIllIlIlIIlI .append (IlIIlllIllIlIIlIl )
	IllIIlllllllIlIII =[]
	for (IlIlllllIIIIIIlll ,IIIlIIllIlIlllllI ,IIlIllIIlllIlllll )in os .walk (IIIllllIlllllIllI ,topdown =_IllIlIllIlIlIllII ):
		for IlIIlllIllIlIIlIl in IIlIllIIlllIlllll :
			if IlIIlllIllIlIIlIl .endswith (_IlllIlllIllllllIl )and _IlIIlIIlIllllIIII not in IlIIlllIllIlIIlIl :IllIIlllllllIlIII .append (_IllIIIllIIIllllII %(IlIlllllIIIIIIlll ,IlIIlllIllIlIIlIl ))
	return {_IIlllIllIllllIIlI :sorted (IIlIllIllIlIlIIlI ),_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },{_IIlllIllIllllIIlI :sorted (IllIIlllllllIlIII ),_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
def IlIlIIllIIlIIIlIl ():return {_IlIllIIIIllllllIl :'',_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
IlIlIlllIIIIIllIl ={_IIIlIIllIIllIIIll :32000 ,_IIlIlIIIllIIlIIll :40000 ,_IIIIllIIIIIlIlIll :48000 }
def IllIlIlllllllIlIl (IlIIlIllIlllIlIlI ,IlllIllIIlIlllIlI ):
	while 1 :
		if IlllIllIIlIlllIlI .poll ()is _IIllllIIIlIllIIIl :sleep (.5 )
		else :break 
	IlIIlIllIlllIlIlI [0 ]=_IIlIIIIllIlIIlIII 
def IIlIIIIlllIlIlIIl (IIIlIllllllIllIll ,IlllllllIllIIllll ):
	while 1 :
		IIlIlIIllIllIIIIl =1 
		for IlIlIlIIlllIIlIll in IlllllllIllIIllll :
			if IlIlIlIIlllIIlIll .poll ()is _IIllllIIIlIllIIIl :IIlIlIIllIllIIIIl =0 ;sleep (.5 );break 
		if IIlIlIIllIllIIIIl ==1 :break 
	IIIlIllllllIllIll [0 ]=_IIlIIIIllIlIIlIII 
def IllIIIlIlIlllIIll (IIlIlIllIlllIIIll ,IIllIIIlIlIlIlllI ,IlIIlIIllIIlIlIII ,IIIlIlIlIIllIlllI ):
	IIlIllIIIIlIIIlII ='%s/logs/%s/preprocess.log';IlIIlIIllIIlIlIII =IlIlIlllIIIIIllIl [IlIIlIIllIIlIlIII ];os .makedirs (_IlllIlIIlIIIllIIl %(IIlllllllIIlIIIII ,IIllIIIlIlIlIlllI ),exist_ok =_IIlIIIIllIlIIlIII );IIlIIIllllIIlIlIl =open (IIlIllIIIIlIIIlII %(IIlllllllIIlIIIII ,IIllIIIlIlIlIlllI ),'w');IIlIIIllllIIlIlIl .close ();IIIllIIlIlIIlIIll =IIlIIIIllIIlIlllI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IIlIlIllIlllIIIll ,IlIIlIIllIIlIlIII ,IIIlIlIlIIllIlllI ,IIlllllllIIlIIIII ,IIllIIIlIlIlIlllI )+str (IIlIIIIllIIlIlllI .noparallel );print (IIIllIIlIlIIlIIll );IllIllIIlIIIIIIlI =Popen (IIIllIIlIlIIlIIll ,shell =_IIlIIIIllIlIIlIII );IllIIlIIIlIIlIlIl =[_IllIlIllIlIlIllII ];threading .Thread (target =IllIlIlllllllIlIl ,args =(IllIIlIIIlIIlIlIl ,IllIllIIlIIIIIIlI )).start ()
	while 1 :
		with open (IIlIllIIIIlIIIlII %(IIlllllllIIlIIIII ,IIllIIIlIlIlIlllI ),_IIIlIlIIIIllIlIlI )as IIlIIIllllIIlIlIl :yield IIlIIIllllIIlIlIl .read ()
		sleep (1 )
		if IllIIlIIIlIIlIlIl [0 ]:break 
	with open (IIlIllIIIIlIIIlII %(IIlllllllIIlIIIII ,IIllIIIlIlIlIlllI ),_IIIlIlIIIIllIlIlI )as IIlIIIllllIIlIlIl :IIlIIIllIlIllllIl =IIlIIIllllIIlIlIl .read ()
	print (IIlIIIllIlIllllIl );yield IIlIIIllIlIllllIl 
def IIlIIIlllllllIIll (IIlllllIIIllIIllI ,IlIllllIlIlIlIIll ,IIIIlIIllIIlIIllI ,IIIIIllIIIllIlIlI ,IlIlllllIIlIllllI ,IIIIIlIIIllIIIIlI ,IIIlIlllIIlllllll ):
	IllIIIIlIlIlIIlll ='%s/logs/%s/extract_fl_feature.log';IIlllllIIIllIIllI =IIlllllIIIllIIllI .split ('-');os .makedirs (_IlllIlIIlIIIllIIl %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),exist_ok =_IIlIIIIllIlIIlIII );IIIlIllIllIIlllIl =open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),'w');IIIlIllIllIIlllIl .close ()
	if IIIIIllIIIllIlIlI :
		if IIIIlIIllIIlIIllI !=_IllIlIIIlIIllIlII :
			IlIIIllllllllllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ,IlIllllIlIlIlIIll ,IIIIlIIllIIlIIllI );print (IlIIIllllllllllIl );IllIIllIlllllIIII =Popen (IlIIIllllllllllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IlIIlIlIIllIIllIl =[_IllIlIllIlIlIllII ];threading .Thread (target =IllIlIlllllllIlIl ,args =(IlIIlIlIIllIIllIl ,IllIIllIlllllIIII )).start ()
			while 1 :
				with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :yield IIIlIllIllIIlllIl .read ()
				sleep (1 )
				if IlIIlIlIIllIIllIl [0 ]:break 
			with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :IIIlllIlIIlIllIIl =IIIlIllIllIIlllIl .read ()
			print (IIIlllIlIIlIllIIl );yield IIIlllIlIIlIllIIl 
		else :
			IIIlIlllIIlllllll =IIIlIlllIIlllllll .split ('-');IIIIIIIIllIlllIll =len (IIIlIlllIIlllllll );IlllIIIIIIIlIIllI =[]
			for (IllIlIIIIIIIIlIII ,IIlIIlIllIlIlIIII )in enumerate (IIIlIlllIIlllllll ):IlIIIllllllllllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IIIIIIIIllIlllIll ,IllIlIIIIIIIIlIII ,IIlIIlIllIlIlIIII ,IIlllllllIIlIIIII ,IlIlllllIIlIllllI ,IIlIIIIllIIlIlllI .is_half );print (IlIIIllllllllllIl );IllIIllIlllllIIII =Popen (IlIIIllllllllllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IlllIIIIIIIlIIllI .append (IllIIllIlllllIIII )
			IlIIlIlIIllIIllIl =[_IllIlIllIlIlIllII ];threading .Thread (target =IIlIIIIlllIlIlIIl ,args =(IlIIlIlIIllIIllIl ,IlllIIIIIIIlIIllI )).start ()
			while 1 :
				with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :yield IIIlIllIllIIlllIl .read ()
				sleep (1 )
				if IlIIlIlIIllIIllIl [0 ]:break 
			with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :IIIlllIlIIlIllIIl =IIIlIllIllIIlllIl .read ()
			print (IIIlllIlIIlIllIIl );yield IIIlllIlIIlIllIIl 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IIIIIIIIllIlllIll =len (IIlllllIIIllIIllI );IlllIIIIIIIlIIllI =[]
	for (IllIlIIIIIIIIlIII ,IIlIIlIllIlIlIIII )in enumerate (IIlllllIIIllIIllI ):IlIIIllllllllllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IIlIIIIllIIlIlllI .device ,IIIIIIIIllIlllIll ,IllIlIIIIIIIIlIII ,IIlIIlIllIlIlIIII ,IIlllllllIIlIIIII ,IlIlllllIIlIllllI ,IIIIIlIIIllIIIIlI );print (IlIIIllllllllllIl );IllIIllIlllllIIII =Popen (IlIIIllllllllllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IlllIIIIIIIlIIllI .append (IllIIllIlllllIIII )
	IlIIlIlIIllIIllIl =[_IllIlIllIlIlIllII ];threading .Thread (target =IIlIIIIlllIlIlIIl ,args =(IlIIlIlIIllIIllIl ,IlllIIIIIIIlIIllI )).start ()
	while 1 :
		with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :yield IIIlIllIllIIlllIl .read ()
		sleep (1 )
		if IlIIlIlIIllIIllIl [0 ]:break 
	with open (IllIIIIlIlIlIIlll %(IIlllllllIIlIIIII ,IlIlllllIIlIllllI ),_IIIlIlIIIIllIlIlI )as IIIlIllIllIIlllIl :IIIlllIlIIlIllIIl =IIIlIllIllIIlllIl .read ()
	print (IIIlllIlIIlIllIIl );yield IIIlllIlIIlIllIIl 
def IIIlIIllIIlIllllI (IllllIlllIlIlIIIl ,IIlIIIlIIIlllllIl ,IIllIllllIlIIllII ):
	IIIlIlIllIIIIlIII =''if IIllIllllIlIIllII ==_IIIlIIIllIIIIlIlI else _IIlIlIIlIllllIlIl ;IlIlIIlllllIIIIIl =_IlIIlIlIIlIllIlIl if IIlIIIlIIIlllllIl else '';IlllIIIIlllIIIlII =os .access (_IllIllllIIIlIlllI %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl ),os .F_OK );IIlIIIIlIlIlIIlll =os .access (_IllIIIIIlIIIIIlII %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl ),os .F_OK )
	if not IlllIIIIlllIIIlII :print (_IllIllllIIIlIlllI %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl ),_IIlllIllIlllIllll )
	if not IIlIIIIlIlIlIIlll :print (_IllIIIIIlIIIIIlII %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl ),_IIlllIllIlllIllll )
	return _IllIllllIIIlIlllI %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl )if IlllIIIIlllIIIlII else '',_IllIIIIIlIIIIIlII %(IIIlIlIllIIIIlIII ,IlIlIIlllllIIIIIl ,IllllIlllIlIlIIIl )if IIlIIIIlIlIlIIlll else ''
def IlIlllIIllIlIIlII (IIIlIllIIIIIlIIlI ,IIIlIIIlIIIllIIlI ,IllIIIIlllIIllIII ):
	IIIIIlllIIIIlllII =''if IllIIIIlllIIllIII ==_IIIlIIIllIIIIlIlI else _IIlIlIIlIllllIlIl 
	if IIIlIllIIIIIlIIlI ==_IIIlIIllIIllIIIll and IllIIIIlllIIllIII ==_IIIlIIIllIIIIlIlI :IIIlIllIIIIIlIIlI =_IIlIlIIIllIIlIIll 
	IlIllIllIllIlllII ={_IIlllIllIllllIIlI :[_IIlIlIIIllIIlIIll ,_IIIIllIIIIIlIlIll ],_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII ,_IlIllIIIIllllllIl :IIIlIllIIIIIlIIlI }if IllIIIIlllIIllIII ==_IIIlIIIllIIIIlIlI else {_IIlllIllIllllIIlI :[_IIlIlIIIllIIlIIll ,_IIIIllIIIIIlIlIll ,_IIIlIIllIIllIIIll ],_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII ,_IlIllIIIIllllllIl :IIIlIllIIIIIlIIlI };IllllIllIlIlIIIll =_IlIIlIlIIlIllIlIl if IIIlIIIlIIIllIIlI else '';IIIlIlIIIIllIIIII =os .access (_IllIllllIIIlIlllI %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI ),os .F_OK );IllIlIlIIlIIIlllI =os .access (_IllIIIIIlIIIIIlII %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI ),os .F_OK )
	if not IIIlIlIIIIllIIIII :print (_IllIllllIIIlIlllI %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI ),_IIlllIllIlllIllll )
	if not IllIlIlIIlIIIlllI :print (_IllIIIIIlIIIIIlII %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI ),_IIlllIllIlllIllll )
	return _IllIllllIIIlIlllI %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI )if IIIlIlIIIIllIIIII else '',_IllIIIIIlIIIIIlII %(IIIIIlllIIIIlllII ,IllllIllIlIlIIIll ,IIIlIllIIIIIlIIlI )if IllIlIlIIlIIIlllI else '',IlIllIllIllIlllII 
def IIIllIIllIlIIlIII (IllllIIIlIIlIIlIl ,IIllllIIIIIllIIll ,IlIlllIllIlllllII ):
	IIIlIlllIIIlIlIII ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IlIlIllIlIllIIlII ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IllIlIllIlIllIIlI =''if IlIlllIllIlllllII ==_IIIlIIIllIIIIlIlI else _IIlIlIIlIllllIlIl ;IIIIllIlllIlIIIlI =os .access (IlIlIllIlIllIIlII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll ),os .F_OK );IlllllIllllllIIll =os .access (IIIlIlllIIIlIlIII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll ),os .F_OK )
	if not IIIIllIlllIlIIIlI :print (IlIlIllIlIllIIlII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll ),_IIlllIllIlllIllll )
	if not IlllllIllllllIIll :print (IIIlIlllIIIlIlIII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll ),_IIlllIllIlllIllll )
	if IllllIIIlIIlIIlIl :return {_IIllIIIlIIIIIlllI :_IIlIIIIllIlIIlIII ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },IlIlIllIlIllIIlII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll )if IIIIllIlllIlIIIlI else '',IIIlIlllIIIlIlIII %(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll )if IlllllIllllllIIll else ''
	return {_IIllIIIlIIIIIlllI :_IllIlIllIlIlIllII ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll )if IIIIllIlllIlIIIlI else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IllIlIllIlIllIIlI ,IIllllIIIIIllIIll )if IlllllIllllllIIll else ''
def IllllllIlIlIlllII (IlIlllIlIlIIIIlIl ,IllIIlIIIIIIIllII ,IllIIIlIIllIIlIlI ,IIIlIIllllIlIIIll ,IIlIlIIIlIIIIIlII ,IIIIlIlIIlIlllIIl ,IlllIIIIIIIlIlIIl ,IIIlllIIIlllIIlIl ,IlllllIlllIIIIlIl ,IlIlIlllIIIlIIIll ,IIIlllIlIlIIIlIIl ,IIllIIlIlIIIIllII ,IllIllIIllIllIlll ,IllIIllIlIIIIIlIl ):
	IlIlIIIIlIIlIIlII ='\x08';IllIlIlIIIIlIIlIl =_IlllIlIIlIIIllIIl %(IIlllllllIIlIIIII ,IlIlllIlIlIIIIlIl );os .makedirs (IllIlIlIIIIlIIlIl ,exist_ok =_IIlIIIIllIlIIlIII );IIIIIIIIlIlllllIl =_IIIlIlllIlIlIIIll %IllIlIlIIIIlIIlIl ;IIIIlIllIIlIlIlll =_IIlIlIlIIlIlIlIll %IllIlIlIIIIlIIlIl if IllIIllIlIIIIIlIl ==_IIIlIIIllIIIIlIlI else _IIlIIIlIIIIlllIll %IllIlIlIIIIlIIlIl 
	if IllIIIlIIllIIlIlI :IlIIlllIlllIIlIIl ='%s/2a_f0'%IllIlIlIIIIlIIlIl ;IllIIlllllIIlIIIl =_IllIlIIllllIIllIl %IllIlIlIIIIlIIlIl ;IIllIIIlIIllIlIII =set ([IlIIlIllIIlIllIlI .split (_IlIlllIIIIlllIIII )[0 ]for IlIIlIllIIlIllIlI in os .listdir (IIIIIIIIlIlllllIl )])&set ([IIllIIlIlIllIllII .split (_IlIlllIIIIlllIIII )[0 ]for IIllIIlIlIllIllII in os .listdir (IIIIlIllIIlIlIlll )])&set ([IllIIIllIIIlIIIlI .split (_IlIlllIIIIlllIIII )[0 ]for IllIIIllIIIlIIIlI in os .listdir (IlIIlllIlllIIlIIl )])&set ([IlIIIlIlIlIlIIIIl .split (_IlIlllIIIIlllIIII )[0 ]for IlIIIlIlIlIlIIIIl in os .listdir (IllIIlllllIIlIIIl )])
	else :IIllIIIlIIllIlIII =set ([IllIIlIllllIllIII .split (_IlIlllIIIIlllIIII )[0 ]for IllIIlIllllIllIII in os .listdir (IIIIIIIIlIlllllIl )])&set ([IllIlIlIllIIlllIl .split (_IlIlllIIIIlllIIII )[0 ]for IllIlIlIllIIlllIl in os .listdir (IIIIlIllIIlIlIlll )])
	IIllIIlIIlIIIIIII =[]
	for IIlIlIllllIlIIIII in IIllIIIlIIllIlIII :
		if IllIIIlIIllIIlIlI :IIllIIlIIlIIIIIII .append (_IIIIlllIlIIlIIlII %(IIIIIIIIlIlllllIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IIIIlIllIIlIlIlll .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IlIIlllIlllIIlIIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IllIIlllllIIlIIIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IIIlIIllllIlIIIll ))
		else :IIllIIlIIlIIIIIII .append (_IllllIIlIlIlllIlI %(IIIIIIIIlIlllllIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IIIIlIllIIlIlIlll .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IIlIlIllllIlIIIII ,IIIlIIllllIlIIIll ))
	IIlllIIlIIIIIllll =256 if IllIIllIlIIIIIlIl ==_IIIlIIIllIIIIlIlI else 768 
	if IllIIIlIIllIIlIlI :
		for _IIIlIIIlIlIllIIII in range (2 ):IIllIIlIIlIIIIIII .append (_IIlIIIIIIllIlIlIl %(IIlllllllIIlIIIII ,IllIIlIIIIIIIllII ,IIlllllllIIlIIIII ,IIlllIIlIIIIIllll ,IIlllllllIIlIIIII ,IIlllllllIIlIIIII ,IIIlIIllllIlIIIll ))
	else :
		for _IIIlIIIlIlIllIIII in range (2 ):IIllIIlIIlIIIIIII .append (_IlIIIIIIIIIIIIIlI %(IIlllllllIIlIIIII ,IllIIlIIIIIIIllII ,IIlllllllIIlIIIII ,IIlllIIlIIIIIllll ,IIIlIIllllIlIIIll ))
	shuffle (IIllIIlIIlIIIIIII )
	with open (_IIlIIIIlIIlllIIlI %IllIlIlIIIIlIIlIl ,'w')as IIllIIIlllllIlllI :IIllIIIlllllIlllI .write (_IIllIIlIIlllIllll .join (IIllIIlIIlIIIIIII ))
	print (_IIIIIIIllllIIIllI );print ('use gpus:',IIIlllIlIlIIIlIIl )
	if IlllllIlllIIIIlIl =='':print ('no pretrained Generator')
	if IlIlIlllIIIlIIIll =='':print ('no pretrained Discriminator')
	if IIIlllIlIlIIIlIIl :IIIIlIIlIIIIIlllI =IIlIIIIllIIlIlllI .python_cmd +_IllllIIIlIlIlIllI %(IlIlllIlIlIIIIlIl ,IllIIlIIIIIIIllII ,1 if IllIIIlIIllIIlIlI else 0 ,IlllIIIIIIIlIlIIl ,IIIlllIlIlIIIlIIl ,IIIIlIlIIlIlllIIl ,IIlIlIIIlIIIIIlII ,_IIlIIIllIllIllIIl %IlllllIlllIIIIlIl if IlllllIlllIIIIlIl !=''else '',_IlIIllIIIlIlllllI %IlIlIlllIIIlIIIll if IlIlIlllIIIlIIIll !=''else '',1 if IIIlllIIIlllIIlIl ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IIllIIlIlIIIIllII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IllIllIIllIllIlll ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,IllIIllIlIIIIIlIl )
	else :IIIIlIIlIIIIIlllI =IIlIIIIllIIlIlllI .python_cmd +_IllllIlIllIllIIIl %(IlIlllIlIlIIIIlIl ,IllIIlIIIIIIIllII ,1 if IllIIIlIIllIIlIlI else 0 ,IlllIIIIIIIlIlIIl ,IIIIlIlIIlIlllIIl ,IIlIlIIIlIIIIIlII ,_IIlIIIllIllIllIIl %IlllllIlllIIIIlIl if IlllllIlllIIIIlIl !=''else IlIlIIIIlIIlIIlII ,_IlIIllIIIlIlllllI %IlIlIlllIIIlIIIll if IlIlIlllIIIlIIIll !=''else IlIlIIIIlIIlIIlII ,1 if IIIlllIIIlllIIlIl ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IIllIIlIlIIIIllII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IllIllIIllIllIlll ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,IllIIllIlIIIIIlIl )
	print (IIIIlIIlIIIIIlllI );IllIllIllIIIIIllI =Popen (IIIIlIIlIIIIIlllI ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IllIllIllIIIIIllI .wait ();return _IIllIIllllllIIIII 
def IIIIIIIIIIIlIIlll (IllIllIlIIIlIIlll ,IIIllIIIIllIlIlll ):
	IIlIIIIlIlIIllIll =_IlllIlIIlIIIllIIl %(IIlllllllIIlIIIII ,IllIllIlIIIlIIlll );os .makedirs (IIlIIIIlIlIIllIll ,exist_ok =_IIlIIIIllIlIIlIII );IIIIllIllIlIIlllI =_IIlIlIlIIlIlIlIll %IIlIIIIlIlIIllIll if IIIllIIIIllIlIlll ==_IIIlIIIllIIIIlIlI else _IIlIIIlIIIIlllIll %IIlIIIIlIlIIllIll 
	if not os .path .exists (IIIIllIllIlIIlllI ):return '请先进行特征提取!'
	IIlllllllllllllIl =list (os .listdir (IIIIllIllIlIIlllI ))
	if len (IIlllllllllllllIl )==0 :return '请先进行特征提取！'
	IIlIllIIlIIlIIIIl =[];IllIIlIIllIIllllI =[]
	for IIIIIIIIIlIIlIIIl in sorted (IIlllllllllllllIl ):IllIlIIllIllIIIll =np .load (_IllIIIllIIIllllII %(IIIIllIllIlIIlllI ,IIIIIIIIIlIIlIIIl ));IllIIlIIllIIllllI .append (IllIlIIllIllIIIll )
	IIIIlIlIIllllIIll =np .concatenate (IllIIlIIllIIllllI ,0 );IIlllIIlIIIIlIlll =np .arange (IIIIlIlIIllllIIll .shape [0 ]);np .random .shuffle (IIlllIIlIIIIlIlll );IIIIlIlIIllllIIll =IIIIlIlIIllllIIll [IIlllIIlIIIIlIlll ]
	if IIIIlIlIIllllIIll .shape [0 ]>2e5 :
		IIlIllIIlIIlIIIIl .append (_IlIIIIIIIIlIlIllI %IIIIlIlIIllllIIll .shape [0 ]);yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl )
		try :IIIIlIlIIllllIIll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIlIIIIllIlIIlIII ,batch_size =256 *IIlIIIIllIIlIlllI .n_cpu ,compute_labels =_IllIlIllIlIlIllII ,init ='random').fit (IIIIlIlIIllllIIll ).cluster_centers_ 
		except :IlIlIlllIlIlIlIIl =traceback .format_exc ();print (IlIlIlllIlIlIlIIl );IIlIllIIlIIlIIIIl .append (IlIlIlllIlIlIlIIl );yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl )
	np .save (_IlIlIIllllIIIllll %IIlIIIIlIlIIllIll ,IIIIlIlIIllllIIll );IIllIlllIlllIllII =min (int (16 *np .sqrt (IIIIlIlIIllllIIll .shape [0 ])),IIIIlIlIIllllIIll .shape [0 ]//39 );IIlIllIIlIIlIIIIl .append ('%s,%s'%(IIIIlIlIIllllIIll .shape ,IIllIlllIlllIllII ));yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl );IlIIlllIllIIIIlIl =faiss .index_factory (256 if IIIllIIIIllIlIlll ==_IIIlIIIllIIIIlIlI else 768 ,_IlIIlIllIIllIIIIl %IIllIlllIlllIllII );IIlIllIIlIIlIIIIl .append ('training');yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl );IlIIIlIlIlIlIIlIl =faiss .extract_index_ivf (IlIIlllIllIIIIlIl );IlIIIlIlIlIlIIlIl .nprobe =1 ;IlIIlllIllIIIIlIl .train (IIIIlIlIIllllIIll );faiss .write_index (IlIIlllIllIIIIlIl ,_IIIIllIIlIllIlllI %(IIlIIIIlIlIIllIll ,IIllIlllIlllIllII ,IlIIIlIlIlIlIIlIl .nprobe ,IllIllIlIIIlIIlll ,IIIllIIIIllIlIlll ));IIlIllIIlIIlIIIIl .append ('adding');yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl );IlIIIllIIIlIlIllI =8192 
	for IllIlIlIIlIllllll in range (0 ,IIIIlIlIIllllIIll .shape [0 ],IlIIIllIIIlIlIllI ):IlIIlllIllIIIIlIl .add (IIIIlIlIIllllIIll [IllIlIlIIlIllllll :IllIlIlIIlIllllll +IlIIIllIIIlIlIllI ])
	faiss .write_index (IlIIlllIllIIIIlIl ,_IIIIIllIlIlIIlIll %(IIlIIIIlIlIIllIll ,IIllIlllIlllIllII ,IlIIIlIlIlIlIIlIl .nprobe ,IllIllIlIIIlIIlll ,IIIllIIIIllIlIlll ));IIlIllIIlIIlIIIIl .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIllIlllIlllIllII ,IlIIIlIlIlIlIIlIl .nprobe ,IllIllIlIIIlIIlll ,IIIllIIIIllIlIlll ));yield _IIllIIlIIlllIllll .join (IIlIllIIlIIlIIIIl )
def IlllllIIllIIlIIIl (IIlllIIlIlIIIlIll ,IIllIlIllllIlIlII ,IlIIlllllIlIIllIl ,IllIIIlIIllIlIlll ,IIllllIlIlIllIlIl ,IIlllIIIlIIlIllll ,IlIllIIIIllIlIlII ,IlIIIlllIllIllllI ,IIlllIlIllIlllIlI ,IlIIlIlIlIIllIIll ,IIlIIIlIlIlllIIII ,IIlIllllIIIIIIIll ,IllIIIlllllIIlIII ,IIIlIlIIlIlIIlIIl ,IllIlIIlIIIlIlIII ,IIlIlIIIIIIlIIlII ,IlIIIlIIlIlIlIlIl ,IlIIIIIlIIlllIIll ):
	IIIIlllIllllIllIl =[]
	def IIlIIllIIlIIIlIll (IllIIIIIIllIlIllI ):IIIIlllIllllIllIl .append (IllIIIIIIllIlIllI );return _IIllIIlIIlllIllll .join (IIIIlllIllllIllIl )
	IIlllllllIlIIIlll =_IlllIlIIlIIIllIIl %(IIlllllllIIlIIIII ,IIlllIIlIlIIIlIll );IlllIlIIllllIllII ='%s/preprocess.log'%IIlllllllIlIIIlll ;IIIlIlllIIIIllllI ='%s/extract_fl_feature.log'%IIlllllllIlIIIlll ;IllllIIlIIlIIlIll =_IIIlIlllIlIlIIIll %IIlllllllIlIIIlll ;IIIlIIlIllllIIIIl =_IIlIlIlIIlIlIlIll %IIlllllllIlIIIlll if IlIIIlIIlIlIlIlIl ==_IIIlIIIllIIIIlIlI else _IIlIIIlIIIIlllIll %IIlllllllIlIIIlll ;os .makedirs (IIlllllllIlIIIlll ,exist_ok =_IIlIIIIllIlIIlIII );open (IlllIlIIllllIllII ,'w').close ();IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IllIIIlIIllIlIlll ,IlIlIlllIIIIIllIl [IIllIlIllllIlIlII ],IIlllIIIlIIlIllll ,IIlllllllIlIIIlll )+str (IIlIIIIllIIlIlllI .noparallel );yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI ('step1:正在处理数据'));yield IIlIIllIIlIIIlIll (IllllIIlIIlIlllIl );IllllllIlllIIlIII =Popen (IllllIIlIIlIlllIl ,shell =_IIlIIIIllIlIIlIII );IllllllIlllIIlIII .wait ()
	with open (IlllIlIIllllIllII ,_IIIlIlIIIIllIlIlI )as IlIIllIlIIIlIlIIl :print (IlIIllIlIIIlIlIIl .read ())
	open (IIIlIlllIIIIllllI ,'w')
	if IlIIlllllIlIIllIl :
		yield IIlIIllIIlIIIlIll ('step2a:正在提取音高')
		if IlIllIIIIllIlIlII !=_IllIlIIIlIIllIlII :IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_fl_print.py "%s" %s %s'%(IIlllllllIlIIIlll ,IIlllIIIlIIlIllll ,IlIllIIIIllIlIlII );yield IIlIIllIIlIIIlIll (IllllIIlIIlIlllIl );IllllllIlllIIlIII =Popen (IllllIIlIIlIlllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IllllllIlllIIlIII .wait ()
		else :
			IlIIIIIlIIlllIIll =IlIIIIIlIIlllIIll .split ('-');IlIIIIIIIllllIlIl =len (IlIIIIIlIIlllIIll );IIlIIIlIIIlIlIllI =[]
			for (IIlIlIlIIlIIlIIll ,IlIlllIlIllIIIIll )in enumerate (IlIIIIIlIIlllIIll ):IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IlIIIIIIIllllIlIl ,IIlIlIlIIlIIlIIll ,IlIlllIlIllIIIIll ,IIlllllllIlIIIlll ,IIlIIIIllIIlIlllI .is_half );yield IIlIIllIIlIIIlIll (IllllIIlIIlIlllIl );IllllllIlllIIlIII =Popen (IllllIIlIIlIlllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IIlIIIlIIIlIlIllI .append (IllllllIlllIIlIII )
			for IllllllIlllIIlIII in IIlIIIlIIIlIlIllI :IllllllIlllIIlIII .wait ()
		with open (IIIlIlllIIIIllllI ,_IIIlIlIIIIllIlIlI )as IlIIllIlIIIlIlIIl :print (IlIIllIlIIIlIlIIl .read ())
	else :yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI ('step2a:无需提取音高'))
	yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI ('step2b:正在提取特征'));IIllIlIIIlIIIIIlI =IIIlIlIIlIlIIlIIl .split ('-');IlIIIIIIIllllIlIl =len (IIllIlIIIlIIIIIlI );IIlIIIlIIIlIlIllI =[]
	for (IIlIlIlIIlIIlIIll ,IlIlllIlIllIIIIll )in enumerate (IIllIlIIIlIIIIIlI ):IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IIlIIIIllIIlIlllI .device ,IlIIIIIIIllllIlIl ,IIlIlIlIIlIIlIIll ,IlIlllIlIllIIIIll ,IIlllllllIlIIIlll ,IlIIIlIIlIlIlIlIl );yield IIlIIllIIlIIIlIll (IllllIIlIIlIlllIl );IllllllIlllIIlIII =Popen (IllllIIlIIlIlllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IIlIIIlIIIlIlIllI .append (IllllllIlllIIlIII )
	for IllllllIlllIIlIII in IIlIIIlIIIlIlIllI :IllllllIlllIIlIII .wait ()
	with open (IIIlIlllIIIIllllI ,_IIIlIlIIIIllIlIlI )as IlIIllIlIIIlIlIIl :print (IlIIllIlIIIlIlIIl .read ())
	yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI ('step3a:正在训练模型'))
	if IlIIlllllIlIIllIl :IIllIIlllIlllIIII ='%s/2a_f0'%IIlllllllIlIIIlll ;IlllIIllllllIIlII =_IllIlIIllllIIllIl %IIlllllllIlIIIlll ;IlIIlllllllIlllII =set ([IllIllIllllllllIl .split (_IlIlllIIIIlllIIII )[0 ]for IllIllIllllllllIl in os .listdir (IllllIIlIIlIIlIll )])&set ([IIIllIlllllllIlll .split (_IlIlllIIIIlllIIII )[0 ]for IIIllIlllllllIlll in os .listdir (IIIlIIlIllllIIIIl )])&set ([IllIllIIIIIIIIIll .split (_IlIlllIIIIlllIIII )[0 ]for IllIllIIIIIIIIIll in os .listdir (IIllIIlllIlllIIII )])&set ([IIIIIIlIlIllIlIIl .split (_IlIlllIIIIlllIIII )[0 ]for IIIIIIlIlIllIlIIl in os .listdir (IlllIIllllllIIlII )])
	else :IlIIlllllllIlllII =set ([IlIlIIlIIIlllIIIl .split (_IlIlllIIIIlllIIII )[0 ]for IlIlIIlIIIlllIIIl in os .listdir (IllllIIlIIlIIlIll )])&set ([IIlIllIlIIllllIll .split (_IlIlllIIIIlllIIII )[0 ]for IIlIllIlIIllllIll in os .listdir (IIIlIIlIllllIIIIl )])
	IlIIlIlIllllIlIII =[]
	for IlIIlIIIIIllllllI in IlIIlllllllIlllII :
		if IlIIlllllIlIIllIl :IlIIlIlIllllIlIII .append (_IIIIlllIlIIlIIlII %(IllllIIlIIlIIlIll .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IIIlIIlIllllIIIIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IIllIIlllIlllIIII .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IlllIIllllllIIlII .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IIllllIlIlIllIlIl ))
		else :IlIIlIlIllllIlIII .append (_IllllIIlIlIlllIlI %(IllllIIlIIlIIlIll .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IIIlIIlIllllIIIIl .replace (_IIIIllllllllIIlII ,_IlIIIllllllIIlIlI ),IlIIlIIIIIllllllI ,IIllllIlIlIllIlIl ))
	IlIllllIlllIllIll =256 if IlIIIlIIlIlIlIlIl ==_IIIlIIIllIIIIlIlI else 768 
	if IlIIlllllIlIIllIl :
		for _IlIIlIIIIIIIIIIll in range (2 ):IlIIlIlIllllIlIII .append (_IIlIIIIIIllIlIlIl %(IIlllllllIIlIIIII ,IIllIlIllllIlIlII ,IIlllllllIIlIIIII ,IlIllllIlllIllIll ,IIlllllllIIlIIIII ,IIlllllllIIlIIIII ,IIllllIlIlIllIlIl ))
	else :
		for _IlIIlIIIIIIIIIIll in range (2 ):IlIIlIlIllllIlIII .append (_IlIIIIIIIIIIIIIlI %(IIlllllllIIlIIIII ,IIllIlIllllIlIlII ,IIlllllllIIlIIIII ,IlIllllIlllIllIll ,IIllllIlIlIllIlIl ))
	shuffle (IlIIlIlIllllIlIII )
	with open (_IIlIIIIlIIlllIIlI %IIlllllllIlIIIlll ,'w')as IlIIllIlIIIlIlIIl :IlIIllIlIIIlIlIIl .write (_IIllIIlIIlllIllll .join (IlIIlIlIllllIlIII ))
	yield IIlIIllIIlIIIlIll (_IIIIIIIllllIIIllI )
	if IIIlIlIIlIlIIlIIl :IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +_IllllIIIlIlIlIllI %(IIlllIIlIlIIIlIll ,IIllIlIllllIlIlII ,1 if IlIIlllllIlIIllIl else 0 ,IlIIlIlIlIIllIIll ,IIIlIlIIlIlIIlIIl ,IIlllIlIllIlllIlI ,IlIIIlllIllIllllI ,_IIlIIIllIllIllIIl %IIlIllllIIIIIIIll if IIlIllllIIIIIIIll !=''else '',_IlIIllIIIlIlllllI %IllIIIlllllIIlIII if IllIIIlllllIIlIII !=''else '',1 if IIlIIIlIlIlllIIII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IllIlIIlIIIlIlIII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IIlIlIIIIIIlIIlII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,IlIIIlIIlIlIlIlIl )
	else :IllllIIlIIlIlllIl =IIlIIIIllIIlIlllI .python_cmd +_IllllIlIllIllIIIl %(IIlllIIlIlIIIlIll ,IIllIlIllllIlIlII ,1 if IlIIlllllIlIIllIl else 0 ,IlIIlIlIlIIllIIll ,IIlllIlIllIlllIlI ,IlIIIlllIllIllllI ,_IIlIIIllIllIllIIl %IIlIllllIIIIIIIll if IIlIllllIIIIIIIll !=''else '',_IlIIllIIIlIlllllI %IllIIIlllllIIlIII if IllIIIlllllIIlIII !=''else '',1 if IIlIIIlIlIlllIIII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IllIlIIlIIIlIlIII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,1 if IIlIlIIIIIIlIIlII ==IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl )else 0 ,IlIIIlIIlIlIlIlIl )
	yield IIlIIllIIlIIIlIll (IllllIIlIIlIlllIl );IllllllIlllIIlIII =Popen (IllllIIlIIlIlllIl ,shell =_IIlIIIIllIlIIlIII ,cwd =IIlllllllIIlIIIII );IllllllIlllIIlIII .wait ();yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI (_IIllIIllllllIIIII ));IllllllIllllIlIII =[];IIlllIllllIIllIll =list (os .listdir (IIIlIIlIllllIIIIl ))
	for IlIIlIIIIIllllllI in sorted (IIlllIllllIIllIll ):IIlIlllIlllIIIlIl =np .load (_IllIIIllIIIllllII %(IIIlIIlIllllIIIIl ,IlIIlIIIIIllllllI ));IllllllIllllIlIII .append (IIlIlllIlllIIIlIl )
	IlllllIIIlIllIIlI =np .concatenate (IllllllIllllIlIII ,0 );IIllIIIlIIlIIIIlI =np .arange (IlllllIIIlIllIIlI .shape [0 ]);np .random .shuffle (IIllIIIlIIlIIIIlI );IlllllIIIlIllIIlI =IlllllIIIlIllIIlI [IIllIIIlIIlIIIIlI ]
	if IlllllIIIlIllIIlI .shape [0 ]>2e5 :
		IlllllIlIIlllIlII =_IlIIIIIIIIlIlIllI %IlllllIIIlIllIIlI .shape [0 ];print (IlllllIlIIlllIlII );yield IIlIIllIIlIIIlIll (IlllllIlIIlllIlII )
		try :IlllllIIIlIllIIlI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIlIIIIllIlIIlIII ,batch_size =256 *IIlIIIIllIIlIlllI .n_cpu ,compute_labels =_IllIlIllIlIlIllII ,init ='random').fit (IlllllIIIlIllIIlI ).cluster_centers_ 
		except :IlllllIlIIlllIlII =traceback .format_exc ();print (IlllllIlIIlllIlII );yield IIlIIllIIlIIIlIll (IlllllIlIIlllIlII )
	np .save (_IlIlIIllllIIIllll %IIlllllllIlIIIlll ,IlllllIIIlIllIIlI );IlIIIlIIIllllIlIl =min (int (16 *np .sqrt (IlllllIIIlIllIIlI .shape [0 ])),IlllllIIIlIllIIlI .shape [0 ]//39 );yield IIlIIllIIlIIIlIll ('%s,%s'%(IlllllIIIlIllIIlI .shape ,IlIIIlIIIllllIlIl ));IllIlIlllIIlIlllI =faiss .index_factory (256 if IlIIIlIIlIlIlIlIl ==_IIIlIIIllIIIIlIlI else 768 ,_IlIIlIllIIllIIIIl %IlIIIlIIIllllIlIl );yield IIlIIllIIlIIIlIll ('training index');IllllIIIIIlllIIII =faiss .extract_index_ivf (IllIlIlllIIlIlllI );IllllIIIIIlllIIII .nprobe =1 ;IllIlIlllIIlIlllI .train (IlllllIIIlIllIIlI );faiss .write_index (IllIlIlllIIlIlllI ,_IIIIllIIlIllIlllI %(IIlllllllIlIIIlll ,IlIIIlIIIllllIlIl ,IllllIIIIIlllIIII .nprobe ,IIlllIIlIlIIIlIll ,IlIIIlIIlIlIlIlIl ));yield IIlIIllIIlIIIlIll ('adding index');IlllllIlllIIIlIIl =8192 
	for IllIlIIlllIIIllII in range (0 ,IlllllIIIlIllIIlI .shape [0 ],IlllllIlllIIIlIIl ):IllIlIlllIIlIlllI .add (IlllllIIIlIllIIlI [IllIlIIlllIIIllII :IllIlIIlllIIIllII +IlllllIlllIIIlIIl ])
	faiss .write_index (IllIlIlllIIlIlllI ,_IIIIIllIlIlIIlIll %(IIlllllllIlIIIlll ,IlIIIlIIIllllIlIl ,IllllIIIIIlllIIII .nprobe ,IIlllIIlIlIIIlIll ,IlIIIlIIlIlIlIlIl ));yield IIlIIllIIlIIIlIll ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIIlIIIllllIlIl ,IllllIIIIIlllIIII .nprobe ,IIlllIIlIlIIIlIll ,IlIIIlIIlIlIlIlIl ));yield IIlIIllIIlIIIlIll (IllIIIIIIlIllIIlI ('全流程结束！'))
def IlIIllIIlIIIIlIlI (IIIlIllIIIIlIllll ):
	IIlIlllIlIIllIlII ='train.log'
	if not os .path .exists (IIIlIllIIIIlIllll .replace (os .path .basename (IIIlIllIIIIlIllll ),IIlIlllIlIIllIlII )):return {_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },{_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },{_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
	try :
		with open (IIIlIllIIIIlIllll .replace (os .path .basename (IIIlIllIIIIlIllll ),IIlIlllIlIIllIlII ),_IIIlIlIIIIllIlIlI )as IllIIIlllIlIlIlII :IlIlIIllIlIIIIIII =eval (IllIIIlllIlIlIlII .read ().strip (_IIllIIlIIlllIllll ).split (_IIllIIlIIlllIllll )[0 ].split ('\t')[-1 ]);IIllIllllllIIllII ,IlIllIIIIlIIIIllI =IlIlIIllIlIIIIIII [_IIllIllIlllIlIlIl ],IlIlIIllIlIIIIIII ['if_f0'];IIIlllIllIIIIIlIl =_IIIlllIIIIIIlIlIl if _IIIllllllllIIlIII in IlIlIIllIlIIIIIII and IlIlIIllIlIIIIIII [_IIIllllllllIIlIII ]==_IIIlllIIIIIIlIlIl else _IIIlIIIllIIIIlIlI ;return IIllIllllllIIllII ,str (IlIllIIIIlIIIIllI ),IIIlllIllIIIIIlIl 
	except :traceback .print_exc ();return {_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },{_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII },{_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
def IlIIIIlIIIIIIlIIl (IIIllllIIIIIIlIIl ):
	if IIIllllIIIIIIlIIl ==_IllIlIIIlIIllIlII :IllllllIIlIIlIIll =_IIlIIIIllIlIIlIII 
	else :IllllllIIlIIlIIll =_IllIlIllIlIlIllII 
	return {_IIllIIIlIIIIIlllI :IllllllIIlIIlIIll ,_IIIIIlIIlllIlllII :_IIIlIlllIIlllIIII }
def IIlIlllIIlIlllllI (IIIllIIlllIlIIlII ,IlIlIIlllllIIIlII ):IIlIIlIlllIIIIlII ='rnd';IllllIIIIIlllIIlI ='pitchf';IIllIIIIllIIIIlll ='pitch';IlllIIIIlllIIlIIl ='phone';global IIIIllllllIIllIIl ;IIIIllllllIIllIIl =torch .load (IIIllIIlllIlIIlII ,map_location =_IIlIIIIIllIIlllII );IIIIllllllIIllIIl [_IlIllIIllIIllIllI ][-3 ]=IIIIllllllIIllIIl [_IlIlIlIIIIIlIlIll ][_IIIIllIlllIllllll ].shape [0 ];IIIIlIIlIlIllllll =256 if IIIIllllllIIllIIl .get (_IIIllllllllIIlIII ,_IIIlIIIllIIIIlIlI )==_IIIlIIIllIIIIlIlI else 768 ;IlIlIllIlllIIIllI =torch .rand (1 ,200 ,IIIIlIIlIlIllllll );IIlIlIllIIlIlllll =torch .tensor ([200 ]).long ();IIllIIllllIlIllll =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IlIIlIIlIIlllIlll =torch .rand (1 ,200 );IIIIIIlIlIIlIllll =torch .LongTensor ([0 ]);IIIlIllIllIlIIIll =torch .rand (1 ,192 ,200 );IllIIllIlllIlllIl =_IIlIIIIIllIIlllII ;IllIlIllIIIlllllI =SynthesizerTrnMsNSFsidM (*IIIIllllllIIllIIl [_IlIllIIllIIllIllI ],is_half =_IllIlIllIlIlIllII ,version =IIIIllllllIIllIIl .get (_IIIllllllllIIlIII ,_IIIlIIIllIIIIlIlI ));IllIlIllIIIlllllI .load_state_dict (IIIIllllllIIllIIl [_IlIlIlIIIIIlIlIll ],strict =_IllIlIllIlIlIllII );IllllIIIIIIIlllIl =[IlllIIIIlllIIlIIl ,'phone_lengths',IIllIIIIllIIIIlll ,IllllIIIIIlllIIlI ,'ds',IIlIIlIlllIIIIlII ];IIIIlIIlIIIIlllll =['audio'];torch .onnx .export (IllIlIllIIIlllllI ,(IlIlIllIlllIIIllI .to (IllIIllIlllIlllIl ),IIlIlIllIIlIlllll .to (IllIIllIlllIlllIl ),IIllIIllllIlIllll .to (IllIIllIlllIlllIl ),IlIIlIIlIIlllIlll .to (IllIIllIlllIlllIl ),IIIIIIlIlIIlIllll .to (IllIIllIlllIlllIl ),IIIlIllIllIlIIIll .to (IllIIllIlllIlllIl )),IlIlIIlllllIIIlII ,dynamic_axes ={IlllIIIIlllIIlIIl :[1 ],IIllIIIIllIIIIlll :[1 ],IllllIIIIIlllIIlI :[1 ],IIlIIlIlllIIIIlII :[2 ]},do_constant_folding =_IllIlIllIlIlIllII ,opset_version =13 ,verbose =_IllIlIllIlIlIllII ,input_names =IllllIIIIIIIlllIl ,output_names =IIIIlIIlIIIIlllll );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IIllIIlIIIlIlIlIl :
	gr .Markdown (value =IllIIIIIIlIllIIlI ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IllIIIIIIlIllIIlI ('模型推理')):
			with gr .Row ():IllIlIlIIIlIIllIl =gr .Dropdown (label =IllIIIIIIlIllIIlI ('推理音色'),choices =sorted (IlIIlllllIIlIllII ));IlIllIlIlllllllII =gr .Button (IllIIIIIIlIllIIlI ('刷新音色列表和索引路径'),variant =_IIlIlIlIlIlllIlll );IIIlIIlIlIlllllll =gr .Button (IllIIIIIIlIllIIlI ('卸载音色省显存'),variant =_IIlIlIlIlIlllIlll );IllIllllIlllllIII =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IllIIIIIIlIllIIlI ('请选择说话人id'),value =0 ,visible =_IllIlIllIlIlIllII ,interactive =_IIlIIIIllIlIIlIII );IIIlIIlIlIlllllll .click (fn =IlIlIIllIIlIIIlIl ,inputs =[],outputs =[IllIlIlIIIlIIllIl ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IIIIlIIlllIIIIIIl =gr .Number (label =IllIIIIIIlIllIIlI (_IlIIIIIlIIllIIIIl ),value =0 );IIlIlIlIllIIlIIlI =gr .Textbox (label =IllIIIIIIlIllIIlI ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IllIlIllIIlIlIlll =gr .Radio (label =IllIIIIIIlIllIIlI (_IlIlIlIllIlIIIllI ),choices =[_IIlIIllIIIIlIIIll ,_IllIlIIIIlIIIIlII ,'crepe',_IlllIIllIIlllllll ],value =_IIlIIllIIIIlIIIll ,interactive =_IIlIIIIllIlIIlIII );IIIlIIlllIlIlIlII =gr .Slider (minimum =0 ,maximum =7 ,label =IllIIIIIIlIllIIlI (_IlIllIlIlllIlllIl ),value =3 ,step =1 ,interactive =_IIlIIIIllIlIIlIII )
					with gr .Column ():IlllIlllIIllIlIIl =gr .Textbox (label =IllIIIIIIlIllIIlI (_IlIlIIIIIlIlIllIl ),value ='',interactive =_IIlIIIIllIlIIlIII );IIllllllllIlIIlII =gr .Dropdown (label =IllIIIIIIlIllIIlI (_IIllIlIIIIllIlIlI ),choices =sorted (IIllIIIIllIllIIlI ),interactive =_IIlIIIIllIlIIlIII );IlIllIlIlllllllII .click (fn =IlIIIlllIllllIllI ,inputs =[],outputs =[IllIlIlIIIlIIllIl ,IIllllllllIlIIlII ],api_name ='infer_refresh');IlIIllIIlllllIlIl =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIIIIIlIllIIlI ('检索特征占比'),value =.75 ,interactive =_IIlIIIIllIlIIlIII )
					with gr .Column ():IlIIlIIlllllIIIll =gr .Slider (minimum =0 ,maximum =48000 ,label =IllIIIIIIlIllIIlI (_IlIlllllIIllIlIII ),value =0 ,step =1 ,interactive =_IIlIIIIllIlIIlIII );IIlllIIIIIllllIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIIIIIlIllIIlI (_IIlIlIlIIlIlIlIIl ),value =.25 ,interactive =_IIlIIIIllIlIIlIII );IllIlIIIlllIIIlIl =gr .Slider (minimum =0 ,maximum =.5 ,label =IllIIIIIIlIllIIlI (_IIllIllllIlllIllI ),value =.33 ,step =.01 ,interactive =_IIlIIIIllIlIIlIII )
					IllIIllIIIllllIll =gr .File (label =IllIIIIIIlIllIIlI ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IIllIllIllllllIII =gr .Button (IllIIIIIIlIllIIlI ('转换'),variant =_IIlIlIlIlIlllIlll )
					with gr .Row ():IllIIIlIlllIlllII =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ));IlIllIllllIIllIlI =gr .Audio (label =IllIIIIIIlIllIIlI ('输出音频(右下角三个点,点了可以下载)'))
					IIllIllIllllllIII .click (IIllIlIlIllIllIll ,[IllIllllIlllllIII ,IIlIlIlIllIIlIIlI ,IIIIlIIlllIIIIIIl ,IllIIllIIIllllIll ,IllIlIllIIlIlIlll ,IlllIlllIIllIlIIl ,IIllllllllIlIIlII ,IlIIllIIlllllIlIl ,IIIlIIlllIlIlIlII ,IlIIlIIlllllIIIll ,IIlllIIIIIllllIIl ,IllIlIIIlllIIIlIl ],[IllIIIlIlllIlllII ,IlIllIllllIIllIlI ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IlIlIllllIlllIIll =gr .Number (label =IllIIIIIIlIllIIlI (_IlIIIIIlIIllIIIIl ),value =0 );IlIlIIlIlIIIlIllI =gr .Textbox (label =IllIIIIIIlIllIIlI ('指定输出文件夹'),value =_IllIllIIlllIlIllI );IllIllllIIIIlIlll =gr .Radio (label =IllIIIIIIlIllIIlI (_IlIlIlIllIlIIIllI ),choices =[_IIlIIllIIIIlIIIll ,_IllIlIIIIlIIIIlII ,'crepe',_IlllIIllIIlllllll ],value =_IIlIIllIIIIlIIIll ,interactive =_IIlIIIIllIlIIlIII );IIlIlIlIIlIIlllIl =gr .Slider (minimum =0 ,maximum =7 ,label =IllIIIIIIlIllIIlI (_IlIllIlIlllIlllIl ),value =3 ,step =1 ,interactive =_IIlIIIIllIlIIlIII )
					with gr .Column ():IIlIlllllIllIllIl =gr .Textbox (label =IllIIIIIIlIllIIlI (_IlIlIIIIIlIlIllIl ),value ='',interactive =_IIlIIIIllIlIIlIII );IIllIlIllIIllIlII =gr .Dropdown (label =IllIIIIIIlIllIIlI (_IIllIlIIIIllIlIlI ),choices =sorted (IIllIIIIllIllIIlI ),interactive =_IIlIIIIllIlIIlIII );IlIllIlIlllllllII .click (fn =lambda :IlIIIlllIllllIllI ()[1 ],inputs =[],outputs =IIllIlIllIIllIlII ,api_name ='infer_refresh_batch');IIlllIllIIIIIIllI =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIIIIIlIllIIlI ('检索特征占比'),value =1 ,interactive =_IIlIIIIllIlIIlIII )
					with gr .Column ():IIllllIllIIllllll =gr .Slider (minimum =0 ,maximum =48000 ,label =IllIIIIIIlIllIIlI (_IlIlllllIIllIlIII ),value =0 ,step =1 ,interactive =_IIlIIIIllIlIIlIII );IIlIIlIllllIlIlll =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIIIIIlIllIIlI (_IIlIlIlIIlIlIlIIl ),value =1 ,interactive =_IIlIIIIllIlIIlIII );IIIlIIllllIIIIIII =gr .Slider (minimum =0 ,maximum =.5 ,label =IllIIIIIIlIllIIlI (_IIllIllllIlllIllI ),value =.33 ,step =.01 ,interactive =_IIlIIIIllIlIIlIII )
					with gr .Column ():IllIIlllIlIllllll =gr .Textbox (label =IllIIIIIIlIllIIlI ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IIlIIIlllllIlIlIl =gr .File (file_count ='multiple',label =IllIIIIIIlIllIIlI (_IIlllIllIllllIIll ))
					with gr .Row ():IIIllIlIlIIIIIIIl =gr .Radio (label =IllIIIIIIlIllIIlI ('导出文件格式'),choices =[_IllIlllIIIlIIIIIl ,_IlllIIIlllIlIIIII ,'mp3','m4a'],value =_IlllIIIlllIlIIIII ,interactive =_IIlIIIIllIlIIlIII );IlIIllIlIIIllIIII =gr .Button (IllIIIIIIlIllIIlI ('转换'),variant =_IIlIlIlIlIlllIlll );IlIlIIIlIIIIIIIll =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ))
					IlIIllIlIIIllIIII .click (IlIlllIIIIlIIlllI ,[IllIllllIlllllIII ,IllIIlllIlIllllll ,IlIlIIlIlIIIlIllI ,IIlIIIlllllIlIlIl ,IlIlIllllIlllIIll ,IllIllllIIIIlIlll ,IIlIlllllIllIllIl ,IIllIlIllIIllIlII ,IIlllIllIIIIIIllI ,IIlIlIlIIlIIlllIl ,IIllllIllIIllllll ,IIlIIlIllllIlIlll ,IIIlIIllllIIIIIII ,IIIllIlIlIIIIIIIl ],[IlIlIIIlIIIIIIIll ],api_name ='infer_convert_batch')
			IllIlIlIIIlIIllIl .change (fn =IIlIlIlllIllIIlII ,inputs =[IllIlIlIIIlIIllIl ,IllIlIIIlllIIIlIl ,IIIlIIllllIIIIIII ],outputs =[IllIllllIlllllIII ,IllIlIIIlllIIIlIl ,IIIlIIllllIIIIIII ,IIllllllllIlIIlII ])
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IIIllIlIIIllllllI =gr .Textbox (label =IllIIIIIIlIllIIlI ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IIIIllIIIIIlIlIIl =gr .File (file_count ='multiple',label =IllIIIIIIlIllIIlI (_IIlllIllIllllIIll ))
					with gr .Column ():IllllIlIIIIlIlIlI =gr .Dropdown (label =IllIIIIIIlIllIIlI ('模型'),choices =IlIlIIIIlIllIIlll );IIIIllllIIIIIIIlI =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IIlIIIIllIlIIlIII ,visible =_IllIlIllIlIlIllII );IlIIlIlIlIIIIIIlI =gr .Textbox (label =IllIIIIIIlIllIIlI ('指定输出主人声文件夹'),value =_IllIllIIlllIlIllI );IlllIllllIIIllIII =gr .Textbox (label =IllIIIIIIlIllIIlI ('指定输出非主人声文件夹'),value =_IllIllIIlllIlIllI );IIlIllIllIlIlllII =gr .Radio (label =IllIIIIIIlIllIIlI ('导出文件格式'),choices =[_IllIlllIIIlIIIIIl ,_IlllIIIlllIlIIIII ,'mp3','m4a'],value =_IlllIIIlllIlIIIII ,interactive =_IIlIIIIllIlIIlIII )
					IlIIlllIllIIlIIlI =gr .Button (IllIIIIIIlIllIIlI ('转换'),variant =_IIlIlIlIlIlllIlll );IIIIllIlllllIIIll =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ));IlIIlllIllIIlIIlI .click (IIIlllIIIIIlIIIII ,[IllllIlIIIIlIlIlI ,IIIllIlIIIllllllI ,IlIIlIlIlIIIIIIlI ,IIIIllIIIIIlIlIIl ,IlllIllllIIIllIII ,IIIIllllIIIIIIIlI ,IIlIllIllIlIlllII ],[IIIIllIlllllIIIll ],api_name ='uvr_convert')
		with gr .TabItem (IllIIIIIIlIllIIlI ('训练')):
			gr .Markdown (value =IllIIIIIIlIllIIlI ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IlIlIIIIIIlIIIlll =gr .Textbox (label =IllIIIIIIlIllIIlI ('输入实验名'),value ='mi-test');IIllIIIIIlIIllIlI =gr .Radio (label =IllIIIIIIlIllIIlI ('目标采样率'),choices =[_IIlIlIIIllIIlIIll ],value =_IIlIlIIIllIIlIIll ,interactive =_IIlIIIIllIlIIlIII );IlIlIllIIIllIlIII =gr .Radio (label =IllIIIIIIlIllIIlI ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IIlIIIIllIlIIlIII ,_IllIlIllIlIlIllII ],value =_IIlIIIIllIlIIlIII ,interactive =_IIlIIIIllIlIIlIII );IlIllllIIllIIIlll =gr .Radio (label =IllIIIIIIlIllIIlI ('版本'),choices =[_IIIlllIIIIIIlIlIl ],value =_IIIlllIIIIIIlIlIl ,interactive =_IIlIIIIllIlIIlIII ,visible =_IIlIIIIllIlIIlIII );IlllllIIIIIIlIIll =gr .Slider (minimum =0 ,maximum =IIlIIIIllIIlIlllI .n_cpu ,step =1 ,label =IllIIIIIIlIllIIlI ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IIlIIIIllIIlIlllI .n_cpu /1.5 )),interactive =_IIlIIIIllIlIIlIII )
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IllIllIlIIIIIIlll =gr .Textbox (label =IllIIIIIIlIllIIlI ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IIIlIIIIlIIIIlIll =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IllIIIIIIlIllIIlI ('请指定说话人id'),value =0 ,interactive =_IIlIIIIllIlIIlIII );IlIIllIlIIIllIIII =gr .Button (IllIIIIIIlIllIIlI ('处理数据'),variant =_IIlIlIlIlIlllIlll );IIIIllllIlIllIIII =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ),value ='');IlIIllIlIIIllIIII .click (IllIIIlIlIlllIIll ,[IllIllIlIIIIIIlll ,IlIlIIIIIIlIIIlll ,IIllIIIIIlIIllIlI ,IlllllIIIIIIlIIll ],[IIIIllllIlIllIIII ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IllIIlllIlIlIlIII =gr .Textbox (label =IllIIIIIIlIllIIlI (_IlllIllIIlIllIlll ),value =IIlIllllIllIlIIIl ,interactive =_IIlIIIIllIlIIlIII );IllllIllIIIIIIlll =gr .Textbox (label =IllIIIIIIlIllIIlI ('显卡信息'),value =IlIlIIlIIIlIIlIll )
					with gr .Column ():IlIllllIIIIllIlII =gr .Radio (label =IllIIIIIIlIllIIlI ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IIlIIllIIIIlIIIll ,_IllIlIIIIlIIIIlII ,'dio',_IlllIIllIIlllllll ,_IllIlIIIlIIllIlII ],value =_IllIlIIIlIIllIlII ,interactive =_IIlIIIIllIlIIlIII );IIIlIIlllIIlIlIII =gr .Textbox (label =IllIIIIIIlIllIIlI ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IIlIllllIllIlIIIl ,IIlIllllIllIlIIIl ),interactive =_IIlIIIIllIlIIlIII ,visible =_IIlIIIIllIlIIlIII )
					IlIIlllIllIIlIIlI =gr .Button (IllIIIIIIlIllIIlI ('特征提取'),variant =_IIlIlIlIlIlllIlll );IIIllIlllIlllIlll =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ),value ='',max_lines =8 );IlIllllIIIIllIlII .change (fn =IlIIIIlIIIIIIlIIl ,inputs =[IlIllllIIIIllIlII ],outputs =[IIIlIIlllIIlIlIII ]);IlIIlllIllIIlIIlI .click (IIlIIIlllllllIIll ,[IllIIlllIlIlIlIII ,IlllllIIIIIIlIIll ,IlIllllIIIIllIlII ,IlIlIllIIIllIlIII ,IlIlIIIIIIlIIIlll ,IlIllllIIllIIIlll ,IIIlIIlllIIlIlIII ],[IIIllIlllIlllIlll ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IllIIIIIIlIllIIlI ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IllIIlIlIIIIIIlll =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IllIIIIIIlIllIIlI ('保存频率save_every_epoch'),value =5 ,interactive =_IIlIIIIllIlIIlIII );IllIIlIlllIllIlII =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IllIIIIIIlIllIIlI ('总训练轮数total_epoch'),value =300 ,interactive =_IIlIIIIllIlIIlIII );IllIlIlIIIIIlIllI =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IllIIIIIIlIllIIlI ('每张显卡的batch_size'),value =IllIIlllIIIlllllI ,interactive =_IIlIIIIllIlIIlIII );IlIIllIlIlllIIlll =gr .Radio (label =IllIIIIIIlIllIIlI ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl ),IllIIIIIIlIllIIlI ('否')],value =IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl ),interactive =_IIlIIIIllIlIIlIII );IIllIlIIIIIIlIlIl =gr .Radio (label =IllIIIIIIlIllIIlI ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl ),IllIIIIIIlIllIIlI ('否')],value =IllIIIIIIlIllIIlI ('否'),interactive =_IIlIIIIllIlIIlIII );IlIllllllIIIlIlIl =gr .Radio (label =IllIIIIIIlIllIIlI ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl ),IllIIIIIIlIllIIlI ('否')],value =IllIIIIIIlIllIIlI (_IIllIllIIllIlIIIl ),interactive =_IIlIIIIllIlIIlIII )
				with gr .Row ():IlllIlllIlIllIIII =gr .Textbox (label =IllIIIIIIlIllIIlI ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IIlIIIIllIlIIlIII );IlIIIlIIlIlIIIIlI =gr .Textbox (label =IllIIIIIIlIllIIlI ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IIlIIIIllIlIIlIII );IIllIIIIIlIIllIlI .change (IIIlIIllIIlIllllI ,[IIllIIIIIlIIllIlI ,IlIlIllIIIllIlIII ,IlIllllIIllIIIlll ],[IlllIlllIlIllIIII ,IlIIIlIIlIlIIIIlI ]);IlIllllIIllIIIlll .change (IlIlllIIllIlIIlII ,[IIllIIIIIlIIllIlI ,IlIlIllIIIllIlIII ,IlIllllIIllIIIlll ],[IlllIlllIlIllIIII ,IlIIIlIIlIlIIIIlI ,IIllIIIIIlIIllIlI ]);IlIlIllIIIllIlIII .change (IIIllIIllIlIIlIII ,[IlIlIllIIIllIlIII ,IIllIIIIIlIIllIlI ,IlIllllIIllIIIlll ],[IlIllllIIIIllIlII ,IlllIlllIlIllIIII ,IlIIIlIIlIlIIIIlI ]);IllllllllIlllIlll =gr .Textbox (label =IllIIIIIIlIllIIlI (_IlllIllIIlIllIlll ),value =IIlIllllIllIlIIIl ,interactive =_IIlIIIIllIlIIlIII );IIllllIllIIlIIIII =gr .Button (IllIIIIIIlIllIIlI ('训练模型'),variant =_IIlIlIlIlIlllIlll );IlIlIIlIlIIllIllI =gr .Button (IllIIIIIIlIllIIlI ('训练特征索引'),variant =_IIlIlIlIlIlllIlll );IIIIlIllIlllIllIl =gr .Button (IllIIIIIIlIllIIlI ('一键训练'),variant =_IIlIlIlIlIlllIlll );IllIIlIIIIllllllI =gr .Textbox (label =IllIIIIIIlIllIIlI (_IIlllIIIIlIlIIIIl ),value ='',max_lines =10 );IIllllIllIIlIIIII .click (IllllllIlIlIlllII ,[IlIlIIIIIIlIIIlll ,IIllIIIIIlIIllIlI ,IlIlIllIIIllIlIII ,IIIlIIIIlIIIIlIll ,IllIIlIlIIIIIIlll ,IllIIlIlllIllIlII ,IllIlIlIIIIIlIllI ,IlIIllIlIlllIIlll ,IlllIlllIlIllIIII ,IlIIIlIIlIlIIIIlI ,IllllllllIlllIlll ,IIllIlIIIIIIlIlIl ,IlIllllllIIIlIlIl ,IlIllllIIllIIIlll ],IllIIlIIIIllllllI ,api_name ='train_start');IlIlIIlIlIIllIllI .click (IIIIIIIIIIIlIIlll ,[IlIlIIIIIIlIIIlll ,IlIllllIIllIIIlll ],IllIIlIIIIllllllI );IIIIlIllIlllIllIl .click (IlllllIIllIIlIIIl ,[IlIlIIIIIIlIIIlll ,IIllIIIIIlIIllIlI ,IlIlIllIIIllIlIII ,IllIllIlIIIIIIlll ,IIIlIIIIlIIIIlIll ,IlllllIIIIIIlIIll ,IlIllllIIIIllIlII ,IllIIlIlIIIIIIlll ,IllIIlIlllIllIlII ,IllIlIlIIIIIlIllI ,IlIIllIlIlllIIlll ,IlllIlllIlIllIIII ,IlIIIlIIlIlIIIIlI ,IllllllllIlllIlll ,IIllIlIIIIIIlIlIl ,IlIllllllIIIlIlIl ,IlIllllIIllIIIlll ,IIIlIIlllIIlIlIII ],IllIIlIIIIllllllI ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IIIlIlIIIIllIlIlI ,encoding ='utf8')as IllIIIIIIlIlIlIll :IIlIIIIIllIIIlIll =IllIIIIIIlIlIlIll .read ()
				else :
					with open ('docs/faq_en.md',_IIIlIlIIIIllIlIlI ,encoding ='utf8')as IllIIIIIIlIlIlIll :IIlIIIIIllIIIlIll =IllIIIIIIlIlIlIll .read ()
				gr .Markdown (value =IIlIIIIIllIIIlIll )
			except :gr .Markdown (traceback .format_exc ())
	if IIlIIIIllIIlIlllI .iscolab :IIllIIlIIIlIlIlIl .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IIlIIIIllIIlIlllI .listen_port ,share =_IllIlIllIlIlIllII )
	else :IIllIIlIIIlIlIlIl .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IIlIIIIllIIlIlllI .noautoopen ,server_port =IIlIIIIllIIlIlllI .listen_port ,quiet =_IllIlIllIlIlIllII ,share =_IllIlIllIlIlIllII )