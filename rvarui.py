_IlIIIllIIlIIIIllI ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IllIIIlllIlllllll ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IIllIllllllIllllI ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IllIlIlIIllIIlIII ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IIlllllllllllIlII ='后处理重采样至最终采样率，0为不进行重采样'
_IlllllIllIllIllIl ='自动检测index路径,下拉式选择(dropdown)'
_IlIIlIIIIlIlllIII ='特征检索库文件路径,为空则使用下拉的选择结果'
_IIllIIIlIIlIllIII ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IIlIllIIlIIllllll ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IIlIIlllIllIlIIlI ='变调(整数, 半音数量, 升八度12降八度-12)'
_IIlIIlllllIlIlIII ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IllllIIIIlIlIlllI ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIIIIllllIIIlIIl ='IVF%s,Flat'
_IlIIlIIlIlIIlIlIl ='%s/total_fea.npy'
_IIIIlIIlIIlllIlII ='Trying doing kmeans %s shape to 10k centers.'
_IlllIIlIlIIlIlIll ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IlIlIllIIIllIllll =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIIllllIlIllIIlll =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IlIIIIlIIlIIllIIl ='write filelist done'
_IllllIllIIlIlIIlI ='%s/filelist.txt'
_IIlllIlIlIlllIllI ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IllllIlIIlIIlIIll ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IIIlIIIllIlllIIIl ='%s/%s.wav|%s/%s.npy|%s'
_IIlllIlllIlllIIIl ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IllllIIllllIIllIl ='%s/2b-f0nsf'
_IIlIllIIIIllIlIII ='%s/0_gt_wavs'
_IlllllIIIIIIIIlIl ='emb_g.weight'
_IlIllIIlllIlllIIl ='clean_empty_cache'
_IlIIllIIlIIlIllII ='sample_rate'
_IIllIllIIIlIllIIl ='%s->%s'
_IIIlIIIlIlllIIIIl ='.index'
_IllIlIllIlIIIIIll ='weights'
_IlIIIIlllIIIllIII ='opt'
_IIIIlIlllIllIIIIl ='rmvpe'
_IIllIlIlIIIIIllIl ='harvest'
_IIllIlllIIlIIIIIl ='%s/3_feature768'
_IIllIlIlIIlllIlII ='%s/3_feature256'
_IIllIIllIIIIlIlII ='_v2'
_IlIIlllIlIlIIlIll ='48k'
_IllIlIIllIIIlIlIl ='32k'
_IlIlIIlIllIllIIll ='cpu'
_IIIIIIIlIIIlllIIl ='wav'
_IllllIlIIIIIIIlII ='trained'
_IlllIIIIIlIllIIll ='logs'
_IIIlIIIllIIIlIllI ='-pd %s'
_IlIlllIIIllIllIll ='-pg %s'
_IIIlIlIlIIllIllIl ='choices'
_IIIIIllIIIIIIIlll ='weight'
_IlIIlIllIllIlIllI ='pm'
_IIlIllllIlIlIlIll ='rmvpe_gpu'
_IIllIIlllllIIIllI ='%s/logs/%s'
_IlllIlllIIIIllIlI ='flac'
_IIllIllIlllIIllIl ='f0'
_IIlllIIlIIlIIllll ='%s/%s'
_IlIIIIllllIIIIlII ='.pth'
_IIIIllllllIlllIlI ='输出信息'
_IlIlllIllIIlIIlII ='not exist, will not use pretrained model'
_IllIIIIllllIlllll ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IllllIlIIlIIIIIlI ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IIIllIIlIlIllIIll ='40k'
_IIllIlIlIIIlIlIII ='value'
_IIIlllllllIllIIll ='v2'
_IIIIlIIIlIlllIlll ='version'
_IIIIlIlIlllllIlII ='visible'
_IIIllIlIIIlllllll ='primary'
_IllIIIIIllIIIIIlI =None 
_IllIIIlIlllllIlll ='\\\\'
_IIlIllllIIllllIll ='\\'
_IIlIllllIlllIIIIl ='"'
_IIIlIllIIIIlIllll =' '
_IlllllllIlllllllI ='config'
_IlIIIIllIIIlIllll ='.'
_IllIIIIlllllIllIl ='r'
_IlllIlllIlllIllIl ='是'
_IIlIlIIIIllIIIIll ='update'
_IlIIIllIlIlIIlllI ='__type__'
_IlllllIlIllIIllII ='v1'
_IIlIlIIIlIIlIIIll ='\n'
_IIIIlIllllIIIlllI =False 
_IlIlIIllIlIIIllIl =True 
import os ,shutil ,sys 
IIIIlIlIIIlIllIlI =os .getcwd ()
sys .path .append (IIIIlIlIIIlIllIlI )
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
IIIIlIlIIIlIllIlI =os .getcwd ()
IIIlIlllIlIIIIIlI =os .path .join (IIIIlIlIIIlIllIlI ,'TEMP')
shutil .rmtree (IIIlIlllIlIIIIIlI ,ignore_errors =_IlIlIIllIlIIIllIl )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IIIIlIlIIIlIllIlI ,ignore_errors =_IlIlIIllIlIIIllIl )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IIIIlIlIIIlIllIlI ,ignore_errors =_IlIlIIllIlIIIllIl )
os .makedirs (IIIlIlllIlIIIIIlI ,exist_ok =_IlIlIIllIlIIIllIl )
os .makedirs (os .path .join (IIIIlIlIIIlIllIlI ,_IlllIIIIIlIllIIll ),exist_ok =_IlIlIIllIlIIIllIl )
os .makedirs (os .path .join (IIIIlIlIIIlIllIlI ,_IllIlIllIlIIIIIll ),exist_ok =_IlIlIIllIlIIIllIl )
os .environ ['TEMP']=IIIlIlllIlIIIIIlI 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IlllIIlIlIIIIlIlI =Config ()
IlIIlllIlllIlllII =I18nAuto ()
IlIIlllIlllIlllII .print ()
IlIIIIlllllIlIIll =torch .cuda .device_count ()
IIlIIllIlIIllIIII =[]
IlIIIllIIllIIlIlI =[]
IllIlIIIlllIlIlIl =_IIIIlIllllIIIlllI 
if torch .cuda .is_available ()or IlIIIIlllllIlIIll !=0 :
	for IIllllIIIIIllIIIl in range (IlIIIIlllllIlIIll ):
		IIlIIIlIIIIlIIllI =torch .cuda .get_device_name (IIllllIIIIIllIIIl )
		if any (IllIIIIlIIIIllIll in IIlIIIlIIIIlIIllI .upper ()for IllIIIIlIIIIllIll in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IllIlIIIlllIlIlIl =_IlIlIIllIlIIIllIl ;IIlIIllIlIIllIIII .append ('%s\t%s'%(IIllllIIIIIllIIIl ,IIlIIIlIIIIlIIllI ));IlIIIllIIllIIlIlI .append (int (torch .cuda .get_device_properties (IIllllIIIIIllIIIl ).total_memory /1024 /1024 /1024 +.4 ))
if IllIlIIIlllIlIlIl and len (IIlIIllIlIIllIIII )>0 :IIIlIlIIIIIlIllII =_IIlIlIIIlIIlIIIll .join (IIlIIllIlIIllIIII );IlllIIlllllIllIII =min (IlIIIllIIllIIlIlI )//2 
else :IIIlIlIIIIIlIllII =IlIIlllIlllIlllII ('很遗憾您这没有能用的显卡来支持您训练');IlllIIlllllIllIII =1 
IlIIIllIIllIllIll ='-'.join ([IIIlllIlllIlllIlI [0 ]for IIIlllIlllIlllIlI in IIlIIllIlIIllIIII ])
class IIllllIIllllllIll (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IllIIllIIlllIlIlI ,**IIIlIlllllIlIllIl ):super ().__init__ (variant ='tool',**IIIlIlllllIlIllIl )
	def get_block_name (IlIlIlIlllIlIlIII ):return 'button'
IlllIIIIllIlIllll =_IllIIIIIllIIIIIlI 
def IIlllllIIIlIllIlI ():
	global IlllIIIIllIlIllll ;IlIIIllIlIIlIlIll ,_IIlIIIllIlIlIllIl ,_IIlIIIllIlIlIllIl =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IlllIIIIllIlIllll =IlIIIllIlIIlIlIll [0 ];IlllIIIIllIlIllll =IlllIIIIllIlIllll .to (IlllIIlIlIIIIlIlI .device )
	if IlllIIlIlIIIIlIlI .is_half :IlllIIIIllIlIllll =IlllIIIIllIlIllll .half ()
	else :IlllIIIIllIlIllll =IlllIIIIllIlIllll .float ()
	IlllIIIIllIlIllll .eval ()
IIIIIlIIlllIIlllI =_IllIlIllIlIIIIIll 
IIIllIllIlIIlIlII ='uvr5_weights'
IIlIIIIIIIlIIlIII =_IlllIIIIIlIllIIll 
IIlIlllIlIIlllllI =[]
for IlIIlIllIIIIlllIl in os .listdir (IIIIIlIIlllIIlllI ):
	if IlIIlIllIIIIlllIl .endswith (_IlIIIIllllIIIIlII ):IIlIlllIlIIlllllI .append (IlIIlIllIIIIlllIl )
IIlIllIlIIIlllIll =[]
for (IIIllIIlllIllIIll ,IllllIlllIIlIlllI ,IIllllllIlIlllIlI )in os .walk (IIlIIIIIIIlIIlIII ,topdown =_IIIIlIllllIIIlllI ):
	for IlIIlIllIIIIlllIl in IIllllllIlIlllIlI :
		if IlIIlIllIIIIlllIl .endswith (_IIIlIIIlIlllIIIIl )and _IllllIlIIIIIIIlII not in IlIIlIllIIIIlllIl :IIlIllIlIIIlllIll .append (_IIlllIIlIIlIIllll %(IIIllIIlllIllIIll ,IlIIlIllIIIIlllIl ))
IIlllllIlIlIlllIl =[]
for IlIIlIllIIIIlllIl in os .listdir (IIIllIllIlIIlIlII ):
	if IlIIlIllIIIIlllIl .endswith (_IlIIIIllllIIIIlII )or 'onnx'in IlIIlIllIIIIlllIl :IIlllllIlIlIlllIl .append (IlIIlIllIIIIlllIl .replace (_IlIIIIllllIIIIlII ,''))
IllIllIlllIllIIll =_IllIIIIIllIIIIIlI 
def IIIlIlIlllIlIIIIl (IllIllIIIllllIlII ,IlllIIIlIlIllIlll ,IllIIllIIIlIllIlI ,IIlllllllIlllllll ,IIlllIlllIIIIIlIl ,IIIIllllIIllIllII ,IIllllIlIIIlIllII ,IIllIIlIIlIllIllI ,IIIlllIIIIllIIlII ,IlIllllllIIllIIll ,IlIIIIIIIllIlIllI ,IllllIlIIIllIlIll ):
	global IlIlIIIIlIllIllll ,IIIlIlIlIllIIIIII ,IllIlIIIlIIIIIlll ,IlllIIIIllIlIllll ,IllIIlIllIlIIIllI ,IllIllIlllIllIIll 
	if IlllIIIlIlIllIlll is _IllIIIIIllIIIIIlI :return 'You need to upload an audio',_IllIIIIIllIIIIIlI 
	IllIIllIIIlIllIlI =int (IllIIllIIIlIllIlI )
	try :
		IIlIllIlIIIlllIIl =load_audio (IlllIIIlIlIllIlll ,16000 );IlllllIIlIIlllIIl =np .abs (IIlIllIlIIIlllIIl ).max ()/.95 
		if IlllllIIlIIlllIIl >1 :IIlIllIlIIIlllIIl /=IlllllIIlIIlllIIl 
		IllllllIIIlIlllll =[0 ,0 ,0 ]
		if not IlllIIIIllIlIllll :IIlllllIIIlIllIlI ()
		IlllIIIIIIllIllIl =IllIllIlllIllIIll .get (_IIllIllIlllIIllIl ,1 );IIIIllllIIllIllII =IIIIllllIIllIllII .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll ).replace (_IllllIlIIIIIIIlII ,'added')if IIIIllllIIllIllII !=''else IIllllIlIIIlIllII ;IlIllIIlIIIlIIlII =IllIlIIIlIIIIIlll .pipeline (IlllIIIIllIlIllll ,IIIlIlIlIllIIIIII ,IllIllIIIllllIlII ,IIlIllIlIIIlllIIl ,IlllIIIlIlIllIlll ,IllllllIIIlIlllll ,IllIIllIIIlIllIlI ,IIlllIlllIIIIIlIl ,IIIIllllIIllIllII ,IIllIIlIIlIllIllI ,IlllIIIIIIllIllIl ,IIIlllIIIIllIIlII ,IlIlIIIIlIllIllll ,IlIllllllIIllIIll ,IlIIIIIIIllIlIllI ,IllIIlIllIlIIIllI ,IllllIlIIIllIlIll ,f0_file =IIlllllllIlllllll )
		if IlIlIIIIlIllIllll !=IlIllllllIIllIIll >=16000 :IlIlIIIIlIllIllll =IlIllllllIIllIIll 
		IlIlIIIlIlIIllIIl ='Using index:%s.'%IIIIllllIIllIllII if os .path .exists (IIIIllllIIllIllII )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IlIlIIIlIlIIllIIl ,IllllllIIIlIlllll [0 ],IllllllIIIlIlllll [1 ],IllllllIIIlIlllll [2 ]),(IlIlIIIIlIllIllll ,IlIllIIlIIIlIIlII )
	except :IIIIIllIIlllllIll =traceback .format_exc ();print (IIIIIllIIlllllIll );return IIIIIllIIlllllIll ,(_IllIIIIIllIIIIIlI ,_IllIIIIIllIIIIIlI )
def IIIlIIllIIIlllIII (IIllllIllIllIlIll ,IlIIIIllIIIlIlllI ,IIlllIIIIIIIIlIlI ,IIIllIIlIIIllIllI ,IIIlIIIlllIlIllIl ,IlIlIlIIlIlIllIIl ,IlIlIIIIlIIllllII ,IIIIIlIlIIIIlIllI ,IllIlIIlIlIIIlllI ,IlIIIIllIIllIIlIl ,IllllIlIlIIlIllII ,IlIlIIIlIlIlIlIlI ,IllIlIlllIIlllIlI ,IlIlIlIIIIllIIIII ):
	try :
		IlIIIIllIIIlIlllI =IlIIIIllIIIlIlllI .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll );IIlllIIIIIIIIlIlI =IIlllIIIIIIIIlIlI .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll );os .makedirs (IIlllIIIIIIIIlIlI ,exist_ok =_IlIlIIllIlIIIllIl )
		try :
			if IlIIIIllIIIlIlllI !='':IIIllIIlIIIllIllI =[os .path .join (IlIIIIllIIIlIlllI ,IlIIIIIIlllIIllIl )for IlIIIIIIlllIIllIl in os .listdir (IlIIIIllIIIlIlllI )]
			else :IIIllIIlIIIllIllI =[IIllllllIlIIIllll .name for IIllllllIlIIIllll in IIIllIIlIIIllIllI ]
		except :traceback .print_exc ();IIIllIIlIIIllIllI =[IlIlIIllllIllllII .name for IlIlIIllllIllllII in IIIllIIlIIIllIllI ]
		IIlllllIIlIlllllI =[]
		for IlIlllllIIlllllIl in IIIllIIlIIIllIllI :
			IIllIlIllIIllllll ,IIlllIIIIIIllIIIl =IIIlIlIlllIlIIIIl (IIllllIllIllIlIll ,IlIlllllIIlllllIl ,IIIlIIIlllIlIllIl ,_IllIIIIIllIIIIIlI ,IlIlIlIIlIlIllIIl ,IlIlIIIIlIIllllII ,IIIIIlIlIIIIlIllI ,IllIlIIlIlIIIlllI ,IlIIIIllIIllIIlIl ,IllllIlIlIIlIllII ,IlIlIIIlIlIlIlIlI ,IllIlIlllIIlllIlI )
			if 'Success'in IIllIlIllIIllllll :
				try :
					IlIllIlllIlIIlIII ,IlllllIllIIIIllll =IIlllIIIIIIllIIIl 
					if IlIlIlIIIIllIIIII in [_IIIIIIIlIIIlllIIl ,_IlllIlllIIIIllIlI ]:sf .write ('%s/%s.%s'%(IIlllIIIIIIIIlIlI ,os .path .basename (IlIlllllIIlllllIl ),IlIlIlIIIIllIIIII ),IlllllIllIIIIllll ,IlIllIlllIlIIlIII )
					else :
						IlIlllllIIlllllIl ='%s/%s.wav'%(IIlllIIIIIIIIlIlI ,os .path .basename (IlIlllllIIlllllIl ));sf .write (IlIlllllIIlllllIl ,IlllllIllIIIIllll ,IlIllIlllIlIIlIII )
						if os .path .exists (IlIlllllIIlllllIl ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IlIlllllIIlllllIl ,IlIlllllIIlllllIl [:-4 ]+'.%s'%IlIlIlIIIIllIIIII ))
				except :IIllIlIllIIllllll +=traceback .format_exc ()
			IIlllllIIlIlllllI .append (_IIllIllIIIlIllIIl %(os .path .basename (IlIlllllIIlllllIl ),IIllIlIllIIllllll ));yield _IIlIlIIIlIIlIIIll .join (IIlllllIIlIlllllI )
		yield _IIlIlIIIlIIlIIIll .join (IIlllllIIlIlllllI )
	except :yield traceback .format_exc ()
def IlllIIlIIIllIlIlI (IIIIlIIIIlllllIlI ,IllllllllIIlIlIlI ,IlIIllIllIIIIIIlI ,IlIllIlIIIlIlllll ,IIIlllIIIIIIIlIIl ,IIlIllIIlIlIIIlII ,IlllIllIlIlIllIII ):
	IlIllIlIIIIllllII ='streams';IIIlIlllIIllllIII ='onnx_dereverb_By_FoxJoy';IlIIIIIllllllIIIl =[]
	try :
		IllllllllIIlIlIlI =IllllllllIIlIlIlI .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll );IlIIllIllIIIIIIlI =IlIIllIllIIIIIIlI .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll );IIIlllIIIIIIIlIIl =IIIlllIIIIIIIlIIl .strip (_IIIlIllIIIIlIllll ).strip (_IIlIllllIlllIIIIl ).strip (_IIlIlIIIlIIlIIIll ).strip (_IIlIllllIlllIIIIl ).strip (_IIIlIllIIIIlIllll )
		if IIIIlIIIIlllllIlI ==IIIlIlllIIllllIII :from MDXNet import MDXNetDereverb ;IIlllllIlIIlIIlIl =MDXNetDereverb (15 )
		else :IllIllIIIIIIIllIl =_audio_pre_ if 'DeEcho'not in IIIIlIIIIlllllIlI else _audio_pre_new ;IIlllllIlIIlIIlIl =IllIllIIIIIIIllIl (agg =int (IIlIllIIlIlIIIlII ),model_path =os .path .join (IIIllIllIlIIlIlII ,IIIIlIIIIlllllIlI +_IlIIIIllllIIIIlII ),device =IlllIIlIlIIIIlIlI .device ,is_half =IlllIIlIlIIIIlIlI .is_half )
		if IllllllllIIlIlIlI !='':IlIllIlIIIlIlllll =[os .path .join (IllllllllIIlIlIlI ,IlIlllIlIlIIlIlIl )for IlIlllIlIlIIlIlIl in os .listdir (IllllllllIIlIlIlI )]
		else :IlIllIlIIIlIlllll =[IIlIlIIIlIIIllIll .name for IIlIlIIIlIIIllIll in IlIllIlIIIlIlllll ]
		for IlIIllIlIIIIIllll in IlIllIlIIIlIlllll :
			IIIlllllllIlllIll =os .path .join (IllllllllIIlIlIlI ,IlIIllIlIIIIIllll );IIIlIIIIIIlIIIIlI =1 ;IllIIIIIllllIIlII =0 
			try :
				IlIIllIIIlIIllIII =ffmpeg .probe (IIIlllllllIlllIll ,cmd ='ffprobe')
				if IlIIllIIIlIIllIII [IlIllIlIIIIllllII ][0 ]['channels']==2 and IlIIllIIIlIIllIII [IlIllIlIIIIllllII ][0 ][_IlIIllIIlIIlIllII ]=='44100':IIIlIIIIIIlIIIIlI =0 ;IIlllllIlIIlIIlIl ._path_audio_ (IIIlllllllIlllIll ,IIIlllIIIIIIIlIIl ,IlIIllIllIIIIIIlI ,IlllIllIlIlIllIII );IllIIIIIllllIIlII =1 
			except :IIIlIIIIIIlIIIIlI =1 ;traceback .print_exc ()
			if IIIlIIIIIIlIIIIlI ==1 :IIlllIIIIIllIlIII ='%s/%s.reformatted.wav'%(IIIlIlllIlIIIIIlI ,os .path .basename (IIIlllllllIlllIll ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIIlllllllIlllIll ,IIlllIIIIIllIlIII ));IIIlllllllIlllIll =IIlllIIIIIllIlIII 
			try :
				if IllIIIIIllllIIlII ==0 :IIlllllIlIIlIIlIl ._path_audio_ (IIIlllllllIlllIll ,IIIlllIIIIIIIlIIl ,IlIIllIllIIIIIIlI ,IlllIllIlIlIllIII )
				IlIIIIIllllllIIIl .append ('%s->Success'%os .path .basename (IIIlllllllIlllIll ));yield _IIlIlIIIlIIlIIIll .join (IlIIIIIllllllIIIl )
			except :IlIIIIIllllllIIIl .append (_IIllIllIIIlIllIIl %(os .path .basename (IIIlllllllIlllIll ),traceback .format_exc ()));yield _IIlIlIIIlIIlIIIll .join (IlIIIIIllllllIIIl )
	except :IlIIIIIllllllIIIl .append (traceback .format_exc ());yield _IIlIlIIIlIIlIIIll .join (IlIIIIIllllllIIIl )
	finally :
		try :
			if IIIIlIIIIlllllIlI ==IIIlIlllIIllllIII :del IIlllllIlIIlIIlIl .pred .model ;del IIlllllIlIIlIIlIl .pred .model_ 
			else :del IIlllllIlIIlIIlIl .model ;del IIlllllIlIIlIIlIl 
		except :traceback .print_exc ()
		print (_IlIllIIlllIlllIIl )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IIlIlIIIlIIlIIIll .join (IlIIIIIllllllIIIl )
def IIlllllIIlIlIIIII (IlIlIIIlIIlllIlll ):
	IIIIllllIIlIIlIlI ='';IIllIlIlIlIlIlIll =os .path .join (_IlllIIIIIlIllIIll ,IlIlIIIlIIlllIlll .split (_IlIIIIllIIIlIllll )[0 ],'')
	for IlIIlIlIlIllllIlI in IIlIllIlIIIlllIll :
		if IIllIlIlIlIlIlIll in IlIIlIlIlIllllIlI :IIIIllllIIlIIlIlI =IlIIlIlIlIllllIlI ;break 
	return IIIIllllIIlIIlIlI 
def IIIllllIIIIIIIIll (IllIIlIllIIIIlIIl ,IllllIIlllllIIlIl ,IIIllllIllllIIIll ):
	global IIIIIlIIlIlIIIIII ,IlIlIIIIlIllIllll ,IIIlIlIlIllIIIIII ,IllIlIIIlIIIIIlll ,IllIllIlllIllIIll ,IllIIlIllIlIIIllI 
	if IllIIlIllIIIIlIIl ==''or IllIIlIllIIIIlIIl ==[]:
		global IlllIIIIllIlIllll 
		if IlllIIIIllIlIllll is not _IllIIIIIllIIIIIlI :
			print (_IlIllIIlllIlllIIl );del IIIlIlIlIllIIIIII ,IIIIIlIIlIlIIIIII ,IllIlIIIlIIIIIlll ,IlllIIIIllIlIllll ,IlIlIIIIlIllIllll ;IlllIIIIllIlIllll =IIIlIlIlIllIIIIII =IIIIIlIIlIlIIIIII =IllIlIIIlIIIIIlll =IlllIIIIllIlIllll =IlIlIIIIlIllIllll =_IllIIIIIllIIIIIlI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IIIIIIIlIllIIlIIl =IllIllIlllIllIIll .get (_IIllIllIlllIIllIl ,1 );IllIIlIllIlIIIllI =IllIllIlllIllIIll .get (_IIIIlIIIlIlllIlll ,_IlllllIlIllIIllII )
			if IllIIlIllIlIIIllI ==_IlllllIlIllIIllII :
				if IIIIIIIlIllIIlIIl ==1 :IIIlIlIlIllIIIIII =SynthesizerTrnMs256NSFsid (*IllIllIlllIllIIll [_IlllllllIlllllllI ],is_half =IlllIIlIlIIIIlIlI .is_half )
				else :IIIlIlIlIllIIIIII =SynthesizerTrnMs256NSFsid_nono (*IllIllIlllIllIIll [_IlllllllIlllllllI ])
			elif IllIIlIllIlIIIllI ==_IIIlllllllIllIIll :
				if IIIIIIIlIllIIlIIl ==1 :IIIlIlIlIllIIIIII =SynthesizerTrnMs768NSFsid (*IllIllIlllIllIIll [_IlllllllIlllllllI ],is_half =IlllIIlIlIIIIlIlI .is_half )
				else :IIIlIlIlIllIIIIII =SynthesizerTrnMs768NSFsid_nono (*IllIllIlllIllIIll [_IlllllllIlllllllI ])
			del IIIlIlIlIllIIIIII ,IllIllIlllIllIIll 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IIIIlIlIlllllIlII :_IIIIlIllllIIIlllI ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
	IIlIllllIIIlIlIlI =_IIlllIIlIIlIIllll %(IIIIIlIIlllIIlllI ,IllIIlIllIIIIlIIl );print ('loading %s'%IIlIllllIIIlIlIlI );IllIllIlllIllIIll =torch .load (IIlIllllIIIlIlIlI ,map_location =_IlIlIIlIllIllIIll );IlIlIIIIlIllIllll =IllIllIlllIllIIll [_IlllllllIlllllllI ][-1 ];IllIllIlllIllIIll [_IlllllllIlllllllI ][-3 ]=IllIllIlllIllIIll [_IIIIIllIIIIIIIlll ][_IlllllIIIIIIIIlIl ].shape [0 ];IIIIIIIlIllIIlIIl =IllIllIlllIllIIll .get (_IIllIllIlllIIllIl ,1 )
	if IIIIIIIlIllIIlIIl ==0 :IllllIIlllllIIlIl =IIIllllIllllIIIll ={_IIIIlIlIlllllIlII :_IIIIlIllllIIIlllI ,_IIllIlIlIIIlIlIII :.5 ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
	else :IllllIIlllllIIlIl ={_IIIIlIlIlllllIlII :_IlIlIIllIlIIIllIl ,_IIllIlIlIIIlIlIII :IllllIIlllllIIlIl ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll };IIIllllIllllIIIll ={_IIIIlIlIlllllIlII :_IlIlIIllIlIIIllIl ,_IIllIlIlIIIlIlIII :IIIllllIllllIIIll ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
	IllIIlIllIlIIIllI =IllIllIlllIllIIll .get (_IIIIlIIIlIlllIlll ,_IlllllIlIllIIllII )
	if IllIIlIllIlIIIllI ==_IlllllIlIllIIllII :
		if IIIIIIIlIllIIlIIl ==1 :IIIlIlIlIllIIIIII =SynthesizerTrnMs256NSFsid (*IllIllIlllIllIIll [_IlllllllIlllllllI ],is_half =IlllIIlIlIIIIlIlI .is_half )
		else :IIIlIlIlIllIIIIII =SynthesizerTrnMs256NSFsid_nono (*IllIllIlllIllIIll [_IlllllllIlllllllI ])
	elif IllIIlIllIlIIIllI ==_IIIlllllllIllIIll :
		if IIIIIIIlIllIIlIIl ==1 :IIIlIlIlIllIIIIII =SynthesizerTrnMs768NSFsid (*IllIllIlllIllIIll [_IlllllllIlllllllI ],is_half =IlllIIlIlIIIIlIlI .is_half )
		else :IIIlIlIlIllIIIIII =SynthesizerTrnMs768NSFsid_nono (*IllIllIlllIllIIll [_IlllllllIlllllllI ])
	del IIIlIlIlIllIIIIII .enc_q ;print (IIIlIlIlIllIIIIII .load_state_dict (IllIllIlllIllIIll [_IIIIIllIIIIIIIlll ],strict =_IIIIlIllllIIIlllI ));IIIlIlIlIllIIIIII .eval ().to (IlllIIlIlIIIIlIlI .device )
	if IlllIIlIlIIIIlIlI .is_half :IIIlIlIlIllIIIIII =IIIlIlIlIllIIIIII .half ()
	else :IIIlIlIlIllIIIIII =IIIlIlIlIllIIIIII .float ()
	IllIlIIIlIIIIIlll =VC (IlIlIIIIlIllIllll ,IlllIIlIlIIIIlIlI );IIIIIlIIlIlIIIIII =IllIllIlllIllIIll [_IlllllllIlllllllI ][-3 ];return {_IIIIlIlIlllllIlII :_IlIlIIllIlIIIllIl ,'maximum':IIIIIlIIlIlIIIIII ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },IllllIIlllllIIlIl ,IIIllllIllllIIIll ,IIlllllIIlIlIIIII (IllIIlIllIIIIlIIl )
def IlIIIIllllIlIlIlI ():
	IlIIllIIIllIIIllI =[]
	for IlIIIllIIIIlIlIIl in os .listdir (IIIIIlIIlllIIlllI ):
		if IlIIIllIIIIlIlIIl .endswith (_IlIIIIllllIIIIlII ):IlIIllIIIllIIIllI .append (IlIIIllIIIIlIlIIl )
	IlllIIIIlIlIIIlll =[]
	for (IlIIllIIllIIllIII ,IlIllIIIIlIlIIIlI ,IIlIIllIIllIIlIII )in os .walk (IIlIIIIIIIlIIlIII ,topdown =_IIIIlIllllIIIlllI ):
		for IlIIIllIIIIlIlIIl in IIlIIllIIllIIlIII :
			if IlIIIllIIIIlIlIIl .endswith (_IIIlIIIlIlllIIIIl )and _IllllIlIIIIIIIlII not in IlIIIllIIIIlIlIIl :IlllIIIIlIlIIIlll .append (_IIlllIIlIIlIIllll %(IlIIllIIllIIllIII ,IlIIIllIIIIlIlIIl ))
	return {_IIIlIlIlIIllIllIl :sorted (IlIIllIIIllIIIllI ),_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },{_IIIlIlIlIIllIllIl :sorted (IlllIIIIlIlIIIlll ),_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
def IlIIllIIlIllIIIll ():return {_IIllIlIlIIIlIlIII :'',_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
IlIIIllIlllllIlII ={_IllIlIIllIIIlIlIl :32000 ,_IIIllIIlIlIllIIll :40000 ,_IlIIlllIlIlIIlIll :48000 }
def IlIllllIIIIIllIIl (IIIlllIIIIIlIIIIl ,IlllIllIIllllIIII ):
	while 1 :
		if IlllIllIIllllIIII .poll ()is _IllIIIIIllIIIIIlI :sleep (.5 )
		else :break 
	IIIlllIIIIIlIIIIl [0 ]=_IlIlIIllIlIIIllIl 
def IIlIlIIIIllIlIlIl (IIlllIlIlIIlIlIll ,IllIlIIIIlllIIIll ):
	while 1 :
		IIllllIIIllIlIllI =1 
		for IllIlIlllIlIIIIlI in IllIlIIIIlllIIIll :
			if IllIlIlllIlIIIIlI .poll ()is _IllIIIIIllIIIIIlI :IIllllIIIllIlIllI =0 ;sleep (.5 );break 
		if IIllllIIIllIlIllI ==1 :break 
	IIlllIlIlIIlIlIll [0 ]=_IlIlIIllIlIIIllIl 
def IllIIIllllIlIIlIl (IlIlllIllIIlIlIII ,IlIlIIlIlllIIllIl ,IIllIlIlIIllllIIl ,IllllIlIllIIIIIll ):
	IllllIlIlllIllIlI ='%s/logs/%s/preprocess.log';IIllIlIlIIllllIIl =IlIIIllIlllllIlII [IIllIlIlIIllllIIl ];os .makedirs (_IIllIIlllllIIIllI %(IIIIlIlIIIlIllIlI ,IlIlIIlIlllIIllIl ),exist_ok =_IlIlIIllIlIIIllIl );IlllIIIllllIlIllI =open (IllllIlIlllIllIlI %(IIIIlIlIIIlIllIlI ,IlIlIIlIlllIIllIl ),'w');IlllIIIllllIlIllI .close ();IIllIIlllIIIlIIIl =IlllIIlIlIIIIlIlI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IlIlllIllIIlIlIII ,IIllIlIlIIllllIIl ,IllllIlIllIIIIIll ,IIIIlIlIIIlIllIlI ,IlIlIIlIlllIIllIl )+str (IlllIIlIlIIIIlIlI .noparallel );print (IIllIIlllIIIlIIIl );IlIIIIIIIlIIIlIII =Popen (IIllIIlllIIIlIIIl ,shell =_IlIlIIllIlIIIllIl );IlIIIIlllllIllIlI =[_IIIIlIllllIIIlllI ];threading .Thread (target =IlIllllIIIIIllIIl ,args =(IlIIIIlllllIllIlI ,IlIIIIIIIlIIIlIII )).start ()
	while 1 :
		with open (IllllIlIlllIllIlI %(IIIIlIlIIIlIllIlI ,IlIlIIlIlllIIllIl ),_IllIIIIlllllIllIl )as IlllIIIllllIlIllI :yield IlllIIIllllIlIllI .read ()
		sleep (1 )
		if IlIIIIlllllIllIlI [0 ]:break 
	with open (IllllIlIlllIllIlI %(IIIIlIlIIIlIllIlI ,IlIlIIlIlllIIllIl ),_IllIIIIlllllIllIl )as IlllIIIllllIlIllI :IIIlIIlIIIllllIll =IlllIIIllllIlIllI .read ()
	print (IIIlIIlIIIllllIll );yield IIIlIIlIIIllllIll 
def IIlllIIIIIIIlIlIl (IlIIIIIlIlIIIllIl ,IIIllIlllIlIlIlIl ,IlIlllllIlIIlIIll ,IIlIIlIlIllIIIlIl ,IIIlllIIllllIIIII ,IllIIllIlIIlIlIII ,IllllIIIlllIIlllI ):
	IIIIlIIIIlIlIIlIl ='%s/logs/%s/extract_fl_feature.log';IlIIIIIlIlIIIllIl =IlIIIIIlIlIIIllIl .split ('-');os .makedirs (_IIllIIlllllIIIllI %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),exist_ok =_IlIlIIllIlIIIllIl );IIllIIllIllIIIIII =open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),'w');IIllIIllIllIIIIII .close ()
	if IIlIIlIlIllIIIlIl :
		if IlIlllllIlIIlIIll !=_IIlIllllIlIlIlIll :
			IllIIIIlIlIllllll =IlllIIlIlIIIIlIlI .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ,IIIllIlllIlIlIlIl ,IlIlllllIlIIlIIll );print (IllIIIIlIlIllllll );IlIllIlIllllIIlll =Popen (IllIIIIlIlIllllll ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIIlIllIlIIIlIlIl =[_IIIIlIllllIIIlllI ];threading .Thread (target =IlIllllIIIIIllIIl ,args =(IIIlIllIlIIIlIlIl ,IlIllIlIllllIIlll )).start ()
			while 1 :
				with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :yield IIllIIllIllIIIIII .read ()
				sleep (1 )
				if IIIlIllIlIIIlIlIl [0 ]:break 
			with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :IIlIlIIllIlIlllIl =IIllIIllIllIIIIII .read ()
			print (IIlIlIIllIlIlllIl );yield IIlIlIIllIlIlllIl 
		else :
			IllllIIIlllIIlllI =IllllIIIlllIIlllI .split ('-');IlIllIllIIIlIIIIl =len (IllllIIIlllIIlllI );IIIlIIlIIlIllIIII =[]
			for (IllIIIIlllIIlIllI ,IllllIIlIIlIlllIl )in enumerate (IllllIIIlllIIlllI ):IllIIIIlIlIllllll =IlllIIlIlIIIIlIlI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IlIllIllIIIlIIIIl ,IllIIIIlllIIlIllI ,IllllIIlIIlIlllIl ,IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ,IlllIIlIlIIIIlIlI .is_half );print (IllIIIIlIlIllllll );IlIllIlIllllIIlll =Popen (IllIIIIlIlIllllll ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIIlIIlIIlIllIIII .append (IlIllIlIllllIIlll )
			IIIlIllIlIIIlIlIl =[_IIIIlIllllIIIlllI ];threading .Thread (target =IIlIlIIIIllIlIlIl ,args =(IIIlIllIlIIIlIlIl ,IIIlIIlIIlIllIIII )).start ()
			while 1 :
				with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :yield IIllIIllIllIIIIII .read ()
				sleep (1 )
				if IIIlIllIlIIIlIlIl [0 ]:break 
			with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :IIlIlIIllIlIlllIl =IIllIIllIllIIIIII .read ()
			print (IIlIlIIllIlIlllIl );yield IIlIlIIllIlIlllIl 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IlIllIllIIIlIIIIl =len (IlIIIIIlIlIIIllIl );IIIlIIlIIlIllIIII =[]
	for (IllIIIIlllIIlIllI ,IllllIIlIIlIlllIl )in enumerate (IlIIIIIlIlIIIllIl ):IllIIIIlIlIllllll =IlllIIlIlIIIIlIlI .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IlllIIlIlIIIIlIlI .device ,IlIllIllIIIlIIIIl ,IllIIIIlllIIlIllI ,IllllIIlIIlIlllIl ,IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ,IllIIllIlIIlIlIII );print (IllIIIIlIlIllllll );IlIllIlIllllIIlll =Popen (IllIIIIlIlIllllll ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIIlIIlIIlIllIIII .append (IlIllIlIllllIIlll )
	IIIlIllIlIIIlIlIl =[_IIIIlIllllIIIlllI ];threading .Thread (target =IIlIlIIIIllIlIlIl ,args =(IIIlIllIlIIIlIlIl ,IIIlIIlIIlIllIIII )).start ()
	while 1 :
		with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :yield IIllIIllIllIIIIII .read ()
		sleep (1 )
		if IIIlIllIlIIIlIlIl [0 ]:break 
	with open (IIIIlIIIIlIlIIlIl %(IIIIlIlIIIlIllIlI ,IIIlllIIllllIIIII ),_IllIIIIlllllIllIl )as IIllIIllIllIIIIII :IIlIlIIllIlIlllIl =IIllIIllIllIIIIII .read ()
	print (IIlIlIIllIlIlllIl );yield IIlIlIIllIlIlllIl 
def IIIlIIllIIlllIlll (IllIlIllllllIlIll ,IlIIIlllIllIlIllI ,IIIlIIlIIllIllIll ):
	IIIllIlIIllIIIIIl =''if IIIlIIlIIllIllIll ==_IlllllIlIllIIllII else _IIllIIllIIIIlIlII ;IllIlIlIIIllIIlII =_IIllIllIlllIIllIl if IlIIIlllIllIlIllI else '';IlllIIIIIlIllllIl =os .access (_IllllIlIIlIIIIIlI %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll ),os .F_OK );IlllllIIIlIlIlIIl =os .access (_IllIIIIllllIlllll %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll ),os .F_OK )
	if not IlllIIIIIlIllllIl :print (_IllllIlIIlIIIIIlI %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll ),_IlIlllIllIIlIIlII )
	if not IlllllIIIlIlIlIIl :print (_IllIIIIllllIlllll %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll ),_IlIlllIllIIlIIlII )
	return _IllllIlIIlIIIIIlI %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll )if IlllIIIIIlIllllIl else '',_IllIIIIllllIlllll %(IIIllIlIIllIIIIIl ,IllIlIlIIIllIIlII ,IllIlIllllllIlIll )if IlllllIIIlIlIlIIl else ''
def IIIIIlIIllllIlIll (IIlIlIllllIIlIlll ,IIIlllIllllIIllll ,IllllllIlIIIIlIlI ):
	IlIIIIIIllIIIlIlI =''if IllllllIlIIIIlIlI ==_IlllllIlIllIIllII else _IIllIIllIIIIlIlII 
	if IIlIlIllllIIlIlll ==_IllIlIIllIIIlIlIl and IllllllIlIIIIlIlI ==_IlllllIlIllIIllII :IIlIlIllllIIlIlll =_IIIllIIlIlIllIIll 
	IlIlIlllIIllIlIIl ={_IIIlIlIlIIllIllIl :[_IIIllIIlIlIllIIll ,_IlIIlllIlIlIIlIll ],_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll ,_IIllIlIlIIIlIlIII :IIlIlIllllIIlIlll }if IllllllIlIIIIlIlI ==_IlllllIlIllIIllII else {_IIIlIlIlIIllIllIl :[_IIIllIIlIlIllIIll ,_IlIIlllIlIlIIlIll ,_IllIlIIllIIIlIlIl ],_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll ,_IIllIlIlIIIlIlIII :IIlIlIllllIIlIlll };IIIIIIlIlIlllIlIl =_IIllIllIlllIIllIl if IIIlllIllllIIllll else '';IIllllllIIllllIIl =os .access (_IllllIlIIlIIIIIlI %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll ),os .F_OK );IlIIIIlIllllIIIIl =os .access (_IllIIIIllllIlllll %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll ),os .F_OK )
	if not IIllllllIIllllIIl :print (_IllllIlIIlIIIIIlI %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll ),_IlIlllIllIIlIIlII )
	if not IlIIIIlIllllIIIIl :print (_IllIIIIllllIlllll %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll ),_IlIlllIllIIlIIlII )
	return _IllllIlIIlIIIIIlI %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll )if IIllllllIIllllIIl else '',_IllIIIIllllIlllll %(IlIIIIIIllIIIlIlI ,IIIIIIlIlIlllIlIl ,IIlIlIllllIIlIlll )if IlIIIIlIllllIIIIl else '',IlIlIlllIIllIlIIl 
def IlllIIIllIlIIIIlI (IllIllllIllIlIIll ,IlIIIIIIlllIIlIll ,IIllIlIllIIIIIIIl ):
	IIlIllllIlIllIIIl ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IIIIllIlIIIllllIl ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IllIlIIIIIlllIIIl =''if IIllIlIllIIIIIIIl ==_IlllllIlIllIIllII else _IIllIIllIIIIlIlII ;IllIIlllIIlIIIlII =os .access (IIIIllIlIIIllllIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll ),os .F_OK );IIIIIllIIIllIlIll =os .access (IIlIllllIlIllIIIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll ),os .F_OK )
	if not IllIIlllIIlIIIlII :print (IIIIllIlIIIllllIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll ),_IlIlllIllIIlIIlII )
	if not IIIIIllIIIllIlIll :print (IIlIllllIlIllIIIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll ),_IlIlllIllIIlIIlII )
	if IllIllllIllIlIIll :return {_IIIIlIlIlllllIlII :_IlIlIIllIlIIIllIl ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },IIIIllIlIIIllllIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll )if IllIIlllIIlIIIlII else '',IIlIllllIlIllIIIl %(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll )if IIIIIllIIIllIlIll else ''
	return {_IIIIlIlIlllllIlII :_IIIIlIllllIIIlllI ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll )if IllIIlllIIlIIIlII else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IllIlIIIIIlllIIIl ,IlIIIIIIlllIIlIll )if IIIIIllIIIllIlIll else ''
def IlllIlIIIIllllIlI (IllIlllIllIllIIlI ,IllIIIIIIlIIlIIII ,IIlIIIIlIlIlIIIll ,IIllllIlIlIIIllll ,IIIlllIllIlllIllI ,IIIIlIllIllIIllIl ,IlIIIlIllIllIIlIl ,IlllllIllIlIIIlII ,IIIllIIIIIlllIlII ,IlllIlIlIIIIlIlIl ,IlIIIlIIlIllIllll ,IIIIIlIIlIlIIlIll ,IlIIlllIIIlllllII ,IIllIllllIIIlIIll ):
	IIIlIIlIIIIlIIIIl ='\x08';IlIllllIIlIIIIlII =_IIllIIlllllIIIllI %(IIIIlIlIIIlIllIlI ,IllIlllIllIllIIlI );os .makedirs (IlIllllIIlIIIIlII ,exist_ok =_IlIlIIllIlIIIllIl );IIIIlIllllIIIllll =_IIlIllIIIIllIlIII %IlIllllIIlIIIIlII ;IlIIlllIIlIIIllII =_IIllIlIlIIlllIlII %IlIllllIIlIIIIlII if IIllIllllIIIlIIll ==_IlllllIlIllIIllII else _IIllIlllIIlIIIIIl %IlIllllIIlIIIIlII 
	if IIlIIIIlIlIlIIIll :IIllIIIIIlllIIIlI ='%s/2a_f0'%IlIllllIIlIIIIlII ;IlIIlIIlllIIlllll =_IllllIIllllIIllIl %IlIllllIIlIIIIlII ;IllIlIIIlIlllIlIl =set ([IlIllIlIlIIlllIIl .split (_IlIIIIllIIIlIllll )[0 ]for IlIllIlIlIIlllIIl in os .listdir (IIIIlIllllIIIllll )])&set ([IIllIlIIIIlllllll .split (_IlIIIIllIIIlIllll )[0 ]for IIllIlIIIIlllllll in os .listdir (IlIIlllIIlIIIllII )])&set ([IlIlllIllIIIllIlI .split (_IlIIIIllIIIlIllll )[0 ]for IlIlllIllIIIllIlI in os .listdir (IIllIIIIIlllIIIlI )])&set ([IIIlIIIIllIllIIlI .split (_IlIIIIllIIIlIllll )[0 ]for IIIlIIIIllIllIIlI in os .listdir (IlIIlIIlllIIlllll )])
	else :IllIlIIIlIlllIlIl =set ([IIIlllllIllIlIlll .split (_IlIIIIllIIIlIllll )[0 ]for IIIlllllIllIlIlll in os .listdir (IIIIlIllllIIIllll )])&set ([IIIlIlIIlIIIIIlll .split (_IlIIIIllIIIlIllll )[0 ]for IIIlIlIIlIIIIIlll in os .listdir (IlIIlllIIlIIIllII )])
	IlIIllIlIllIIlIII =[]
	for IIIlIlIIllIlllIII in IllIlIIIlIlllIlIl :
		if IIlIIIIlIlIlIIIll :IlIIllIlIllIIlIII .append (_IIlllIlllIlllIIIl %(IIIIlIllllIIIllll .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IlIIlllIIlIIIllII .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IIllIIIIIlllIIIlI .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IlIIlIIlllIIlllll .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IIllllIlIlIIIllll ))
		else :IlIIllIlIllIIlIII .append (_IIIlIIIllIlllIIIl %(IIIIlIllllIIIllll .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IlIIlllIIlIIIllII .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIIlIlIIllIlllIII ,IIllllIlIlIIIllll ))
	IlIlIllIlIlllllII =256 if IIllIllllIIIlIIll ==_IlllllIlIllIIllII else 768 
	if IIlIIIIlIlIlIIIll :
		for _IIlllIlIlIlIIllII in range (2 ):IlIIllIlIllIIlIII .append (_IllllIlIIlIIlIIll %(IIIIlIlIIIlIllIlI ,IllIIIIIIlIIlIIII ,IIIIlIlIIIlIllIlI ,IlIlIllIlIlllllII ,IIIIlIlIIIlIllIlI ,IIIIlIlIIIlIllIlI ,IIllllIlIlIIIllll ))
	else :
		for _IIlllIlIlIlIIllII in range (2 ):IlIIllIlIllIIlIII .append (_IIlllIlIlIlllIllI %(IIIIlIlIIIlIllIlI ,IllIIIIIIlIIlIIII ,IIIIlIlIIIlIllIlI ,IlIlIllIlIlllllII ,IIllllIlIlIIIllll ))
	shuffle (IlIIllIlIllIIlIII )
	with open (_IllllIllIIlIlIIlI %IlIllllIIlIIIIlII ,'w')as IIIIIIIIIIIIlIlIl :IIIIIIIIIIIIlIlIl .write (_IIlIlIIIlIIlIIIll .join (IlIIllIlIllIIlIII ))
	print (_IlIIIIlIIlIIllIIl );print ('use gpus:',IlIIIlIIlIllIllll )
	if IIIllIIIIIlllIlII =='':print ('no pretrained Generator')
	if IlllIlIlIIIIlIlIl =='':print ('no pretrained Discriminator')
	if IlIIIlIIlIllIllll :IIIIIIIIlllllIlII =IlllIIlIlIIIIlIlI .python_cmd +_IIIllllIlIllIIlll %(IllIlllIllIllIIlI ,IllIIIIIIlIIlIIII ,1 if IIlIIIIlIlIlIIIll else 0 ,IlIIIlIllIllIIlIl ,IlIIIlIIlIllIllll ,IIIIlIllIllIIllIl ,IIIlllIllIlllIllI ,_IlIlllIIIllIllIll %IIIllIIIIIlllIlII if IIIllIIIIIlllIlII !=''else '',_IIIlIIIllIIIlIllI %IlllIlIlIIIIlIlIl if IlllIlIlIIIIlIlIl !=''else '',1 if IlllllIllIlIIIlII ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IIIIIlIIlIlIIlIll ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IlIIlllIIIlllllII ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,IIllIllllIIIlIIll )
	else :IIIIIIIIlllllIlII =IlllIIlIlIIIIlIlI .python_cmd +_IlIlIllIIIllIllll %(IllIlllIllIllIIlI ,IllIIIIIIlIIlIIII ,1 if IIlIIIIlIlIlIIIll else 0 ,IlIIIlIllIllIIlIl ,IIIIlIllIllIIllIl ,IIIlllIllIlllIllI ,_IlIlllIIIllIllIll %IIIllIIIIIlllIlII if IIIllIIIIIlllIlII !=''else IIIlIIlIIIIlIIIIl ,_IIIlIIIllIIIlIllI %IlllIlIlIIIIlIlIl if IlllIlIlIIIIlIlIl !=''else IIIlIIlIIIIlIIIIl ,1 if IlllllIllIlIIIlII ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IIIIIlIIlIlIIlIll ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IlIIlllIIIlllllII ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,IIllIllllIIIlIIll )
	print (IIIIIIIIlllllIlII );IIllIllllllllIIlI =Popen (IIIIIIIIlllllIlII ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIllIllllllllIIlI .wait ();return _IlllIIlIlIIlIlIll 
def IIlIlIllIlllIIIII (IIIIIIlIIlIIIlIII ,IllIIlIIlIllIIIII ):
	IIIIlIlIIlIIIlIlI =_IIllIIlllllIIIllI %(IIIIlIlIIIlIllIlI ,IIIIIIlIIlIIIlIII );os .makedirs (IIIIlIlIIlIIIlIlI ,exist_ok =_IlIlIIllIlIIIllIl );IIIIIlIllllllIllI =_IIllIlIlIIlllIlII %IIIIlIlIIlIIIlIlI if IllIIlIIlIllIIIII ==_IlllllIlIllIIllII else _IIllIlllIIlIIIIIl %IIIIlIlIIlIIIlIlI 
	if not os .path .exists (IIIIIlIllllllIllI ):return '请先进行特征提取!'
	IIllIllIlIIIIlIII =list (os .listdir (IIIIIlIllllllIllI ))
	if len (IIllIllIlIIIIlIII )==0 :return '请先进行特征提取！'
	IIlllIIIlllIlIIlI =[];IlIIlIlllIIlIIIlI =[]
	for IllIlIlIlIlIIlllI in sorted (IIllIllIlIIIIlIII ):IIIIlllIIIIlIIIII =np .load (_IIlllIIlIIlIIllll %(IIIIIlIllllllIllI ,IllIlIlIlIlIIlllI ));IlIIlIlllIIlIIIlI .append (IIIIlllIIIIlIIIII )
	IlIllIIIIIIllIlll =np .concatenate (IlIIlIlllIIlIIIlI ,0 );IllIllIllllIllllI =np .arange (IlIllIIIIIIllIlll .shape [0 ]);np .random .shuffle (IllIllIllllIllllI );IlIllIIIIIIllIlll =IlIllIIIIIIllIlll [IllIllIllllIllllI ]
	if IlIllIIIIIIllIlll .shape [0 ]>2e5 :
		IIlllIIIlllIlIIlI .append (_IIIIlIIlIIlllIlII %IlIllIIIIIIllIlll .shape [0 ]);yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI )
		try :IlIllIIIIIIllIlll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IlIlIIllIlIIIllIl ,batch_size =256 *IlllIIlIlIIIIlIlI .n_cpu ,compute_labels =_IIIIlIllllIIIlllI ,init ='random').fit (IlIllIIIIIIllIlll ).cluster_centers_ 
		except :IlIIIllIIllIIIIlI =traceback .format_exc ();print (IlIIIllIIllIIIIlI );IIlllIIIlllIlIIlI .append (IlIIIllIIllIIIIlI );yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI )
	np .save (_IlIIlIIlIlIIlIlIl %IIIIlIlIIlIIIlIlI ,IlIllIIIIIIllIlll );IlIIIIlIIlIlllIII =min (int (16 *np .sqrt (IlIllIIIIIIllIlll .shape [0 ])),IlIllIIIIIIllIlll .shape [0 ]//39 );IIlllIIIlllIlIIlI .append ('%s,%s'%(IlIllIIIIIIllIlll .shape ,IlIIIIlIIlIlllIII ));yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI );IlIIlIlIIlIllIIIl =faiss .index_factory (256 if IllIIlIIlIllIIIII ==_IlllllIlIllIIllII else 768 ,_IlIIIIllllIIIlIIl %IlIIIIlIIlIlllIII );IIlllIIIlllIlIIlI .append ('training');yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI );IIlIlIIIIIllllIll =faiss .extract_index_ivf (IlIIlIlIIlIllIIIl );IIlIlIIIIIllllIll .nprobe =1 ;IlIIlIlIIlIllIIIl .train (IlIllIIIIIIllIlll );faiss .write_index (IlIIlIlIIlIllIIIl ,_IllllIIIIlIlIlllI %(IIIIlIlIIlIIIlIlI ,IlIIIIlIIlIlllIII ,IIlIlIIIIIllllIll .nprobe ,IIIIIIlIIlIIIlIII ,IllIIlIIlIllIIIII ));IIlllIIIlllIlIIlI .append ('adding');yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI );IIIlIlllIIIIIlllI =8192 
	for IllllllIIlIIlIlII in range (0 ,IlIllIIIIIIllIlll .shape [0 ],IIIlIlllIIIIIlllI ):IlIIlIlIIlIllIIIl .add (IlIllIIIIIIllIlll [IllllllIIlIIlIlII :IllllllIIlIIlIlII +IIIlIlllIIIIIlllI ])
	faiss .write_index (IlIIlIlIIlIllIIIl ,_IIlIIlllllIlIlIII %(IIIIlIlIIlIIIlIlI ,IlIIIIlIIlIlllIII ,IIlIlIIIIIllllIll .nprobe ,IIIIIIlIIlIIIlIII ,IllIIlIIlIllIIIII ));IIlllIIIlllIlIIlI .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIIIlIIlIlllIII ,IIlIlIIIIIllllIll .nprobe ,IIIIIIlIIlIIIlIII ,IllIIlIIlIllIIIII ));yield _IIlIlIIIlIIlIIIll .join (IIlllIIIlllIlIIlI )
def IllllllIlllIIllII (IIlIlllIllIIlllll ,IlIllIIIllIlIIIII ,IIlIIlIllIIllIIlI ,IIIIIlIlllIlIllIl ,IIlIIllIlllllIIlI ,IlIlllIllIlIIIlll ,IIIlllIlIllIIlIII ,IIlllIllllIIIIllI ,IlIllIlIllIIIlIII ,IIIIIIIIIIlIllIlI ,IlIlllIlIlIIlIIlI ,IlIIlIlIllIIIlIII ,IIlIIIIlIIlIIIlIl ,IlllIllIllllIIlll ,IllIIIIIllIIIIIll ,IlllIIIIlIIIlIlIl ,IIlIllIlllIlIlllI ,IIlIIIllIIllIllII ):
	IIlIllIlIlllIIIll =[]
	def IIIlllllIIlIIlIII (IIIlllIlIIIIIIlIl ):IIlIllIlIlllIIIll .append (IIIlllIlIIIIIIlIl );return _IIlIlIIIlIIlIIIll .join (IIlIllIlIlllIIIll )
	IllllIlIlIIllllIl =_IIllIIlllllIIIllI %(IIIIlIlIIIlIllIlI ,IIlIlllIllIIlllll );IllllllIIIIlIIllI ='%s/preprocess.log'%IllllIlIlIIllllIl ;IlIIIIllIlIlllllI ='%s/extract_fl_feature.log'%IllllIlIlIIllllIl ;IIlIIlIlIIIlIlIII =_IIlIllIIIIllIlIII %IllllIlIlIIllllIl ;IllIIlIllIIIllIlI =_IIllIlIlIIlllIlII %IllllIlIlIIllllIl if IIlIllIlllIlIlllI ==_IlllllIlIllIIllII else _IIllIlllIIlIIIIIl %IllllIlIlIIllllIl ;os .makedirs (IllllIlIlIIllllIl ,exist_ok =_IlIlIIllIlIIIllIl );open (IllllllIIIIlIIllI ,'w').close ();IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IIIIIlIlllIlIllIl ,IlIIIllIlllllIlII [IlIllIIIllIlIIIII ],IlIlllIllIlIIIlll ,IllllIlIlIIllllIl )+str (IlllIIlIlIIIIlIlI .noparallel );yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII ('step1:正在处理数据'));yield IIIlllllIIlIIlIII (IIllIllIIIlIlllIl );IlllIlIlllIlIlIll =Popen (IIllIllIIIlIlllIl ,shell =_IlIlIIllIlIIIllIl );IlllIlIlllIlIlIll .wait ()
	with open (IllllllIIIIlIIllI ,_IllIIIIlllllIllIl )as IllIIlllllIllIlll :print (IllIIlllllIllIlll .read ())
	open (IlIIIIllIlIlllllI ,'w')
	if IIlIIlIllIIllIIlI :
		yield IIIlllllIIlIIlIII ('step2a:正在提取音高')
		if IIIlllIlIllIIlIII !=_IIlIllllIlIlIlIll :IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +' extract_fl_print.py "%s" %s %s'%(IllllIlIlIIllllIl ,IlIlllIllIlIIIlll ,IIIlllIlIllIIlIII );yield IIIlllllIIlIIlIII (IIllIllIIIlIlllIl );IlllIlIlllIlIlIll =Popen (IIllIllIIIlIlllIl ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IlllIlIlllIlIlIll .wait ()
		else :
			IIlIIIllIIllIllII =IIlIIIllIIllIllII .split ('-');IllIlIllIIllIllIl =len (IIlIIIllIIllIllII );IIIIlIllIIIlllIII =[]
			for (IIlIIlllIIIllIIlI ,IllIlIlIIlIIlIlII )in enumerate (IIlIIIllIIllIllII ):IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IllIlIllIIllIllIl ,IIlIIlllIIIllIIlI ,IllIlIlIIlIIlIlII ,IllllIlIlIIllllIl ,IlllIIlIlIIIIlIlI .is_half );yield IIIlllllIIlIIlIII (IIllIllIIIlIlllIl );IlllIlIlllIlIlIll =Popen (IIllIllIIIlIlllIl ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIIIlIllIIIlllIII .append (IlllIlIlllIlIlIll )
			for IlllIlIlllIlIlIll in IIIIlIllIIIlllIII :IlllIlIlllIlIlIll .wait ()
		with open (IlIIIIllIlIlllllI ,_IllIIIIlllllIllIl )as IllIIlllllIllIlll :print (IllIIlllllIllIlll .read ())
	else :yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII ('step2a:无需提取音高'))
	yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII ('step2b:正在提取特征'));IllIIIlIIlIIIlIll =IlllIllIllllIIlll .split ('-');IllIlIllIIllIllIl =len (IllIIIlIIlIIIlIll );IIIIlIllIIIlllIII =[]
	for (IIlIIlllIIIllIIlI ,IllIlIlIIlIIlIlII )in enumerate (IllIIIlIIlIIIlIll ):IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IlllIIlIlIIIIlIlI .device ,IllIlIllIIllIllIl ,IIlIIlllIIIllIIlI ,IllIlIlIIlIIlIlII ,IllllIlIlIIllllIl ,IIlIllIlllIlIlllI );yield IIIlllllIIlIIlIII (IIllIllIIIlIlllIl );IlllIlIlllIlIlIll =Popen (IIllIllIIIlIlllIl ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IIIIlIllIIIlllIII .append (IlllIlIlllIlIlIll )
	for IlllIlIlllIlIlIll in IIIIlIllIIIlllIII :IlllIlIlllIlIlIll .wait ()
	with open (IlIIIIllIlIlllllI ,_IllIIIIlllllIllIl )as IllIIlllllIllIlll :print (IllIIlllllIllIlll .read ())
	yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII ('step3a:正在训练模型'))
	if IIlIIlIllIIllIIlI :IlIllIllIllIlllII ='%s/2a_f0'%IllllIlIlIIllllIl ;IIlIlIIllIIIlIIll =_IllllIIllllIIllIl %IllllIlIlIIllllIl ;IIlIIIIIllIIIIlIl =set ([IIllllllIlIIlIlII .split (_IlIIIIllIIIlIllll )[0 ]for IIllllllIlIIlIlII in os .listdir (IIlIIlIlIIIlIlIII )])&set ([IlIIlIIlIIIlIlllI .split (_IlIIIIllIIIlIllll )[0 ]for IlIIlIIlIIIlIlllI in os .listdir (IllIIlIllIIIllIlI )])&set ([IllIIlllIIIlIIIII .split (_IlIIIIllIIIlIllll )[0 ]for IllIIlllIIIlIIIII in os .listdir (IlIllIllIllIlllII )])&set ([IllIlllIlIIlllllI .split (_IlIIIIllIIIlIllll )[0 ]for IllIlllIlIIlllllI in os .listdir (IIlIlIIllIIIlIIll )])
	else :IIlIIIIIllIIIIlIl =set ([IIllIllIIIIlIlIlI .split (_IlIIIIllIIIlIllll )[0 ]for IIllIllIIIIlIlIlI in os .listdir (IIlIIlIlIIIlIlIII )])&set ([IIllIlIlllIlIIIll .split (_IlIIIIllIIIlIllll )[0 ]for IIllIlIlllIlIIIll in os .listdir (IllIIlIllIIIllIlI )])
	IIIlIIIIIlIllIIlI =[]
	for IIlIIIlIllllllIII in IIlIIIIIllIIIIlIl :
		if IIlIIlIllIIllIIlI :IIIlIIIIIlIllIIlI .append (_IIlllIlllIlllIIIl %(IIlIIlIlIIIlIlIII .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IllIIlIllIIIllIlI .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IlIllIllIllIlllII .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IIlIlIIllIIIlIIll .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IIlIIllIlllllIIlI ))
		else :IIIlIIIIIlIllIIlI .append (_IIIlIIIllIlllIIIl %(IIlIIlIlIIIlIlIII .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IllIIlIllIIIllIlI .replace (_IIlIllllIIllllIll ,_IllIIIlIlllllIlll ),IIlIIIlIllllllIII ,IIlIIllIlllllIIlI ))
	IIIIIIllllIlllIIl =256 if IIlIllIlllIlIlllI ==_IlllllIlIllIIllII else 768 
	if IIlIIlIllIIllIIlI :
		for _IIIIIIllIIlIlIlII in range (2 ):IIIlIIIIIlIllIIlI .append (_IllllIlIIlIIlIIll %(IIIIlIlIIIlIllIlI ,IlIllIIIllIlIIIII ,IIIIlIlIIIlIllIlI ,IIIIIIllllIlllIIl ,IIIIlIlIIIlIllIlI ,IIIIlIlIIIlIllIlI ,IIlIIllIlllllIIlI ))
	else :
		for _IIIIIIllIIlIlIlII in range (2 ):IIIlIIIIIlIllIIlI .append (_IIlllIlIlIlllIllI %(IIIIlIlIIIlIllIlI ,IlIllIIIllIlIIIII ,IIIIlIlIIIlIllIlI ,IIIIIIllllIlllIIl ,IIlIIllIlllllIIlI ))
	shuffle (IIIlIIIIIlIllIIlI )
	with open (_IllllIllIIlIlIIlI %IllllIlIlIIllllIl ,'w')as IllIIlllllIllIlll :IllIIlllllIllIlll .write (_IIlIlIIIlIIlIIIll .join (IIIlIIIIIlIllIIlI ))
	yield IIIlllllIIlIIlIII (_IlIIIIlIIlIIllIIl )
	if IlllIllIllllIIlll :IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +_IIIllllIlIllIIlll %(IIlIlllIllIIlllll ,IlIllIIIllIlIIIII ,1 if IIlIIlIllIIllIIlI else 0 ,IIIIIIIIIIlIllIlI ,IlllIllIllllIIlll ,IlIllIlIllIIIlIII ,IIlllIllllIIIIllI ,_IlIlllIIIllIllIll %IlIIlIlIllIIIlIII if IlIIlIlIllIIIlIII !=''else '',_IIIlIIIllIIIlIllI %IIlIIIIlIIlIIIlIl if IIlIIIIlIIlIIIlIl !=''else '',1 if IlIlllIlIlIIlIIlI ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IllIIIIIllIIIIIll ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IlllIIIIlIIIlIlIl ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,IIlIllIlllIlIlllI )
	else :IIllIllIIIlIlllIl =IlllIIlIlIIIIlIlI .python_cmd +_IlIlIllIIIllIllll %(IIlIlllIllIIlllll ,IlIllIIIllIlIIIII ,1 if IIlIIlIllIIllIIlI else 0 ,IIIIIIIIIIlIllIlI ,IlIllIlIllIIIlIII ,IIlllIllllIIIIllI ,_IlIlllIIIllIllIll %IlIIlIlIllIIIlIII if IlIIlIlIllIIIlIII !=''else '',_IIIlIIIllIIIlIllI %IIlIIIIlIIlIIIlIl if IIlIIIIlIIlIIIlIl !=''else '',1 if IlIlllIlIlIIlIIlI ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IllIIIIIllIIIIIll ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,1 if IlllIIIIlIIIlIlIl ==IlIIlllIlllIlllII (_IlllIlllIlllIllIl )else 0 ,IIlIllIlllIlIlllI )
	yield IIIlllllIIlIIlIII (IIllIllIIIlIlllIl );IlllIlIlllIlIlIll =Popen (IIllIllIIIlIlllIl ,shell =_IlIlIIllIlIIIllIl ,cwd =IIIIlIlIIIlIllIlI );IlllIlIlllIlIlIll .wait ();yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII (_IlllIIlIlIIlIlIll ));IIlllIIIlIIlllIlI =[];IlllIIlIlIllIIIII =list (os .listdir (IllIIlIllIIIllIlI ))
	for IIlIIIlIllllllIII in sorted (IlllIIlIlIllIIIII ):IIlIIlIlIllllIIll =np .load (_IIlllIIlIIlIIllll %(IllIIlIllIIIllIlI ,IIlIIIlIllllllIII ));IIlllIIIlIIlllIlI .append (IIlIIlIlIllllIIll )
	IIIIlIIIIlIIIIllI =np .concatenate (IIlllIIIlIIlllIlI ,0 );IIllIIIIlIllIlIII =np .arange (IIIIlIIIIlIIIIllI .shape [0 ]);np .random .shuffle (IIllIIIIlIllIlIII );IIIIlIIIIlIIIIllI =IIIIlIIIIlIIIIllI [IIllIIIIlIllIlIII ]
	if IIIIlIIIIlIIIIllI .shape [0 ]>2e5 :
		IlIllllIIlllIllII =_IIIIlIIlIIlllIlII %IIIIlIIIIlIIIIllI .shape [0 ];print (IlIllllIIlllIllII );yield IIIlllllIIlIIlIII (IlIllllIIlllIllII )
		try :IIIIlIIIIlIIIIllI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IlIlIIllIlIIIllIl ,batch_size =256 *IlllIIlIlIIIIlIlI .n_cpu ,compute_labels =_IIIIlIllllIIIlllI ,init ='random').fit (IIIIlIIIIlIIIIllI ).cluster_centers_ 
		except :IlIllllIIlllIllII =traceback .format_exc ();print (IlIllllIIlllIllII );yield IIIlllllIIlIIlIII (IlIllllIIlllIllII )
	np .save (_IlIIlIIlIlIIlIlIl %IllllIlIlIIllllIl ,IIIIlIIIIlIIIIllI );IlIIlIIllIIIllIIl =min (int (16 *np .sqrt (IIIIlIIIIlIIIIllI .shape [0 ])),IIIIlIIIIlIIIIllI .shape [0 ]//39 );yield IIIlllllIIlIIlIII ('%s,%s'%(IIIIlIIIIlIIIIllI .shape ,IlIIlIIllIIIllIIl ));IlllIIlIIIlllIIIl =faiss .index_factory (256 if IIlIllIlllIlIlllI ==_IlllllIlIllIIllII else 768 ,_IlIIIIllllIIIlIIl %IlIIlIIllIIIllIIl );yield IIIlllllIIlIIlIII ('training index');IlIIlIllIIlIIlIlI =faiss .extract_index_ivf (IlllIIlIIIlllIIIl );IlIIlIllIIlIIlIlI .nprobe =1 ;IlllIIlIIIlllIIIl .train (IIIIlIIIIlIIIIllI );faiss .write_index (IlllIIlIIIlllIIIl ,_IllllIIIIlIlIlllI %(IllllIlIlIIllllIl ,IlIIlIIllIIIllIIl ,IlIIlIllIIlIIlIlI .nprobe ,IIlIlllIllIIlllll ,IIlIllIlllIlIlllI ));yield IIIlllllIIlIIlIII ('adding index');IIIlIIIlllllllIlI =8192 
	for IlIllIIIlIllllIIl in range (0 ,IIIIlIIIIlIIIIllI .shape [0 ],IIIlIIIlllllllIlI ):IlllIIlIIIlllIIIl .add (IIIIlIIIIlIIIIllI [IlIllIIIlIllllIIl :IlIllIIIlIllllIIl +IIIlIIIlllllllIlI ])
	faiss .write_index (IlllIIlIIIlllIIIl ,_IIlIIlllllIlIlIII %(IllllIlIlIIllllIl ,IlIIlIIllIIIllIIl ,IlIIlIllIIlIIlIlI .nprobe ,IIlIlllIllIIlllll ,IIlIllIlllIlIlllI ));yield IIIlllllIIlIIlIII ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIlIIllIIIllIIl ,IlIIlIllIIlIIlIlI .nprobe ,IIlIlllIllIIlllll ,IIlIllIlllIlIlllI ));yield IIIlllllIIlIIlIII (IlIIlllIlllIlllII ('全流程结束！'))
def IlIIllIIlIIllIIIl (IlllIIlIlIIllIIIl ):
	IIIlIIllIlllIlIII ='train.log'
	if not os .path .exists (IlllIIlIlIIllIIIl .replace (os .path .basename (IlllIIlIlIIllIIIl ),IIIlIIllIlllIlIII )):return {_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },{_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },{_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
	try :
		with open (IlllIIlIlIIllIIIl .replace (os .path .basename (IlllIIlIlIIllIIIl ),IIIlIIllIlllIlIII ),_IllIIIIlllllIllIl )as IlIIIlIIlIlIIIIll :IlIllIIIlIllllIll =eval (IlIIIlIIlIlIIIIll .read ().strip (_IIlIlIIIlIIlIIIll ).split (_IIlIlIIIlIIlIIIll )[0 ].split ('\t')[-1 ]);IIIIlIlIIlIllIIlI ,IIlllIIlllllIlIll =IlIllIIIlIllllIll [_IlIIllIIlIIlIllII ],IlIllIIIlIllllIll ['if_f0'];IlIIllIIIlIllllII =_IIIlllllllIllIIll if _IIIIlIIIlIlllIlll in IlIllIIIlIllllIll and IlIllIIIlIllllIll [_IIIIlIIIlIlllIlll ]==_IIIlllllllIllIIll else _IlllllIlIllIIllII ;return IIIIlIlIIlIllIIlI ,str (IIlllIIlllllIlIll ),IlIIllIIIlIllllII 
	except :traceback .print_exc ();return {_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },{_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll },{_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
def IlIllIllllIlIlIII (IlIIIIlIIllIlIIll ):
	if IlIIIIlIIllIlIIll ==_IIlIllllIlIlIlIll :IlIIIIIIIIIIIIIII =_IlIlIIllIlIIIllIl 
	else :IlIIIIIIIIIIIIIII =_IIIIlIllllIIIlllI 
	return {_IIIIlIlIlllllIlII :IlIIIIIIIIIIIIIII ,_IlIIIllIlIlIIlllI :_IIlIlIIIIllIIIIll }
def IIIIlIIlIIlIlllIl (IlllIIIlIlIlllIIl ,IIllIIlIIIllllIIl ):IIIIlllllIllllIII ='rnd';IIIllIllIlIIIllII ='pitchf';IlIlllIIIllIlIlll ='pitch';IIIIIlIlllIIIllIl ='phone';global IllIllIlllIllIIll ;IllIllIlllIllIIll =torch .load (IlllIIIlIlIlllIIl ,map_location =_IlIlIIlIllIllIIll );IllIllIlllIllIIll [_IlllllllIlllllllI ][-3 ]=IllIllIlllIllIIll [_IIIIIllIIIIIIIlll ][_IlllllIIIIIIIIlIl ].shape [0 ];IlIlIllIIIllIlIll =256 if IllIllIlllIllIIll .get (_IIIIlIIIlIlllIlll ,_IlllllIlIllIIllII )==_IlllllIlIllIIllII else 768 ;IIIIlIIIllllllIll =torch .rand (1 ,200 ,IlIlIllIIIllIlIll );IIlllIIlIlIlIIIlI =torch .tensor ([200 ]).long ();IllllIlIlIIIIIIII =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IIIlIllIlIIllllIl =torch .rand (1 ,200 );IlIllllIllllIlllI =torch .LongTensor ([0 ]);IllIllIlIIIlIlIll =torch .rand (1 ,192 ,200 );IllIIIlIIllllllII =_IlIlIIlIllIllIIll ;IllIlIllIIIllIlIl =SynthesizerTrnMsNSFsidM (*IllIllIlllIllIIll [_IlllllllIlllllllI ],is_half =_IIIIlIllllIIIlllI ,version =IllIllIlllIllIIll .get (_IIIIlIIIlIlllIlll ,_IlllllIlIllIIllII ));IllIlIllIIIllIlIl .load_state_dict (IllIllIlllIllIIll [_IIIIIllIIIIIIIlll ],strict =_IIIIlIllllIIIlllI );IllllIIlIlIllIlIl =[IIIIIlIlllIIIllIl ,'phone_lengths',IlIlllIIIllIlIlll ,IIIllIllIlIIIllII ,'ds',IIIIlllllIllllIII ];IlIlIlllIIIlIIlIl =['audio'];torch .onnx .export (IllIlIllIIIllIlIl ,(IIIIlIIIllllllIll .to (IllIIIlIIllllllII ),IIlllIIlIlIlIIIlI .to (IllIIIlIIllllllII ),IllllIlIlIIIIIIII .to (IllIIIlIIllllllII ),IIIlIllIlIIllllIl .to (IllIIIlIIllllllII ),IlIllllIllllIlllI .to (IllIIIlIIllllllII ),IllIllIlIIIlIlIll .to (IllIIIlIIllllllII )),IIllIIlIIIllllIIl ,dynamic_axes ={IIIIIlIlllIIIllIl :[1 ],IlIlllIIIllIlIlll :[1 ],IIIllIllIlIIIllII :[1 ],IIIIlllllIllllIII :[2 ]},do_constant_folding =_IIIIlIllllIIIlllI ,opset_version =13 ,verbose =_IIIIlIllllIIIlllI ,input_names =IllllIIlIlIllIlIl ,output_names =IlIlIlllIIIlIIlIl );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IlllIlllIIIIllllI :
	gr .Markdown (value =IlIIlllIlllIlllII ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IlIIlllIlllIlllII ('模型推理')):
			with gr .Row ():IlllIlllIllIIlIlI =gr .Dropdown (label =IlIIlllIlllIlllII ('推理音色'),choices =sorted (IIlIlllIlIIlllllI ));IIllIlIlIlllIlIII =gr .Button (IlIIlllIlllIlllII ('刷新音色列表和索引路径'),variant =_IIIllIlIIIlllllll );IllIlIlllIlIlIllI =gr .Button (IlIIlllIlllIlllII ('卸载音色省显存'),variant =_IIIllIlIIIlllllll );IlllIIlIlIlIlIIIl =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IlIIlllIlllIlllII ('请选择说话人id'),value =0 ,visible =_IIIIlIllllIIIlllI ,interactive =_IlIlIIllIlIIIllIl );IllIlIlllIlIlIllI .click (fn =IlIIllIIlIllIIIll ,inputs =[],outputs =[IlllIlllIllIIlIlI ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IlllIlIIIllllIlII =gr .Number (label =IlIIlllIlllIlllII (_IIlIIlllIllIlIIlI ),value =0 );IlIlIIIllllIlIllI =gr .Textbox (label =IlIIlllIlllIlllII ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IIlIIIIIIIIlllIlI =gr .Radio (label =IlIIlllIlllIlllII (_IIlIllIIlIIllllll ),choices =[_IlIIlIllIllIlIllI ,_IIllIlIlIIIIIllIl ,'crepe',_IIIIlIlllIllIIIIl ],value =_IlIIlIllIllIlIllI ,interactive =_IlIlIIllIlIIIllIl );IlIllllIIllIIlllI =gr .Slider (minimum =0 ,maximum =7 ,label =IlIIlllIlllIlllII (_IIllIIIlIIlIllIII ),value =3 ,step =1 ,interactive =_IlIlIIllIlIIIllIl )
					with gr .Column ():IllIIllIlIIIlIIll =gr .Textbox (label =IlIIlllIlllIlllII (_IlIIlIIIIlIlllIII ),value ='',interactive =_IlIlIIllIlIIIllIl );IlllllIIlIllIIlIl =gr .Dropdown (label =IlIIlllIlllIlllII (_IlllllIllIllIllIl ),choices =sorted (IIlIllIlIIIlllIll ),interactive =_IlIlIIllIlIIIllIl );IIllIlIlIlllIlIII .click (fn =IlIIIIllllIlIlIlI ,inputs =[],outputs =[IlllIlllIllIIlIlI ,IlllllIIlIllIIlIl ],api_name ='infer_refresh');IllIlllllIIlIllll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIlllIlllIlllII ('检索特征占比'),value =.75 ,interactive =_IlIlIIllIlIIIllIl )
					with gr .Column ():IllIIIlllIllIIIll =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIIlllIlllIlllII (_IIlllllllllllIlII ),value =0 ,step =1 ,interactive =_IlIlIIllIlIIIllIl );IllIIlIIIlIlIlIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIlllIlllIlllII (_IllIlIlIIllIIlIII ),value =.25 ,interactive =_IlIlIIllIlIIIllIl );IIllIlIIlIllIlIIl =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIIlllIlllIlllII (_IIllIllllllIllllI ),value =.33 ,step =.01 ,interactive =_IlIlIIllIlIIIllIl )
					IlIlIIIIIllIlllll =gr .File (label =IlIIlllIlllIlllII ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IlIlIIlIIIlIIIlIl =gr .Button (IlIIlllIlllIlllII ('转换'),variant =_IIIllIlIIIlllllll )
					with gr .Row ():IIlllIlIlllllIIll =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ));IllIIlIIIllllIllI =gr .Audio (label =IlIIlllIlllIlllII ('输出音频(右下角三个点,点了可以下载)'))
					IlIlIIlIIIlIIIlIl .click (IIIlIlIlllIlIIIIl ,[IlllIIlIlIlIlIIIl ,IlIlIIIllllIlIllI ,IlllIlIIIllllIlII ,IlIlIIIIIllIlllll ,IIlIIIIIIIIlllIlI ,IllIIllIlIIIlIIll ,IlllllIIlIllIIlIl ,IllIlllllIIlIllll ,IlIllllIIllIIlllI ,IllIIIlllIllIIIll ,IllIIlIIIlIlIlIIl ,IIllIlIIlIllIlIIl ],[IIlllIlIlllllIIll ,IllIIlIIIllllIllI ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IIlIIIIIlIlllllII =gr .Number (label =IlIIlllIlllIlllII (_IIlIIlllIllIlIIlI ),value =0 );IIIllllllIllIlllI =gr .Textbox (label =IlIIlllIlllIlllII ('指定输出文件夹'),value =_IlIIIIlllIIIllIII );IIIIlIIlIIIIlllII =gr .Radio (label =IlIIlllIlllIlllII (_IIlIllIIlIIllllll ),choices =[_IlIIlIllIllIlIllI ,_IIllIlIlIIIIIllIl ,'crepe',_IIIIlIlllIllIIIIl ],value =_IlIIlIllIllIlIllI ,interactive =_IlIlIIllIlIIIllIl );IllllIIIIlIIllIll =gr .Slider (minimum =0 ,maximum =7 ,label =IlIIlllIlllIlllII (_IIllIIIlIIlIllIII ),value =3 ,step =1 ,interactive =_IlIlIIllIlIIIllIl )
					with gr .Column ():IIIIIIIIllIlIlIll =gr .Textbox (label =IlIIlllIlllIlllII (_IlIIlIIIIlIlllIII ),value ='',interactive =_IlIlIIllIlIIIllIl );IlIllIlIIIIIIIllI =gr .Dropdown (label =IlIIlllIlllIlllII (_IlllllIllIllIllIl ),choices =sorted (IIlIllIlIIIlllIll ),interactive =_IlIlIIllIlIIIllIl );IIllIlIlIlllIlIII .click (fn =lambda :IlIIIIllllIlIlIlI ()[1 ],inputs =[],outputs =IlIllIlIIIIIIIllI ,api_name ='infer_refresh_batch');IIIllIlIIIIlIlIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIlllIlllIlllII ('检索特征占比'),value =1 ,interactive =_IlIlIIllIlIIIllIl )
					with gr .Column ():IIlIIIlIIllIlllII =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIIlllIlllIlllII (_IIlllllllllllIlII ),value =0 ,step =1 ,interactive =_IlIlIIllIlIIIllIl );IllllIIlIIIlllIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIlllIlllIlllII (_IllIlIlIIllIIlIII ),value =1 ,interactive =_IlIlIIllIlIIIllIl );IIlIlIIlIlIIIIIll =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIIlllIlllIlllII (_IIllIllllllIllllI ),value =.33 ,step =.01 ,interactive =_IlIlIIllIlIIIllIl )
					with gr .Column ():IllIllllIlllIlllI =gr .Textbox (label =IlIIlllIlllIlllII ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IlIllllllIIllIlll =gr .File (file_count ='multiple',label =IlIIlllIlllIlllII (_IllIIIlllIlllllll ))
					with gr .Row ():IIlllllllIIIllIlI =gr .Radio (label =IlIIlllIlllIlllII ('导出文件格式'),choices =[_IIIIIIIlIIIlllIIl ,_IlllIlllIIIIllIlI ,'mp3','m4a'],value =_IlllIlllIIIIllIlI ,interactive =_IlIlIIllIlIIIllIl );IlIlIIIIlIlIllIIl =gr .Button (IlIIlllIlllIlllII ('转换'),variant =_IIIllIlIIIlllllll );IIIllIIIlIlIIlllI =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ))
					IlIlIIIIlIlIllIIl .click (IIIlIIllIIIlllIII ,[IlllIIlIlIlIlIIIl ,IllIllllIlllIlllI ,IIIllllllIllIlllI ,IlIllllllIIllIlll ,IIlIIIIIlIlllllII ,IIIIlIIlIIIIlllII ,IIIIIIIIllIlIlIll ,IlIllIlIIIIIIIllI ,IIIllIlIIIIlIlIIl ,IllllIIIIlIIllIll ,IIlIIIlIIllIlllII ,IllllIIlIIIlllIIl ,IIlIlIIlIlIIIIIll ,IIlllllllIIIllIlI ],[IIIllIIIlIlIIlllI ],api_name ='infer_convert_batch')
			IlllIlllIllIIlIlI .change (fn =IIIllllIIIIIIIIll ,inputs =[IlllIlllIllIIlIlI ,IIllIlIIlIllIlIIl ,IIlIlIIlIlIIIIIll ],outputs =[IlllIIlIlIlIlIIIl ,IIllIlIIlIllIlIIl ,IIlIlIIlIlIIIIIll ,IlllllIIlIllIIlIl ])
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IIIIllIIllIIIlIll =gr .Textbox (label =IlIIlllIlllIlllII ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IIIlIlIIIlIIlIlIl =gr .File (file_count ='multiple',label =IlIIlllIlllIlllII (_IllIIIlllIlllllll ))
					with gr .Column ():IllIlIllIIlIIlllI =gr .Dropdown (label =IlIIlllIlllIlllII ('模型'),choices =IIlllllIlIlIlllIl );IIIIIIllllllIllIl =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IlIlIIllIlIIIllIl ,visible =_IIIIlIllllIIIlllI );IlIIIIllIIIllIIlI =gr .Textbox (label =IlIIlllIlllIlllII ('指定输出主人声文件夹'),value =_IlIIIIlllIIIllIII );IIIllIlIIIIIllIll =gr .Textbox (label =IlIIlllIlllIlllII ('指定输出非主人声文件夹'),value =_IlIIIIlllIIIllIII );IllIIIIIlIIllIlll =gr .Radio (label =IlIIlllIlllIlllII ('导出文件格式'),choices =[_IIIIIIIlIIIlllIIl ,_IlllIlllIIIIllIlI ,'mp3','m4a'],value =_IlllIlllIIIIllIlI ,interactive =_IlIlIIllIlIIIllIl )
					IlllIlllIIIIllIlI =gr .Button (IlIIlllIlllIlllII ('转换'),variant =_IIIllIlIIIlllllll );IIllIllIIllIlIIll =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ));IlllIlllIIIIllIlI .click (IlllIIlIIIllIlIlI ,[IllIlIllIIlIIlllI ,IIIIllIIllIIIlIll ,IlIIIIllIIIllIIlI ,IIIlIlIIIlIIlIlIl ,IIIllIlIIIIIllIll ,IIIIIIllllllIllIl ,IllIIIIIlIIllIlll ],[IIllIllIIllIlIIll ],api_name ='uvr_convert')
		with gr .TabItem (IlIIlllIlllIlllII ('训练')):
			gr .Markdown (value =IlIIlllIlllIlllII ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIIIIIllIlllIIlIl =gr .Textbox (label =IlIIlllIlllIlllII ('输入实验名'),value ='mi-test');IIIlllIlIlIIlIlll =gr .Radio (label =IlIIlllIlllIlllII ('目标采样率'),choices =[_IIIllIIlIlIllIIll ],value =_IIIllIIlIlIllIIll ,interactive =_IlIlIIllIlIIIllIl );IlIlllIlIIllllllI =gr .Radio (label =IlIIlllIlllIlllII ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IlIlIIllIlIIIllIl ,_IIIIlIllllIIIlllI ],value =_IlIlIIllIlIIIllIl ,interactive =_IlIlIIllIlIIIllIl );IlIllllllIIlllIll =gr .Radio (label =IlIIlllIlllIlllII ('版本'),choices =[_IIIlllllllIllIIll ],value =_IIIlllllllIllIIll ,interactive =_IlIlIIllIlIIIllIl ,visible =_IlIlIIllIlIIIllIl );IllIIlIllIIlIIIIl =gr .Slider (minimum =0 ,maximum =IlllIIlIlIIIIlIlI .n_cpu ,step =1 ,label =IlIIlllIlllIlllII ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IlllIIlIlIIIIlIlI .n_cpu /1.5 )),interactive =_IlIlIIllIlIIIllIl )
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IIlllllIIlIlIlIlI =gr .Textbox (label =IlIIlllIlllIlllII ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IIlIIIlllIlIlIlll =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IlIIlllIlllIlllII ('请指定说话人id'),value =0 ,interactive =_IlIlIIllIlIIIllIl );IlIlIIIIlIlIllIIl =gr .Button (IlIIlllIlllIlllII ('处理数据'),variant =_IIIllIlIIIlllllll );IlllIIllIlllllIll =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ),value ='');IlIlIIIIlIlIllIIl .click (IllIIIllllIlIIlIl ,[IIlllllIIlIlIlIlI ,IIIIIIllIlllIIlIl ,IIIlllIlIlIIlIlll ,IllIIlIllIIlIIIIl ],[IlllIIllIlllllIll ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IlIIllIIlIIllIlll =gr .Textbox (label =IlIIlllIlllIlllII (_IlIIIllIIlIIIIllI ),value =IlIIIllIIllIllIll ,interactive =_IlIlIIllIlIIIllIl );IIIIllllIIIIllllI =gr .Textbox (label =IlIIlllIlllIlllII ('显卡信息'),value =IIIlIlIIIIIlIllII )
					with gr .Column ():IIIlIlIlIllIllIll =gr .Radio (label =IlIIlllIlllIlllII ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IlIIlIllIllIlIllI ,_IIllIlIlIIIIIllIl ,'dio',_IIIIlIlllIllIIIIl ,_IIlIllllIlIlIlIll ],value =_IIlIllllIlIlIlIll ,interactive =_IlIlIIllIlIIIllIl );IlIllIIlllllIIlII =gr .Textbox (label =IlIIlllIlllIlllII ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IlIIIllIIllIllIll ,IlIIIllIIllIllIll ),interactive =_IlIlIIllIlIIIllIl ,visible =_IlIlIIllIlIIIllIl )
					IlllIlllIIIIllIlI =gr .Button (IlIIlllIlllIlllII ('特征提取'),variant =_IIIllIlIIIlllllll );IIlIIllIIlIllIIlI =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ),value ='',max_lines =8 );IIIlIlIlIllIllIll .change (fn =IlIllIllllIlIlIII ,inputs =[IIIlIlIlIllIllIll ],outputs =[IlIllIIlllllIIlII ]);IlllIlllIIIIllIlI .click (IIlllIIIIIIIlIlIl ,[IlIIllIIlIIllIlll ,IllIIlIllIIlIIIIl ,IIIlIlIlIllIllIll ,IlIlllIlIIllllllI ,IIIIIIllIlllIIlIl ,IlIllllllIIlllIll ,IlIllIIlllllIIlII ],[IIlIIllIIlIllIIlI ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IlIIlllIlllIlllII ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IlIlIllIIllllIIll =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IlIIlllIlllIlllII ('保存频率save_every_epoch'),value =5 ,interactive =_IlIlIIllIlIIIllIl );IlIlIlIllIlIllIll =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IlIIlllIlllIlllII ('总训练轮数total_epoch'),value =300 ,interactive =_IlIlIIllIlIIIllIl );IIlIlIllIIIIIIIlI =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IlIIlllIlllIlllII ('每张显卡的batch_size'),value =IlllIIlllllIllIII ,interactive =_IlIlIIllIlIIIllIl );IIIIlllIIllIIlIll =gr .Radio (label =IlIIlllIlllIlllII ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IlIIlllIlllIlllII (_IlllIlllIlllIllIl ),IlIIlllIlllIlllII ('否')],value =IlIIlllIlllIlllII (_IlllIlllIlllIllIl ),interactive =_IlIlIIllIlIIIllIl );IlIlllIllllIIllII =gr .Radio (label =IlIIlllIlllIlllII ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IlIIlllIlllIlllII (_IlllIlllIlllIllIl ),IlIIlllIlllIlllII ('否')],value =IlIIlllIlllIlllII ('否'),interactive =_IlIlIIllIlIIIllIl );IlllllIIIIIllllIl =gr .Radio (label =IlIIlllIlllIlllII ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IlIIlllIlllIlllII (_IlllIlllIlllIllIl ),IlIIlllIlllIlllII ('否')],value =IlIIlllIlllIlllII (_IlllIlllIlllIllIl ),interactive =_IlIlIIllIlIIIllIl )
				with gr .Row ():IlIIIlIlllllIlIIl =gr .Textbox (label =IlIIlllIlllIlllII ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IlIlIIllIlIIIllIl );IlIIllIllIlIllIlI =gr .Textbox (label =IlIIlllIlllIlllII ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IlIlIIllIlIIIllIl );IIIlllIlIlIIlIlll .change (IIIlIIllIIlllIlll ,[IIIlllIlIlIIlIlll ,IlIlllIlIIllllllI ,IlIllllllIIlllIll ],[IlIIIlIlllllIlIIl ,IlIIllIllIlIllIlI ]);IlIllllllIIlllIll .change (IIIIIlIIllllIlIll ,[IIIlllIlIlIIlIlll ,IlIlllIlIIllllllI ,IlIllllllIIlllIll ],[IlIIIlIlllllIlIIl ,IlIIllIllIlIllIlI ,IIIlllIlIlIIlIlll ]);IlIlllIlIIllllllI .change (IlllIIIllIlIIIIlI ,[IlIlllIlIIllllllI ,IIIlllIlIlIIlIlll ,IlIllllllIIlllIll ],[IIIlIlIlIllIllIll ,IlIIIlIlllllIlIIl ,IlIIllIllIlIllIlI ]);IlIIllIlIllIIIlIl =gr .Textbox (label =IlIIlllIlllIlllII (_IlIIIllIIlIIIIllI ),value =IlIIIllIIllIllIll ,interactive =_IlIlIIllIlIIIllIl );IlIlIllIlIIllIIll =gr .Button (IlIIlllIlllIlllII ('训练模型'),variant =_IIIllIlIIIlllllll );IlllllllIIlIllIll =gr .Button (IlIIlllIlllIlllII ('训练特征索引'),variant =_IIIllIlIIIlllllll );IlIIIIIIIIIIlIIlI =gr .Button (IlIIlllIlllIlllII ('一键训练'),variant =_IIIllIlIIIlllllll );IlIIIIIIIlIlIlIlI =gr .Textbox (label =IlIIlllIlllIlllII (_IIIIllllllIlllIlI ),value ='',max_lines =10 );IlIlIllIlIIllIIll .click (IlllIlIIIIllllIlI ,[IIIIIIllIlllIIlIl ,IIIlllIlIlIIlIlll ,IlIlllIlIIllllllI ,IIlIIIlllIlIlIlll ,IlIlIllIIllllIIll ,IlIlIlIllIlIllIll ,IIlIlIllIIIIIIIlI ,IIIIlllIIllIIlIll ,IlIIIlIlllllIlIIl ,IlIIllIllIlIllIlI ,IlIIllIlIllIIIlIl ,IlIlllIllllIIllII ,IlllllIIIIIllllIl ,IlIllllllIIlllIll ],IlIIIIIIIlIlIlIlI ,api_name ='train_start');IlllllllIIlIllIll .click (IIlIlIllIlllIIIII ,[IIIIIIllIlllIIlIl ,IlIllllllIIlllIll ],IlIIIIIIIlIlIlIlI );IlIIIIIIIIIIlIIlI .click (IllllllIlllIIllII ,[IIIIIIllIlllIIlIl ,IIIlllIlIlIIlIlll ,IlIlllIlIIllllllI ,IIlllllIIlIlIlIlI ,IIlIIIlllIlIlIlll ,IllIIlIllIIlIIIIl ,IIIlIlIlIllIllIll ,IlIlIllIIllllIIll ,IlIlIlIllIlIllIll ,IIlIlIllIIIIIIIlI ,IIIIlllIIllIIlIll ,IlIIIlIlllllIlIIl ,IlIIllIllIlIllIlI ,IlIIllIlIllIIIlIl ,IlIlllIllllIIllII ,IlllllIIIIIllllIl ,IlIllllllIIlllIll ,IlIllIIlllllIIlII ],IlIIIIIIIlIlIlIlI ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IllIIIIlllllIllIl ,encoding ='utf8')as IlIIllllIIIIIIIlI :IlIIllIIlIlIlIllI =IlIIllllIIIIIIIlI .read ()
				else :
					with open ('docs/faq_en.md',_IllIIIIlllllIllIl ,encoding ='utf8')as IlIIllllIIIIIIIlI :IlIIllIIlIlIlIllI =IlIIllllIIIIIIIlI .read ()
				gr .Markdown (value =IlIIllIIlIlIlIllI )
			except :gr .Markdown (traceback .format_exc ())
	if IlllIIlIlIIIIlIlI .iscolab :IlllIlllIIIIllllI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IlllIIlIlIIIIlIlI .listen_port ,share =_IIIIlIllllIIIlllI )
	else :IlllIlllIIIIllllI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IlllIIlIlIIIIlIlI .noautoopen ,server_port =IlllIIlIlIIIIlIlI .listen_port ,quiet =_IIIIlIllllIIIlllI ,share =_IIIIlIllllIIIlllI )
