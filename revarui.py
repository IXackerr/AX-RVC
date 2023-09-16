_IIIIIIllIllIllllI ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IIlIIllllllllIlIl ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IlIlIIIIlIlllllII ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIIllllIlllIIIIll ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IlIlIIIIlIIlIllIl ='后处理重采样至最终采样率，0为不进行重采样'
_IllllllIIIIlIIIlI ='自动检测index路径,下拉式选择(dropdown)'
_IllIIIIIIllIIIIIl ='特征检索库文件路径,为空则使用下拉的选择结果'
_IIllIllIlIIIlIlll ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IlIlIIlIIlllIllII ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IlIllIllllIIllIIl ='变调(整数, 半音数量, 升八度12降八度-12)'
_IIIlIllIlIllIIlll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IIlIIlIIlllllIlIl ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IllIlIIIlIIIllIIl ='IVF%s,Flat'
_IIllIlIllIIIIlIlI ='%s/total_fea.npy'
_IlIIlIIlIIIIlllIl ='Trying doing kmeans %s shape to 10k centers.'
_IllIIlIllllIIlIII ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IlllIlIlIlIlIlIll =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIlllIIIIllIIIIlI =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IllIllIlIlIIIlIlI ='write filelist done'
_IIlIIIlIIIllIlIlI ='%s/filelist.txt'
_IlIIllllIlllIllll ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IIllIIllIIlIllIII ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IIIlIIIIlIlIIllIl ='%s/%s.wav|%s/%s.npy|%s'
_IlIIIIllIIIIlIIll ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IIlIlllIIIlllIIII ='%s/2b-f0nsf'
_IIIIlIIllIlIlIIIl ='%s/0_gt_wavs'
_IlIlIlllIllllIIIl ='emb_g.weight'
_IIIIIlIIlllllIlII ='clean_empty_cache'
_IlIllllIlIlIIIlII ='sample_rate'
_IIllIIIIIllIIIIlI ='%s->%s'
_IIIIllIllllIlIlIl ='.index'
_IlIIlIIIIlllIllIl ='weights'
_IIIllIIIlllIIIIlI ='opt'
_IIIlIllIlIIllIIIl ='rmvpe'
_IlIlIIIlIIllllIlI ='harvest'
_IlIlIIIIlIllllIII ='%s/3_feature768'
_IlIlllIlIllllIIIl ='%s/3_feature256'
_IIIlIIIlIIllIIlIl ='_v2'
_IIlIIIlIIlIlIllIl ='48k'
_IlIlIlllIllllIIlI ='32k'
_IIIIlIIIllllIIllI ='cpu'
_IllIlIIllIlIllIlI ='wav'
_IIlllIIllIIllllll ='trained'
_IlllIlIIIllIlIlll ='logs'
_IIIIllIlIIlIllIll ='-pd %s'
_IIlllIllIlIIlIlIl ='-pg %s'
_IlIlIlIlIlIllIlll ='choices'
_IIIIllIlIIIIlIIlI ='weight'
_IIIlIllIlIIllIIIl ='pm'
_IIIllllIIllllIIlI ='rmvpe_gpu'
_IlIlIlIllIIlIIlIl ='%s/logs/%s'
_IllIlllIllllIIlIl ='flac'
_IIIIlIllIlllIIllI ='f0'
_IIlIllIlllIlIllII ='%s/%s'
_IIllIllIlIlIIIllI ='.pth'
_IlllIIllIlIIlllII ='输出信息'
_IIllIIllIlIIIIlll ='not exist, will not use pretrained model'
_IIlIIIlllllIllIlI ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IllllIIIIlllllIll ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IllIIIllIllIIlIII ='40k'
_IIIlIIIIIllIIIllI ='value'
_IIIIlIllllIIIlIIl ='v2'
_IIllIllllllIllIII ='version'
_IlllIIlIlIlIlllII ='visible'
_IllIIlIllllllllll ='primary'
_IlIlIllIlIlIIlIIl =None 
_IlIIlIllllIIIlIll ='\\\\'
_IIIIIllllllIllIll ='\\'
_IllIIIllllIlIlIII ='"'
_IlIlIIIlIIllIllIl =' '
_IlIlIIIIIIllIIIll ='config'
_IIlIIlllllIIlllIl ='.'
_IlIIlIIllllIIllII ='r'
_IIIIllIllIlllIIll ='是'
_IlIlIlIlIlIIIIlIl ='update'
_IllIllllIllIIIIll ='__type__'
_IlIlIIIlIlllIlIll ='v1'
_IlIlIlIlIIIIIIlIl =False 
_IlIlIllIIIlIIIlII ='\n'
_IllllIIIllIlIlIlI =True 
import os ,shutil ,sys 
IIIlllIlIlIlIllll =os .getcwd ()
sys .path .append (IIIlllIlIlIlIllll )
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
IIIlllIlIlIlIllll =os .getcwd ()
IlIlllIIIIllIlllI =os .path .join (IIIlllIlIlIlIllll ,'TEMP')
shutil .rmtree (IlIlllIIIIllIlllI ,ignore_errors =_IllllIIIllIlIlIlI )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IIIlllIlIlIlIllll ,ignore_errors =_IllllIIIllIlIlIlI )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IIIlllIlIlIlIllll ,ignore_errors =_IllllIIIllIlIlIlI )
os .makedirs (IlIlllIIIIllIlllI ,exist_ok =_IllllIIIllIlIlIlI )
os .makedirs (os .path .join (IIIlllIlIlIlIllll ,_IlllIlIIIllIlIlll ),exist_ok =_IllllIIIllIlIlIlI )
os .makedirs (os .path .join (IIIlllIlIlIlIllll ,_IlIIlIIIIlllIllIl ),exist_ok =_IllllIIIllIlIlIlI )
os .environ ['TEMP']=IlIlllIIIIllIlllI 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IIlIIIlIllIIIllll =Config ()
IllIIllllIlllIIll =I18nAuto ()
IllIIllllIlllIIll .print ()
IIIlIlIlIlIIIIllI =torch .cuda .device_count ()
IIIIIIlIIIlIllIll =[]
IIlIllIIllIIllllI =[]
IlIlIIIlllIlIllIl =_IlIlIlIlIIIIIIlIl 
if torch .cuda .is_available ()or IIIlIlIlIlIIIIllI !=0 :
	for IIIIIlllIllIIllIl in range (IIIlIlIlIlIIIIllI ):
		IlllIIIlIlIIllIIl =torch .cuda .get_device_name (IIIIIlllIllIIllIl )
		if any (IIlIIlIIllIIlIllI in IlllIIIlIlIIllIIl .upper ()for IIlIIlIIllIIlIllI in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IlIlIIIlllIlIllIl =_IllllIIIllIlIlIlI ;IIIIIIlIIIlIllIll .append ('%s\t%s'%(IIIIIlllIllIIllIl ,IlllIIIlIlIIllIIl ));IIlIllIIllIIllllI .append (int (torch .cuda .get_device_properties (IIIIIlllIllIIllIl ).total_memory /1024 /1024 /1024 +.4 ))
if IlIlIIIlllIlIllIl and len (IIIIIIlIIIlIllIll )>0 :IllllIIIIlIIlllII =_IlIlIllIIIlIIIlII .join (IIIIIIlIIIlIllIll );IIlIIIIIIlllllIlI =min (IIlIllIIllIIllllI )//2 
else :IllllIIIIlIIlllII =IllIIllllIlllIIll ('很遗憾您这没有能用的显卡来支持您训练');IIlIIIIIIlllllIlI =1 
IllIIllIlllllIIll ='-'.join ([IlIlIllIIIIIIIIll [0 ]for IlIlIllIIIIIIIIll in IIIIIIlIIIlIllIll ])
class IllIllllIllIIllll (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IIlIIlIIlIllllIIl ,**IlllIllIIIIllllll ):super ().__init__ (variant ='tool',**IlllIllIIIIllllll )
	def get_block_name (IIIIIIIllIIllllII ):return 'button'
IIlIlllllIIIllIII =_IlIlIllIlIlIIlIIl 
def IlIlllIlllIIllIll ():
	global IIlIlllllIIIllIII ;IlIIllIlIIllIllIl ,_IIIlIIllIIIIIlIll ,_IIIlIIllIIIIIlIll =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IIlIlllllIIIllIII =IlIIllIlIIllIllIl [0 ];IIlIlllllIIIllIII =IIlIlllllIIIllIII .to (IIlIIIlIllIIIllll .device )
	if IIlIIIlIllIIIllll .is_half :IIlIlllllIIIllIII =IIlIlllllIIIllIII .half ()
	else :IIlIlllllIIIllIII =IIlIlllllIIIllIII .float ()
	IIlIlllllIIIllIII .eval ()
IIlIlIIlIlIlIllII =_IlIIlIIIIlllIllIl 
IllIIIlllIlIIIIIl ='uvr5_weights'
IIIIIlllIlllllIII =_IlllIlIIIllIlIlll 
IlIlllIIIIllIIIll =[]
for IlIlllIIIIIlIlIll in os .listdir (IIlIlIIlIlIlIllII ):
	if IlIlllIIIIIlIlIll .endswith (_IIllIllIlIlIIIllI ):IlIlllIIIIllIIIll .append (IlIlllIIIIIlIlIll )
IIIIlllIIIlllIlIl =[]
for (IlIIIIllIIlIIIlIl ,IlllllIllIIlIIIII ,IlIIlIIIllIIlllll )in os .walk (IIIIIlllIlllllIII ,topdown =_IlIlIlIlIIIIIIlIl ):
	for IlIlllIIIIIlIlIll in IlIIlIIIllIIlllll :
		if IlIlllIIIIIlIlIll .endswith (_IIIIllIllllIlIlIl )and _IIlllIIllIIllllll not in IlIlllIIIIIlIlIll :IIIIlllIIIlllIlIl .append (_IIlIllIlllIlIllII %(IlIIIIllIIlIIIlIl ,IlIlllIIIIIlIlIll ))
IlllIlllllIlIlllI =[]
for IlIlllIIIIIlIlIll in os .listdir (IllIIIlllIlIIIIIl ):
	if IlIlllIIIIIlIlIll .endswith (_IIllIllIlIlIIIllI )or 'onnx'in IlIlllIIIIIlIlIll :IlllIlllllIlIlllI .append (IlIlllIIIIIlIlIll .replace (_IIllIllIlIlIIIllI ,''))
IlIllllllIlIllIlI =_IlIlIllIlIlIIlIIl 
def IlllllIIIIIlIlIll (IlIlllIllIlIIIIII ,IllIllIIlIIIIIlIl ,IlllllIlIIllIIIII ,IlIIIllIIIIlllllI ,IlllIIlllllllIlll ,IllIlllIllIllllIl ,IIllIlllIIlIlIIll ,IIIlIlIlIIlIIllIl ,IIllIIlIIIllllIll ,IIIllIlllIlIllIII ,IlIIIIlIlIIlIllll ,IllIllIlIllIIllII ):
	global IIIllIlllIllIllIl ,IIIIlllIIlIllIllI ,IIllIllIIIllIllIl ,IIlIlllllIIIllIII ,IlIllllllllIllllI ,IlIllllllIlIllIlI 
	if IllIllIIlIIIIIlIl is _IlIlIllIlIlIIlIIl :return 'You need to upload an audio',_IlIlIllIlIlIIlIIl 
	IlllllIlIIllIIIII =int (IlllllIlIIllIIIII )
	try :
		IIIlIIIIIlIlIIlII =load_audio (IllIllIIlIIIIIlIl ,16000 );IIIllIlIlllIlIIll =np .abs (IIIlIIIIIlIlIIlII ).max ()/.95 
		if IIIllIlIlllIlIIll >1 :IIIlIIIIIlIlIIlII /=IIIllIlIlllIlIIll 
		IIIIlIlIIIlIlIlII =[0 ,0 ,0 ]
		if not IIlIlllllIIIllIII :IlIlllIlllIIllIll ()
		IIlIlIIIIlIlllIll =IlIllllllIlIllIlI .get (_IIIIlIllIlllIIllI ,1 );IllIlllIllIllllIl =IllIlllIllIllllIl .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl ).replace (_IIlllIIllIIllllll ,'added')if IllIlllIllIllllIl !=''else IIllIlllIIlIlIIll ;IIIlIIIIlllIIIIlI =IIllIllIIIllIllIl .pipeline (IIlIlllllIIIllIII ,IIIIlllIIlIllIllI ,IlIlllIllIlIIIIII ,IIIlIIIIIlIlIIlII ,IllIllIIlIIIIIlIl ,IIIIlIlIIIlIlIlII ,IlllllIlIIllIIIII ,IlllIIlllllllIlll ,IllIlllIllIllllIl ,IIIlIlIlIIlIIllIl ,IIlIlIIIIlIlllIll ,IIllIIlIIIllllIll ,IIIllIlllIllIllIl ,IIIllIlllIlIllIII ,IlIIIIlIlIIlIllll ,IlIllllllllIllllI ,IllIllIlIllIIllII ,f0_file =IlIIIllIIIIlllllI )
		if IIIllIlllIllIllIl !=IIIllIlllIlIllIII >=16000 :IIIllIlllIllIllIl =IIIllIlllIlIllIII 
		IllllIIlIIIlIIlIl ='Using index:%s.'%IllIlllIllIllllIl if os .path .exists (IllIlllIllIllllIl )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IllllIIlIIIlIIlIl ,IIIIlIlIIIlIlIlII [0 ],IIIIlIlIIIlIlIlII [1 ],IIIIlIlIIIlIlIlII [2 ]),(IIIllIlllIllIllIl ,IIIlIIIIlllIIIIlI )
	except :IIllIIllIlIlllllI =traceback .format_exc ();print (IIllIIllIlIlllllI );return IIllIIllIlIlllllI ,(_IlIlIllIlIlIIlIIl ,_IlIlIllIlIlIIlIIl )
def IIlIllIIlIIlIIllI (IIIIlIllIllIIIllI ,IIIlllIIlIIIIlIII ,IlIIIIllIllIIIIll ,IlIlIIlIlllIlIllI ,IlIIIllIIlllIIlII ,IlllIIIllllIIIlll ,IllllIIIIllIlIIll ,IIlllllIlllIlIIIl ,IllIIlIIlIIllIlII ,IIIIlIllIIlllIIlI ,IllIIIIlIIIIIllll ,IIIIIIllllIlIllII ,IIlIllIlIlIllllll ,IIIIIlllllIlllIll ):
	try :
		IIIlllIIlIIIIlIII =IIIlllIIlIIIIlIII .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl );IlIIIIllIllIIIIll =IlIIIIllIllIIIIll .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl );os .makedirs (IlIIIIllIllIIIIll ,exist_ok =_IllllIIIllIlIlIlI )
		try :
			if IIIlllIIlIIIIlIII !='':IlIlIIlIlllIlIllI =[os .path .join (IIIlllIIlIIIIlIII ,IIIIlllIlIIIIIlIl )for IIIIlllIlIIIIIlIl in os .listdir (IIIlllIIlIIIIlIII )]
			else :IlIlIIlIlllIlIllI =[IllIIIIllIlIlllll .name for IllIIIIllIlIlllll in IlIlIIlIlllIlIllI ]
		except :traceback .print_exc ();IlIlIIlIlllIlIllI =[IIIllIIlIllIllllI .name for IIIllIIlIllIllllI in IlIlIIlIlllIlIllI ]
		IIllIlIllllIllIII =[]
		for IlllllllllIIIIllI in IlIlIIlIlllIlIllI :
			IIIIIlIIIIIIIlIII ,IIIlIIIllIllIIIlI =IlllllIIIIIlIlIll (IIIIlIllIllIIIllI ,IlllllllllIIIIllI ,IlIIIllIIlllIIlII ,_IlIlIllIlIlIIlIIl ,IlllIIIllllIIIlll ,IllllIIIIllIlIIll ,IIlllllIlllIlIIIl ,IllIIlIIlIIllIlII ,IIIIlIllIIlllIIlI ,IllIIIIlIIIIIllll ,IIIIIIllllIlIllII ,IIlIllIlIlIllllll )
			if 'Success'in IIIIIlIIIIIIIlIII :
				try :
					IIIIIlIlllIlIIIll ,IIllIIlIIIIlIIIII =IIIlIIIllIllIIIlI 
					if IIIIIlllllIlllIll in [_IllIlIIllIlIllIlI ,_IllIlllIllllIIlIl ]:sf .write ('%s/%s.%s'%(IlIIIIllIllIIIIll ,os .path .basename (IlllllllllIIIIllI ),IIIIIlllllIlllIll ),IIllIIlIIIIlIIIII ,IIIIIlIlllIlIIIll )
					else :
						IlllllllllIIIIllI ='%s/%s.wav'%(IlIIIIllIllIIIIll ,os .path .basename (IlllllllllIIIIllI ));sf .write (IlllllllllIIIIllI ,IIllIIlIIIIlIIIII ,IIIIIlIlllIlIIIll )
						if os .path .exists (IlllllllllIIIIllI ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IlllllllllIIIIllI ,IlllllllllIIIIllI [:-4 ]+'.%s'%IIIIIlllllIlllIll ))
				except :IIIIIlIIIIIIIlIII +=traceback .format_exc ()
			IIllIlIllllIllIII .append (_IIllIIIIIllIIIIlI %(os .path .basename (IlllllllllIIIIllI ),IIIIIlIIIIIIIlIII ));yield _IlIlIllIIIlIIIlII .join (IIllIlIllllIllIII )
		yield _IlIlIllIIIlIIIlII .join (IIllIlIllllIllIII )
	except :yield traceback .format_exc ()
def IIlIlllIIIlIIIIIl (IlIIlllIlIlllIIII ,IlIIIIlllIIIIlIlI ,IlIlIIIlllIIlIIlI ,IllIlIllIllIllIII ,IIlllIlIlllIlllIl ,IllllIIllllIIllll ,IIIIIIIIIIIIIlIlI ):
	IIllIllIlllllllll ='streams';IIIllIIlllllllIIl ='onnx_dereverb_By_FoxJoy';IlIIIIllllIlIIIIl =[]
	try :
		IlIIIIlllIIIIlIlI =IlIIIIlllIIIIlIlI .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl );IlIlIIIlllIIlIIlI =IlIlIIIlllIIlIIlI .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl );IIlllIlIlllIlllIl =IIlllIlIlllIlllIl .strip (_IlIlIIIlIIllIllIl ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIllIIIlIIIlII ).strip (_IllIIIllllIlIlIII ).strip (_IlIlIIIlIIllIllIl )
		if IlIIlllIlIlllIIII ==IIIllIIlllllllIIl :from MDXNet import MDXNetDereverb ;IlIIlllIIIlIIlIll =MDXNetDereverb (15 )
		else :IIIllIIIIIIlIIIIl =_audio_pre_ if 'DeEcho'not in IlIIlllIlIlllIIII else _audio_pre_new ;IlIIlllIIIlIIlIll =IIIllIIIIIIlIIIIl (agg =int (IllllIIllllIIllll ),model_path =os .path .join (IllIIIlllIlIIIIIl ,IlIIlllIlIlllIIII +_IIllIllIlIlIIIllI ),device =IIlIIIlIllIIIllll .device ,is_half =IIlIIIlIllIIIllll .is_half )
		if IlIIIIlllIIIIlIlI !='':IllIlIllIllIllIII =[os .path .join (IlIIIIlllIIIIlIlI ,IIIlIllllIlIlIIII )for IIIlIllllIlIlIIII in os .listdir (IlIIIIlllIIIIlIlI )]
		else :IllIlIllIllIllIII =[IlllIIllIIIlIIlIl .name for IlllIIllIIIlIIlIl in IllIlIllIllIllIII ]
		for IIIlIIIIIIlIllIII in IllIlIllIllIllIII :
			IIllllIIlIIlIllll =os .path .join (IlIIIIlllIIIIlIlI ,IIIlIIIIIIlIllIII );IllllIlIlllllllII =1 ;IIIlIIlIIlIllIlIl =0 
			try :
				IIlIIIIIllIIIlIII =ffmpeg .probe (IIllllIIlIIlIllll ,cmd ='ffprobe')
				if IIlIIIIIllIIIlIII [IIllIllIlllllllll ][0 ]['channels']==2 and IIlIIIIIllIIIlIII [IIllIllIlllllllll ][0 ][_IlIllllIlIlIIIlII ]=='44100':IllllIlIlllllllII =0 ;IlIIlllIIIlIIlIll ._path_audio_ (IIllllIIlIIlIllll ,IIlllIlIlllIlllIl ,IlIlIIIlllIIlIIlI ,IIIIIIIIIIIIIlIlI );IIIlIIlIIlIllIlIl =1 
			except :IllllIlIlllllllII =1 ;traceback .print_exc ()
			if IllllIlIlllllllII ==1 :IllllIllIIllIllIl ='%s/%s.reformatted.wav'%(IlIlllIIIIllIlllI ,os .path .basename (IIllllIIlIIlIllll ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIllllIIlIIlIllll ,IllllIllIIllIllIl ));IIllllIIlIIlIllll =IllllIllIIllIllIl 
			try :
				if IIIlIIlIIlIllIlIl ==0 :IlIIlllIIIlIIlIll ._path_audio_ (IIllllIIlIIlIllll ,IIlllIlIlllIlllIl ,IlIlIIIlllIIlIIlI ,IIIIIIIIIIIIIlIlI )
				IlIIIIllllIlIIIIl .append ('%s->Success'%os .path .basename (IIllllIIlIIlIllll ));yield _IlIlIllIIIlIIIlII .join (IlIIIIllllIlIIIIl )
			except :IlIIIIllllIlIIIIl .append (_IIllIIIIIllIIIIlI %(os .path .basename (IIllllIIlIIlIllll ),traceback .format_exc ()));yield _IlIlIllIIIlIIIlII .join (IlIIIIllllIlIIIIl )
	except :IlIIIIllllIlIIIIl .append (traceback .format_exc ());yield _IlIlIllIIIlIIIlII .join (IlIIIIllllIlIIIIl )
	finally :
		try :
			if IlIIlllIlIlllIIII ==IIIllIIlllllllIIl :del IlIIlllIIIlIIlIll .pred .model ;del IlIIlllIIIlIIlIll .pred .model_ 
			else :del IlIIlllIIIlIIlIll .model ;del IlIIlllIIIlIIlIll 
		except :traceback .print_exc ()
		print (_IIIIIlIIlllllIlII )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IlIlIllIIIlIIIlII .join (IlIIIIllllIlIIIIl )
def IllIllllIIllIlIIl (IllIIllIlIlIIIlII ):
	IIIlIlllllIllIIll ='';IlIIlIlIIIlllllII =os .path .join (_IlllIlIIIllIlIlll ,IllIIllIlIlIIIlII .split (_IIlIIlllllIIlllIl )[0 ],'')
	for IlIlIIIlllIllIllI in IIIIlllIIIlllIlIl :
		if IlIIlIlIIIlllllII in IlIlIIIlllIllIllI :IIIlIlllllIllIIll =IlIlIIIlllIllIllI ;break 
	return IIIlIlllllIllIIll 
def IllIlIlIlllllllII (IIllIIlIIIlIlllll ,IlIllIIllIIlllllI ,IIlIllllllIIllIIl ):
	global IIIIIlIIIIIIlllll ,IIIllIlllIllIllIl ,IIIIlllIIlIllIllI ,IIllIllIIIllIllIl ,IlIllllllIlIllIlI ,IlIllllllllIllllI 
	if IIllIIlIIIlIlllll ==''or IIllIIlIIIlIlllll ==[]:
		global IIlIlllllIIIllIII 
		if IIlIlllllIIIllIII is not _IlIlIllIlIlIIlIIl :
			print (_IIIIIlIIlllllIlII );del IIIIlllIIlIllIllI ,IIIIIlIIIIIIlllll ,IIllIllIIIllIllIl ,IIlIlllllIIIllIII ,IIIllIlllIllIllIl ;IIlIlllllIIIllIII =IIIIlllIIlIllIllI =IIIIIlIIIIIIlllll =IIllIllIIIllIllIl =IIlIlllllIIIllIII =IIIllIlllIllIllIl =_IlIlIllIlIlIIlIIl 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IlllIlIIIIllIlIll =IlIllllllIlIllIlI .get (_IIIIlIllIlllIIllI ,1 );IlIllllllllIllllI =IlIllllllIlIllIlI .get (_IIllIllllllIllIII ,_IlIlIIIlIlllIlIll )
			if IlIllllllllIllllI ==_IlIlIIIlIlllIlIll :
				if IlllIlIIIIllIlIll ==1 :IIIIlllIIlIllIllI =SynthesizerTrnMs256NSFsid (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ],is_half =IIlIIIlIllIIIllll .is_half )
				else :IIIIlllIIlIllIllI =SynthesizerTrnMs256NSFsid_nono (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ])
			elif IlIllllllllIllllI ==_IIIIlIllllIIIlIIl :
				if IlllIlIIIIllIlIll ==1 :IIIIlllIIlIllIllI =SynthesizerTrnMs768NSFsid (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ],is_half =IIlIIIlIllIIIllll .is_half )
				else :IIIIlllIIlIllIllI =SynthesizerTrnMs768NSFsid_nono (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ])
			del IIIIlllIIlIllIllI ,IlIllllllIlIllIlI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IlllIIlIlIlIlllII :_IlIlIlIlIIIIIIlIl ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
	IlIllIIIIIlIllIIl =_IIlIllIlllIlIllII %(IIlIlIIlIlIlIllII ,IIllIIlIIIlIlllll );print ('loading %s'%IlIllIIIIIlIllIIl );IlIllllllIlIllIlI =torch .load (IlIllIIIIIlIllIIl ,map_location =_IIIIlIIIllllIIllI );IIIllIlllIllIllIl =IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ][-1 ];IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ][-3 ]=IlIllllllIlIllIlI [_IIIIllIlIIIIlIIlI ][_IlIlIlllIllllIIIl ].shape [0 ];IlllIlIIIIllIlIll =IlIllllllIlIllIlI .get (_IIIIlIllIlllIIllI ,1 )
	if IlllIlIIIIllIlIll ==0 :IlIllIIllIIlllllI =IIlIllllllIIllIIl ={_IlllIIlIlIlIlllII :_IlIlIlIlIIIIIIlIl ,_IIIlIIIIIllIIIllI :.5 ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
	else :IlIllIIllIIlllllI ={_IlllIIlIlIlIlllII :_IllllIIIllIlIlIlI ,_IIIlIIIIIllIIIllI :IlIllIIllIIlllllI ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl };IIlIllllllIIllIIl ={_IlllIIlIlIlIlllII :_IllllIIIllIlIlIlI ,_IIIlIIIIIllIIIllI :IIlIllllllIIllIIl ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
	IlIllllllllIllllI =IlIllllllIlIllIlI .get (_IIllIllllllIllIII ,_IlIlIIIlIlllIlIll )
	if IlIllllllllIllllI ==_IlIlIIIlIlllIlIll :
		if IlllIlIIIIllIlIll ==1 :IIIIlllIIlIllIllI =SynthesizerTrnMs256NSFsid (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ],is_half =IIlIIIlIllIIIllll .is_half )
		else :IIIIlllIIlIllIllI =SynthesizerTrnMs256NSFsid_nono (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ])
	elif IlIllllllllIllllI ==_IIIIlIllllIIIlIIl :
		if IlllIlIIIIllIlIll ==1 :IIIIlllIIlIllIllI =SynthesizerTrnMs768NSFsid (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ],is_half =IIlIIIlIllIIIllll .is_half )
		else :IIIIlllIIlIllIllI =SynthesizerTrnMs768NSFsid_nono (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ])
	del IIIIlllIIlIllIllI .enc_q ;print (IIIIlllIIlIllIllI .load_state_dict (IlIllllllIlIllIlI [_IIIIllIlIIIIlIIlI ],strict =_IlIlIlIlIIIIIIlIl ));IIIIlllIIlIllIllI .eval ().to (IIlIIIlIllIIIllll .device )
	if IIlIIIlIllIIIllll .is_half :IIIIlllIIlIllIllI =IIIIlllIIlIllIllI .half ()
	else :IIIIlllIIlIllIllI =IIIIlllIIlIllIllI .float ()
	IIllIllIIIllIllIl =VC (IIIllIlllIllIllIl ,IIlIIIlIllIIIllll );IIIIIlIIIIIIlllll =IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ][-3 ];return {_IlllIIlIlIlIlllII :_IllllIIIllIlIlIlI ,'maximum':IIIIIlIIIIIIlllll ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },IlIllIIllIIlllllI ,IIlIllllllIIllIIl ,IllIllllIIllIlIIl (IIllIIlIIIlIlllll )
def IIllIlIIIIlIIIlII ():
	IllllIlIIIIlIIIII =[]
	for IlIlllIIIlIlIIIII in os .listdir (IIlIlIIlIlIlIllII ):
		if IlIlllIIIlIlIIIII .endswith (_IIllIllIlIlIIIllI ):IllllIlIIIIlIIIII .append (IlIlllIIIlIlIIIII )
	IlIlIIIllllIlIIll =[]
	for (IllIIlllIIIlIIllI ,IllIllIIIlllllllI ,IllIlIIlIIlIlIIll )in os .walk (IIIIIlllIlllllIII ,topdown =_IlIlIlIlIIIIIIlIl ):
		for IlIlllIIIlIlIIIII in IllIlIIlIIlIlIIll :
			if IlIlllIIIlIlIIIII .endswith (_IIIIllIllllIlIlIl )and _IIlllIIllIIllllll not in IlIlllIIIlIlIIIII :IlIlIIIllllIlIIll .append (_IIlIllIlllIlIllII %(IllIIlllIIIlIIllI ,IlIlllIIIlIlIIIII ))
	return {_IlIlIlIlIlIllIlll :sorted (IllllIlIIIIlIIIII ),_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },{_IlIlIlIlIlIllIlll :sorted (IlIlIIIllllIlIIll ),_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
def IIlllIIlIlIlllIIl ():return {_IIIlIIIIIllIIIllI :'',_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
IIIllllllllIlIIlI ={_IlIlIlllIllllIIlI :32000 ,_IllIIIllIllIIlIII :40000 ,_IIlIIIlIIlIlIllIl :48000 }
def IIIlIIIllIlIIlIII (IIIlIlIIIIIIIIllI ,IllIlIIlIllIIIlII ):
	while 1 :
		if IllIlIIlIllIIIlII .poll ()is _IlIlIllIlIlIIlIIl :sleep (.5 )
		else :break 
	IIIlIlIIIIIIIIllI [0 ]=_IllllIIIllIlIlIlI 
def IllIlIIIllIIlllIl (IlIlIIIlIIlIlIlII ,IIIlIllIlIllIIIII ):
	while 1 :
		IIllIllllIlIIllIl =1 
		for IllIlllIlIllIlIIl in IIIlIllIlIllIIIII :
			if IllIlllIlIllIlIIl .poll ()is _IlIlIllIlIlIIlIIl :IIllIllllIlIIllIl =0 ;sleep (.5 );break 
		if IIllIllllIlIIllIl ==1 :break 
	IlIlIIIlIIlIlIlII [0 ]=_IllllIIIllIlIlIlI 
def IlIIIIIlIIIlIlllI (IIIllllIIIlIIIIII ,IllllIIllllllIIIl ,IIllIllllIIlllIlI ,IllIlIIIllllIIllI ):
	IIllllIIIIIIIllll ='%s/logs/%s/preprocess.log';IIllIllllIIlllIlI =IIIllllllllIlIIlI [IIllIllllIIlllIlI ];os .makedirs (_IlIlIlIllIIlIIlIl %(IIIlllIlIlIlIllll ,IllllIIllllllIIIl ),exist_ok =_IllllIIIllIlIlIlI );IllIlllIIIllllllI =open (IIllllIIIIIIIllll %(IIIlllIlIlIlIllll ,IllllIIllllllIIIl ),'w');IllIlllIIIllllllI .close ();IlllIIIIlllIIIlll =IIlIIIlIllIIIllll .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IIIllllIIIlIIIIII ,IIllIllllIIlllIlI ,IllIlIIIllllIIllI ,IIIlllIlIlIlIllll ,IllllIIllllllIIIl )+str (IIlIIIlIllIIIllll .noparallel );print (IlllIIIIlllIIIlll );IIlIllIIIlIlIlIll =Popen (IlllIIIIlllIIIlll ,shell =_IllllIIIllIlIlIlI );IlllIIIlIIllIlllI =[_IlIlIlIlIIIIIIlIl ];threading .Thread (target =IIIlIIIllIlIIlIII ,args =(IlllIIIlIIllIlllI ,IIlIllIIIlIlIlIll )).start ()
	while 1 :
		with open (IIllllIIIIIIIllll %(IIIlllIlIlIlIllll ,IllllIIllllllIIIl ),_IlIIlIIllllIIllII )as IllIlllIIIllllllI :yield IllIlllIIIllllllI .read ()
		sleep (1 )
		if IlllIIIlIIllIlllI [0 ]:break 
	with open (IIllllIIIIIIIllll %(IIIlllIlIlIlIllll ,IllllIIllllllIIIl ),_IlIIlIIllllIIllII )as IllIlllIIIllllllI :IIIIllIIIIIlIIIII =IllIlllIIIllllllI .read ()
	print (IIIIllIIIIIlIIIII );yield IIIIllIIIIIlIIIII 
def IlIllllllllIIlIIl (IIIllIlllIIlIIIIl ,IllllIIllIIllIllI ,IIlIlIIlllIIIIlIl ,IIIlIIIIllIIIIllI ,IlIIlllIllIlIllII ,IlIIlIIIllIIIlIlI ,IllIIlIIlllIlIIIl ):
	IllIIlIllIIIlIIII ='%s/logs/%s/extract_fl_feature.log';IIIllIlllIIlIIIIl =IIIllIlllIIlIIIIl .split ('-');os .makedirs (_IlIlIlIllIIlIIlIl %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),exist_ok =_IllllIIIllIlIlIlI );IIllIllllIlIIIIll =open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),'w');IIllIllllIlIIIIll .close ()
	if IIIlIIIIllIIIIllI :
		if IIlIlIIlllIIIIlIl !=_IIIllllIIllllIIlI :
			IIIIIIlIIlllIIllI =IIlIIIlIllIIIllll .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ,IllllIIllIIllIllI ,IIlIlIIlllIIIIlIl );print (IIIIIIlIIlllIIllI );IlIllIIlllIlIIllI =Popen (IIIIIIlIIlllIIllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IlllIlIlIIllIllIl =[_IlIlIlIlIIIIIIlIl ];threading .Thread (target =IIIlIIIllIlIIlIII ,args =(IlllIlIlIIllIllIl ,IlIllIIlllIlIIllI )).start ()
			while 1 :
				with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :yield IIllIllllIlIIIIll .read ()
				sleep (1 )
				if IlllIlIlIIllIllIl [0 ]:break 
			with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :IlIIlIlIllllIllII =IIllIllllIlIIIIll .read ()
			print (IlIIlIlIllllIllII );yield IlIIlIlIllllIllII 
		else :
			IllIIlIIlllIlIIIl =IllIIlIIlllIlIIIl .split ('-');IIllIlllIIIlIIlII =len (IllIIlIIlllIlIIIl );IIllllIlllIIIllIl =[]
			for (IlllIIlIIlIIIlIII ,IlIIlIlIIlIIIllll )in enumerate (IllIIlIIlllIlIIIl ):IIIIIIlIIlllIIllI =IIlIIIlIllIIIllll .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IIllIlllIIIlIIlII ,IlllIIlIIlIIIlIII ,IlIIlIlIIlIIIllll ,IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ,IIlIIIlIllIIIllll .is_half );print (IIIIIIlIIlllIIllI );IlIllIIlllIlIIllI =Popen (IIIIIIlIIlllIIllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IIllllIlllIIIllIl .append (IlIllIIlllIlIIllI )
			IlllIlIlIIllIllIl =[_IlIlIlIlIIIIIIlIl ];threading .Thread (target =IllIlIIIllIIlllIl ,args =(IlllIlIlIIllIllIl ,IIllllIlllIIIllIl )).start ()
			while 1 :
				with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :yield IIllIllllIlIIIIll .read ()
				sleep (1 )
				if IlllIlIlIIllIllIl [0 ]:break 
			with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :IlIIlIlIllllIllII =IIllIllllIlIIIIll .read ()
			print (IlIIlIlIllllIllII );yield IlIIlIlIllllIllII 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IIllIlllIIIlIIlII =len (IIIllIlllIIlIIIIl );IIllllIlllIIIllIl =[]
	for (IlllIIlIIlIIIlIII ,IlIIlIlIIlIIIllll )in enumerate (IIIllIlllIIlIIIIl ):IIIIIIlIIlllIIllI =IIlIIIlIllIIIllll .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IIlIIIlIllIIIllll .device ,IIllIlllIIIlIIlII ,IlllIIlIIlIIIlIII ,IlIIlIlIIlIIIllll ,IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ,IlIIlIIIllIIIlIlI );print (IIIIIIlIIlllIIllI );IlIllIIlllIlIIllI =Popen (IIIIIIlIIlllIIllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IIllllIlllIIIllIl .append (IlIllIIlllIlIIllI )
	IlllIlIlIIllIllIl =[_IlIlIlIlIIIIIIlIl ];threading .Thread (target =IllIlIIIllIIlllIl ,args =(IlllIlIlIIllIllIl ,IIllllIlllIIIllIl )).start ()
	while 1 :
		with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :yield IIllIllllIlIIIIll .read ()
		sleep (1 )
		if IlllIlIlIIllIllIl [0 ]:break 
	with open (IllIIlIllIIIlIIII %(IIIlllIlIlIlIllll ,IlIIlllIllIlIllII ),_IlIIlIIllllIIllII )as IIllIllllIlIIIIll :IlIIlIlIllllIllII =IIllIllllIlIIIIll .read ()
	print (IlIIlIlIllllIllII );yield IlIIlIlIllllIllII 
def IIIIllIIlllllllII (IllIIlIllIllIllIl ,IIllIlIIIlllllIIl ,IIIIllllIlllIIllI ):
	IlIllIllllIlIIIIl =''if IIIIllllIlllIIllI ==_IlIlIIIlIlllIlIll else _IIIlIIIlIIllIIlIl ;IIlIllIIlIllIIIlI =_IIIIlIllIlllIIllI if IIllIlIIIlllllIIl else '';IlllIllIIIlllIIlI =os .access (_IllllIIIIlllllIll %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl ),os .F_OK );IlIlIlIlllIllIllI =os .access (_IIlIIIlllllIllIlI %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl ),os .F_OK )
	if not IlllIllIIIlllIIlI :print (_IllllIIIIlllllIll %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl ),_IIllIIllIlIIIIlll )
	if not IlIlIlIlllIllIllI :print (_IIlIIIlllllIllIlI %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl ),_IIllIIllIlIIIIlll )
	return _IllllIIIIlllllIll %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl )if IlllIllIIIlllIIlI else '',_IIlIIIlllllIllIlI %(IlIllIllllIlIIIIl ,IIlIllIIlIllIIIlI ,IllIIlIllIllIllIl )if IlIlIlIlllIllIllI else ''
def IlIIIIlIIIlIIlllI (IIlIIIIlIlIIIIllI ,IIllIIIlIlIIllIlI ,IIIllIllIIllllllI ):
	IlIllIlllllIlIIII =''if IIIllIllIIllllllI ==_IlIlIIIlIlllIlIll else _IIIlIIIlIIllIIlIl 
	if IIlIIIIlIlIIIIllI ==_IlIlIlllIllllIIlI and IIIllIllIIllllllI ==_IlIlIIIlIlllIlIll :IIlIIIIlIlIIIIllI =_IllIIIllIllIIlIII 
	IIIIlllIlIIIIllII ={_IlIlIlIlIlIllIlll :[_IllIIIllIllIIlIII ,_IIlIIIlIIlIlIllIl ],_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl ,_IIIlIIIIIllIIIllI :IIlIIIIlIlIIIIllI }if IIIllIllIIllllllI ==_IlIlIIIlIlllIlIll else {_IlIlIlIlIlIllIlll :[_IllIIIllIllIIlIII ,_IIlIIIlIIlIlIllIl ,_IlIlIlllIllllIIlI ],_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl ,_IIIlIIIIIllIIIllI :IIlIIIIlIlIIIIllI };IlIIllIIlIIIlIlII =_IIIIlIllIlllIIllI if IIllIIIlIlIIllIlI else '';IIlIlIIlIIlllIIlI =os .access (_IllllIIIIlllllIll %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI ),os .F_OK );IlIlllllIIllllllI =os .access (_IIlIIIlllllIllIlI %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI ),os .F_OK )
	if not IIlIlIIlIIlllIIlI :print (_IllllIIIIlllllIll %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI ),_IIllIIllIlIIIIlll )
	if not IlIlllllIIllllllI :print (_IIlIIIlllllIllIlI %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI ),_IIllIIllIlIIIIlll )
	return _IllllIIIIlllllIll %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI )if IIlIlIIlIIlllIIlI else '',_IIlIIIlllllIllIlI %(IlIllIlllllIlIIII ,IlIIllIIlIIIlIlII ,IIlIIIIlIlIIIIllI )if IlIlllllIIllllllI else '',IIIIlllIlIIIIllII 
def IllIlIIllllllIIII (IIIIllIIllllIIlIl ,IlllIlIllIIlIIIIl ,IlIIlIlllIllIlIll ):
	IIlllIlIllllIlIIl ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IIIllIllIIIIIllIl ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IlllIIlIIlIlIIlll =''if IlIIlIlllIllIlIll ==_IlIlIIIlIlllIlIll else _IIIlIIIlIIllIIlIl ;IlIIIlllIIlIlllIl =os .access (IIIllIllIIIIIllIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl ),os .F_OK );IlIIlIIlIlIlllIlI =os .access (IIlllIlIllllIlIIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl ),os .F_OK )
	if not IlIIIlllIIlIlllIl :print (IIIllIllIIIIIllIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl ),_IIllIIllIlIIIIlll )
	if not IlIIlIIlIlIlllIlI :print (IIlllIlIllllIlIIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl ),_IIllIIllIlIIIIlll )
	if IIIIllIIllllIIlIl :return {_IlllIIlIlIlIlllII :_IllllIIIllIlIlIlI ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },IIIllIllIIIIIllIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl )if IlIIIlllIIlIlllIl else '',IIlllIlIllllIlIIl %(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl )if IlIIlIIlIlIlllIlI else ''
	return {_IlllIIlIlIlIlllII :_IlIlIlIlIIIIIIlIl ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl )if IlIIIlllIIlIlllIl else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IlllIIlIIlIlIIlll ,IlllIlIllIIlIIIIl )if IlIIlIIlIlIlllIlI else ''
def IlIllIIlIlllIIllI (IlIIIIIIllllllIlI ,IlIlIIIllllIIlIIl ,IlIIllIllIIIlllll ,IIIlIIlIlIIlllIIl ,IIlIlllIllIllIlll ,IllIllIIlIIIIllIl ,IlIIIlIIlllIIIIll ,IllIllIllIlIllIIl ,IlIlllIlIIlllllIl ,IIIIlIlIIlIllIIIl ,IlllIIlIIllllIlIl ,IlIIllIIlllIlIllI ,IlIlIIIlIIIlIllll ,IllIIlllIlIlIIllI ):
	IIIIlIlIllIIlIIII ='\x08';IIlIIIIIllllllIll =_IlIlIlIllIIlIIlIl %(IIIlllIlIlIlIllll ,IlIIIIIIllllllIlI );os .makedirs (IIlIIIIIllllllIll ,exist_ok =_IllllIIIllIlIlIlI );IlIIIlIllIIIlIIll =_IIIIlIIllIlIlIIIl %IIlIIIIIllllllIll ;IllllIIllllIlIlIl =_IlIlllIlIllllIIIl %IIlIIIIIllllllIll if IllIIlllIlIlIIllI ==_IlIlIIIlIlllIlIll else _IlIlIIIIlIllllIII %IIlIIIIIllllllIll 
	if IlIIllIllIIIlllll :IIlllIIllIIlIIIII ='%s/2a_f0'%IIlIIIIIllllllIll ;IIlIIlIIIIIIIIlll =_IIlIlllIIIlllIIII %IIlIIIIIllllllIll ;IllllIlIIlIIlIllI =set ([IlllIIIIllIlIIIII .split (_IIlIIlllllIIlllIl )[0 ]for IlllIIIIllIlIIIII in os .listdir (IlIIIlIllIIIlIIll )])&set ([IllIlIlIllIlIIIII .split (_IIlIIlllllIIlllIl )[0 ]for IllIlIlIllIlIIIII in os .listdir (IllllIIllllIlIlIl )])&set ([IlllIllllIIllIIlI .split (_IIlIIlllllIIlllIl )[0 ]for IlllIllllIIllIIlI in os .listdir (IIlllIIllIIlIIIII )])&set ([IlllIIIIIllIIIIlI .split (_IIlIIlllllIIlllIl )[0 ]for IlllIIIIIllIIIIlI in os .listdir (IIlIIlIIIIIIIIlll )])
	else :IllllIlIIlIIlIllI =set ([IlIIlIlIlllllIlll .split (_IIlIIlllllIIlllIl )[0 ]for IlIIlIlIlllllIlll in os .listdir (IlIIIlIllIIIlIIll )])&set ([IllIIIlIIlIIIIlIl .split (_IIlIIlllllIIlllIl )[0 ]for IllIIIlIIlIIIIlIl in os .listdir (IllllIIllllIlIlIl )])
	IlIllllIlIlIIIIll =[]
	for IIlllIlllIllIIlII in IllllIlIIlIIlIllI :
		if IlIIllIllIIIlllll :IlIllllIlIlIIIIll .append (_IlIIIIllIIIIlIIll %(IlIIIlIllIIIlIIll .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IllllIIllllIlIlIl .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IIlllIIllIIlIIIII .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IIlIIlIIIIIIIIlll .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IIIlIIlIlIIlllIIl ))
		else :IlIllllIlIlIIIIll .append (_IIIlIIIIlIlIIllIl %(IlIIIlIllIIIlIIll .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IllllIIllllIlIlIl .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IIlllIlllIllIIlII ,IIIlIIlIlIIlllIIl ))
	IlIIlIIIllIIIIIII =256 if IllIIlllIlIlIIllI ==_IlIlIIIlIlllIlIll else 768 
	if IlIIllIllIIIlllll :
		for _IIlIlIIlIIIlIIlll in range (2 ):IlIllllIlIlIIIIll .append (_IIllIIllIIlIllIII %(IIIlllIlIlIlIllll ,IlIlIIIllllIIlIIl ,IIIlllIlIlIlIllll ,IlIIlIIIllIIIIIII ,IIIlllIlIlIlIllll ,IIIlllIlIlIlIllll ,IIIlIIlIlIIlllIIl ))
	else :
		for _IIlIlIIlIIIlIIlll in range (2 ):IlIllllIlIlIIIIll .append (_IlIIllllIlllIllll %(IIIlllIlIlIlIllll ,IlIlIIIllllIIlIIl ,IIIlllIlIlIlIllll ,IlIIlIIIllIIIIIII ,IIIlIIlIlIIlllIIl ))
	shuffle (IlIllllIlIlIIIIll )
	with open (_IIlIIIlIIIllIlIlI %IIlIIIIIllllllIll ,'w')as IIIlllllIlllIlIlI :IIIlllllIlllIlIlI .write (_IlIlIllIIIlIIIlII .join (IlIllllIlIlIIIIll ))
	print (_IllIllIlIlIIIlIlI );print ('use gpus:',IlllIIlIIllllIlIl )
	if IlIlllIlIIlllllIl =='':print ('no pretrained Generator')
	if IIIIlIlIIlIllIIIl =='':print ('no pretrained Discriminator')
	if IlllIIlIIllllIlIl :IllIllIlllIlllIII =IIlIIIlIllIIIllll .python_cmd +_IIlllIIIIllIIIIlI %(IlIIIIIIllllllIlI ,IlIlIIIllllIIlIIl ,1 if IlIIllIllIIIlllll else 0 ,IlIIIlIIlllIIIIll ,IlllIIlIIllllIlIl ,IllIllIIlIIIIllIl ,IIlIlllIllIllIlll ,_IIlllIllIlIIlIlIl %IlIlllIlIIlllllIl if IlIlllIlIIlllllIl !=''else '',_IIIIllIlIIlIllIll %IIIIlIlIIlIllIIIl if IIIIlIlIIlIllIIIl !=''else '',1 if IllIllIllIlIllIIl ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIIllIIlllIlIllI ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIlIIIlIIIlIllll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,IllIIlllIlIlIIllI )
	else :IllIllIlllIlllIII =IIlIIIlIllIIIllll .python_cmd +_IlllIlIlIlIlIlIll %(IlIIIIIIllllllIlI ,IlIlIIIllllIIlIIl ,1 if IlIIllIllIIIlllll else 0 ,IlIIIlIIlllIIIIll ,IllIllIIlIIIIllIl ,IIlIlllIllIllIlll ,_IIlllIllIlIIlIlIl %IlIlllIlIIlllllIl if IlIlllIlIIlllllIl !=''else IIIIlIlIllIIlIIII ,_IIIIllIlIIlIllIll %IIIIlIlIIlIllIIIl if IIIIlIlIIlIllIIIl !=''else IIIIlIlIllIIlIIII ,1 if IllIllIllIlIllIIl ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIIllIIlllIlIllI ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIlIIIlIIIlIllll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,IllIIlllIlIlIIllI )
	print (IllIllIlllIlllIII );IllIlllIlIlllllII =Popen (IllIllIlllIlllIII ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IllIlllIlIlllllII .wait ();return _IllIIlIllllIIlIII 
def IlIIIIIIIIllllIll (IllIllIIIlIllllll ,IIIllIIIIIlIllIII ):
	IIlllIIIIlIIIlIIl =_IlIlIlIllIIlIIlIl %(IIIlllIlIlIlIllll ,IllIllIIIlIllllll );os .makedirs (IIlllIIIIlIIIlIIl ,exist_ok =_IllllIIIllIlIlIlI );IIIIlIIIIIllIlllI =_IlIlllIlIllllIIIl %IIlllIIIIlIIIlIIl if IIIllIIIIIlIllIII ==_IlIlIIIlIlllIlIll else _IlIlIIIIlIllllIII %IIlllIIIIlIIIlIIl 
	if not os .path .exists (IIIIlIIIIIllIlllI ):return '请先进行特征提取!'
	IlIIIIlIlIlIllIII =list (os .listdir (IIIIlIIIIIllIlllI ))
	if len (IlIIIIlIlIlIllIII )==0 :return '请先进行特征提取！'
	IIIIIIllIIlIIlIII =[];IIIIlIIlIIIIlllll =[]
	for IllllllIIllIIIlIl in sorted (IlIIIIlIlIlIllIII ):IllIllIllIIIlllIl =np .load (_IIlIllIlllIlIllII %(IIIIlIIIIIllIlllI ,IllllllIIllIIIlIl ));IIIIlIIlIIIIlllll .append (IllIllIllIIIlllIl )
	IIIllIIllIIlIllIl =np .concatenate (IIIIlIIlIIIIlllll ,0 );IlllIIlIIlIllIlII =np .arange (IIIllIIllIIlIllIl .shape [0 ]);np .random .shuffle (IlllIIlIIlIllIlII );IIIllIIllIIlIllIl =IIIllIIllIIlIllIl [IlllIIlIIlIllIlII ]
	if IIIllIIllIIlIllIl .shape [0 ]>2e5 :
		IIIIIIllIIlIIlIII .append (_IlIIlIIlIIIIlllIl %IIIllIIllIIlIllIl .shape [0 ]);yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII )
		try :IIIllIIllIIlIllIl =MiniBatchKMeans (n_clusters =10000 ,verbose =_IllllIIIllIlIlIlI ,batch_size =256 *IIlIIIlIllIIIllll .n_cpu ,compute_labels =_IlIlIlIlIIIIIIlIl ,init ='random').fit (IIIllIIllIIlIllIl ).cluster_centers_ 
		except :IIIlllIlIllIIIIll =traceback .format_exc ();print (IIIlllIlIllIIIIll );IIIIIIllIIlIIlIII .append (IIIlllIlIllIIIIll );yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII )
	np .save (_IIllIlIllIIIIlIlI %IIlllIIIIlIIIlIIl ,IIIllIIllIIlIllIl );IlIIIIllIIlIIllIl =min (int (16 *np .sqrt (IIIllIIllIIlIllIl .shape [0 ])),IIIllIIllIIlIllIl .shape [0 ]//39 );IIIIIIllIIlIIlIII .append ('%s,%s'%(IIIllIIllIIlIllIl .shape ,IlIIIIllIIlIIllIl ));yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII );IIllIIIIIIIIlIlIl =faiss .index_factory (256 if IIIllIIIIIlIllIII ==_IlIlIIIlIlllIlIll else 768 ,_IllIlIIIlIIIllIIl %IlIIIIllIIlIIllIl );IIIIIIllIIlIIlIII .append ('training');yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII );IllIIllIlIIIlIIII =faiss .extract_index_ivf (IIllIIIIIIIIlIlIl );IllIIllIlIIIlIIII .nprobe =1 ;IIllIIIIIIIIlIlIl .train (IIIllIIllIIlIllIl );faiss .write_index (IIllIIIIIIIIlIlIl ,_IIlIIlIIlllllIlIl %(IIlllIIIIlIIIlIIl ,IlIIIIllIIlIIllIl ,IllIIllIlIIIlIIII .nprobe ,IllIllIIIlIllllll ,IIIllIIIIIlIllIII ));IIIIIIllIIlIIlIII .append ('adding');yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII );IlllIllllIlIlIIlI =8192 
	for IlIlIIlllIlIlIlll in range (0 ,IIIllIIllIIlIllIl .shape [0 ],IlllIllllIlIlIIlI ):IIllIIIIIIIIlIlIl .add (IIIllIIllIIlIllIl [IlIlIIlllIlIlIlll :IlIlIIlllIlIlIlll +IlllIllllIlIlIIlI ])
	faiss .write_index (IIllIIIIIIIIlIlIl ,_IIIlIllIlIllIIlll %(IIlllIIIIlIIIlIIl ,IlIIIIllIIlIIllIl ,IllIIllIlIIIlIIII .nprobe ,IllIllIIIlIllllll ,IIIllIIIIIlIllIII ));IIIIIIllIIlIIlIII .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIIIllIIlIIllIl ,IllIIllIlIIIlIIII .nprobe ,IllIllIIIlIllllll ,IIIllIIIIIlIllIII ));yield _IlIlIllIIIlIIIlII .join (IIIIIIllIIlIIlIII )
def IlIIllIlIllIlIllI (IIlIIlIIlIIllIIIl ,IlIIllIIllIllIllI ,IlllllIlIIIIllIlI ,IllIllIlIIIllIIll ,IllIIIlllllIIIIlI ,IIIlIlIlIIIlIlllI ,IlIlllIllIIIIlllI ,IlIIIIIIllllllIII ,IllIIIIllIlIIIIll ,IlIIllIIllIIlIllI ,IIlllIllIllIlllll ,IIIlllIIIlIIIlIll ,IIIIIlIIllIIlIlll ,IlIlIlIIlIllIIIIl ,IIIllIlllIlIIlIll ,IlIIllIlIlllIlIlI ,IllllIlIIIIIIlIII ,IIIIIlllIlllIllll ):
	IlIIlIlIlIlIllIII =[]
	def IIIlIllIIIIIlllIl (IlIlIlIIIllIIIIlI ):IlIIlIlIlIlIllIII .append (IlIlIlIIIllIIIIlI );return _IlIlIllIIIlIIIlII .join (IlIIlIlIlIlIllIII )
	IlIllIlIIlIlllIII =_IlIlIlIllIIlIIlIl %(IIIlllIlIlIlIllll ,IIlIIlIIlIIllIIIl );IIIllllllIlIIlIIl ='%s/preprocess.log'%IlIllIlIIlIlllIII ;IlIllIIllIllIIIlI ='%s/extract_fl_feature.log'%IlIllIlIIlIlllIII ;IIlIIIIlIlIIIIIIl =_IIIIlIIllIlIlIIIl %IlIllIlIIlIlllIII ;IIIlllIIIllIlIIlI =_IlIlllIlIllllIIIl %IlIllIlIIlIlllIII if IllllIlIIIIIIlIII ==_IlIlIIIlIlllIlIll else _IlIlIIIIlIllllIII %IlIllIlIIlIlllIII ;os .makedirs (IlIllIlIIlIlllIII ,exist_ok =_IllllIIIllIlIlIlI );open (IIIllllllIlIIlIIl ,'w').close ();IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IllIllIlIIIllIIll ,IIIllllllllIlIIlI [IlIIllIIllIllIllI ],IIIlIlIlIIIlIlllI ,IlIllIlIIlIlllIII )+str (IIlIIIlIllIIIllll .noparallel );yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll ('step1:正在处理数据'));yield IIIlIllIIIIIlllIl (IIIlIlllIllIIlllI );IIlIIllllllIlIIll =Popen (IIIlIlllIllIIlllI ,shell =_IllllIIIllIlIlIlI );IIlIIllllllIlIIll .wait ()
	with open (IIIllllllIlIIlIIl ,_IlIIlIIllllIIllII )as IlIIIIIIIIIIlllIl :print (IlIIIIIIIIIIlllIl .read ())
	open (IlIllIIllIllIIIlI ,'w')
	if IlllllIlIIIIllIlI :
		yield IIIlIllIIIIIlllIl ('step2a:正在提取音高')
		if IlIlllIllIIIIlllI !=_IIIllllIIllllIIlI :IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +' extract_fl_print.py "%s" %s %s'%(IlIllIlIIlIlllIII ,IIIlIlIlIIIlIlllI ,IlIlllIllIIIIlllI );yield IIIlIllIIIIIlllIl (IIIlIlllIllIIlllI );IIlIIllllllIlIIll =Popen (IIIlIlllIllIIlllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IIlIIllllllIlIIll .wait ()
		else :
			IIIIIlllIlllIllll =IIIIIlllIlllIllll .split ('-');IIlllIIllIIIlIIll =len (IIIIIlllIlllIllll );IlIlIlllIlllIIlIl =[]
			for (IlIIIIlIIlIllIllI ,IlIIlllIlIIlIllIl )in enumerate (IIIIIlllIlllIllll ):IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IIlllIIllIIIlIIll ,IlIIIIlIIlIllIllI ,IlIIlllIlIIlIllIl ,IlIllIlIIlIlllIII ,IIlIIIlIllIIIllll .is_half );yield IIIlIllIIIIIlllIl (IIIlIlllIllIIlllI );IIlIIllllllIlIIll =Popen (IIIlIlllIllIIlllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IlIlIlllIlllIIlIl .append (IIlIIllllllIlIIll )
			for IIlIIllllllIlIIll in IlIlIlllIlllIIlIl :IIlIIllllllIlIIll .wait ()
		with open (IlIllIIllIllIIIlI ,_IlIIlIIllllIIllII )as IlIIIIIIIIIIlllIl :print (IlIIIIIIIIIIlllIl .read ())
	else :yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll ('step2a:无需提取音高'))
	yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll ('step2b:正在提取特征'));IIIIIlllllIlIIIlI =IlIlIlIIlIllIIIIl .split ('-');IIlllIIllIIIlIIll =len (IIIIIlllllIlIIIlI );IlIlIlllIlllIIlIl =[]
	for (IlIIIIlIIlIllIllI ,IlIIlllIlIIlIllIl )in enumerate (IIIIIlllllIlIIIlI ):IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IIlIIIlIllIIIllll .device ,IIlllIIllIIIlIIll ,IlIIIIlIIlIllIllI ,IlIIlllIlIIlIllIl ,IlIllIlIIlIlllIII ,IllllIlIIIIIIlIII );yield IIIlIllIIIIIlllIl (IIIlIlllIllIIlllI );IIlIIllllllIlIIll =Popen (IIIlIlllIllIIlllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IlIlIlllIlllIIlIl .append (IIlIIllllllIlIIll )
	for IIlIIllllllIlIIll in IlIlIlllIlllIIlIl :IIlIIllllllIlIIll .wait ()
	with open (IlIllIIllIllIIIlI ,_IlIIlIIllllIIllII )as IlIIIIIIIIIIlllIl :print (IlIIIIIIIIIIlllIl .read ())
	yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll ('step3a:正在训练模型'))
	if IlllllIlIIIIllIlI :IlllIIllllIIIIIll ='%s/2a_f0'%IlIllIlIIlIlllIII ;IllllIlIlllIIlIll =_IIlIlllIIIlllIIII %IlIllIlIIlIlllIII ;IllIIIIIIllIIIllI =set ([IllllIIIIIllIlIII .split (_IIlIIlllllIIlllIl )[0 ]for IllllIIIIIllIlIII in os .listdir (IIlIIIIlIlIIIIIIl )])&set ([IlllIlllllIlIlIlI .split (_IIlIIlllllIIlllIl )[0 ]for IlllIlllllIlIlIlI in os .listdir (IIIlllIIIllIlIIlI )])&set ([IIIIIlllIIIlIlIII .split (_IIlIIlllllIIlllIl )[0 ]for IIIIIlllIIIlIlIII in os .listdir (IlllIIllllIIIIIll )])&set ([IIIIlIIllIIIIlllI .split (_IIlIIlllllIIlllIl )[0 ]for IIIIlIIllIIIIlllI in os .listdir (IllllIlIlllIIlIll )])
	else :IllIIIIIIllIIIllI =set ([IIIIlIlIllIIIlllI .split (_IIlIIlllllIIlllIl )[0 ]for IIIIlIlIllIIIlllI in os .listdir (IIlIIIIlIlIIIIIIl )])&set ([IIlllIllIIIIIlllI .split (_IIlIIlllllIIlllIl )[0 ]for IIlllIllIIIIIlllI in os .listdir (IIIlllIIIllIlIIlI )])
	IlIIlllIlllIlIlII =[]
	for IlIIlllIllIllllII in IllIIIIIIllIIIllI :
		if IlllllIlIIIIllIlI :IlIIlllIlllIlIlII .append (_IlIIIIllIIIIlIIll %(IIlIIIIlIlIIIIIIl .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IIIlllIIIllIlIIlI .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IlllIIllllIIIIIll .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IllllIlIlllIIlIll .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IllIIIlllllIIIIlI ))
		else :IlIIlllIlllIlIlII .append (_IIIlIIIIlIlIIllIl %(IIlIIIIlIlIIIIIIl .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IIIlllIIIllIlIIlI .replace (_IIIIIllllllIllIll ,_IlIIlIllllIIIlIll ),IlIIlllIllIllllII ,IllIIIlllllIIIIlI ))
	IIIIIlIlIlIIlllIl =256 if IllllIlIIIIIIlIII ==_IlIlIIIlIlllIlIll else 768 
	if IlllllIlIIIIllIlI :
		for _IIIIlIllIIIlIIllI in range (2 ):IlIIlllIlllIlIlII .append (_IIllIIllIIlIllIII %(IIIlllIlIlIlIllll ,IlIIllIIllIllIllI ,IIIlllIlIlIlIllll ,IIIIIlIlIlIIlllIl ,IIIlllIlIlIlIllll ,IIIlllIlIlIlIllll ,IllIIIlllllIIIIlI ))
	else :
		for _IIIIlIllIIIlIIllI in range (2 ):IlIIlllIlllIlIlII .append (_IlIIllllIlllIllll %(IIIlllIlIlIlIllll ,IlIIllIIllIllIllI ,IIIlllIlIlIlIllll ,IIIIIlIlIlIIlllIl ,IllIIIlllllIIIIlI ))
	shuffle (IlIIlllIlllIlIlII )
	with open (_IIlIIIlIIIllIlIlI %IlIllIlIIlIlllIII ,'w')as IlIIIIIIIIIIlllIl :IlIIIIIIIIIIlllIl .write (_IlIlIllIIIlIIIlII .join (IlIIlllIlllIlIlII ))
	yield IIIlIllIIIIIlllIl (_IllIllIlIlIIIlIlI )
	if IlIlIlIIlIllIIIIl :IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +_IIlllIIIIllIIIIlI %(IIlIIlIIlIIllIIIl ,IlIIllIIllIllIllI ,1 if IlllllIlIIIIllIlI else 0 ,IlIIllIIllIIlIllI ,IlIlIlIIlIllIIIIl ,IllIIIIllIlIIIIll ,IlIIIIIIllllllIII ,_IIlllIllIlIIlIlIl %IIIlllIIIlIIIlIll if IIIlllIIIlIIIlIll !=''else '',_IIIIllIlIIlIllIll %IIIIIlIIllIIlIlll if IIIIIlIIllIIlIlll !=''else '',1 if IIlllIllIllIlllll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IIIllIlllIlIIlIll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIIllIlIlllIlIlI ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,IllllIlIIIIIIlIII )
	else :IIIlIlllIllIIlllI =IIlIIIlIllIIIllll .python_cmd +_IlllIlIlIlIlIlIll %(IIlIIlIIlIIllIIIl ,IlIIllIIllIllIllI ,1 if IlllllIlIIIIllIlI else 0 ,IlIIllIIllIIlIllI ,IllIIIIllIlIIIIll ,IlIIIIIIllllllIII ,_IIlllIllIlIIlIlIl %IIIlllIIIlIIIlIll if IIIlllIIIlIIIlIll !=''else '',_IIIIllIlIIlIllIll %IIIIIlIIllIIlIlll if IIIIIlIIllIIlIlll !=''else '',1 if IIlllIllIllIlllll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IIIllIlllIlIIlIll ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,1 if IlIIllIlIlllIlIlI ==IllIIllllIlllIIll (_IIIIllIllIlllIIll )else 0 ,IllllIlIIIIIIlIII )
	yield IIIlIllIIIIIlllIl (IIIlIlllIllIIlllI );IIlIIllllllIlIIll =Popen (IIIlIlllIllIIlllI ,shell =_IllllIIIllIlIlIlI ,cwd =IIIlllIlIlIlIllll );IIlIIllllllIlIIll .wait ();yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll (_IllIIlIllllIIlIII ));IIlIlIlIIlIIIlIIl =[];IIIllIIIllIllllIl =list (os .listdir (IIIlllIIIllIlIIlI ))
	for IlIIlllIllIllllII in sorted (IIIllIIIllIllllIl ):IllIIIlllIIIIIIII =np .load (_IIlIllIlllIlIllII %(IIIlllIIIllIlIIlI ,IlIIlllIllIllllII ));IIlIlIlIIlIIIlIIl .append (IllIIIlllIIIIIIII )
	IllIlIIIlIlIIllll =np .concatenate (IIlIlIlIIlIIIlIIl ,0 );IllIIllIIllIIIIIl =np .arange (IllIlIIIlIlIIllll .shape [0 ]);np .random .shuffle (IllIIllIIllIIIIIl );IllIlIIIlIlIIllll =IllIlIIIlIlIIllll [IllIIllIIllIIIIIl ]
	if IllIlIIIlIlIIllll .shape [0 ]>2e5 :
		IllIIlllllIIlIlll =_IlIIlIIlIIIIlllIl %IllIlIIIlIlIIllll .shape [0 ];print (IllIIlllllIIlIlll );yield IIIlIllIIIIIlllIl (IllIIlllllIIlIlll )
		try :IllIlIIIlIlIIllll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IllllIIIllIlIlIlI ,batch_size =256 *IIlIIIlIllIIIllll .n_cpu ,compute_labels =_IlIlIlIlIIIIIIlIl ,init ='random').fit (IllIlIIIlIlIIllll ).cluster_centers_ 
		except :IllIIlllllIIlIlll =traceback .format_exc ();print (IllIIlllllIIlIlll );yield IIIlIllIIIIIlllIl (IllIIlllllIIlIlll )
	np .save (_IIllIlIllIIIIlIlI %IlIllIlIIlIlllIII ,IllIlIIIlIlIIllll );IIIIIllIIIllIlIIl =min (int (16 *np .sqrt (IllIlIIIlIlIIllll .shape [0 ])),IllIlIIIlIlIIllll .shape [0 ]//39 );yield IIIlIllIIIIIlllIl ('%s,%s'%(IllIlIIIlIlIIllll .shape ,IIIIIllIIIllIlIIl ));IlIIllIIIlllIIIIl =faiss .index_factory (256 if IllllIlIIIIIIlIII ==_IlIlIIIlIlllIlIll else 768 ,_IllIlIIIlIIIllIIl %IIIIIllIIIllIlIIl );yield IIIlIllIIIIIlllIl ('training index');IlIlllIIIlIIIllll =faiss .extract_index_ivf (IlIIllIIIlllIIIIl );IlIlllIIIlIIIllll .nprobe =1 ;IlIIllIIIlllIIIIl .train (IllIlIIIlIlIIllll );faiss .write_index (IlIIllIIIlllIIIIl ,_IIlIIlIIlllllIlIl %(IlIllIlIIlIlllIII ,IIIIIllIIIllIlIIl ,IlIlllIIIlIIIllll .nprobe ,IIlIIlIIlIIllIIIl ,IllllIlIIIIIIlIII ));yield IIIlIllIIIIIlllIl ('adding index');IllIIIIllIlllIIIl =8192 
	for IlIIIllllllllIllI in range (0 ,IllIlIIIlIlIIllll .shape [0 ],IllIIIIllIlllIIIl ):IlIIllIIIlllIIIIl .add (IllIlIIIlIlIIllll [IlIIIllllllllIllI :IlIIIllllllllIllI +IllIIIIllIlllIIIl ])
	faiss .write_index (IlIIllIIIlllIIIIl ,_IIIlIllIlIllIIlll %(IlIllIlIIlIlllIII ,IIIIIllIIIllIlIIl ,IlIlllIIIlIIIllll .nprobe ,IIlIIlIIlIIllIIIl ,IllllIlIIIIIIlIII ));yield IIIlIllIIIIIlllIl ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIIIIllIIIllIlIIl ,IlIlllIIIlIIIllll .nprobe ,IIlIIlIIlIIllIIIl ,IllllIlIIIIIIlIII ));yield IIIlIllIIIIIlllIl (IllIIllllIlllIIll ('全流程结束！'))
def IIIlIIIIIIIlIIIII (IlllIIIlllIIIllII ):
	IlIIlIlIIIlIlIlII ='train.log'
	if not os .path .exists (IlllIIIlllIIIllII .replace (os .path .basename (IlllIIIlllIIIllII ),IlIIlIlIIIlIlIlII )):return {_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },{_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },{_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
	try :
		with open (IlllIIIlllIIIllII .replace (os .path .basename (IlllIIIlllIIIllII ),IlIIlIlIIIlIlIlII ),_IlIIlIIllllIIllII )as IIlIIlIIlllIIllll :IIIIllIlIIIllIIII =eval (IIlIIlIIlllIIllll .read ().strip (_IlIlIllIIIlIIIlII ).split (_IlIlIllIIIlIIIlII )[0 ].split ('\t')[-1 ]);IllIllIllllllllII ,IlIIlllllllIlIlII =IIIIllIlIIIllIIII [_IlIllllIlIlIIIlII ],IIIIllIlIIIllIIII ['if_f0'];IllIllllIllIlIIIl =_IIIIlIllllIIIlIIl if _IIllIllllllIllIII in IIIIllIlIIIllIIII and IIIIllIlIIIllIIII [_IIllIllllllIllIII ]==_IIIIlIllllIIIlIIl else _IlIlIIIlIlllIlIll ;return IllIllIllllllllII ,str (IlIIlllllllIlIlII ),IllIllllIllIlIIIl 
	except :traceback .print_exc ();return {_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },{_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl },{_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
def IIllllIIIIllIllIl (IllIIllIllIIlIlll ):
	if IllIIllIllIIlIlll ==_IIIllllIIllllIIlI :IlIlIlIllIllIlIII =_IllllIIIllIlIlIlI 
	else :IlIlIlIllIllIlIII =_IlIlIlIlIIIIIIlIl 
	return {_IlllIIlIlIlIlllII :IlIlIlIllIllIlIII ,_IllIllllIllIIIIll :_IlIlIlIlIlIIIIlIl }
def IIIlllIIlIIlIlIIl (IIlllIlIIIIllllIl ,IllIIIllIIlIIIlll ):IIllIlIIIlllllIlI ='rnd';IIlllIlIIIlIIlllI ='pitchf';IlIIlIIIIIlllIlIl ='pitch';IlIlIlIllIIIllIII ='phone';global IlIllllllIlIllIlI ;IlIllllllIlIllIlI =torch .load (IIlllIlIIIIllllIl ,map_location =_IIIIlIIIllllIIllI );IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ][-3 ]=IlIllllllIlIllIlI [_IIIIllIlIIIIlIIlI ][_IlIlIlllIllllIIIl ].shape [0 ];IIIIIlIlIIIIlIIIl =256 if IlIllllllIlIllIlI .get (_IIllIllllllIllIII ,_IlIlIIIlIlllIlIll )==_IlIlIIIlIlllIlIll else 768 ;IIlIllIIIlIIIllII =torch .rand (1 ,200 ,IIIIIlIlIIIIlIIIl );IllIlIIIlIIlllIIl =torch .tensor ([200 ]).long ();IlllllIllIIlIIIlI =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IlIIlllllllIlllIl =torch .rand (1 ,200 );IllIIIIIIlIIIlIll =torch .LongTensor ([0 ]);IlIlIIIllllllIIII =torch .rand (1 ,192 ,200 );IIllIlIIlIIlIIIIl =_IIIIlIIIllllIIllI ;IlIIIllIIllIllllI =SynthesizerTrnMsNSFsidM (*IlIllllllIlIllIlI [_IlIlIIIIIIllIIIll ],is_half =_IlIlIlIlIIIIIIlIl ,version =IlIllllllIlIllIlI .get (_IIllIllllllIllIII ,_IlIlIIIlIlllIlIll ));IlIIIllIIllIllllI .load_state_dict (IlIllllllIlIllIlI [_IIIIllIlIIIIlIIlI ],strict =_IlIlIlIlIIIIIIlIl );IlIllllIlIlIllIlI =[IlIlIlIllIIIllIII ,'phone_lengths',IlIIlIIIIIlllIlIl ,IIlllIlIIIlIIlllI ,'ds',IIllIlIIIlllllIlI ];IllllIllIllIllIIl =['audio'];torch .onnx .export (IlIIIllIIllIllllI ,(IIlIllIIIlIIIllII .to (IIllIlIIlIIlIIIIl ),IllIlIIIlIIlllIIl .to (IIllIlIIlIIlIIIIl ),IlllllIllIIlIIIlI .to (IIllIlIIlIIlIIIIl ),IlIIlllllllIlllIl .to (IIllIlIIlIIlIIIIl ),IllIIIIIIlIIIlIll .to (IIllIlIIlIIlIIIIl ),IlIlIIIllllllIIII .to (IIllIlIIlIIlIIIIl )),IllIIIllIIlIIIlll ,dynamic_axes ={IlIlIlIllIIIllIII :[1 ],IlIIlIIIIIlllIlIl :[1 ],IIlllIlIIIlIIlllI :[1 ],IIllIlIIIlllllIlI :[2 ]},do_constant_folding =_IlIlIlIlIIIIIIlIl ,opset_version =13 ,verbose =_IlIlIlIlIIIIIIlIl ,input_names =IlIllllIlIlIllIlI ,output_names =IllllIllIllIllIIl );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IllIIIlIIIllllIIl :
	gr .Markdown (value =IllIIllllIlllIIll ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IllIIllllIlllIIll ('模型推理')):
			with gr .Row ():IlIlIlllIlIllIlII =gr .Dropdown (label =IllIIllllIlllIIll ('推理音色'),choices =sorted (IlIlllIIIIllIIIll ));IIIIlIIlIIllIIlll =gr .Button (IllIIllllIlllIIll ('刷新音色列表和索引路径'),variant =_IllIIlIllllllllll );IlllIIlIlllllllll =gr .Button (IllIIllllIlllIIll ('卸载音色省显存'),variant =_IllIIlIllllllllll );IIlIIlIlIIlIlIIIl =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IllIIllllIlllIIll ('请选择说话人id'),value =0 ,visible =_IlIlIlIlIIIIIIlIl ,interactive =_IllllIIIllIlIlIlI );IlllIIlIlllllllll .click (fn =IIlllIIlIlIlllIIl ,inputs =[],outputs =[IlIlIlllIlIllIlII ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IlIIIllllIlllIllI =gr .Number (label =IllIIllllIlllIIll (_IlIllIllllIIllIIl ),value =0 );IIIIlIIIIIIllIIll =gr .Textbox (label =IllIIllllIlllIIll ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IlIlllllIIIllllII =gr .Radio (label =IllIIllllIlllIIll (_IlIlIIlIIlllIllII ),choices =[_IIIlIllIlIIllIIIl ,_IlIlIIIlIIllllIlI ,'crepe',_IIIlIllIlIIllIIIl ],value =_IIIlIllIlIIllIIIl ,interactive =_IllllIIIllIlIlIlI );IIlllIIlIlIIlIlll =gr .Slider (minimum =0 ,maximum =7 ,label =IllIIllllIlllIIll (_IIllIllIlIIIlIlll ),value =3 ,step =1 ,interactive =_IllllIIIllIlIlIlI )
					with gr .Column ():IlllllIlIlIIIlIIl =gr .Textbox (label =IllIIllllIlllIIll (_IllIIIIIIllIIIIIl ),value ='',interactive =_IllllIIIllIlIlIlI );IIIIIIIlllIIllllI =gr .Dropdown (label =IllIIllllIlllIIll (_IllllllIIIIlIIIlI ),choices =sorted (IIIIlllIIIlllIlIl ),interactive =_IllllIIIllIlIlIlI );IIIIlIIlIIllIIlll .click (fn =IIllIlIIIIlIIIlII ,inputs =[],outputs =[IlIlIlllIlIllIlII ,IIIIIIIlllIIllllI ],api_name ='infer_refresh');IIIlllIIllIIllIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIllllIlllIIll ('检索特征占比'),value =.75 ,interactive =_IllllIIIllIlIlIlI )
					with gr .Column ():IllllIIIlIIlIIlll =gr .Slider (minimum =0 ,maximum =48000 ,label =IllIIllllIlllIIll (_IlIlIIIIlIIlIllIl ),value =0 ,step =1 ,interactive =_IllllIIIllIlIlIlI );IIIllIIIlIIllllll =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIllllIlllIIll (_IIIllllIlllIIIIll ),value =.25 ,interactive =_IllllIIIllIlIlIlI );IIlIIIIIlllIIIIll =gr .Slider (minimum =0 ,maximum =.5 ,label =IllIIllllIlllIIll (_IlIlIIIIlIlllllII ),value =.33 ,step =.01 ,interactive =_IllllIIIllIlIlIlI )
					IIlIIIIIIIIllIIIl =gr .File (label =IllIIllllIlllIIll ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IIIlIIlIIIIIIlIIl =gr .Button (IllIIllllIlllIIll ('转换'),variant =_IllIIlIllllllllll )
					with gr .Row ():IllIlIllllIlIIIll =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ));IlIIlIlIIllIllIII =gr .Audio (label =IllIIllllIlllIIll ('输出音频(右下角三个点,点了可以下载)'))
					IIIlIIlIIIIIIlIIl .click (IlllllIIIIIlIlIll ,[IIlIIlIlIIlIlIIIl ,IIIIlIIIIIIllIIll ,IlIIIllllIlllIllI ,IIlIIIIIIIIllIIIl ,IlIlllllIIIllllII ,IlllllIlIlIIIlIIl ,IIIIIIIlllIIllllI ,IIIlllIIllIIllIIl ,IIlllIIlIlIIlIlll ,IllllIIIlIIlIIlll ,IIIllIIIlIIllllll ,IIlIIIIIlllIIIIll ],[IllIlIllllIlIIIll ,IlIIlIlIIllIllIII ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IlIIIIIIIlIlllIIl =gr .Number (label =IllIIllllIlllIIll (_IlIllIllllIIllIIl ),value =0 );IlllllIIlIIIllIlI =gr .Textbox (label =IllIIllllIlllIIll ('指定输出文件夹'),value =_IIIllIIIlllIIIIlI );IlIIIlIIlIIlllIlI =gr .Radio (label =IllIIllllIlllIIll (_IlIlIIlIIlllIllII ),choices =[_IIIlIllIlIIllIIIl ,_IlIlIIIlIIllllIlI ,'crepe',_IIIlIllIlIIllIIIl ],value =_IIIlIllIlIIllIIIl ,interactive =_IllllIIIllIlIlIlI );IIlIIllIlllIlIIIl =gr .Slider (minimum =0 ,maximum =7 ,label =IllIIllllIlllIIll (_IIllIllIlIIIlIlll ),value =3 ,step =1 ,interactive =_IllllIIIllIlIlIlI )
					with gr .Column ():IIlIlIIlllllIIlll =gr .Textbox (label =IllIIllllIlllIIll (_IllIIIIIIllIIIIIl ),value ='',interactive =_IllllIIIllIlIlIlI );IlIllIlllIlIlllll =gr .Dropdown (label =IllIIllllIlllIIll (_IllllllIIIIlIIIlI ),choices =sorted (IIIIlllIIIlllIlIl ),interactive =_IllllIIIllIlIlIlI );IIIIlIIlIIllIIlll .click (fn =lambda :IIllIlIIIIlIIIlII ()[1 ],inputs =[],outputs =IlIllIlllIlIlllll ,api_name ='infer_refresh_batch');IIIlIlIlIIlIIIIll =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIllllIlllIIll ('检索特征占比'),value =1 ,interactive =_IllllIIIllIlIlIlI )
					with gr .Column ():IIIIIlIIlIlIIIlIl =gr .Slider (minimum =0 ,maximum =48000 ,label =IllIIllllIlllIIll (_IlIlIIIIlIIlIllIl ),value =0 ,step =1 ,interactive =_IllllIIIllIlIlIlI );IIIlIIlIlllllIIll =gr .Slider (minimum =0 ,maximum =1 ,label =IllIIllllIlllIIll (_IIIllllIlllIIIIll ),value =1 ,interactive =_IllllIIIllIlIlIlI );IlIllllIlIIIIIIII =gr .Slider (minimum =0 ,maximum =.5 ,label =IllIIllllIlllIIll (_IlIlIIIIlIlllllII ),value =.33 ,step =.01 ,interactive =_IllllIIIllIlIlIlI )
					with gr .Column ():IlIllIllIlIlIlIII =gr .Textbox (label =IllIIllllIlllIIll ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IIllllIIlllIIlIll =gr .File (file_count ='multiple',label =IllIIllllIlllIIll (_IIlIIllllllllIlIl ))
					with gr .Row ():IIIIIlIllIIIlIlII =gr .Radio (label =IllIIllllIlllIIll ('导出文件格式'),choices =[_IllIlIIllIlIllIlI ,_IllIlllIllllIIlIl ,'mp3','m4a'],value =_IllIlllIllllIIlIl ,interactive =_IllllIIIllIlIlIlI );IlIIIIIIIllIlllII =gr .Button (IllIIllllIlllIIll ('转换'),variant =_IllIIlIllllllllll );IIIlllllllIIlllll =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ))
					IlIIIIIIIllIlllII .click (IIlIllIIlIIlIIllI ,[IIlIIlIlIIlIlIIIl ,IlIllIllIlIlIlIII ,IlllllIIlIIIllIlI ,IIllllIIlllIIlIll ,IlIIIIIIIlIlllIIl ,IlIIIlIIlIIlllIlI ,IIlIlIIlllllIIlll ,IlIllIlllIlIlllll ,IIIlIlIlIIlIIIIll ,IIlIIllIlllIlIIIl ,IIIIIlIIlIlIIIlIl ,IIIlIIlIlllllIIll ,IlIllllIlIIIIIIII ,IIIIIlIllIIIlIlII ],[IIIlllllllIIlllll ],api_name ='infer_convert_batch')
			IlIlIlllIlIllIlII .change (fn =IllIlIlIlllllllII ,inputs =[IlIlIlllIlIllIlII ,IIlIIIIIlllIIIIll ,IlIllllIlIIIIIIII ],outputs =[IIlIIlIlIIlIlIIIl ,IIlIIIIIlllIIIIll ,IlIllllIlIIIIIIII ,IIIIIIIlllIIllllI ])
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IIIlIIIlIIIIIlIlI =gr .Textbox (label =IllIIllllIlllIIll ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IlIIllIlIIlIllIII =gr .File (file_count ='multiple',label =IllIIllllIlllIIll (_IIlIIllllllllIlIl ))
					with gr .Column ():IIlIIIlIllIlIlIII =gr .Dropdown (label =IllIIllllIlllIIll ('模型'),choices =IlllIlllllIlIlllI );IlIIIllIlllIlIlll =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IllllIIIllIlIlIlI ,visible =_IlIlIlIlIIIIIIlIl );IIlIIlIlllIlIIlll =gr .Textbox (label =IllIIllllIlllIIll ('指定输出主人声文件夹'),value =_IIIllIIIlllIIIIlI );IIlIIlIIIIlIlIlII =gr .Textbox (label =IllIIllllIlllIIll ('指定输出非主人声文件夹'),value =_IIIllIIIlllIIIIlI );IlllIlllIIlIIlllI =gr .Radio (label =IllIIllllIlllIIll ('导出文件格式'),choices =[_IllIlIIllIlIllIlI ,_IllIlllIllllIIlIl ,'mp3','m4a'],value =_IllIlllIllllIIlIl ,interactive =_IllllIIIllIlIlIlI )
					IIIlllIlIIlllllII =gr .Button (IllIIllllIlllIIll ('转换'),variant =_IllIIlIllllllllll );IIlIlIIIllIllllII =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ));IIIlllIlIIlllllII .click (IIlIlllIIIlIIIIIl ,[IIlIIIlIllIlIlIII ,IIIlIIIlIIIIIlIlI ,IIlIIlIlllIlIIlll ,IlIIllIlIIlIllIII ,IIlIIlIIIIlIlIlII ,IlIIIllIlllIlIlll ,IlllIlllIIlIIlllI ],[IIlIlIIIllIllllII ],api_name ='uvr_convert')
		with gr .TabItem (IllIIllllIlllIIll ('训练')):
			gr .Markdown (value =IllIIllllIlllIIll ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIlIIlIlIlIIlIlII =gr .Textbox (label =IllIIllllIlllIIll ('输入实验名'),value ='mi-test');IIIIIlIIIIIIlIlll =gr .Radio (label =IllIIllllIlllIIll ('目标采样率'),choices =[_IllIIIllIllIIlIII ],value =_IllIIIllIllIIlIII ,interactive =_IllllIIIllIlIlIlI );IIIlIllIIllllIlII =gr .Radio (label =IllIIllllIlllIIll ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IllllIIIllIlIlIlI ,_IlIlIlIlIIIIIIlIl ],value =_IllllIIIllIlIlIlI ,interactive =_IllllIIIllIlIlIlI );IIIlIlIIIIIIlIlll =gr .Radio (label =IllIIllllIlllIIll ('版本'),choices =[_IIIIlIllllIIIlIIl ],value =_IIIIlIllllIIIlIIl ,interactive =_IllllIIIllIlIlIlI ,visible =_IllllIIIllIlIlIlI );IllIIIIIIIIlIIIlI =gr .Slider (minimum =0 ,maximum =IIlIIIlIllIIIllll .n_cpu ,step =1 ,label =IllIIllllIlllIIll ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IIlIIIlIllIIIllll .n_cpu /1.5 )),interactive =_IllllIIIllIlIlIlI )
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IllIlIIllllllIllI =gr .Textbox (label =IllIIllllIlllIIll ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IIIllIIllllIIIlIl =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IllIIllllIlllIIll ('请指定说话人id'),value =0 ,interactive =_IllllIIIllIlIlIlI );IlIIIIIIIllIlllII =gr .Button (IllIIllllIlllIIll ('处理数据'),variant =_IllIIlIllllllllll );IlIllIIlllIIlIIII =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ),value ='');IlIIIIIIIllIlllII .click (IlIIIIIlIIIlIlllI ,[IllIlIIllllllIllI ,IIlIIlIlIlIIlIlII ,IIIIIlIIIIIIlIlll ,IllIIIIIIIIlIIIlI ],[IlIllIIlllIIlIIII ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IIIIlllIllIIlIlIl =gr .Textbox (label =IllIIllllIlllIIll (_IIIIIIllIllIllllI ),value =IllIIllIlllllIIll ,interactive =_IllllIIIllIlIlIlI );IIllIIIlIIllllIIl =gr .Textbox (label =IllIIllllIlllIIll ('显卡信息'),value =IllllIIIIlIIlllII )
					with gr .Column ():IIIIIIllllllIllIl =gr .Radio (label =IllIIllllIlllIIll ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IIIlIllIlIIllIIIl ,_IlIlIIIlIIllllIlI ,'dio',_IIIlIllIlIIllIIIl ,_IIIllllIIllllIIlI ],value =_IIIllllIIllllIIlI ,interactive =_IllllIIIllIlIlIlI );IIlIIlIIIIIllllll =gr .Textbox (label =IllIIllllIlllIIll ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IllIIllIlllllIIll ,IllIIllIlllllIIll ),interactive =_IllllIIIllIlIlIlI ,visible =_IllllIIIllIlIlIlI )
					IIIlllIlIIlllllII =gr .Button (IllIIllllIlllIIll ('特征提取'),variant =_IllIIlIllllllllll );IllIIIllIllIIllll =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ),value ='',max_lines =8 );IIIIIIllllllIllIl .change (fn =IIllllIIIIllIllIl ,inputs =[IIIIIIllllllIllIl ],outputs =[IIlIIlIIIIIllllll ]);IIIlllIlIIlllllII .click (IlIllllllllIIlIIl ,[IIIIlllIllIIlIlIl ,IllIIIIIIIIlIIIlI ,IIIIIIllllllIllIl ,IIIlIllIIllllIlII ,IIlIIlIlIlIIlIlII ,IIIlIlIIIIIIlIlll ,IIlIIlIIIIIllllll ],[IllIIIllIllIIllll ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IllIIllllIlllIIll ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IIIlIIIlIIllIIIlI =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IllIIllllIlllIIll ('保存频率save_every_epoch'),value =5 ,interactive =_IllllIIIllIlIlIlI );IIllIlllIlllIIlIl =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IllIIllllIlllIIll ('总训练轮数total_epoch'),value =300 ,interactive =_IllllIIIllIlIlIlI );IlllIllIllIIIllII =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IllIIllllIlllIIll ('每张显卡的batch_size'),value =IIlIIIIIIlllllIlI ,interactive =_IllllIIIllIlIlIlI );IIlIIlIIlIIIIlllI =gr .Radio (label =IllIIllllIlllIIll ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IllIIllllIlllIIll (_IIIIllIllIlllIIll ),IllIIllllIlllIIll ('否')],value =IllIIllllIlllIIll (_IIIIllIllIlllIIll ),interactive =_IllllIIIllIlIlIlI );IlIIllIIIIlIIlIIl =gr .Radio (label =IllIIllllIlllIIll ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IllIIllllIlllIIll (_IIIIllIllIlllIIll ),IllIIllllIlllIIll ('否')],value =IllIIllllIlllIIll ('否'),interactive =_IllllIIIllIlIlIlI );IIIIIlIlllIlllIIl =gr .Radio (label =IllIIllllIlllIIll ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IllIIllllIlllIIll (_IIIIllIllIlllIIll ),IllIIllllIlllIIll ('否')],value =IllIIllllIlllIIll (_IIIIllIllIlllIIll ),interactive =_IllllIIIllIlIlIlI )
				with gr .Row ():IllIIlIIlllIIllIl =gr .Textbox (label =IllIIllllIlllIIll ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IllllIIIllIlIlIlI );IIIllllllIIIllIlI =gr .Textbox (label =IllIIllllIlllIIll ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IllllIIIllIlIlIlI );IIIIIlIIIIIIlIlll .change (IIIIllIIlllllllII ,[IIIIIlIIIIIIlIlll ,IIIlIllIIllllIlII ,IIIlIlIIIIIIlIlll ],[IllIIlIIlllIIllIl ,IIIllllllIIIllIlI ]);IIIlIlIIIIIIlIlll .change (IlIIIIlIIIlIIlllI ,[IIIIIlIIIIIIlIlll ,IIIlIllIIllllIlII ,IIIlIlIIIIIIlIlll ],[IllIIlIIlllIIllIl ,IIIllllllIIIllIlI ,IIIIIlIIIIIIlIlll ]);IIIlIllIIllllIlII .change (IllIlIIllllllIIII ,[IIIlIllIIllllIlII ,IIIIIlIIIIIIlIlll ,IIIlIlIIIIIIlIlll ],[IIIIIIllllllIllIl ,IllIIlIIlllIIllIl ,IIIllllllIIIllIlI ]);IIIIllllIIIllllIl =gr .Textbox (label =IllIIllllIlllIIll (_IIIIIIllIllIllllI ),value =IllIIllIlllllIIll ,interactive =_IllllIIIllIlIlIlI );IlIIllIllllIlIIll =gr .Button (IllIIllllIlllIIll ('训练模型'),variant =_IllIIlIllllllllll );IIIIIlllIllIIlIlI =gr .Button (IllIIllllIlllIIll ('训练特征索引'),variant =_IllIIlIllllllllll );IIIlIIIlllIlllIll =gr .Button (IllIIllllIlllIIll ('一键训练'),variant =_IllIIlIllllllllll );IIllIlIllIlIllIlI =gr .Textbox (label =IllIIllllIlllIIll (_IlllIIllIlIIlllII ),value ='',max_lines =10 );IlIIllIllllIlIIll .click (IlIllIIlIlllIIllI ,[IIlIIlIlIlIIlIlII ,IIIIIlIIIIIIlIlll ,IIIlIllIIllllIlII ,IIIllIIllllIIIlIl ,IIIlIIIlIIllIIIlI ,IIllIlllIlllIIlIl ,IlllIllIllIIIllII ,IIlIIlIIlIIIIlllI ,IllIIlIIlllIIllIl ,IIIllllllIIIllIlI ,IIIIllllIIIllllIl ,IlIIllIIIIlIIlIIl ,IIIIIlIlllIlllIIl ,IIIlIlIIIIIIlIlll ],IIllIlIllIlIllIlI ,api_name ='train_start');IIIIIlllIllIIlIlI .click (IlIIIIIIIIllllIll ,[IIlIIlIlIlIIlIlII ,IIIlIlIIIIIIlIlll ],IIllIlIllIlIllIlI );IIIlIIIlllIlllIll .click (IlIIllIlIllIlIllI ,[IIlIIlIlIlIIlIlII ,IIIIIlIIIIIIlIlll ,IIIlIllIIllllIlII ,IllIlIIllllllIllI ,IIIllIIllllIIIlIl ,IllIIIIIIIIlIIIlI ,IIIIIIllllllIllIl ,IIIlIIIlIIllIIIlI ,IIllIlllIlllIIlIl ,IlllIllIllIIIllII ,IIlIIlIIlIIIIlllI ,IllIIlIIlllIIllIl ,IIIllllllIIIllIlI ,IIIIllllIIIllllIl ,IlIIllIIIIlIIlIIl ,IIIIIlIlllIlllIIl ,IIIlIlIIIIIIlIlll ,IIlIIlIIIIIllllll ],IIllIlIllIlIllIlI ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IlIIlIIllllIIllII ,encoding ='utf8')as IlllIlllIllIIlIlI :IllIllIlIIIIIlIlI =IlllIlllIllIIlIlI .read ()
				else :
					with open ('docs/faq_en.md',_IlIIlIIllllIIllII ,encoding ='utf8')as IlllIlllIllIIlIlI :IllIllIlIIIIIlIlI =IlllIlllIllIIlIlI .read ()
				gr .Markdown (value =IllIllIlIIIIIlIlI )
			except :gr .Markdown (traceback .format_exc ())
	if IIlIIIlIllIIIllll .iscolab :IllIIIlIIIllllIIl .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IIlIIIlIllIIIllll .listen_port ,share =_IlIlIlIlIIIIIIlIl )
	else :IllIIIlIIIllllIIl .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IIlIIIlIllIIIllll .noautoopen ,server_port =IIlIIIlIllIIIllll .listen_port ,quiet =_IlIlIlIlIIIIIIlIl )