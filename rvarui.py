_IllIIlIIIlIIIllll ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IlIllIlIIIlllIIII ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IlIllIIlllIlIIlll ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIlllllllIllllIII ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IllIlIllIIllIlIlI ='后处理重采样至最终采样率，0为不进行重采样'
_IIIlIlIIIlIllllII ='自动检测index路径,下拉式选择(dropdown)'
_IIIIlllIlIIlIllll ='特征检索库文件路径,为空则使用下拉的选择结果'
_IIIlllIlIIIllllll ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IlIlIlIIlllllIlII ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IIlllIlIIllIIIllI ='变调(整数, 半音数量, 升八度12降八度-12)'
_IlIIIlllIllIlIIll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIIIIIIIIlllIlIl ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IIIIIIlllllllllll ='IVF%s,Flat'
_IllIIIIIIIIIllIII ='%s/total_fea.npy'
_IIIIllIllIlllIIlI ='Trying doing kmeans %s shape to 10k centers.'
_IIlIIIlIIllIIIlIl ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IIIllIllllllllIII =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIlIllIlllIllIIlI =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIIIllIlIllIIllIl ='write filelist done'
_IllIlIIlIlIIllIIl ='%s/filelist.txt'
_IlIllIIIllIIIlIlI ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IIIIlIlIIlIlllllI ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IlIllIlIlIIIllIlI ='%s/%s.wav|%s/%s.npy|%s'
_IllllIIIlIlIIIIlI ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IlllIIIIIllllIlIl ='%s/2b-f0nsf'
_IlllllIlIIIIIlIlI ='%s/0_gt_wavs'
_IIIIllllIIlIIllll ='emb_g.weight'
_IllIlIIIIlIlIllIl ='clean_empty_cache'
_IIIIllIlIIlllllll ='sample_rate'
_IlllIIllIIIIlIlIl ='%s->%s'
_IIllIIIIllllIlllI ='.index'
_IIIIlIlIIlIIlIlIl ='weights'
_IIIlIIIIIlIllIlll ='opt'
_IIIllIlIIIIIlIIll ='rmvpe'
_IIIlIIIlllllIlIII ='harvest'
_IllIllIIIIIIlIIII ='%s/3_feature768'
_IIlIlIlllIllIIlII ='%s/3_feature256'
_IllIIllIlIlllIIIl ='_v2'
_IllllIIIlllIIlIII ='48k'
_IlIIIIlIIIlllIlII ='32k'
_IlIlIIlllIIllIlIl ='cpu'
_IIlIlIlIlIIllIIII ='wav'
_IIllIlIIlllIlIlIl ='trained'
_IIIIIIIIIlIIlIIII ='logs'
_IlIllllllIIIlllIl ='-pd %s'
_IIlIIIlIIIllIIlIl ='-pg %s'
_IIIlllllIIIlIIlll ='choices'
_IIIlIllIIlIIIlIll ='weight'
_IlIlIlIIlIIlIllll ='pm'
_IIllIIIllIlIlllIl ='rmvpe_gpu'
_IIllllIlIIllllllI ='%s/logs/%s'
_IlIlIIlllIlIIllll ='flac'
_IllIIlllIllIlIIIl ='f0'
_IIlIlIIIlIllIIIlI ='%s/%s'
_IlllIIlIIllIIIlIl ='.pth'
_IIIIllllIIIIIlIII ='输出信息'
_IlIIlllIIlllIlllI ='not exist, will not use pretrained model'
_IllIlllIllIlIIIII ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IlIlIllIIllIIIlII ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IIIllIIlIIIllIIII ='40k'
_IIIIlIIIIllIlllll ='value'
_IlllIIIIllIIIlllI ='v2'
_IIIIIlIIlIIllIlll ='version'
_IlIllllIIIIIlllll ='visible'
_IIIIlllIlIllIlIlI ='primary'
_IIlIIIllIllIIIllI =None 
_IIIlIlIIIIIIlIlII ='\\\\'
_IlIllIIIIIllllllI ='\\'
_IIIlllIIIlIIIllll ='"'
_IIIIIIlIllIIIllIl =' '
_IlIIIIIIlIlllIIlI ='config'
_IllIIlIlIIIIllIIl ='.'
_IIIlIIIIIIlIIIIll ='r'
_IlIIlllIIlllIIIll ='是'
_IIlIlIIlllIllIIlI ='update'
_IIIIllllIlIIIlIlI ='__type__'
_IlIlIllIIIlIIIllI ='v1'
_IlllIIIlllllIIIll ='\n'
_IlIlllIlIlllIlllI =False 
_IIIIIlIIllIlllIIl =True 
import os ,shutil ,sys 
IllllIIlIllIIIlII =os .getcwd ()
sys .path .append (IllllIIlIllIIIlII )
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
IllllIIlIllIIIlII =os .getcwd ()
IIIllIllllIlIllII =os .path .join (IllllIIlIllIIIlII ,'TEMP')
shutil .rmtree (IIIllIllllIlIllII ,ignore_errors =_IIIIIlIIllIlllIIl )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IllllIIlIllIIIlII ,ignore_errors =_IIIIIlIIllIlllIIl )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IllllIIlIllIIIlII ,ignore_errors =_IIIIIlIIllIlllIIl )
os .makedirs (IIIllIllllIlIllII ,exist_ok =_IIIIIlIIllIlllIIl )
os .makedirs (os .path .join (IllllIIlIllIIIlII ,_IIIIIIIIIlIIlIIII ),exist_ok =_IIIIIlIIllIlllIIl )
os .makedirs (os .path .join (IllllIIlIllIIIlII ,_IIIIlIlIIlIIlIlIl ),exist_ok =_IIIIIlIIllIlllIIl )
os .environ ['TEMP']=IIIllIllllIlIllII 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IIllIIIlIllllIllI =Config ()
IlIllIIIlIIIIlIIl =I18nAuto ()
IlIllIIIlIIIIlIIl .print ()
IIIIlIlIIIIlllIIl =torch .cuda .device_count ()
IlIIllIllIIIllIIl =[]
IIlllIIIIIIIlIIIl =[]
IIIIllllIIlllllII =_IlIlllIlIlllIlllI 
if torch .cuda .is_available ()or IIIIlIlIIIIlllIIl !=0 :
	for IllIlIlllIlIllIII in range (IIIIlIlIIIIlllIIl ):
		IllIIIIIIlIllIllI =torch .cuda .get_device_name (IllIlIlllIlIllIII )
		if any (IIlIlIIIlIIlllIlI in IllIIIIIIlIllIllI .upper ()for IIlIlIIIlIIlllIlI in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IIIIllllIIlllllII =_IIIIIlIIllIlllIIl ;IlIIllIllIIIllIIl .append ('%s\t%s'%(IllIlIlllIlIllIII ,IllIIIIIIlIllIllI ));IIlllIIIIIIIlIIIl .append (int (torch .cuda .get_device_properties (IllIlIlllIlIllIII ).total_memory /1024 /1024 /1024 +.4 ))
if IIIIllllIIlllllII and len (IlIIllIllIIIllIIl )>0 :IlIIlIlllIlIIIIlI =_IlllIIIlllllIIIll .join (IlIIllIllIIIllIIl );IIIlIllIIIIllllII =min (IIlllIIIIIIIlIIIl )//2 
else :IlIIlIlllIlIIIIlI =IlIllIIIlIIIIlIIl ('很遗憾您这没有能用的显卡来支持您训练');IIIlIllIIIIllllII =1 
IIlIIIlIllIIlIIII ='-'.join ([IlIllIlIlllIlllll [0 ]for IlIllIlIlllIlllll in IlIIllIllIIIllIIl ])
class IlIIllllIlIIllIII (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IIIlIIIIlIlIlIlIl ,**IlllIIIIIIIlIlIIl ):super ().__init__ (variant ='tool',**IlllIIIIIIIlIlIIl )
	def get_block_name (IllIllIllIlIllIlI ):return 'button'
IIlIIllllIIllllII =_IIlIIIllIllIIIllI 
def IllIIlllIIllIllll ():
	global IIlIIllllIIllllII ;IIIlIlIIIllIlIIII ,_IlIIlIIIllIIIIIII ,_IlIIlIIIllIIIIIII =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IIlIIllllIIllllII =IIIlIlIIIllIlIIII [0 ];IIlIIllllIIllllII =IIlIIllllIIllllII .to (IIllIIIlIllllIllI .device )
	if IIllIIIlIllllIllI .is_half :IIlIIllllIIllllII =IIlIIllllIIllllII .half ()
	else :IIlIIllllIIllllII =IIlIIllllIIllllII .float ()
	IIlIIllllIIllllII .eval ()
IIIlIllIIlIlllllI =_IIIIlIlIIlIIlIlIl 
IIlIIlIllIIIIlIIl ='uvr5_weights'
IlIIllllIIllIIIlI =_IIIIIIIIIlIIlIIII 
IIlIlIIllllIIIlII =[]
for IlIlIIIllIllllIII in os .listdir (IIIlIllIIlIlllllI ):
	if IlIlIIIllIllllIII .endswith (_IlllIIlIIllIIIlIl ):IIlIlIIllllIIIlII .append (IlIlIIIllIllllIII )
IlIllIIIllIllllII =[]
for (IIlIIIIIlllIllIIl ,IIIIIllIIIllIlllI ,IllIIIIllIlIIIlII )in os .walk (IlIIllllIIllIIIlI ,topdown =_IlIlllIlIlllIlllI ):
	for IlIlIIIllIllllIII in IllIIIIllIlIIIlII :
		if IlIlIIIllIllllIII .endswith (_IIllIIIIllllIlllI )and _IIllIlIIlllIlIlIl not in IlIlIIIllIllllIII :IlIllIIIllIllllII .append (_IIlIlIIIlIllIIIlI %(IIlIIIIIlllIllIIl ,IlIlIIIllIllllIII ))
IIllIlllIIlllIIll =[]
for IlIlIIIllIllllIII in os .listdir (IIlIIlIllIIIIlIIl ):
	if IlIlIIIllIllllIII .endswith (_IlllIIlIIllIIIlIl )or 'onnx'in IlIlIIIllIllllIII :IIllIlllIIlllIIll .append (IlIlIIIllIllllIII .replace (_IlllIIlIIllIIIlIl ,''))
IllIIlIIIlllllIIl =_IIlIIIllIllIIIllI 
def IIllIIIlllIIlIlll (IIllIIIlIIlIlIIlI ,IIIIIlIllIIlIllIl ,IlIlIlIIIIIllIIII ,IlllIIllIIIlIlIlI ,IIlIlIllllllIlIII ,IlllllllllllIIlII ,IllIIIlIIIIllIIll ,IlIlIIIIIlIllIIll ,IlIIlIllllIlllllI ,IIIlllIllIIlIIIII ,IlIIlIllIllIlllIl ,IlIlIllIIlIIIllIl ):
	global IlllIIIIlIIlIIlll ,IlIIlIIlllllllIll ,IlIlIIllllIIIIIIl ,IIlIIllllIIllllII ,IlIllIIlIlllIllII ,IllIIlIIIlllllIIl 
	if IIIIIlIllIIlIllIl is _IIlIIIllIllIIIllI :return 'You need to upload an audio',_IIlIIIllIllIIIllI 
	IlIlIlIIIIIllIIII =int (IlIlIlIIIIIllIIII )
	try :
		IIlllIIIlllIlIIIl =load_audio (IIIIIlIllIIlIllIl ,16000 );IIlIllIIIlIlIllll =np .abs (IIlllIIIlllIlIIIl ).max ()/.95 
		if IIlIllIIIlIlIllll >1 :IIlllIIIlllIlIIIl /=IIlIllIIIlIlIllll 
		IIIllllIIlIlIlllI =[0 ,0 ,0 ]
		if not IIlIIllllIIllllII :IllIIlllIIllIllll ()
		IIlIIllIIllIIllll =IllIIlIIIlllllIIl .get (_IllIIlllIllIlIIIl ,1 );IlllllllllllIIlII =IlllllllllllIIlII .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl ).replace (_IIllIlIIlllIlIlIl ,'added')if IlllllllllllIIlII !=''else IllIIIlIIIIllIIll ;IllllIIllIIlIlIll =IlIlIIllllIIIIIIl .pipeline (IIlIIllllIIllllII ,IlIIlIIlllllllIll ,IIllIIIlIIlIlIIlI ,IIlllIIIlllIlIIIl ,IIIIIlIllIIlIllIl ,IIIllllIIlIlIlllI ,IlIlIlIIIIIllIIII ,IIlIlIllllllIlIII ,IlllllllllllIIlII ,IlIlIIIIIlIllIIll ,IIlIIllIIllIIllll ,IlIIlIllllIlllllI ,IlllIIIIlIIlIIlll ,IIIlllIllIIlIIIII ,IlIIlIllIllIlllIl ,IlIllIIlIlllIllII ,IlIlIllIIlIIIllIl ,f0_file =IlllIIllIIIlIlIlI )
		if IlllIIIIlIIlIIlll !=IIIlllIllIIlIIIII >=16000 :IlllIIIIlIIlIIlll =IIIlllIllIIlIIIII 
		IIlIIllllIIllIlII ='Using index:%s.'%IlllllllllllIIlII if os .path .exists (IlllllllllllIIlII )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IIlIIllllIIllIlII ,IIIllllIIlIlIlllI [0 ],IIIllllIIlIlIlllI [1 ],IIIllllIIlIlIlllI [2 ]),(IlllIIIIlIIlIIlll ,IllllIIllIIlIlIll )
	except :IlIlIlllIlIIlIIlI =traceback .format_exc ();print (IlIlIlllIlIIlIIlI );return IlIlIlllIlIIlIIlI ,(_IIlIIIllIllIIIllI ,_IIlIIIllIllIIIllI )
def IIlIllIlIlIIllIII (IIlllllIlllIIllll ,IIlIlIllIlIllllII ,IIIIlIllllIlIIlII ,IIIIIIIllIIlllIlI ,IlIIlIlIIIlIIllIl ,IllllIlllIlllIIlI ,IllIllIlIllIIlIII ,IllIIllIlIllllIII ,IlllIllIllllIlIII ,IlIlIIlIIlllIIlII ,IlIIlllIlIllIIlIl ,IIIIlIlllllllIlIl ,IIIIIllIIIIlllllI ,IIlIllIlIIIIIIIlI ):
	try :
		IIlIlIllIlIllllII =IIlIlIllIlIllllII .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl );IIIIlIllllIlIIlII =IIIIlIllllIlIIlII .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl );os .makedirs (IIIIlIllllIlIIlII ,exist_ok =_IIIIIlIIllIlllIIl )
		try :
			if IIlIlIllIlIllllII !='':IIIIIIIllIIlllIlI =[os .path .join (IIlIlIllIlIllllII ,IIIllllllIIIIlIll )for IIIllllllIIIIlIll in os .listdir (IIlIlIllIlIllllII )]
			else :IIIIIIIllIIlllIlI =[IlIIllIIlIIIlIlll .name for IlIIllIIlIIIlIlll in IIIIIIIllIIlllIlI ]
		except :traceback .print_exc ();IIIIIIIllIIlllIlI =[IlllIllIllllllIlI .name for IlllIllIllllllIlI in IIIIIIIllIIlllIlI ]
		IlllIIIllIlIlIIll =[]
		for IlIIIIIIIllIIIIll in IIIIIIIllIIlllIlI :
			IIIlllIIlIlIIlIlI ,IIIIIIIllIIlIlIlI =IIllIIIlllIIlIlll (IIlllllIlllIIllll ,IlIIIIIIIllIIIIll ,IlIIlIlIIIlIIllIl ,_IIlIIIllIllIIIllI ,IllllIlllIlllIIlI ,IllIllIlIllIIlIII ,IllIIllIlIllllIII ,IlllIllIllllIlIII ,IlIlIIlIIlllIIlII ,IlIIlllIlIllIIlIl ,IIIIlIlllllllIlIl ,IIIIIllIIIIlllllI )
			if 'Success'in IIIlllIIlIlIIlIlI :
				try :
					IlIlIIlllIllllIII ,IIIlIIIllllIllllI =IIIIIIIllIIlIlIlI 
					if IIlIllIlIIIIIIIlI in [_IIlIlIlIlIIllIIII ,_IlIlIIlllIlIIllll ]:sf .write ('%s/%s.%s'%(IIIIlIllllIlIIlII ,os .path .basename (IlIIIIIIIllIIIIll ),IIlIllIlIIIIIIIlI ),IIIlIIIllllIllllI ,IlIlIIlllIllllIII )
					else :
						IlIIIIIIIllIIIIll ='%s/%s.wav'%(IIIIlIllllIlIIlII ,os .path .basename (IlIIIIIIIllIIIIll ));sf .write (IlIIIIIIIllIIIIll ,IIIlIIIllllIllllI ,IlIlIIlllIllllIII )
						if os .path .exists (IlIIIIIIIllIIIIll ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IlIIIIIIIllIIIIll ,IlIIIIIIIllIIIIll [:-4 ]+'.%s'%IIlIllIlIIIIIIIlI ))
				except :IIIlllIIlIlIIlIlI +=traceback .format_exc ()
			IlllIIIllIlIlIIll .append (_IlllIIllIIIIlIlIl %(os .path .basename (IlIIIIIIIllIIIIll ),IIIlllIIlIlIIlIlI ));yield _IlllIIIlllllIIIll .join (IlllIIIllIlIlIIll )
		yield _IlllIIIlllllIIIll .join (IlllIIIllIlIlIIll )
	except :yield traceback .format_exc ()
def IIIIIIlIlllIlIIll (IIllllllIIIIlIlII ,IIlIIIIIIllIIllIl ,IllIllIIIlIIIIIII ,IIlllllIllIllllll ,IlIlllIllIlllIIlI ,IlIlIlIIIIIIlllll ,IIIllIIllIllIllll ):
	IlllllllIlIIIlllI ='streams';IIllllIllIllIllll ='onnx_dereverb_By_FoxJoy';IllIllIllIlllIIII =[]
	try :
		IIlIIIIIIllIIllIl =IIlIIIIIIllIIllIl .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl );IllIllIIIlIIIIIII =IllIllIIIlIIIIIII .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl );IlIlllIllIlllIIlI =IlIlllIllIlllIIlI .strip (_IIIIIIlIllIIIllIl ).strip (_IIIlllIIIlIIIllll ).strip (_IlllIIIlllllIIIll ).strip (_IIIlllIIIlIIIllll ).strip (_IIIIIIlIllIIIllIl )
		if IIllllllIIIIlIlII ==IIllllIllIllIllll :from MDXNet import MDXNetDereverb ;IlIIllllIIIllllIl =MDXNetDereverb (15 )
		else :IlIlIllllIIIIlIlI =_audio_pre_ if 'DeEcho'not in IIllllllIIIIlIlII else _audio_pre_new ;IlIIllllIIIllllIl =IlIlIllllIIIIlIlI (agg =int (IlIlIlIIIIIIlllll ),model_path =os .path .join (IIlIIlIllIIIIlIIl ,IIllllllIIIIlIlII +_IlllIIlIIllIIIlIl ),device =IIllIIIlIllllIllI .device ,is_half =IIllIIIlIllllIllI .is_half )
		if IIlIIIIIIllIIllIl !='':IIlllllIllIllllll =[os .path .join (IIlIIIIIIllIIllIl ,IIllIlIlIlllIIIIl )for IIllIlIlIlllIIIIl in os .listdir (IIlIIIIIIllIIllIl )]
		else :IIlllllIllIllllll =[IIIlIIlIlllIIIIII .name for IIIlIIlIlllIIIIII in IIlllllIllIllllll ]
		for IIIllIlIlIIlllIIl in IIlllllIllIllllll :
			IllIllllllIlllIII =os .path .join (IIlIIIIIIllIIllIl ,IIIllIlIlIIlllIIl );IIIllIllllllIlIll =1 ;IlIlllIIIIIIllIll =0 
			try :
				IIlllIlIllIIlIllI =ffmpeg .probe (IllIllllllIlllIII ,cmd ='ffprobe')
				if IIlllIlIllIIlIllI [IlllllllIlIIIlllI ][0 ]['channels']==2 and IIlllIlIllIIlIllI [IlllllllIlIIIlllI ][0 ][_IIIIllIlIIlllllll ]=='44100':IIIllIllllllIlIll =0 ;IlIIllllIIIllllIl ._path_audio_ (IllIllllllIlllIII ,IlIlllIllIlllIIlI ,IllIllIIIlIIIIIII ,IIIllIIllIllIllll );IlIlllIIIIIIllIll =1 
			except :IIIllIllllllIlIll =1 ;traceback .print_exc ()
			if IIIllIllllllIlIll ==1 :IIlllIllllIIIIIIl ='%s/%s.reformatted.wav'%(IIIllIllllIlIllII ,os .path .basename (IllIllllllIlllIII ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IllIllllllIlllIII ,IIlllIllllIIIIIIl ));IllIllllllIlllIII =IIlllIllllIIIIIIl 
			try :
				if IlIlllIIIIIIllIll ==0 :IlIIllllIIIllllIl ._path_audio_ (IllIllllllIlllIII ,IlIlllIllIlllIIlI ,IllIllIIIlIIIIIII ,IIIllIIllIllIllll )
				IllIllIllIlllIIII .append ('%s->Success'%os .path .basename (IllIllllllIlllIII ));yield _IlllIIIlllllIIIll .join (IllIllIllIlllIIII )
			except :IllIllIllIlllIIII .append (_IlllIIllIIIIlIlIl %(os .path .basename (IllIllllllIlllIII ),traceback .format_exc ()));yield _IlllIIIlllllIIIll .join (IllIllIllIlllIIII )
	except :IllIllIllIlllIIII .append (traceback .format_exc ());yield _IlllIIIlllllIIIll .join (IllIllIllIlllIIII )
	finally :
		try :
			if IIllllllIIIIlIlII ==IIllllIllIllIllll :del IlIIllllIIIllllIl .pred .model ;del IlIIllllIIIllllIl .pred .model_ 
			else :del IlIIllllIIIllllIl .model ;del IlIIllllIIIllllIl 
		except :traceback .print_exc ()
		print (_IllIlIIIIlIlIllIl )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IlllIIIlllllIIIll .join (IllIllIllIlllIIII )
def IIllIlllIIlIIIllI (IIlIIIIIIIIllIllI ):
	IIIlIlIlIlIIllllI ='';IIIllIIIllIlIIlII =os .path .join (_IIIIIIIIIlIIlIIII ,IIlIIIIIIIIllIllI .split (_IllIIlIlIIIIllIIl )[0 ],'')
	for IlIIIIlIlIlIllIII in IlIllIIIllIllllII :
		if IIIllIIIllIlIIlII in IlIIIIlIlIlIllIII :IIIlIlIlIlIIllllI =IlIIIIlIlIlIllIII ;break 
	return IIIlIlIlIlIIllllI 
def IlIllIIllIIIIIIlI (IIIllIlIlIllIllII ,IIIllIIIllllllllI ,IllllIlllIIllIlll ):
	global IIIlIlIIIlIIIlllI ,IlllIIIIlIIlIIlll ,IlIIlIIlllllllIll ,IlIlIIllllIIIIIIl ,IllIIlIIIlllllIIl ,IlIllIIlIlllIllII 
	if IIIllIlIlIllIllII ==''or IIIllIlIlIllIllII ==[]:
		global IIlIIllllIIllllII 
		if IIlIIllllIIllllII is not _IIlIIIllIllIIIllI :
			print (_IllIlIIIIlIlIllIl );del IlIIlIIlllllllIll ,IIIlIlIIIlIIIlllI ,IlIlIIllllIIIIIIl ,IIlIIllllIIllllII ,IlllIIIIlIIlIIlll ;IIlIIllllIIllllII =IlIIlIIlllllllIll =IIIlIlIIIlIIIlllI =IlIlIIllllIIIIIIl =IIlIIllllIIllllII =IlllIIIIlIIlIIlll =_IIlIIIllIllIIIllI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IIllIIIllIllIIlII =IllIIlIIIlllllIIl .get (_IllIIlllIllIlIIIl ,1 );IlIllIIlIlllIllII =IllIIlIIIlllllIIl .get (_IIIIIlIIlIIllIlll ,_IlIlIllIIIlIIIllI )
			if IlIllIIlIlllIllII ==_IlIlIllIIIlIIIllI :
				if IIllIIIllIllIIlII ==1 :IlIIlIIlllllllIll =SynthesizerTrnMs256NSFsid (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ],is_half =IIllIIIlIllllIllI .is_half )
				else :IlIIlIIlllllllIll =SynthesizerTrnMs256NSFsid_nono (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ])
			elif IlIllIIlIlllIllII ==_IlllIIIIllIIIlllI :
				if IIllIIIllIllIIlII ==1 :IlIIlIIlllllllIll =SynthesizerTrnMs768NSFsid (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ],is_half =IIllIIIlIllllIllI .is_half )
				else :IlIIlIIlllllllIll =SynthesizerTrnMs768NSFsid_nono (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ])
			del IlIIlIIlllllllIll ,IllIIlIIIlllllIIl 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IlIllllIIIIIlllll :_IlIlllIlIlllIlllI ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
	IlIllIlIIllIIlIIl =_IIlIlIIIlIllIIIlI %(IIIlIllIIlIlllllI ,IIIllIlIlIllIllII );print ('loading %s'%IlIllIlIIllIIlIIl );IllIIlIIIlllllIIl =torch .load (IlIllIlIIllIIlIIl ,map_location =_IlIlIIlllIIllIlIl );IlllIIIIlIIlIIlll =IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ][-1 ];IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ][-3 ]=IllIIlIIIlllllIIl [_IIIlIllIIlIIIlIll ][_IIIIllllIIlIIllll ].shape [0 ];IIllIIIllIllIIlII =IllIIlIIIlllllIIl .get (_IllIIlllIllIlIIIl ,1 )
	if IIllIIIllIllIIlII ==0 :IIIllIIIllllllllI =IllllIlllIIllIlll ={_IlIllllIIIIIlllll :_IlIlllIlIlllIlllI ,_IIIIlIIIIllIlllll :.5 ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
	else :IIIllIIIllllllllI ={_IlIllllIIIIIlllll :_IIIIIlIIllIlllIIl ,_IIIIlIIIIllIlllll :IIIllIIIllllllllI ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI };IllllIlllIIllIlll ={_IlIllllIIIIIlllll :_IIIIIlIIllIlllIIl ,_IIIIlIIIIllIlllll :IllllIlllIIllIlll ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
	IlIllIIlIlllIllII =IllIIlIIIlllllIIl .get (_IIIIIlIIlIIllIlll ,_IlIlIllIIIlIIIllI )
	if IlIllIIlIlllIllII ==_IlIlIllIIIlIIIllI :
		if IIllIIIllIllIIlII ==1 :IlIIlIIlllllllIll =SynthesizerTrnMs256NSFsid (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ],is_half =IIllIIIlIllllIllI .is_half )
		else :IlIIlIIlllllllIll =SynthesizerTrnMs256NSFsid_nono (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ])
	elif IlIllIIlIlllIllII ==_IlllIIIIllIIIlllI :
		if IIllIIIllIllIIlII ==1 :IlIIlIIlllllllIll =SynthesizerTrnMs768NSFsid (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ],is_half =IIllIIIlIllllIllI .is_half )
		else :IlIIlIIlllllllIll =SynthesizerTrnMs768NSFsid_nono (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ])
	del IlIIlIIlllllllIll .enc_q ;print (IlIIlIIlllllllIll .load_state_dict (IllIIlIIIlllllIIl [_IIIlIllIIlIIIlIll ],strict =_IlIlllIlIlllIlllI ));IlIIlIIlllllllIll .eval ().to (IIllIIIlIllllIllI .device )
	if IIllIIIlIllllIllI .is_half :IlIIlIIlllllllIll =IlIIlIIlllllllIll .half ()
	else :IlIIlIIlllllllIll =IlIIlIIlllllllIll .float ()
	IlIlIIllllIIIIIIl =VC (IlllIIIIlIIlIIlll ,IIllIIIlIllllIllI );IIIlIlIIIlIIIlllI =IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ][-3 ];return {_IlIllllIIIIIlllll :_IIIIIlIIllIlllIIl ,'maximum':IIIlIlIIIlIIIlllI ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },IIIllIIIllllllllI ,IllllIlllIIllIlll ,IIllIlllIIlIIIllI (IIIllIlIlIllIllII )
def IllIllIIIIllIIlIl ():
	IIllIllIIllIllIlI =[]
	for IIIlllllIIllIIlIl in os .listdir (IIIlIllIIlIlllllI ):
		if IIIlllllIIllIIlIl .endswith (_IlllIIlIIllIIIlIl ):IIllIllIIllIllIlI .append (IIIlllllIIllIIlIl )
	IIIlIlIlIllIIlIII =[]
	for (IllIllIIIIIIIIIlI ,IlIlIIIlIIlIIIlIl ,IIIIlIIllIIlIIllI )in os .walk (IlIIllllIIllIIIlI ,topdown =_IlIlllIlIlllIlllI ):
		for IIIlllllIIllIIlIl in IIIIlIIllIIlIIllI :
			if IIIlllllIIllIIlIl .endswith (_IIllIIIIllllIlllI )and _IIllIlIIlllIlIlIl not in IIIlllllIIllIIlIl :IIIlIlIlIllIIlIII .append (_IIlIlIIIlIllIIIlI %(IllIllIIIIIIIIIlI ,IIIlllllIIllIIlIl ))
	return {_IIIlllllIIIlIIlll :sorted (IIllIllIIllIllIlI ),_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },{_IIIlllllIIIlIIlll :sorted (IIIlIlIlIllIIlIII ),_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
def IIIlllllIIIIlIIIl ():return {_IIIIlIIIIllIlllll :'',_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
IIllIIlIlllIllIII ={_IlIIIIlIIIlllIlII :32000 ,_IIIllIIlIIIllIIII :40000 ,_IllllIIIlllIIlIII :48000 }
def IIlIllIIlIIlIllII (IllllIlIllIlIIlll ,IIIllIllIIlIlIlII ):
	while 1 :
		if IIIllIllIIlIlIlII .poll ()is _IIlIIIllIllIIIllI :sleep (.5 )
		else :break 
	IllllIlIllIlIIlll [0 ]=_IIIIIlIIllIlllIIl 
def IIlllIlIllIlIIIIl (IllIIIIIIlIIIllII ,IIIllIlIIIIlIIllI ):
	while 1 :
		IlIIIIllllIllIIlI =1 
		for IlIIlIIIIIlIIlIlI in IIIllIlIIIIlIIllI :
			if IlIIlIIIIIlIIlIlI .poll ()is _IIlIIIllIllIIIllI :IlIIIIllllIllIIlI =0 ;sleep (.5 );break 
		if IlIIIIllllIllIIlI ==1 :break 
	IllIIIIIIlIIIllII [0 ]=_IIIIIlIIllIlllIIl 
def IIIIlllIIlllIllIl (IlIIllllIlIlllIll ,IllIlIlllIIIlllIl ,IIlIlIIlIllIlIlll ,IIIIIIIIIllIIIIII ):
	IlIllIlIllIlIIIlI ='%s/logs/%s/preprocess.log';IIlIlIIlIllIlIlll =IIllIIlIlllIllIII [IIlIlIIlIllIlIlll ];os .makedirs (_IIllllIlIIllllllI %(IllllIIlIllIIIlII ,IllIlIlllIIIlllIl ),exist_ok =_IIIIIlIIllIlllIIl );IIIlllIlllIIIIIlI =open (IlIllIlIllIlIIIlI %(IllllIIlIllIIIlII ,IllIlIlllIIIlllIl ),'w');IIIlllIlllIIIIIlI .close ();IlIllllIlIllIIllI =IIllIIIlIllllIllI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IlIIllllIlIlllIll ,IIlIlIIlIllIlIlll ,IIIIIIIIIllIIIIII ,IllllIIlIllIIIlII ,IllIlIlllIIIlllIl )+str (IIllIIIlIllllIllI .noparallel );print (IlIllllIlIllIIllI );IlIllllIlIllIIIlI =Popen (IlIllllIlIllIIllI ,shell =_IIIIIlIIllIlllIIl );IIIIIIlllIlIlIlII =[_IlIlllIlIlllIlllI ];threading .Thread (target =IIlIllIIlIIlIllII ,args =(IIIIIIlllIlIlIlII ,IlIllllIlIllIIIlI )).start ()
	while 1 :
		with open (IlIllIlIllIlIIIlI %(IllllIIlIllIIIlII ,IllIlIlllIIIlllIl ),_IIIlIIIIIIlIIIIll )as IIIlllIlllIIIIIlI :yield IIIlllIlllIIIIIlI .read ()
		sleep (1 )
		if IIIIIIlllIlIlIlII [0 ]:break 
	with open (IlIllIlIllIlIIIlI %(IllllIIlIllIIIlII ,IllIlIlllIIIlllIl ),_IIIlIIIIIIlIIIIll )as IIIlllIlllIIIIIlI :IIIlIlllIlIIIIlII =IIIlllIlllIIIIIlI .read ()
	print (IIIlIlllIlIIIIlII );yield IIIlIlllIlIIIIlII 
def IlIIIIlIIlIlIIIll (IllIlIIlIlllIIIll ,IlIlllllIIIIllIll ,IllllIlllllllIIlI ,IlIIIllIIIlllIIlI ,IlllIIIlIIIIllIIl ,IllIlllIIIllllIll ,IIIIIllIlIIlIIllI ):
	IIIlIlIIllllllIII ='%s/logs/%s/extract_fl_feature.log';IllIlIIlIlllIIIll =IllIlIIlIlllIIIll .split ('-');os .makedirs (_IIllllIlIIllllllI %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),exist_ok =_IIIIIlIIllIlllIIl );IllIIIIIIllIlllIl =open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),'w');IllIIIIIIllIlllIl .close ()
	if IlIIIllIIIlllIIlI :
		if IllllIlllllllIIlI !=_IIllIIIllIlIlllIl :
			IllllIIIllIIlllII =IIllIIIlIllllIllI .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ,IlIlllllIIIIllIll ,IllllIlllllllIIlI );print (IllllIIIllIIlllII );IlllIlllIIlIIlIIl =Popen (IllllIIIllIIlllII ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIlIllllllIIIIlII =[_IlIlllIlIlllIlllI ];threading .Thread (target =IIlIllIIlIIlIllII ,args =(IIlIllllllIIIIlII ,IlllIlllIIlIIlIIl )).start ()
			while 1 :
				with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :yield IllIIIIIIllIlllIl .read ()
				sleep (1 )
				if IIlIllllllIIIIlII [0 ]:break 
			with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :IIlIlllIIIlIIlIll =IllIIIIIIllIlllIl .read ()
			print (IIlIlllIIIlIIlIll );yield IIlIlllIIIlIIlIll 
		else :
			IIIIIllIlIIlIIllI =IIIIIllIlIIlIIllI .split ('-');IIllIlllIIIlllIll =len (IIIIIllIlIIlIIllI );IIIIlIllIllIlIllI =[]
			for (IlIllIIlIIlllllIl ,IIllIllIIlIIIIIIl )in enumerate (IIIIIllIlIIlIIllI ):IllllIIIllIIlllII =IIllIIIlIllllIllI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IIllIlllIIIlllIll ,IlIllIIlIIlllllIl ,IIllIllIIlIIIIIIl ,IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ,IIllIIIlIllllIllI .is_half );print (IllllIIIllIIlllII );IlllIlllIIlIIlIIl =Popen (IllllIIIllIIlllII ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIIIlIllIllIlIllI .append (IlllIlllIIlIIlIIl )
			IIlIllllllIIIIlII =[_IlIlllIlIlllIlllI ];threading .Thread (target =IIlllIlIllIlIIIIl ,args =(IIlIllllllIIIIlII ,IIIIlIllIllIlIllI )).start ()
			while 1 :
				with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :yield IllIIIIIIllIlllIl .read ()
				sleep (1 )
				if IIlIllllllIIIIlII [0 ]:break 
			with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :IIlIlllIIIlIIlIll =IllIIIIIIllIlllIl .read ()
			print (IIlIlllIIIlIIlIll );yield IIlIlllIIIlIIlIll 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IIllIlllIIIlllIll =len (IllIlIIlIlllIIIll );IIIIlIllIllIlIllI =[]
	for (IlIllIIlIIlllllIl ,IIllIllIIlIIIIIIl )in enumerate (IllIlIIlIlllIIIll ):IllllIIIllIIlllII =IIllIIIlIllllIllI .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IIllIIIlIllllIllI .device ,IIllIlllIIIlllIll ,IlIllIIlIIlllllIl ,IIllIllIIlIIIIIIl ,IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ,IllIlllIIIllllIll );print (IllllIIIllIIlllII );IlllIlllIIlIIlIIl =Popen (IllllIIIllIIlllII ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIIIlIllIllIlIllI .append (IlllIlllIIlIIlIIl )
	IIlIllllllIIIIlII =[_IlIlllIlIlllIlllI ];threading .Thread (target =IIlllIlIllIlIIIIl ,args =(IIlIllllllIIIIlII ,IIIIlIllIllIlIllI )).start ()
	while 1 :
		with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :yield IllIIIIIIllIlllIl .read ()
		sleep (1 )
		if IIlIllllllIIIIlII [0 ]:break 
	with open (IIIlIlIIllllllIII %(IllllIIlIllIIIlII ,IlllIIIlIIIIllIIl ),_IIIlIIIIIIlIIIIll )as IllIIIIIIllIlllIl :IIlIlllIIIlIIlIll =IllIIIIIIllIlllIl .read ()
	print (IIlIlllIIIlIIlIll );yield IIlIlllIIIlIIlIll 
def IlllIIIIIIlIlIIlI (IIIIlllIIIlIllllI ,IlIIlIlIlIIIIllIl ,IIlIIllIlIIlIllll ):
	IIIlIIIlIlllIIllI =''if IIlIIllIlIIlIllll ==_IlIlIllIIIlIIIllI else _IllIIllIlIlllIIIl ;IIllIlIIlllIlIlIl =_IllIIlllIllIlIIIl if IlIIlIlIlIIIIllIl else '';IIlIlllIIlllIIlll =os .access (_IlIlIllIIllIIIlII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI ),os .F_OK );IIIIllllllllIllll =os .access (_IllIlllIllIlIIIII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI ),os .F_OK )
	if not IIlIlllIIlllIIlll :print (_IlIlIllIIllIIIlII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI ),_IlIIlllIIlllIlllI )
	if not IIIIllllllllIllll :print (_IllIlllIllIlIIIII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI ),_IlIIlllIIlllIlllI )
	return _IlIlIllIIllIIIlII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI )if IIlIlllIIlllIIlll else '',_IllIlllIllIlIIIII %(IIIlIIIlIlllIIllI ,IIllIlIIlllIlIlIl ,IIIIlllIIIlIllllI )if IIIIllllllllIllll else ''
def IlIIIlIllllllIIll (IIIIIllIIIIlIIIIl ,IlIlIIIllIIlIlIII ,IIlIIlIlllllIIIII ):
	IllIlIIlllIIlllIl =''if IIlIIlIlllllIIIII ==_IlIlIllIIIlIIIllI else _IllIIllIlIlllIIIl 
	if IIIIIllIIIIlIIIIl ==_IlIIIIlIIIlllIlII and IIlIIlIlllllIIIII ==_IlIlIllIIIlIIIllI :IIIIIllIIIIlIIIIl =_IIIllIIlIIIllIIII 
	IIIlIllIIIIIIlllI ={_IIIlllllIIIlIIlll :[_IIIllIIlIIIllIIII ,_IllllIIIlllIIlIII ],_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI ,_IIIIlIIIIllIlllll :IIIIIllIIIIlIIIIl }if IIlIIlIlllllIIIII ==_IlIlIllIIIlIIIllI else {_IIIlllllIIIlIIlll :[_IIIllIIlIIIllIIII ,_IllllIIIlllIIlIII ,_IlIIIIlIIIlllIlII ],_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI ,_IIIIlIIIIllIlllll :IIIIIllIIIIlIIIIl };IllIllIIIlIIlIIIl =_IllIIlllIllIlIIIl if IlIlIIIllIIlIlIII else '';IIlllIlllllIllIII =os .access (_IlIlIllIIllIIIlII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl ),os .F_OK );IlllIlIlIllllIIII =os .access (_IllIlllIllIlIIIII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl ),os .F_OK )
	if not IIlllIlllllIllIII :print (_IlIlIllIIllIIIlII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl ),_IlIIlllIIlllIlllI )
	if not IlllIlIlIllllIIII :print (_IllIlllIllIlIIIII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl ),_IlIIlllIIlllIlllI )
	return _IlIlIllIIllIIIlII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl )if IIlllIlllllIllIII else '',_IllIlllIllIlIIIII %(IllIlIIlllIIlllIl ,IllIllIIIlIIlIIIl ,IIIIIllIIIIlIIIIl )if IlllIlIlIllllIIII else '',IIIlIllIIIIIIlllI 
def IllIIIlIllIIIlIIl (IIIIlllIllllIIIlI ,IIIlIlIlIIIllIIII ,IllIIlllllIlllIII ):
	IIlIlIlIlIIllIlII ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IlIlIIIlIIlIlllIl ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IllllIlIIllllIllI =''if IllIIlllllIlllIII ==_IlIlIllIIIlIIIllI else _IllIIllIlIlllIIIl ;IIIlIIllIIIIIIIIl =os .access (IlIlIIIlIIlIlllIl %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII ),os .F_OK );IllIIIllIlIIlIIII =os .access (IIlIlIlIlIIllIlII %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII ),os .F_OK )
	if not IIIlIIllIIIIIIIIl :print (IlIlIIIlIIlIlllIl %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII ),_IlIIlllIIlllIlllI )
	if not IllIIIllIlIIlIIII :print (IIlIlIlIlIIllIlII %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII ),_IlIIlllIIlllIlllI )
	if IIIIlllIllllIIIlI :return {_IlIllllIIIIIlllll :_IIIIIlIIllIlllIIl ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },IlIlIIIlIIlIlllIl %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII )if IIIlIIllIIIIIIIIl else '',IIlIlIlIlIIllIlII %(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII )if IllIIIllIlIIlIIII else ''
	return {_IlIllllIIIIIlllll :_IlIlllIlIlllIlllI ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII )if IIIlIIllIIIIIIIIl else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IllllIlIIllllIllI ,IIIlIlIlIIIllIIII )if IllIIIllIlIIlIIII else ''
def IIllIllIlIIlIlIII (IllIlIllIIlIIlIIl ,IIlIlIIllIIIllIll ,IlIllIlllllllIlII ,IIlIIlIIIlIIlIIll ,IllIIlIlIlIlIlllI ,IIIIIlllllIIIIIlI ,IIlIlIIllIIlIllll ,IIIIllllIlIlIlIll ,IIlIlIllIIIIIIlIl ,IIllllIllIlIIIIll ,IlllIIllIIIIllIII ,IllIIllIllIllIllI ,IlIIllIIIIIIIIlII ,IlIIllIlIllIIIIIl ):
	IlllllllllIIIlIlI ='\x08';IlllIIllIIllIIlll =_IIllllIlIIllllllI %(IllllIIlIllIIIlII ,IllIlIllIIlIIlIIl );os .makedirs (IlllIIllIIllIIlll ,exist_ok =_IIIIIlIIllIlllIIl );IIIlIIIIlIIlIlIlI =_IlllllIlIIIIIlIlI %IlllIIllIIllIIlll ;IIlIIIIIllIlIIlll =_IIlIlIlllIllIIlII %IlllIIllIIllIIlll if IlIIllIlIllIIIIIl ==_IlIlIllIIIlIIIllI else _IllIllIIIIIIlIIII %IlllIIllIIllIIlll 
	if IlIllIlllllllIlII :IIIIIllIlIlllIIII ='%s/2a_f0'%IlllIIllIIllIIlll ;IIllllIIIIlIlIlll =_IlllIIIIIllllIlIl %IlllIIllIIllIIlll ;IllIlllIlIIIIIlll =set ([IlIllllIIlIIllIII .split (_IllIIlIlIIIIllIIl )[0 ]for IlIllllIIlIIllIII in os .listdir (IIIlIIIIlIIlIlIlI )])&set ([IIlllIIIIIIIllllI .split (_IllIIlIlIIIIllIIl )[0 ]for IIlllIIIIIIIllllI in os .listdir (IIlIIIIIllIlIIlll )])&set ([IIIIllIllIllIllIl .split (_IllIIlIlIIIIllIIl )[0 ]for IIIIllIllIllIllIl in os .listdir (IIIIIllIlIlllIIII )])&set ([IIlIlIIlIIllIIlll .split (_IllIIlIlIIIIllIIl )[0 ]for IIlIlIIlIIllIIlll in os .listdir (IIllllIIIIlIlIlll )])
	else :IllIlllIlIIIIIlll =set ([IlllIIlIlIIIIIIll .split (_IllIIlIlIIIIllIIl )[0 ]for IlllIIlIlIIIIIIll in os .listdir (IIIlIIIIlIIlIlIlI )])&set ([IIIIlIIlllIIIlllI .split (_IllIIlIlIIIIllIIl )[0 ]for IIIIlIIlllIIIlllI in os .listdir (IIlIIIIIllIlIIlll )])
	IIllllIllIIlllllI =[]
	for IIlllIlllllIIIllI in IllIlllIlIIIIIlll :
		if IlIllIlllllllIlII :IIllllIllIIlllllI .append (_IllllIIIlIlIIIIlI %(IIIlIIIIlIIlIlIlI .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIlIIIIIllIlIIlll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIIIIllIlIlllIIII .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIllllIIIIlIlIlll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIlIIlIIIlIIlIIll ))
		else :IIllllIllIIlllllI .append (_IlIllIlIlIIIllIlI %(IIIlIIIIlIIlIlIlI .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIlIIIIIllIlIIlll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlllIlllllIIIllI ,IIlIIlIIIlIIlIIll ))
	IllIlIlllIIIllIlI =256 if IlIIllIlIllIIIIIl ==_IlIlIllIIIlIIIllI else 768 
	if IlIllIlllllllIlII :
		for _IlllIIIlllllllllI in range (2 ):IIllllIllIIlllllI .append (_IIIIlIlIIlIlllllI %(IllllIIlIllIIIlII ,IIlIlIIllIIIllIll ,IllllIIlIllIIIlII ,IllIlIlllIIIllIlI ,IllllIIlIllIIIlII ,IllllIIlIllIIIlII ,IIlIIlIIIlIIlIIll ))
	else :
		for _IlllIIIlllllllllI in range (2 ):IIllllIllIIlllllI .append (_IlIllIIIllIIIlIlI %(IllllIIlIllIIIlII ,IIlIlIIllIIIllIll ,IllllIIlIllIIIlII ,IllIlIlllIIIllIlI ,IIlIIlIIIlIIlIIll ))
	shuffle (IIllllIllIIlllllI )
	with open (_IllIlIIlIlIIllIIl %IlllIIllIIllIIlll ,'w')as IlllIIlIIIIIIIIIl :IlllIIlIIIIIIIIIl .write (_IlllIIIlllllIIIll .join (IIllllIllIIlllllI ))
	print (_IIIIllIlIllIIllIl );print ('use gpus:',IlllIIllIIIIllIII )
	if IIlIlIllIIIIIIlIl =='':print ('no pretrained Generator')
	if IIllllIllIlIIIIll =='':print ('no pretrained Discriminator')
	if IlllIIllIIIIllIII :IIIllIllIIlIIlIIl =IIllIIIlIllllIllI .python_cmd +_IIlIllIlllIllIIlI %(IllIlIllIIlIIlIIl ,IIlIlIIllIIIllIll ,1 if IlIllIlllllllIlII else 0 ,IIlIlIIllIIlIllll ,IlllIIllIIIIllIII ,IIIIIlllllIIIIIlI ,IllIIlIlIlIlIlllI ,_IIlIIIlIIIllIIlIl %IIlIlIllIIIIIIlIl if IIlIlIllIIIIIIlIl !=''else '',_IlIllllllIIIlllIl %IIllllIllIlIIIIll if IIllllIllIlIIIIll !=''else '',1 if IIIIllllIlIlIlIll ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IllIIllIllIllIllI ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IlIIllIIIIIIIIlII ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,IlIIllIlIllIIIIIl )
	else :IIIllIllIIlIIlIIl =IIllIIIlIllllIllI .python_cmd +_IIIllIllllllllIII %(IllIlIllIIlIIlIIl ,IIlIlIIllIIIllIll ,1 if IlIllIlllllllIlII else 0 ,IIlIlIIllIIlIllll ,IIIIIlllllIIIIIlI ,IllIIlIlIlIlIlllI ,_IIlIIIlIIIllIIlIl %IIlIlIllIIIIIIlIl if IIlIlIllIIIIIIlIl !=''else IlllllllllIIIlIlI ,_IlIllllllIIIlllIl %IIllllIllIlIIIIll if IIllllIllIlIIIIll !=''else IlllllllllIIIlIlI ,1 if IIIIllllIlIlIlIll ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IllIIllIllIllIllI ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IlIIllIIIIIIIIlII ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,IlIIllIlIllIIIIIl )
	print (IIIllIllIIlIIlIIl );IllIlIIlIIIllIlIl =Popen (IIIllIllIIlIIlIIl ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IllIlIIlIIIllIlIl .wait ();return _IIlIIIlIIllIIIlIl 
def IIIIllIIIIlIIllIl (IIIllIIIIIlIlllll ,IIllllIIlIIlIIIlI ):
	IllIIIlIlIIIlIlII =_IIllllIlIIllllllI %(IllllIIlIllIIIlII ,IIIllIIIIIlIlllll );os .makedirs (IllIIIlIlIIIlIlII ,exist_ok =_IIIIIlIIllIlllIIl );IIlIlllIIlIIIIlIl =_IIlIlIlllIllIIlII %IllIIIlIlIIIlIlII if IIllllIIlIIlIIIlI ==_IlIlIllIIIlIIIllI else _IllIllIIIIIIlIIII %IllIIIlIlIIIlIlII 
	if not os .path .exists (IIlIlllIIlIIIIlIl ):return '请先进行特征提取!'
	IlIIIlIIIllIlllll =list (os .listdir (IIlIlllIIlIIIIlIl ))
	if len (IlIIIlIIIllIlllll )==0 :return '请先进行特征提取！'
	IIlllllIIllIlIIIl =[];IllIIIlIIlIlIIlIl =[]
	for IIIllIIlIIIIIlIIl in sorted (IlIIIlIIIllIlllll ):IlIIIlllIlIIlIIIl =np .load (_IIlIlIIIlIllIIIlI %(IIlIlllIIlIIIIlIl ,IIIllIIlIIIIIlIIl ));IllIIIlIIlIlIIlIl .append (IlIIIlllIlIIlIIIl )
	IIlllIIIIIlIllIll =np .concatenate (IllIIIlIIlIlIIlIl ,0 );IllIlIlIlIlIIllII =np .arange (IIlllIIIIIlIllIll .shape [0 ]);np .random .shuffle (IllIlIlIlIlIIllII );IIlllIIIIIlIllIll =IIlllIIIIIlIllIll [IllIlIlIlIlIIllII ]
	if IIlllIIIIIlIllIll .shape [0 ]>2e5 :
		IIlllllIIllIlIIIl .append (_IIIIllIllIlllIIlI %IIlllIIIIIlIllIll .shape [0 ]);yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl )
		try :IIlllIIIIIlIllIll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIIIIlIIllIlllIIl ,batch_size =256 *IIllIIIlIllllIllI .n_cpu ,compute_labels =_IlIlllIlIlllIlllI ,init ='random').fit (IIlllIIIIIlIllIll ).cluster_centers_ 
		except :IIIIIlllIlIlIIllI =traceback .format_exc ();print (IIIIIlllIlIlIIllI );IIlllllIIllIlIIIl .append (IIIIIlllIlIlIIllI );yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl )
	np .save (_IllIIIIIIIIIllIII %IllIIIlIlIIIlIlII ,IIlllIIIIIlIllIll );IIIlIlllIIIIlIlIl =min (int (16 *np .sqrt (IIlllIIIIIlIllIll .shape [0 ])),IIlllIIIIIlIllIll .shape [0 ]//39 );IIlllllIIllIlIIIl .append ('%s,%s'%(IIlllIIIIIlIllIll .shape ,IIIlIlllIIIIlIlIl ));yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl );IlIlIIIlIlIIlIlIl =faiss .index_factory (256 if IIllllIIlIIlIIIlI ==_IlIlIllIIIlIIIllI else 768 ,_IIIIIIlllllllllll %IIIlIlllIIIIlIlIl );IIlllllIIllIlIIIl .append ('training');yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl );IIIllIlIllIlIIlIl =faiss .extract_index_ivf (IlIlIIIlIlIIlIlIl );IIIllIlIllIlIIlIl .nprobe =1 ;IlIlIIIlIlIIlIlIl .train (IIlllIIIIIlIllIll );faiss .write_index (IlIlIIIlIlIIlIlIl ,_IlIIIIIIIIlllIlIl %(IllIIIlIlIIIlIlII ,IIIlIlllIIIIlIlIl ,IIIllIlIllIlIIlIl .nprobe ,IIIllIIIIIlIlllll ,IIllllIIlIIlIIIlI ));IIlllllIIllIlIIIl .append ('adding');yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl );IllIlIllIIIIlIIIl =8192 
	for IlllIIlIllIllllll in range (0 ,IIlllIIIIIlIllIll .shape [0 ],IllIlIllIIIIlIIIl ):IlIlIIIlIlIIlIlIl .add (IIlllIIIIIlIllIll [IlllIIlIllIllllll :IlllIIlIllIllllll +IllIlIllIIIIlIIIl ])
	faiss .write_index (IlIlIIIlIlIIlIlIl ,_IlIIIlllIllIlIIll %(IllIIIlIlIIIlIlII ,IIIlIlllIIIIlIlIl ,IIIllIlIllIlIIlIl .nprobe ,IIIllIIIIIlIlllll ,IIllllIIlIIlIIIlI ));IIlllllIIllIlIIIl .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIIlIlllIIIIlIlIl ,IIIllIlIllIlIIlIl .nprobe ,IIIllIIIIIlIlllll ,IIllllIIlIIlIIIlI ));yield _IlllIIIlllllIIIll .join (IIlllllIIllIlIIIl )
def IllIlIIIlIIllIllI (IIIlIIIIIlIllIlII ,IIlIlIIllllIlllIl ,IIIlIlIIlIllllIIl ,IllIllIIllllIIllI ,IllllIllIlIlIIIII ,IlIIIlIlIIllllIII ,IlIIIlIlIIlllIlIl ,IllIIlIlIIIIIIlll ,IlIIIIIIllllIlllI ,IllIlIIIlIIlllllI ,IlIlllIIlIllIllIl ,IllIllIIllIIIIIIl ,IIllllIllIIIllIlI ,IIIIIIIIlIIllIlll ,IIIllllIIlIllIlll ,IllIlIllIlIIlIIlI ,IlIIlIIIIIlIIIlII ,IllIlllIlIlIllllI ):
	IlIIlllIlIIlIIllI =[]
	def IIlIllIIlIlIllIII (IIIIIllIIlllIIlIl ):IlIIlllIlIIlIIllI .append (IIIIIllIIlllIIlIl );return _IlllIIIlllllIIIll .join (IlIIlllIlIIlIIllI )
	IIIlIIIlIIIIIIlIl =_IIllllIlIIllllllI %(IllllIIlIllIIIlII ,IIIlIIIIIlIllIlII );IllIlIIIIlllIIIlI ='%s/preprocess.log'%IIIlIIIlIIIIIIlIl ;IllIIlIIIlllllllI ='%s/extract_fl_feature.log'%IIIlIIIlIIIIIIlIl ;IIIlIIlIlIlIlIlll =_IlllllIlIIIIIlIlI %IIIlIIIlIIIIIIlIl ;IlIlIIIIIlIIllIll =_IIlIlIlllIllIIlII %IIIlIIIlIIIIIIlIl if IlIIlIIIIIlIIIlII ==_IlIlIllIIIlIIIllI else _IllIllIIIIIIlIIII %IIIlIIIlIIIIIIlIl ;os .makedirs (IIIlIIIlIIIIIIlIl ,exist_ok =_IIIIIlIIllIlllIIl );open (IllIlIIIIlllIIIlI ,'w').close ();IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IllIllIIllllIIllI ,IIllIIlIlllIllIII [IIlIlIIllllIlllIl ],IlIIIlIlIIllllIII ,IIIlIIIlIIIIIIlIl )+str (IIllIIIlIllllIllI .noparallel );yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl ('step1:正在处理数据'));yield IIlIllIIlIlIllIII (IlIIIllIlIllIllll );IIIIIlllIlllIIIll =Popen (IlIIIllIlIllIllll ,shell =_IIIIIlIIllIlllIIl );IIIIIlllIlllIIIll .wait ()
	with open (IllIlIIIIlllIIIlI ,_IIIlIIIIIIlIIIIll )as IIllIIlIIIlIIIIll :print (IIllIIlIIIlIIIIll .read ())
	open (IllIIlIIIlllllllI ,'w')
	if IIIlIlIIlIllllIIl :
		yield IIlIllIIlIlIllIII ('step2a:正在提取音高')
		if IlIIIlIlIIlllIlIl !=_IIllIIIllIlIlllIl :IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +' extract_fl_print.py "%s" %s %s'%(IIIlIIIlIIIIIIlIl ,IlIIIlIlIIllllIII ,IlIIIlIlIIlllIlIl );yield IIlIllIIlIlIllIII (IlIIIllIlIllIllll );IIIIIlllIlllIIIll =Popen (IlIIIllIlIllIllll ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIIIIlllIlllIIIll .wait ()
		else :
			IllIlllIlIlIllllI =IllIlllIlIlIllllI .split ('-');IlIIIIIllIIIlIlII =len (IllIlllIlIlIllllI );IIlIllIllIIlIlIIl =[]
			for (IIIlIlIIllIIIIIlI ,IllIlllIIllIIllIl )in enumerate (IllIlllIlIlIllllI ):IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IlIIIIIllIIIlIlII ,IIIlIlIIllIIIIIlI ,IllIlllIIllIIllIl ,IIIlIIIlIIIIIIlIl ,IIllIIIlIllllIllI .is_half );yield IIlIllIIlIlIllIII (IlIIIllIlIllIllll );IIIIIlllIlllIIIll =Popen (IlIIIllIlIllIllll ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIlIllIllIIlIlIIl .append (IIIIIlllIlllIIIll )
			for IIIIIlllIlllIIIll in IIlIllIllIIlIlIIl :IIIIIlllIlllIIIll .wait ()
		with open (IllIIlIIIlllllllI ,_IIIlIIIIIIlIIIIll )as IIllIIlIIIlIIIIll :print (IIllIIlIIIlIIIIll .read ())
	else :yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl ('step2a:无需提取音高'))
	yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl ('step2b:正在提取特征'));IlIllIlllIlIIlIIl =IIIIIIIIlIIllIlll .split ('-');IlIIIIIllIIIlIlII =len (IlIllIlllIlIIlIIl );IIlIllIllIIlIlIIl =[]
	for (IIIlIlIIllIIIIIlI ,IllIlllIIllIIllIl )in enumerate (IlIllIlllIlIIlIIl ):IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IIllIIIlIllllIllI .device ,IlIIIIIllIIIlIlII ,IIIlIlIIllIIIIIlI ,IllIlllIIllIIllIl ,IIIlIIIlIIIIIIlIl ,IlIIlIIIIIlIIIlII );yield IIlIllIIlIlIllIII (IlIIIllIlIllIllll );IIIIIlllIlllIIIll =Popen (IlIIIllIlIllIllll ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIlIllIllIIlIlIIl .append (IIIIIlllIlllIIIll )
	for IIIIIlllIlllIIIll in IIlIllIllIIlIlIIl :IIIIIlllIlllIIIll .wait ()
	with open (IllIIlIIIlllllllI ,_IIIlIIIIIIlIIIIll )as IIllIIlIIIlIIIIll :print (IIllIIlIIIlIIIIll .read ())
	yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl ('step3a:正在训练模型'))
	if IIIlIlIIlIllllIIl :IlIIlIIlIlIlllIII ='%s/2a_f0'%IIIlIIIlIIIIIIlIl ;IllIIIlIIIIlllIlI =_IlllIIIIIllllIlIl %IIIlIIIlIIIIIIlIl ;IIIllllIlIllIIIll =set ([IIIlIlIlIIIIIlIlI .split (_IllIIlIlIIIIllIIl )[0 ]for IIIlIlIlIIIIIlIlI in os .listdir (IIIlIIlIlIlIlIlll )])&set ([IlllllIlllIIlIIlI .split (_IllIIlIlIIIIllIIl )[0 ]for IlllllIlllIIlIIlI in os .listdir (IlIlIIIIIlIIllIll )])&set ([IIlIllIIllIlIIIIl .split (_IllIIlIlIIIIllIIl )[0 ]for IIlIllIIllIlIIIIl in os .listdir (IlIIlIIlIlIlllIII )])&set ([IlIlIIlIIIIIIIIll .split (_IllIIlIlIIIIllIIl )[0 ]for IlIlIIlIIIIIIIIll in os .listdir (IllIIIlIIIIlllIlI )])
	else :IIIllllIlIllIIIll =set ([IIIIlIllIIIlIlIll .split (_IllIIlIlIIIIllIIl )[0 ]for IIIIlIllIIIlIlIll in os .listdir (IIIlIIlIlIlIlIlll )])&set ([IlIllIIllIlllIIll .split (_IllIIlIlIIIIllIIl )[0 ]for IlIllIIllIlllIIll in os .listdir (IlIlIIIIIlIIllIll )])
	IlIIlIIIIlIlIllIl =[]
	for IIlIlIllIIlllIllI in IIIllllIlIllIIIll :
		if IIIlIlIIlIllllIIl :IlIIlIIIIlIlIllIl .append (_IllllIIIlIlIIIIlI %(IIIlIIlIlIlIlIlll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IlIlIIIIIlIIllIll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IlIIlIIlIlIlllIII .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IllIIIlIIIIlllIlI .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IllllIllIlIlIIIII ))
		else :IlIIlIIIIlIlIllIl .append (_IlIllIlIlIIIllIlI %(IIIlIIlIlIlIlIlll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IlIlIIIIIlIIllIll .replace (_IlIllIIIIIllllllI ,_IIIlIlIIIIIIlIlII ),IIlIlIllIIlllIllI ,IllllIllIlIlIIIII ))
	IlllIllIlIlIlIIll =256 if IlIIlIIIIIlIIIlII ==_IlIlIllIIIlIIIllI else 768 
	if IIIlIlIIlIllllIIl :
		for _IlIIlIlIIllllIIlI in range (2 ):IlIIlIIIIlIlIllIl .append (_IIIIlIlIIlIlllllI %(IllllIIlIllIIIlII ,IIlIlIIllllIlllIl ,IllllIIlIllIIIlII ,IlllIllIlIlIlIIll ,IllllIIlIllIIIlII ,IllllIIlIllIIIlII ,IllllIllIlIlIIIII ))
	else :
		for _IlIIlIlIIllllIIlI in range (2 ):IlIIlIIIIlIlIllIl .append (_IlIllIIIllIIIlIlI %(IllllIIlIllIIIlII ,IIlIlIIllllIlllIl ,IllllIIlIllIIIlII ,IlllIllIlIlIlIIll ,IllllIllIlIlIIIII ))
	shuffle (IlIIlIIIIlIlIllIl )
	with open (_IllIlIIlIlIIllIIl %IIIlIIIlIIIIIIlIl ,'w')as IIllIIlIIIlIIIIll :IIllIIlIIIlIIIIll .write (_IlllIIIlllllIIIll .join (IlIIlIIIIlIlIllIl ))
	yield IIlIllIIlIlIllIII (_IIIIllIlIllIIllIl )
	if IIIIIIIIlIIllIlll :IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +_IIlIllIlllIllIIlI %(IIIlIIIIIlIllIlII ,IIlIlIIllllIlllIl ,1 if IIIlIlIIlIllllIIl else 0 ,IllIlIIIlIIlllllI ,IIIIIIIIlIIllIlll ,IlIIIIIIllllIlllI ,IllIIlIlIIIIIIlll ,_IIlIIIlIIIllIIlIl %IllIllIIllIIIIIIl if IllIllIIllIIIIIIl !=''else '',_IlIllllllIIIlllIl %IIllllIllIIIllIlI if IIllllIllIIIllIlI !=''else '',1 if IlIlllIIlIllIllIl ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IIIllllIIlIllIlll ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IllIlIllIlIIlIIlI ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,IlIIlIIIIIlIIIlII )
	else :IlIIIllIlIllIllll =IIllIIIlIllllIllI .python_cmd +_IIIllIllllllllIII %(IIIlIIIIIlIllIlII ,IIlIlIIllllIlllIl ,1 if IIIlIlIIlIllllIIl else 0 ,IllIlIIIlIIlllllI ,IlIIIIIIllllIlllI ,IllIIlIlIIIIIIlll ,_IIlIIIlIIIllIIlIl %IllIllIIllIIIIIIl if IllIllIIllIIIIIIl !=''else '',_IlIllllllIIIlllIl %IIllllIllIIIllIlI if IIllllIllIIIllIlI !=''else '',1 if IlIlllIIlIllIllIl ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IIIllllIIlIllIlll ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,1 if IllIlIllIlIIlIIlI ==IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll )else 0 ,IlIIlIIIIIlIIIlII )
	yield IIlIllIIlIlIllIII (IlIIIllIlIllIllll );IIIIIlllIlllIIIll =Popen (IlIIIllIlIllIllll ,shell =_IIIIIlIIllIlllIIl ,cwd =IllllIIlIllIIIlII );IIIIIlllIlllIIIll .wait ();yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl (_IIlIIIlIIllIIIlIl ));IlIllIllIllIlIIII =[];IIlIlllllIlIllllI =list (os .listdir (IlIlIIIIIlIIllIll ))
	for IIlIlIllIIlllIllI in sorted (IIlIlllllIlIllllI ):IlIIlIlIllIlllIII =np .load (_IIlIlIIIlIllIIIlI %(IlIlIIIIIlIIllIll ,IIlIlIllIIlllIllI ));IlIllIllIllIlIIII .append (IlIIlIlIllIlllIII )
	IIIIlIlllllllIIlI =np .concatenate (IlIllIllIllIlIIII ,0 );IIlllllIlllIIlIIl =np .arange (IIIIlIlllllllIIlI .shape [0 ]);np .random .shuffle (IIlllllIlllIIlIIl );IIIIlIlllllllIIlI =IIIIlIlllllllIIlI [IIlllllIlllIIlIIl ]
	if IIIIlIlllllllIIlI .shape [0 ]>2e5 :
		IIlIIIIlIIIIllIIl =_IIIIllIllIlllIIlI %IIIIlIlllllllIIlI .shape [0 ];print (IIlIIIIlIIIIllIIl );yield IIlIllIIlIlIllIII (IIlIIIIlIIIIllIIl )
		try :IIIIlIlllllllIIlI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IIIIIlIIllIlllIIl ,batch_size =256 *IIllIIIlIllllIllI .n_cpu ,compute_labels =_IlIlllIlIlllIlllI ,init ='random').fit (IIIIlIlllllllIIlI ).cluster_centers_ 
		except :IIlIIIIlIIIIllIIl =traceback .format_exc ();print (IIlIIIIlIIIIllIIl );yield IIlIllIIlIlIllIII (IIlIIIIlIIIIllIIl )
	np .save (_IllIIIIIIIIIllIII %IIIlIIIlIIIIIIlIl ,IIIIlIlllllllIIlI );IIIIllIIlIIlllIII =min (int (16 *np .sqrt (IIIIlIlllllllIIlI .shape [0 ])),IIIIlIlllllllIIlI .shape [0 ]//39 );yield IIlIllIIlIlIllIII ('%s,%s'%(IIIIlIlllllllIIlI .shape ,IIIIllIIlIIlllIII ));IIIlIlllIIllIlIlI =faiss .index_factory (256 if IlIIlIIIIIlIIIlII ==_IlIlIllIIIlIIIllI else 768 ,_IIIIIIlllllllllll %IIIIllIIlIIlllIII );yield IIlIllIIlIlIllIII ('training index');IIIIIlIIIlllIllIl =faiss .extract_index_ivf (IIIlIlllIIllIlIlI );IIIIIlIIIlllIllIl .nprobe =1 ;IIIlIlllIIllIlIlI .train (IIIIlIlllllllIIlI );faiss .write_index (IIIlIlllIIllIlIlI ,_IlIIIIIIIIlllIlIl %(IIIlIIIlIIIIIIlIl ,IIIIllIIlIIlllIII ,IIIIIlIIIlllIllIl .nprobe ,IIIlIIIIIlIllIlII ,IlIIlIIIIIlIIIlII ));yield IIlIllIIlIlIllIII ('adding index');IlIIlllllIIIllIIl =8192 
	for IIlllIlllllIlIIll in range (0 ,IIIIlIlllllllIIlI .shape [0 ],IlIIlllllIIIllIIl ):IIIlIlllIIllIlIlI .add (IIIIlIlllllllIIlI [IIlllIlllllIlIIll :IIlllIlllllIlIIll +IlIIlllllIIIllIIl ])
	faiss .write_index (IIIlIlllIIllIlIlI ,_IlIIIlllIllIlIIll %(IIIlIIIlIIIIIIlIl ,IIIIllIIlIIlllIII ,IIIIIlIIIlllIllIl .nprobe ,IIIlIIIIIlIllIlII ,IlIIlIIIIIlIIIlII ));yield IIlIllIIlIlIllIII ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIIIllIIlIIlllIII ,IIIIIlIIIlllIllIl .nprobe ,IIIlIIIIIlIllIlII ,IlIIlIIIIIlIIIlII ));yield IIlIllIIlIlIllIII (IlIllIIIlIIIIlIIl ('全流程结束！'))
def IllllIIllIIlIIllI (IlIllIIlIlIllIlll ):
	IIIIIlIIIlllIIlII ='train.log'
	if not os .path .exists (IlIllIIlIlIllIlll .replace (os .path .basename (IlIllIIlIlIllIlll ),IIIIIlIIIlllIIlII )):return {_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },{_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },{_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
	try :
		with open (IlIllIIlIlIllIlll .replace (os .path .basename (IlIllIIlIlIllIlll ),IIIIIlIIIlllIIlII ),_IIIlIIIIIIlIIIIll )as IlIIIlllIIIIlIlIl :IlIIlIlIIlIIlIIIl =eval (IlIIIlllIIIIlIlIl .read ().strip (_IlllIIIlllllIIIll ).split (_IlllIIIlllllIIIll )[0 ].split ('\t')[-1 ]);IIIIllllIIlllIllI ,IlIIIIlIIIIIIllIl =IlIIlIlIIlIIlIIIl [_IIIIllIlIIlllllll ],IlIIlIlIIlIIlIIIl ['if_f0'];IIIlIIIlIllIIIlIl =_IlllIIIIllIIIlllI if _IIIIIlIIlIIllIlll in IlIIlIlIIlIIlIIIl and IlIIlIlIIlIIlIIIl [_IIIIIlIIlIIllIlll ]==_IlllIIIIllIIIlllI else _IlIlIllIIIlIIIllI ;return IIIIllllIIlllIllI ,str (IlIIIIlIIIIIIllIl ),IIIlIIIlIllIIIlIl 
	except :traceback .print_exc ();return {_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },{_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI },{_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
def IlllIIIllIllIlllI (IlIlIIIIlllIlIlII ):
	if IlIlIIIIlllIlIlII ==_IIllIIIllIlIlllIl :IlIlllllIIlIIIIll =_IIIIIlIIllIlllIIl 
	else :IlIlllllIIlIIIIll =_IlIlllIlIlllIlllI 
	return {_IlIllllIIIIIlllll :IlIlllllIIlIIIIll ,_IIIIllllIlIIIlIlI :_IIlIlIIlllIllIIlI }
def IIIIIIlIIIllIlIII (IIlIlIlIIIlIIlIlI ,IIIIIIIlIIIIIIlll ):IIIlIlllllIIIIIll ='rnd';IIIllIIIIllIlllll ='pitchf';IllIIlIllIIIIIIIl ='pitch';IllIlIllIIlllIIII ='phone';global IllIIlIIIlllllIIl ;IllIIlIIIlllllIIl =torch .load (IIlIlIlIIIlIIlIlI ,map_location =_IlIlIIlllIIllIlIl );IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ][-3 ]=IllIIlIIIlllllIIl [_IIIlIllIIlIIIlIll ][_IIIIllllIIlIIllll ].shape [0 ];IlllIIIllIIlllllI =256 if IllIIlIIIlllllIIl .get (_IIIIIlIIlIIllIlll ,_IlIlIllIIIlIIIllI )==_IlIlIllIIIlIIIllI else 768 ;IIlIlllIIIlIlllll =torch .rand (1 ,200 ,IlllIIIllIIlllllI );IIlIIlllIIlllllll =torch .tensor ([200 ]).long ();IIlIllllIllllIIII =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IlllIlIIllIIIIIlI =torch .rand (1 ,200 );IIlIIIllIllIllIIl =torch .LongTensor ([0 ]);IIIllIlllllIIIIIl =torch .rand (1 ,192 ,200 );IllIIIlIlIlIIlllI =_IlIlIIlllIIllIlIl ;IIlIIIlIIlIlIIlll =SynthesizerTrnMsNSFsidM (*IllIIlIIIlllllIIl [_IlIIIIIIlIlllIIlI ],is_half =_IlIlllIlIlllIlllI ,version =IllIIlIIIlllllIIl .get (_IIIIIlIIlIIllIlll ,_IlIlIllIIIlIIIllI ));IIlIIIlIIlIlIIlll .load_state_dict (IllIIlIIIlllllIIl [_IIIlIllIIlIIIlIll ],strict =_IlIlllIlIlllIlllI );IIIllIIllIlIIlIIl =[IllIlIllIIlllIIII ,'phone_lengths',IllIIlIllIIIIIIIl ,IIIllIIIIllIlllll ,'ds',IIIlIlllllIIIIIll ];IIIIlllIIIIIIllII =['audio'];torch .onnx .export (IIlIIIlIIlIlIIlll ,(IIlIlllIIIlIlllll .to (IllIIIlIlIlIIlllI ),IIlIIlllIIlllllll .to (IllIIIlIlIlIIlllI ),IIlIllllIllllIIII .to (IllIIIlIlIlIIlllI ),IlllIlIIllIIIIIlI .to (IllIIIlIlIlIIlllI ),IIlIIIllIllIllIIl .to (IllIIIlIlIlIIlllI ),IIIllIlllllIIIIIl .to (IllIIIlIlIlIIlllI )),IIIIIIIlIIIIIIlll ,dynamic_axes ={IllIlIllIIlllIIII :[1 ],IllIIlIllIIIIIIIl :[1 ],IIIllIIIIllIlllll :[1 ],IIIlIlllllIIIIIll :[2 ]},do_constant_folding =_IlIlllIlIlllIlllI ,opset_version =13 ,verbose =_IlIlllIlIlllIlllI ,input_names =IIIllIIllIlIIlIIl ,output_names =IIIIlllIIIIIIllII );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IIIlIlIIlllIlllll :
	gr .Markdown (value =IlIllIIIlIIIIlIIl ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IlIllIIIlIIIIlIIl ('模型推理')):
			with gr .Row ():IlllIlllIllIIlIIl =gr .Dropdown (label =IlIllIIIlIIIIlIIl ('推理音色'),choices =sorted (IIlIlIIllllIIIlII ));IlllIlIIIlIllIllI =gr .Button (IlIllIIIlIIIIlIIl ('刷新音色列表和索引路径'),variant =_IIIIlllIlIllIlIlI );IlIlIllIlIllllIlI =gr .Button (IlIllIIIlIIIIlIIl ('卸载音色省显存'),variant =_IIIIlllIlIllIlIlI );IIIIIIlllIllIIllI =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IlIllIIIlIIIIlIIl ('请选择说话人id'),value =0 ,visible =_IlIlllIlIlllIlllI ,interactive =_IIIIIlIIllIlllIIl );IlIlIllIlIllllIlI .click (fn =IIIlllllIIIIlIIIl ,inputs =[],outputs =[IlllIlllIllIIlIIl ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IlllIIllIIIlIllII =gr .Number (label =IlIllIIIlIIIIlIIl (_IIlllIlIIllIIIllI ),value =0 );IlIIIIIllIIlIIlII =gr .Textbox (label =IlIllIIIlIIIIlIIl ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IIIIllIllIIIIllll =gr .Radio (label =IlIllIIIlIIIIlIIl (_IlIlIlIIlllllIlII ),choices =[_IlIlIlIIlIIlIllll ,_IIIlIIIlllllIlIII ,'crepe',_IIIllIlIIIIIlIIll ],value =_IlIlIlIIlIIlIllll ,interactive =_IIIIIlIIllIlllIIl );IIIIllIlIllIllIll =gr .Slider (minimum =0 ,maximum =7 ,label =IlIllIIIlIIIIlIIl (_IIIlllIlIIIllllll ),value =3 ,step =1 ,interactive =_IIIIIlIIllIlllIIl )
					with gr .Column ():IllIIllIIllIIlIll =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIlllIlIIlIllll ),value ='',interactive =_IIIIIlIIllIlllIIl );IllIIllIlllIIIlll =gr .Dropdown (label =IlIllIIIlIIIIlIIl (_IIIlIlIIIlIllllII ),choices =sorted (IlIllIIIllIllllII ),interactive =_IIIIIlIIllIlllIIl );IlllIlIIIlIllIllI .click (fn =IllIllIIIIllIIlIl ,inputs =[],outputs =[IlllIlllIllIIlIIl ,IllIIllIlllIIIlll ],api_name ='infer_refresh');IIIIIlIIlIllIIIII =gr .Slider (minimum =0 ,maximum =1 ,label =IlIllIIIlIIIIlIIl ('检索特征占比'),value =.75 ,interactive =_IIIIIlIIllIlllIIl )
					with gr .Column ():IlIIIlIllIllllIII =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIllIIIlIIIIlIIl (_IllIlIllIIllIlIlI ),value =0 ,step =1 ,interactive =_IIIIIlIIllIlllIIl );IlIIIlllllIlIIIIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIllIIIlIIIIlIIl (_IIlllllllIllllIII ),value =.25 ,interactive =_IIIIIlIIllIlllIIl );IlIllIIlIIIIlIIII =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIllIIIlIIIIlIIl (_IlIllIIlllIlIIlll ),value =.33 ,step =.01 ,interactive =_IIIIIlIIllIlllIIl )
					IllIlllIIlIlllIlI =gr .File (label =IlIllIIIlIIIIlIIl ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IIIlIIIIIllllIIlI =gr .Button (IlIllIIIlIIIIlIIl ('转换'),variant =_IIIIlllIlIllIlIlI )
					with gr .Row ():IlIIIlllllIIIIIll =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ));IIlIIIIIlIllllIIl =gr .Audio (label =IlIllIIIlIIIIlIIl ('输出音频(右下角三个点,点了可以下载)'))
					IIIlIIIIIllllIIlI .click (IIllIIIlllIIlIlll ,[IIIIIIlllIllIIllI ,IlIIIIIllIIlIIlII ,IlllIIllIIIlIllII ,IllIlllIIlIlllIlI ,IIIIllIllIIIIllll ,IllIIllIIllIIlIll ,IllIIllIlllIIIlll ,IIIIIlIIlIllIIIII ,IIIIllIlIllIllIll ,IlIIIlIllIllllIII ,IlIIIlllllIlIIIIl ,IlIllIIlIIIIlIIII ],[IlIIIlllllIIIIIll ,IIlIIIIIlIllllIIl ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IlIllIIIlIlIIIIII =gr .Number (label =IlIllIIIlIIIIlIIl (_IIlllIlIIllIIIllI ),value =0 );IlIllIIIIIllllIII =gr .Textbox (label =IlIllIIIlIIIIlIIl ('指定输出文件夹'),value =_IIIlIIIIIlIllIlll );IIIllllIIIlllllII =gr .Radio (label =IlIllIIIlIIIIlIIl (_IlIlIlIIlllllIlII ),choices =[_IlIlIlIIlIIlIllll ,_IIIlIIIlllllIlIII ,'crepe',_IIIllIlIIIIIlIIll ],value =_IlIlIlIIlIIlIllll ,interactive =_IIIIIlIIllIlllIIl );IllIlIIIllIIIIlll =gr .Slider (minimum =0 ,maximum =7 ,label =IlIllIIIlIIIIlIIl (_IIIlllIlIIIllllll ),value =3 ,step =1 ,interactive =_IIIIIlIIllIlllIIl )
					with gr .Column ():IlIlIlIIIIlIIIlII =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIlllIlIIlIllll ),value ='',interactive =_IIIIIlIIllIlllIIl );IlIIlIlIIIIlIlIlI =gr .Dropdown (label =IlIllIIIlIIIIlIIl (_IIIlIlIIIlIllllII ),choices =sorted (IlIllIIIllIllllII ),interactive =_IIIIIlIIllIlllIIl );IlllIlIIIlIllIllI .click (fn =lambda :IllIllIIIIllIIlIl ()[1 ],inputs =[],outputs =IlIIlIlIIIIlIlIlI ,api_name ='infer_refresh_batch');IIllIlIlIIIIllIll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIllIIIlIIIIlIIl ('检索特征占比'),value =1 ,interactive =_IIIIIlIIllIlllIIl )
					with gr .Column ():IllIlIIIlIIlIIlII =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIllIIIlIIIIlIIl (_IllIlIllIIllIlIlI ),value =0 ,step =1 ,interactive =_IIIIIlIIllIlllIIl );IIlllllIIllllllIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIllIIIlIIIIlIIl (_IIlllllllIllllIII ),value =1 ,interactive =_IIIIIlIIllIlllIIl );IlllIlIIIIIIIIlIl =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIllIIIlIIIIlIIl (_IlIllIIlllIlIIlll ),value =.33 ,step =.01 ,interactive =_IIIIIlIIllIlllIIl )
					with gr .Column ():IIllIIllIIlIllllI =gr .Textbox (label =IlIllIIIlIIIIlIIl ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IlIlllIllIlIlIlII =gr .File (file_count ='multiple',label =IlIllIIIlIIIIlIIl (_IlIllIlIIIlllIIII ))
					with gr .Row ():IIlIlllIlIIIIIIII =gr .Radio (label =IlIllIIIlIIIIlIIl ('导出文件格式'),choices =[_IIlIlIlIlIIllIIII ,_IlIlIIlllIlIIllll ,'mp3','m4a'],value =_IlIlIIlllIlIIllll ,interactive =_IIIIIlIIllIlllIIl );IlIlIllIIIllIIIII =gr .Button (IlIllIIIlIIIIlIIl ('转换'),variant =_IIIIlllIlIllIlIlI );IIIlllIIllIIlllll =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ))
					IlIlIllIIIllIIIII .click (IIlIllIlIlIIllIII ,[IIIIIIlllIllIIllI ,IIllIIllIIlIllllI ,IlIllIIIIIllllIII ,IlIlllIllIlIlIlII ,IlIllIIIlIlIIIIII ,IIIllllIIIlllllII ,IlIlIlIIIIlIIIlII ,IlIIlIlIIIIlIlIlI ,IIllIlIlIIIIllIll ,IllIlIIIllIIIIlll ,IllIlIIIlIIlIIlII ,IIlllllIIllllllIl ,IlllIlIIIIIIIIlIl ,IIlIlllIlIIIIIIII ],[IIIlllIIllIIlllll ],api_name ='infer_convert_batch')
			IlllIlllIllIIlIIl .change (fn =IlIllIIllIIIIIIlI ,inputs =[IlllIlllIllIIlIIl ,IlIllIIlIIIIlIIII ,IlllIlIIIIIIIIlIl ],outputs =[IIIIIIlllIllIIllI ,IlIllIIlIIIIlIIII ,IlllIlIIIIIIIIlIl ,IllIIllIlllIIIlll ])
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IlIlllIllIlIIIIII =gr .Textbox (label =IlIllIIIlIIIIlIIl ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IIIlIllIllIlIIIll =gr .File (file_count ='multiple',label =IlIllIIIlIIIIlIIl (_IlIllIlIIIlllIIII ))
					with gr .Column ():IIIIIIlIllllllIII =gr .Dropdown (label =IlIllIIIlIIIIlIIl ('模型'),choices =IIllIlllIIlllIIll );IIlIIIIIlllIlIIlI =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IIIIIlIIllIlllIIl ,visible =_IlIlllIlIlllIlllI );IlIllIIlIllIIIlll =gr .Textbox (label =IlIllIIIlIIIIlIIl ('指定输出主人声文件夹'),value =_IIIlIIIIIlIllIlll );IIllIIIllIlIIllll =gr .Textbox (label =IlIllIIIlIIIIlIIl ('指定输出非主人声文件夹'),value =_IIIlIIIIIlIllIlll );IlIllllIlIIllIllI =gr .Radio (label =IlIllIIIlIIIIlIIl ('导出文件格式'),choices =[_IIlIlIlIlIIllIIII ,_IlIlIIlllIlIIllll ,'mp3','m4a'],value =_IlIlIIlllIlIIllll ,interactive =_IIIIIlIIllIlllIIl )
					IIllllIlIIIlIIlII =gr .Button (IlIllIIIlIIIIlIIl ('转换'),variant =_IIIIlllIlIllIlIlI );IlllllIlIIllIIlIl =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ));IIllllIlIIIlIIlII .click (IIIIIIlIlllIlIIll ,[IIIIIIlIllllllIII ,IlIlllIllIlIIIIII ,IlIllIIlIllIIIlll ,IIIlIllIllIlIIIll ,IIllIIIllIlIIllll ,IIlIIIIIlllIlIIlI ,IlIllllIlIIllIllI ],[IlllllIlIIllIIlIl ],api_name ='uvr_convert')
		with gr .TabItem (IlIllIIIlIIIIlIIl ('训练')):
			gr .Markdown (value =IlIllIIIlIIIIlIIl ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIlIIlllIIIlllIll =gr .Textbox (label =IlIllIIIlIIIIlIIl ('输入实验名'),value ='mi-test');IlIllIIlllIIlIIIl =gr .Radio (label =IlIllIIIlIIIIlIIl ('目标采样率'),choices =[_IIIllIIlIIIllIIII ],value =_IIIllIIlIIIllIIII ,interactive =_IIIIIlIIllIlllIIl );IIllIIlIlllllllll =gr .Radio (label =IlIllIIIlIIIIlIIl ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IIIIIlIIllIlllIIl ,_IlIlllIlIlllIlllI ],value =_IIIIIlIIllIlllIIl ,interactive =_IIIIIlIIllIlllIIl );IllIlIIllIlllIIII =gr .Radio (label =IlIllIIIlIIIIlIIl ('版本'),choices =[_IlllIIIIllIIIlllI ],value =_IlllIIIIllIIIlllI ,interactive =_IIIIIlIIllIlllIIl ,visible =_IIIIIlIIllIlllIIl );IIIIllIlIIllIlIIl =gr .Slider (minimum =0 ,maximum =IIllIIIlIllllIllI .n_cpu ,step =1 ,label =IlIllIIIlIIIIlIIl ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IIllIIIlIllllIllI .n_cpu /1.5 )),interactive =_IIIIIlIIllIlllIIl )
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IllIlIllIllllIlll =gr .Textbox (label =IlIllIIIlIIIIlIIl ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IIIllIlIlIlllIlll =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IlIllIIIlIIIIlIIl ('请指定说话人id'),value =0 ,interactive =_IIIIIlIIllIlllIIl );IlIlIllIIIllIIIII =gr .Button (IlIllIIIlIIIIlIIl ('处理数据'),variant =_IIIIlllIlIllIlIlI );IllIlIlIlIIIIlIlI =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ),value ='');IlIlIllIIIllIIIII .click (IIIIlllIIlllIllIl ,[IllIlIllIllllIlll ,IIlIIlllIIIlllIll ,IlIllIIlllIIlIIIl ,IIIIllIlIIllIlIIl ],[IllIlIlIlIIIIlIlI ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IIlIlIIllIIIIllIl =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IllIIlIIIlIIIllll ),value =IIlIIIlIllIIlIIII ,interactive =_IIIIIlIIllIlllIIl );IIllIIIlIIIlIllll =gr .Textbox (label =IlIllIIIlIIIIlIIl ('显卡信息'),value =IlIIlIlllIlIIIIlI )
					with gr .Column ():IllIlIlllIIlIllIl =gr .Radio (label =IlIllIIIlIIIIlIIl ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IlIlIlIIlIIlIllll ,_IIIlIIIlllllIlIII ,'dio',_IIIllIlIIIIIlIIll ,_IIllIIIllIlIlllIl ],value =_IIllIIIllIlIlllIl ,interactive =_IIIIIlIIllIlllIIl );IIIlIlIlIlIIIllIl =gr .Textbox (label =IlIllIIIlIIIIlIIl ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IIlIIIlIllIIlIIII ,IIlIIIlIllIIlIIII ),interactive =_IIIIIlIIllIlllIIl ,visible =_IIIIIlIIllIlllIIl )
					IIllllIlIIIlIIlII =gr .Button (IlIllIIIlIIIIlIIl ('特征提取'),variant =_IIIIlllIlIllIlIlI );IIIIIIllllIIlIIll =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ),value ='',max_lines =8 );IllIlIlllIIlIllIl .change (fn =IlllIIIllIllIlllI ,inputs =[IllIlIlllIIlIllIl ],outputs =[IIIlIlIlIlIIIllIl ]);IIllllIlIIIlIIlII .click (IlIIIIlIIlIlIIIll ,[IIlIlIIllIIIIllIl ,IIIIllIlIIllIlIIl ,IllIlIlllIIlIllIl ,IIllIIlIlllllllll ,IIlIIlllIIIlllIll ,IllIlIIllIlllIIII ,IIIlIlIlIlIIIllIl ],[IIIIIIllllIIlIIll ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IlIllIIIlIIIIlIIl ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IIIIIIIlIIlllllll =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IlIllIIIlIIIIlIIl ('保存频率save_every_epoch'),value =5 ,interactive =_IIIIIlIIllIlllIIl );IIllIIlIIIIllIlII =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IlIllIIIlIIIIlIIl ('总训练轮数total_epoch'),value =300 ,interactive =_IIIIIlIIllIlllIIl );IIllIllllIlllIlll =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IlIllIIIlIIIIlIIl ('每张显卡的batch_size'),value =IIIlIllIIIIllllII ,interactive =_IIIIIlIIllIlllIIl );IlIlIlIIIIlllllIl =gr .Radio (label =IlIllIIIlIIIIlIIl ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll ),IlIllIIIlIIIIlIIl ('否')],value =IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll ),interactive =_IIIIIlIIllIlllIIl );IIlIIIlIIIIlllIlI =gr .Radio (label =IlIllIIIlIIIIlIIl ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll ),IlIllIIIlIIIIlIIl ('否')],value =IlIllIIIlIIIIlIIl ('否'),interactive =_IIIIIlIIllIlllIIl );IlIIlllIIllllllII =gr .Radio (label =IlIllIIIlIIIIlIIl ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll ),IlIllIIIlIIIIlIIl ('否')],value =IlIllIIIlIIIIlIIl (_IlIIlllIIlllIIIll ),interactive =_IIIIIlIIllIlllIIl )
				with gr .Row ():IllIIIIlIllIlIllI =gr .Textbox (label =IlIllIIIlIIIIlIIl ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IIIIIlIIllIlllIIl );IIIlllIIIlllIllII =gr .Textbox (label =IlIllIIIlIIIIlIIl ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IIIIIlIIllIlllIIl );IlIllIIlllIIlIIIl .change (IlllIIIIIIlIlIIlI ,[IlIllIIlllIIlIIIl ,IIllIIlIlllllllll ,IllIlIIllIlllIIII ],[IllIIIIlIllIlIllI ,IIIlllIIIlllIllII ]);IllIlIIllIlllIIII .change (IlIIIlIllllllIIll ,[IlIllIIlllIIlIIIl ,IIllIIlIlllllllll ,IllIlIIllIlllIIII ],[IllIIIIlIllIlIllI ,IIIlllIIIlllIllII ,IlIllIIlllIIlIIIl ]);IIllIIlIlllllllll .change (IllIIIlIllIIIlIIl ,[IIllIIlIlllllllll ,IlIllIIlllIIlIIIl ,IllIlIIllIlllIIII ],[IllIlIlllIIlIllIl ,IllIIIIlIllIlIllI ,IIIlllIIIlllIllII ]);IllIllllIIlIlIllI =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IllIIlIIIlIIIllll ),value =IIlIIIlIllIIlIIII ,interactive =_IIIIIlIIllIlllIIl );IIIIlIIIlIllllIlI =gr .Button (IlIllIIIlIIIIlIIl ('训练模型'),variant =_IIIIlllIlIllIlIlI );IIlIIllIIIlIIIIII =gr .Button (IlIllIIIlIIIIlIIl ('训练特征索引'),variant =_IIIIlllIlIllIlIlI );IlllIllllIlIIIlIl =gr .Button (IlIllIIIlIIIIlIIl ('一键训练'),variant =_IIIIlllIlIllIlIlI );IlllIIIIlIIIIllII =gr .Textbox (label =IlIllIIIlIIIIlIIl (_IIIIllllIIIIIlIII ),value ='',max_lines =10 );IIIIlIIIlIllllIlI .click (IIllIllIlIIlIlIII ,[IIlIIlllIIIlllIll ,IlIllIIlllIIlIIIl ,IIllIIlIlllllllll ,IIIllIlIlIlllIlll ,IIIIIIIlIIlllllll ,IIllIIlIIIIllIlII ,IIllIllllIlllIlll ,IlIlIlIIIIlllllIl ,IllIIIIlIllIlIllI ,IIIlllIIIlllIllII ,IllIllllIIlIlIllI ,IIlIIIlIIIIlllIlI ,IlIIlllIIllllllII ,IllIlIIllIlllIIII ],IlllIIIIlIIIIllII ,api_name ='train_start');IIlIIllIIIlIIIIII .click (IIIIllIIIIlIIllIl ,[IIlIIlllIIIlllIll ,IllIlIIllIlllIIII ],IlllIIIIlIIIIllII );IlllIllllIlIIIlIl .click (IllIlIIIlIIllIllI ,[IIlIIlllIIIlllIll ,IlIllIIlllIIlIIIl ,IIllIIlIlllllllll ,IllIlIllIllllIlll ,IIIllIlIlIlllIlll ,IIIIllIlIIllIlIIl ,IllIlIlllIIlIllIl ,IIIIIIIlIIlllllll ,IIllIIlIIIIllIlII ,IIllIllllIlllIlll ,IlIlIlIIIIlllllIl ,IllIIIIlIllIlIllI ,IIIlllIIIlllIllII ,IllIllllIIlIlIllI ,IIlIIIlIIIIlllIlI ,IlIIlllIIllllllII ,IllIlIIllIlllIIII ,IIIlIlIlIlIIIllIl ],IlllIIIIlIIIIllII ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IIIlIIIIIIlIIIIll ,encoding ='utf8')as IllIIIIllIllIIIll :IllIlIIIIllllIIll =IllIIIIllIllIIIll .read ()
				else :
					with open ('docs/faq_en.md',_IIIlIIIIIIlIIIIll ,encoding ='utf8')as IllIIIIllIllIIIll :IllIlIIIIllllIIll =IllIIIIllIllIIIll .read ()
				gr .Markdown (value =IllIlIIIIllllIIll )
			except :gr .Markdown (traceback .format_exc ())
	if IIllIIIlIllllIllI .iscolab :IIIlIlIIlllIlllll .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IIllIIIlIllllIllI .listen_port ,share =_IlIlllIlIlllIlllI )
	else :IIIlIlIIlllIlllll .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IIllIIIlIllllIllI .noautoopen ,server_port =IIllIIIlIllllIllI .listen_port ,quiet =_IlIlllIlIlllIlllI ,share =_IlIlllIlIlllIlllI )