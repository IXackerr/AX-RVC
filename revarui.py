_IIIlIlIIlllllIIIl ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IlIlllIIlIlIIllll ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IIllIllllllllIlII ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIllIIIIlIIlIIlII ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IllllIlIIIIllIIlI ='后处理重采样至最终采样率，0为不进行重采样'
_IIlIlIIlIlIIIlIII ='自动检测index路径,下拉式选择(dropdown)'
_IllIIlIllllIlllIl ='特征检索库文件路径,为空则使用下拉的选择结果'
_IllIlIIllIlIlIlIl ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IIIlIlIIlIllllllI ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IlIIlIIIIIlIllIll ='变调(整数, 半音数量, 升八度12降八度-12)'
_IlllIIIllIlIlIIll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIIIIIllIlllllIl ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IlIllIlllIIlIIllI ='IVF%s,Flat'
_IIllIlIlIlIIlIIII ='%s/total_fea.npy'
_IlllIIlllIIllIIll ='Trying doing kmeans %s shape to 10k centers.'
_IlIIllIIllIIllIII ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IIlIllIlIIllIIIII =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IllIlIllIllllIIll =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIIllIllllIllIIlI ='write filelist done'
_IIIlIlIlIlIIIllll ='%s/filelist.txt'
_IlllIllllIIlIlIIl ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IlIllIIIllIIllIII ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IIIlIIllllIlIIlII ='%s/%s.wav|%s/%s.npy|%s'
_IllIlIlIIIlIIlllI ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IlIIlIlIIIIIIllII ='%s/2b-f0nsf'
_IlIIIllIIlIlIIlII ='%s/0_gt_wavs'
_IIllIIllIIIIIlIll ='emb_g.weight'
_IlllIlIIIIIIllIIl ='clean_empty_cache'
_IIlIlIlIIIlIIIlIl ='sample_rate'
_IIIIllllllIIlIllI ='%s->%s'
_IlIlIlllllIIIIIII ='.index'
_IlIIIIIIlIllllIll ='weights'
_IIIIIlIIIlIlIIIll ='opt'
_IlIllIlIlIllllllI ='rmvpe'
_IllllIlllllllIlll ='harvest'
_IlIIlIlIIlIllIllI ='%s/3_feature768'
_IlIIIlIIlIllllIll ='%s/3_feature256'
_IIllIlIlIIIIIIIll ='_v2'
_IIlllIllIIlIlllll ='48k'
_IlIIllIlIlIIIlIII ='32k'
_IIllIllllIIlIIlII ='cpu'
_IIIIlIIIlIIllIlIl ='wav'
_IIllIIIIIIlIlIlIl ='trained'
_IlIIlllllIIIllIIl ='logs'
_IllIIIIllllIlIllI ='-pd %s'
_IlllIIIlIlIlIIlIl ='-pg %s'
_IIlIIlIIlIIlIIlll ='choices'
_IIIllllllllIIIllI ='weight'
_IIIlIIIlIIIIIIIll ='pm'
_IlIllIllIIlllIlII ='rmvpe_gpu'
_IllllIIIIlllIIlIl ='%s/logs/%s'
_IlIIlIIIIIllIlIll ='flac'
_IIIIIlllIlllIIlll ='f0'
_IIlIllIIlIlllIIll ='%s/%s'
_IlIlIlIIIIllIIIll ='.pth'
_IlIlIllIllllIllII ='输出信息'
_IIIIIllIIIllIlIII ='not exist, will not use pretrained model'
_IlIlIllIIIIlIlIII ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IIllllIlllIIlIIll ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IlIlIIIIlIlIllIll ='40k'
_IllIllllIIIIllIll ='value'
_IllIllllIllIlllIl ='v2'
_IIllIIlllIlIllIll ='version'
_IlIIIIIIIllllIIII ='visible'
_IIIIlllIlIIIIIIlI ='primary'
_IllllllIllIIlIlII =None 
_IlllllIllllllIIlI ='\\\\'
_IIIlIlIlllIIIIlIl ='\\'
_IIIIIIIlIlIlIlIlI ='"'
_IlllIlIIlIIIlIlII =' '
_IllIlIIIIIIIIllll ='config'
_IIlIIIIlIIIIIllll ='.'
_IlllIllIlllIllIIl ='r'
_IlIIllIIIIllIllII ='是'
_IlIIIlIlIIlllIlll ='update'
_IIIllIlllllIlllII ='__type__'
_IIIllllIIIIIIIIlI ='v1'
_IlllllIlIIlIllllI =False 
_IIIIllIIIIIIIIlII ='\n'
_IlIlIlIIIllIIIIll =True 
import os ,shutil ,sys 
IIlIIllIIIIIIIIll =os .getcwd ()
sys .path .append (IIlIIllIIIIIIIIll )
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
IIlIIllIIIIIIIIll =os .getcwd ()
IllIlllIIlIIlIlIl =os .path .join (IIlIIllIIIIIIIIll ,'TEMP')
shutil .rmtree (IllIlllIIlIIlIlIl ,ignore_errors =_IlIlIlIIIllIIIIll )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IIlIIllIIIIIIIIll ,ignore_errors =_IlIlIlIIIllIIIIll )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IIlIIllIIIIIIIIll ,ignore_errors =_IlIlIlIIIllIIIIll )
os .makedirs (IllIlllIIlIIlIlIl ,exist_ok =_IlIlIlIIIllIIIIll )
os .makedirs (os .path .join (IIlIIllIIIIIIIIll ,_IlIIlllllIIIllIIl ),exist_ok =_IlIlIlIIIllIIIIll )
os .makedirs (os .path .join (IIlIIllIIIIIIIIll ,_IlIIIIIIlIllllIll ),exist_ok =_IlIlIlIIIllIIIIll )
os .environ ['TEMP']=IllIlllIIlIIlIlIl 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IIIllIlllIlllllII =Config ()
IlIIIIllIllIIlllI =I18nAuto ()
IlIIIIllIllIIlllI .print ()
IlIllIIlIlIlIllll =torch .cuda .device_count ()
IIllIIIlIlIllllIl =[]
IIlllIlllIlIlllII =[]
IIIIIIlIlIllIIllI =_IlllllIlIIlIllllI 
if torch .cuda .is_available ()or IlIllIIlIlIlIllll !=0 :
	for IIllllIIlllIlllll in range (IlIllIIlIlIlIllll ):
		IlIlIlIIlllIllIII =torch .cuda .get_device_name (IIllllIIlllIlllll )
		if any (IlIIllIlIlIIIIIlI in IlIlIlIIlllIllIII .upper ()for IlIIllIlIlIIIIIlI in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IIIIIIlIlIllIIllI =_IlIlIlIIIllIIIIll ;IIllIIIlIlIllllIl .append ('%s\t%s'%(IIllllIIlllIlllll ,IlIlIlIIlllIllIII ));IIlllIlllIlIlllII .append (int (torch .cuda .get_device_properties (IIllllIIlllIlllll ).total_memory /1024 /1024 /1024 +.4 ))
if IIIIIIlIlIllIIllI and len (IIllIIIlIlIllllIl )>0 :IlIIIIlIIlllIllII =_IIIIllIIIIIIIIlII .join (IIllIIIlIlIllllIl );IlllIIIlIIIllllll =min (IIlllIlllIlIlllII )//2 
else :IlIIIIlIIlllIllII =IlIIIIllIllIIlllI ('很遗憾您这没有能用的显卡来支持您训练');IlllIIIlIIIllllll =1 
IIllIIlllllllllIl ='-'.join ([IlIIIIIIlIlIlllII [0 ]for IlIIIIIIlIlIlllII in IIllIIIlIlIllllIl ])
class IllIIIIlllllIlIll (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IIlIIllIIIllIIllI ,**IllIIlllIIIIIIIll ):super ().__init__ (variant ='tool',**IllIIlllIIIIIIIll )
	def get_block_name (IIIIllIllIIlllllI ):return 'button'
IllIIlllIlIIlllll =_IllllllIllIIlIlII 
def IIlllIIlIllIlllII ():
	global IllIIlllIlIIlllll ;IIllIlIIIllIlIlII ,_IllllIIlllIIllIII ,_IllllIIlllIIllIII =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IllIIlllIlIIlllll =IIllIlIIIllIlIlII [0 ];IllIIlllIlIIlllll =IllIIlllIlIIlllll .to (IIIllIlllIlllllII .device )
	if IIIllIlllIlllllII .is_half :IllIIlllIlIIlllll =IllIIlllIlIIlllll .half ()
	else :IllIIlllIlIIlllll =IllIIlllIlIIlllll .float ()
	IllIIlllIlIIlllll .eval ()
IlIIlIlIllIlIllIl =_IlIIIIIIlIllllIll 
IIllIIIlIIlllIIII ='uvr5_weights'
IllIlIIIIIIIIIlII =_IlIIlllllIIIllIIl 
IIlllllIlllllIIlI =[]
for IlIIIlIllllIIIIIl in os .listdir (IlIIlIlIllIlIllIl ):
	if IlIIIlIllllIIIIIl .endswith (_IlIlIlIIIIllIIIll ):IIlllllIlllllIIlI .append (IlIIIlIllllIIIIIl )
IlIIlIIllIIIIlIIl =[]
for (IlIlIlIIIIlllIIIl ,IlllllIllllIIlIlI ,IllIlIlIIIlIllIlI )in os .walk (IllIlIIIIIIIIIlII ,topdown =_IlllllIlIIlIllllI ):
	for IlIIIlIllllIIIIIl in IllIlIlIIIlIllIlI :
		if IlIIIlIllllIIIIIl .endswith (_IlIlIlllllIIIIIII )and _IIllIIIIIIlIlIlIl not in IlIIIlIllllIIIIIl :IlIIlIIllIIIIlIIl .append (_IIlIllIIlIlllIIll %(IlIlIlIIIIlllIIIl ,IlIIIlIllllIIIIIl ))
IIIlIIIIIIllIIlII =[]
for IlIIIlIllllIIIIIl in os .listdir (IIllIIIlIIlllIIII ):
	if IlIIIlIllllIIIIIl .endswith (_IlIlIlIIIIllIIIll )or 'onnx'in IlIIIlIllllIIIIIl :IIIlIIIIIIllIIlII .append (IlIIIlIllllIIIIIl .replace (_IlIlIlIIIIllIIIll ,''))
IIlIlIlllllIIllll =_IllllllIllIIlIlII 
def IlIIIlIlIllllIlIl (IIIIllIIIlIIllllI ,IllIlIIIIIlllllll ,IlllIllIllllIllll ,IIlIIlllIlIIIllII ,IlIlIIllIIIIIlllI ,IIllIllIIlIlIIllI ,IlIlIIIlIIIllllII ,IIIIllIIlllllIIll ,IIlIIlIIIIlIlIlII ,IIIIlllIlIllIIIlI ,IIIlIIIlIlIIlIllI ,IlllIlIlIlIlIIIlI ):
	global IlIIlIIIlIIlllllI ,IlIIIIIllllIIIlll ,IllIIlllIIlIIlIll ,IllIIlllIlIIlllll ,IIlIlIlllIlIlIllI ,IIlIlIlllllIIllll 
	if IllIlIIIIIlllllll is _IllllllIllIIlIlII :return 'You need to upload an audio',_IllllllIllIIlIlII 
	IlllIllIllllIllll =int (IlllIllIllllIllll )
	try :
		IlIIIIlIIlIIllIlI =load_audio (IllIlIIIIIlllllll ,16000 );IIllIlIllIIIIIIll =np .abs (IlIIIIlIIlIIllIlI ).max ()/.95 
		if IIllIlIllIIIIIIll >1 :IlIIIIlIIlIIllIlI /=IIllIlIllIIIIIIll 
		IIIIIlIlIIIlllIlI =[0 ,0 ,0 ]
		if not IllIIlllIlIIlllll :IIlllIIlIllIlllII ()
		IIlIlllllIIlllIIl =IIlIlIlllllIIllll .get (_IIIIIlllIlllIIlll ,1 );IIllIllIIlIlIIllI =IIllIllIIlIlIIllI .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII ).replace (_IIllIIIIIIlIlIlIl ,'added')if IIllIllIIlIlIIllI !=''else IlIlIIIlIIIllllII ;IIlIlIIIllIllIlll =IllIIlllIIlIIlIll .pipeline (IllIIlllIlIIlllll ,IlIIIIIllllIIIlll ,IIIIllIIIlIIllllI ,IlIIIIlIIlIIllIlI ,IllIlIIIIIlllllll ,IIIIIlIlIIIlllIlI ,IlllIllIllllIllll ,IlIlIIllIIIIIlllI ,IIllIllIIlIlIIllI ,IIIIllIIlllllIIll ,IIlIlllllIIlllIIl ,IIlIIlIIIIlIlIlII ,IlIIlIIIlIIlllllI ,IIIIlllIlIllIIIlI ,IIIlIIIlIlIIlIllI ,IIlIlIlllIlIlIllI ,IlllIlIlIlIlIIIlI ,f0_file =IIlIIlllIlIIIllII )
		if IlIIlIIIlIIlllllI !=IIIIlllIlIllIIIlI >=16000 :IlIIlIIIlIIlllllI =IIIIlllIlIllIIIlI 
		IIIIllIIIIlIIllll ='Using index:%s.'%IIllIllIIlIlIIllI if os .path .exists (IIllIllIIlIlIIllI )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IIIIllIIIIlIIllll ,IIIIIlIlIIIlllIlI [0 ],IIIIIlIlIIIlllIlI [1 ],IIIIIlIlIIIlllIlI [2 ]),(IlIIlIIIlIIlllllI ,IIlIlIIIllIllIlll )
	except :IIIlIlllIlllIlIll =traceback .format_exc ();print (IIIlIlllIlllIlIll );return IIIlIlllIlllIlIll ,(_IllllllIllIIlIlII ,_IllllllIllIIlIlII )
def IllIIIllIllIllIll (IllIIIIIIIIIlIIlI ,IlIlIllIlIllIllII ,IIIllIlIlIIlIIIII ,IIIlllIlllllIlIll ,IIllllIIlIlIlIIll ,IIlIlllIIllIllIII ,IIlIIIIllIIlllIII ,IIlIlIIlIIllllllI ,IlIIIIIIlllllIlll ,IIllIllIllIIlIlll ,IlIIlllIIllllIIlI ,IIIIIllllllIIIIIl ,IlIIIIllIIIIIIIIl ,IlllIllllIIllllII ):
	try :
		IlIlIllIlIllIllII =IlIlIllIlIllIllII .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII );IIIllIlIlIIlIIIII =IIIllIlIlIIlIIIII .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII );os .makedirs (IIIllIlIlIIlIIIII ,exist_ok =_IlIlIlIIIllIIIIll )
		try :
			if IlIlIllIlIllIllII !='':IIIlllIlllllIlIll =[os .path .join (IlIlIllIlIllIllII ,IIllIlllIlIIIlIll )for IIllIlllIlIIIlIll in os .listdir (IlIlIllIlIllIllII )]
			else :IIIlllIlllllIlIll =[IIIIIIlIIIllIIIlI .name for IIIIIIlIIIllIIIlI in IIIlllIlllllIlIll ]
		except :traceback .print_exc ();IIIlllIlllllIlIll =[IIlIlIllIlllIIIll .name for IIlIlIllIlllIIIll in IIIlllIlllllIlIll ]
		IllIlIllIIIIlIlIl =[]
		for IlIIllllllIIlIIll in IIIlllIlllllIlIll :
			IlIlIIIIllIIllIIl ,IIlIlllIllIllIlll =IlIIIlIlIllllIlIl (IllIIIIIIIIIlIIlI ,IlIIllllllIIlIIll ,IIllllIIlIlIlIIll ,_IllllllIllIIlIlII ,IIlIlllIIllIllIII ,IIlIIIIllIIlllIII ,IIlIlIIlIIllllllI ,IlIIIIIIlllllIlll ,IIllIllIllIIlIlll ,IlIIlllIIllllIIlI ,IIIIIllllllIIIIIl ,IlIIIIllIIIIIIIIl )
			if 'Success'in IlIlIIIIllIIllIIl :
				try :
					IllllIIIlIIIlIIIl ,IIIIlIIIlIIIIllll =IIlIlllIllIllIlll 
					if IlllIllllIIllllII in [_IIIIlIIIlIIllIlIl ,_IlIIlIIIIIllIlIll ]:sf .write ('%s/%s.%s'%(IIIllIlIlIIlIIIII ,os .path .basename (IlIIllllllIIlIIll ),IlllIllllIIllllII ),IIIIlIIIlIIIIllll ,IllllIIIlIIIlIIIl )
					else :
						IlIIllllllIIlIIll ='%s/%s.wav'%(IIIllIlIlIIlIIIII ,os .path .basename (IlIIllllllIIlIIll ));sf .write (IlIIllllllIIlIIll ,IIIIlIIIlIIIIllll ,IllllIIIlIIIlIIIl )
						if os .path .exists (IlIIllllllIIlIIll ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IlIIllllllIIlIIll ,IlIIllllllIIlIIll [:-4 ]+'.%s'%IlllIllllIIllllII ))
				except :IlIlIIIIllIIllIIl +=traceback .format_exc ()
			IllIlIllIIIIlIlIl .append (_IIIIllllllIIlIllI %(os .path .basename (IlIIllllllIIlIIll ),IlIlIIIIllIIllIIl ));yield _IIIIllIIIIIIIIlII .join (IllIlIllIIIIlIlIl )
		yield _IIIIllIIIIIIIIlII .join (IllIlIllIIIIlIlIl )
	except :yield traceback .format_exc ()
def IllllllllIlIIIlll (IlllIlllIlIlIIIll ,IlIlllIlIlllllIlI ,IIlIIllIllIlIIIII ,IlIlIlIIIIlIIIIII ,IIlIIllIIlllIIIll ,IlIlIIIIlIIIIIIlI ,IllIIlllIIIlllIlI ):
	IIlIlllIIllIIIIll ='streams';IIIIIIlIIIlIIIllI ='onnx_dereverb_By_FoxJoy';IIIIIIIlllIlIIlII =[]
	try :
		IlIlllIlIlllllIlI =IlIlllIlIlllllIlI .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII );IIlIIllIllIlIIIII =IIlIIllIllIlIIIII .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII );IIlIIllIIlllIIIll =IIlIIllIIlllIIIll .strip (_IlllIlIIlIIIlIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IIIIllIIIIIIIIlII ).strip (_IIIIIIIlIlIlIlIlI ).strip (_IlllIlIIlIIIlIlII )
		if IlllIlllIlIlIIIll ==IIIIIIlIIIlIIIllI :from MDXNet import MDXNetDereverb ;IIllIlIIIllIlIIlI =MDXNetDereverb (15 )
		else :IIlllllIIllIIIIll =_audio_pre_ if 'DeEcho'not in IlllIlllIlIlIIIll else _audio_pre_new ;IIllIlIIIllIlIIlI =IIlllllIIllIIIIll (agg =int (IlIlIIIIlIIIIIIlI ),model_path =os .path .join (IIllIIIlIIlllIIII ,IlllIlllIlIlIIIll +_IlIlIlIIIIllIIIll ),device =IIIllIlllIlllllII .device ,is_half =IIIllIlllIlllllII .is_half )
		if IlIlllIlIlllllIlI !='':IlIlIlIIIIlIIIIII =[os .path .join (IlIlllIlIlllllIlI ,IIIIlIlIlIlIlllll )for IIIIlIlIlIlIlllll in os .listdir (IlIlllIlIlllllIlI )]
		else :IlIlIlIIIIlIIIIII =[IlIlIIlllIlIlIIll .name for IlIlIIlllIlIlIIll in IlIlIlIIIIlIIIIII ]
		for IIIIlIIIllIlIIIII in IlIlIlIIIIlIIIIII :
			IIllIlIIIllIIIIlI =os .path .join (IlIlllIlIlllllIlI ,IIIIlIIIllIlIIIII );IllllIIllIIIlllIl =1 ;IlIIIIIlllllllIlI =0 
			try :
				IIIlIllllllIIIIll =ffmpeg .probe (IIllIlIIIllIIIIlI ,cmd ='ffprobe')
				if IIIlIllllllIIIIll [IIlIlllIIllIIIIll ][0 ]['channels']==2 and IIIlIllllllIIIIll [IIlIlllIIllIIIIll ][0 ][_IIlIlIlIIIlIIIlIl ]=='44100':IllllIIllIIIlllIl =0 ;IIllIlIIIllIlIIlI ._path_audio_ (IIllIlIIIllIIIIlI ,IIlIIllIIlllIIIll ,IIlIIllIllIlIIIII ,IllIIlllIIIlllIlI );IlIIIIIlllllllIlI =1 
			except :IllllIIllIIIlllIl =1 ;traceback .print_exc ()
			if IllllIIllIIIlllIl ==1 :IIllIlIIlIlllIlII ='%s/%s.reformatted.wav'%(IllIlllIIlIIlIlIl ,os .path .basename (IIllIlIIIllIIIIlI ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIllIlIIIllIIIIlI ,IIllIlIIlIlllIlII ));IIllIlIIIllIIIIlI =IIllIlIIlIlllIlII 
			try :
				if IlIIIIIlllllllIlI ==0 :IIllIlIIIllIlIIlI ._path_audio_ (IIllIlIIIllIIIIlI ,IIlIIllIIlllIIIll ,IIlIIllIllIlIIIII ,IllIIlllIIIlllIlI )
				IIIIIIIlllIlIIlII .append ('%s->Success'%os .path .basename (IIllIlIIIllIIIIlI ));yield _IIIIllIIIIIIIIlII .join (IIIIIIIlllIlIIlII )
			except :IIIIIIIlllIlIIlII .append (_IIIIllllllIIlIllI %(os .path .basename (IIllIlIIIllIIIIlI ),traceback .format_exc ()));yield _IIIIllIIIIIIIIlII .join (IIIIIIIlllIlIIlII )
	except :IIIIIIIlllIlIIlII .append (traceback .format_exc ());yield _IIIIllIIIIIIIIlII .join (IIIIIIIlllIlIIlII )
	finally :
		try :
			if IlllIlllIlIlIIIll ==IIIIIIlIIIlIIIllI :del IIllIlIIIllIlIIlI .pred .model ;del IIllIlIIIllIlIIlI .pred .model_ 
			else :del IIllIlIIIllIlIIlI .model ;del IIllIlIIIllIlIIlI 
		except :traceback .print_exc ()
		print (_IlllIlIIIIIIllIIl )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IIIIllIIIIIIIIlII .join (IIIIIIIlllIlIIlII )
def IIllllIlIIlIIllIl (IIlIIlIIIIllIllll ):
	IllIIIIlIlIlllllI ='';IlIIlIIIIIllIlllI =os .path .join (_IlIIlllllIIIllIIl ,IIlIIlIIIIllIllll .split (_IIlIIIIlIIIIIllll )[0 ],'')
	for IllllllIlllIIIllI in IlIIlIIllIIIIlIIl :
		if IlIIlIIIIIllIlllI in IllllllIlllIIIllI :IllIIIIlIlIlllllI =IllllllIlllIIIllI ;break 
	return IllIIIIlIlIlllllI 
def IIlIIllIIIIIllllI (IlIlIlIIIlIlllllI ,IlIllIIIlIIIIIIII ,IlIIllIIlllIIlIll ):
	global IlIIIIIIIIllIIIII ,IlIIlIIIlIIlllllI ,IlIIIIIllllIIIlll ,IllIIlllIIlIIlIll ,IIlIlIlllllIIllll ,IIlIlIlllIlIlIllI 
	if IlIlIlIIIlIlllllI ==''or IlIlIlIIIlIlllllI ==[]:
		global IllIIlllIlIIlllll 
		if IllIIlllIlIIlllll is not _IllllllIllIIlIlII :
			print (_IlllIlIIIIIIllIIl );del IlIIIIIllllIIIlll ,IlIIIIIIIIllIIIII ,IllIIlllIIlIIlIll ,IllIIlllIlIIlllll ,IlIIlIIIlIIlllllI ;IllIIlllIlIIlllll =IlIIIIIllllIIIlll =IlIIIIIIIIllIIIII =IllIIlllIIlIIlIll =IllIIlllIlIIlllll =IlIIlIIIlIIlllllI =_IllllllIllIIlIlII 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IIlIlIIllIlllIIIl =IIlIlIlllllIIllll .get (_IIIIIlllIlllIIlll ,1 );IIlIlIlllIlIlIllI =IIlIlIlllllIIllll .get (_IIllIIlllIlIllIll ,_IIIllllIIIIIIIIlI )
			if IIlIlIlllIlIlIllI ==_IIIllllIIIIIIIIlI :
				if IIlIlIIllIlllIIIl ==1 :IlIIIIIllllIIIlll =SynthesizerTrnMs256NSFsid (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ],is_half =IIIllIlllIlllllII .is_half )
				else :IlIIIIIllllIIIlll =SynthesizerTrnMs256NSFsid_nono (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ])
			elif IIlIlIlllIlIlIllI ==_IllIllllIllIlllIl :
				if IIlIlIIllIlllIIIl ==1 :IlIIIIIllllIIIlll =SynthesizerTrnMs768NSFsid (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ],is_half =IIIllIlllIlllllII .is_half )
				else :IlIIIIIllllIIIlll =SynthesizerTrnMs768NSFsid_nono (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ])
			del IlIIIIIllllIIIlll ,IIlIlIlllllIIllll 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IlIIIIIIIllllIIII :_IlllllIlIIlIllllI ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
	IIIlIIIIlIIllIIII =_IIlIllIIlIlllIIll %(IlIIlIlIllIlIllIl ,IlIlIlIIIlIlllllI );print ('loading %s'%IIIlIIIIlIIllIIII );IIlIlIlllllIIllll =torch .load (IIIlIIIIlIIllIIII ,map_location =_IIllIllllIIlIIlII );IlIIlIIIlIIlllllI =IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ][-1 ];IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ][-3 ]=IIlIlIlllllIIllll [_IIIllllllllIIIllI ][_IIllIIllIIIIIlIll ].shape [0 ];IIlIlIIllIlllIIIl =IIlIlIlllllIIllll .get (_IIIIIlllIlllIIlll ,1 )
	if IIlIlIIllIlllIIIl ==0 :IlIllIIIlIIIIIIII =IlIIllIIlllIIlIll ={_IlIIIIIIIllllIIII :_IlllllIlIIlIllllI ,_IllIllllIIIIllIll :.5 ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
	else :IlIllIIIlIIIIIIII ={_IlIIIIIIIllllIIII :_IlIlIlIIIllIIIIll ,_IllIllllIIIIllIll :IlIllIIIlIIIIIIII ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll };IlIIllIIlllIIlIll ={_IlIIIIIIIllllIIII :_IlIlIlIIIllIIIIll ,_IllIllllIIIIllIll :IlIIllIIlllIIlIll ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
	IIlIlIlllIlIlIllI =IIlIlIlllllIIllll .get (_IIllIIlllIlIllIll ,_IIIllllIIIIIIIIlI )
	if IIlIlIlllIlIlIllI ==_IIIllllIIIIIIIIlI :
		if IIlIlIIllIlllIIIl ==1 :IlIIIIIllllIIIlll =SynthesizerTrnMs256NSFsid (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ],is_half =IIIllIlllIlllllII .is_half )
		else :IlIIIIIllllIIIlll =SynthesizerTrnMs256NSFsid_nono (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ])
	elif IIlIlIlllIlIlIllI ==_IllIllllIllIlllIl :
		if IIlIlIIllIlllIIIl ==1 :IlIIIIIllllIIIlll =SynthesizerTrnMs768NSFsid (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ],is_half =IIIllIlllIlllllII .is_half )
		else :IlIIIIIllllIIIlll =SynthesizerTrnMs768NSFsid_nono (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ])
	del IlIIIIIllllIIIlll .enc_q ;print (IlIIIIIllllIIIlll .load_state_dict (IIlIlIlllllIIllll [_IIIllllllllIIIllI ],strict =_IlllllIlIIlIllllI ));IlIIIIIllllIIIlll .eval ().to (IIIllIlllIlllllII .device )
	if IIIllIlllIlllllII .is_half :IlIIIIIllllIIIlll =IlIIIIIllllIIIlll .half ()
	else :IlIIIIIllllIIIlll =IlIIIIIllllIIIlll .float ()
	IllIIlllIIlIIlIll =VC (IlIIlIIIlIIlllllI ,IIIllIlllIlllllII );IlIIIIIIIIllIIIII =IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ][-3 ];return {_IlIIIIIIIllllIIII :_IlIlIlIIIllIIIIll ,'maximum':IlIIIIIIIIllIIIII ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },IlIllIIIlIIIIIIII ,IlIIllIIlllIIlIll ,IIllllIlIIlIIllIl (IlIlIlIIIlIlllllI )
def IIllllIIIlIIIlIlI ():
	IlIIIlIIlIllIIIlI =[]
	for IllIlIlIIlIIIIIII in os .listdir (IlIIlIlIllIlIllIl ):
		if IllIlIlIIlIIIIIII .endswith (_IlIlIlIIIIllIIIll ):IlIIIlIIlIllIIIlI .append (IllIlIlIIlIIIIIII )
	IIIllIlIllIlIllIl =[]
	for (IlIllIIlllIIlIIll ,IIlIlllIIIlIIlllI ,IIIlIlIIllIIllIlI )in os .walk (IllIlIIIIIIIIIlII ,topdown =_IlllllIlIIlIllllI ):
		for IllIlIlIIlIIIIIII in IIIlIlIIllIIllIlI :
			if IllIlIlIIlIIIIIII .endswith (_IlIlIlllllIIIIIII )and _IIllIIIIIIlIlIlIl not in IllIlIlIIlIIIIIII :IIIllIlIllIlIllIl .append (_IIlIllIIlIlllIIll %(IlIllIIlllIIlIIll ,IllIlIlIIlIIIIIII ))
	return {_IIlIIlIIlIIlIIlll :sorted (IlIIIlIIlIllIIIlI ),_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },{_IIlIIlIIlIIlIIlll :sorted (IIIllIlIllIlIllIl ),_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
def IlIIIlllIIlIIIllI ():return {_IllIllllIIIIllIll :'',_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
IlIllIIlllIIIlIlI ={_IlIIllIlIlIIIlIII :32000 ,_IlIlIIIIlIlIllIll :40000 ,_IIlllIllIIlIlllll :48000 }
def IIlIIIIIIllIIIIIl (IIIIlllIIIlIlIIII ,IllIlIllIlllIIIIl ):
	while 1 :
		if IllIlIllIlllIIIIl .poll ()is _IllllllIllIIlIlII :sleep (.5 )
		else :break 
	IIIIlllIIIlIlIIII [0 ]=_IlIlIlIIIllIIIIll 
def IIIIIIIlIlllllIII (IIIIIIlIlIlIlIlII ,IlIIlllIlllllIIII ):
	while 1 :
		IIIIIIIIIllIllIII =1 
		for IlIIlllIIlllIllll in IlIIlllIlllllIIII :
			if IlIIlllIIlllIllll .poll ()is _IllllllIllIIlIlII :IIIIIIIIIllIllIII =0 ;sleep (.5 );break 
		if IIIIIIIIIllIllIII ==1 :break 
	IIIIIIlIlIlIlIlII [0 ]=_IlIlIlIIIllIIIIll 
def IIlIIIlIlllIlIIII (IlIlIIIllIlIIIIll ,IIlIIlllIlIIIlIIl ,IIIllllIIlIIllIll ,IlIllllllIIlllIll ):
	IllIllIlIIlIlIIII ='%s/logs/%s/preprocess.log';IIIllllIIlIIllIll =IlIllIIlllIIIlIlI [IIIllllIIlIIllIll ];os .makedirs (_IllllIIIIlllIIlIl %(IIlIIllIIIIIIIIll ,IIlIIlllIlIIIlIIl ),exist_ok =_IlIlIlIIIllIIIIll );IIllllIlllllIIIll =open (IllIllIlIIlIlIIII %(IIlIIllIIIIIIIIll ,IIlIIlllIlIIIlIIl ),'w');IIllllIlllllIIIll .close ();IlIIllIlIIIlllIIl =IIIllIlllIlllllII .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IlIlIIIllIlIIIIll ,IIIllllIIlIIllIll ,IlIllllllIIlllIll ,IIlIIllIIIIIIIIll ,IIlIIlllIlIIIlIIl )+str (IIIllIlllIlllllII .noparallel );print (IlIIllIlIIIlllIIl );IIlIIlIIllIIIllIl =Popen (IlIIllIlIIIlllIIl ,shell =_IlIlIlIIIllIIIIll );IIIIllIlIllllIlll =[_IlllllIlIIlIllllI ];threading .Thread (target =IIlIIIIIIllIIIIIl ,args =(IIIIllIlIllllIlll ,IIlIIlIIllIIIllIl )).start ()
	while 1 :
		with open (IllIllIlIIlIlIIII %(IIlIIllIIIIIIIIll ,IIlIIlllIlIIIlIIl ),_IlllIllIlllIllIIl )as IIllllIlllllIIIll :yield IIllllIlllllIIIll .read ()
		sleep (1 )
		if IIIIllIlIllllIlll [0 ]:break 
	with open (IllIllIlIIlIlIIII %(IIlIIllIIIIIIIIll ,IIlIIlllIlIIIlIIl ),_IlllIllIlllIllIIl )as IIllllIlllllIIIll :IlIlllIIlIlllIIII =IIllllIlllllIIIll .read ()
	print (IlIlllIIlIlllIIII );yield IlIlllIIlIlllIIII 
def IIIIlIIllIlIlIIII (IIllIlllIllIIIIII ,IllIIlIIIIIllIIII ,IIlIllIIIlllllIIl ,IIIlIIlIIIIIlllII ,IIIlIllIllIIIIlll ,IlIIlIllIIIIIIlIl ,IIIlIlIIlllIlIIlI ):
	IlIIIIIlllIlIIllI ='%s/logs/%s/extract_fl_feature.log';IIllIlllIllIIIIII =IIllIlllIllIIIIII .split ('-');os .makedirs (_IllllIIIIlllIIlIl %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),exist_ok =_IlIlIlIIIllIIIIll );IllIIIlIIlIIlIIll =open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),'w');IllIIIlIIlIIlIIll .close ()
	if IIIlIIlIIIIIlllII :
		if IIlIllIIIlllllIIl !=_IlIllIllIIlllIlII :
			IIIIlIlIllIIlIIll =IIIllIlllIlllllII .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ,IllIIlIIIIIllIIII ,IIlIllIIIlllllIIl );print (IIIIlIlIllIIlIIll );IIIIIIllIllllIllI =Popen (IIIIlIlIllIIlIIll ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IIIlIIllIllllIlII =[_IlllllIlIIlIllllI ];threading .Thread (target =IIlIIIIIIllIIIIIl ,args =(IIIlIIllIllllIlII ,IIIIIIllIllllIllI )).start ()
			while 1 :
				with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :yield IllIIIlIIlIIlIIll .read ()
				sleep (1 )
				if IIIlIIllIllllIlII [0 ]:break 
			with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :IIllllIllllIlIlII =IllIIIlIIlIIlIIll .read ()
			print (IIllllIllllIlIlII );yield IIllllIllllIlIlII 
		else :
			IIIlIlIIlllIlIIlI =IIIlIlIIlllIlIIlI .split ('-');IlIllIlIIIIIIIIII =len (IIIlIlIIlllIlIIlI );IlllIllIlIlIlIlIl =[]
			for (IlllIIIIllIIlIIII ,IlIlIIIIlllIlIllI )in enumerate (IIIlIlIIlllIlIIlI ):IIIIlIlIllIIlIIll =IIIllIlllIlllllII .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IlIllIlIIIIIIIIII ,IlllIIIIllIIlIIII ,IlIlIIIIlllIlIllI ,IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ,IIIllIlllIlllllII .is_half );print (IIIIlIlIllIIlIIll );IIIIIIllIllllIllI =Popen (IIIIlIlIllIIlIIll ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IlllIllIlIlIlIlIl .append (IIIIIIllIllllIllI )
			IIIlIIllIllllIlII =[_IlllllIlIIlIllllI ];threading .Thread (target =IIIIIIIlIlllllIII ,args =(IIIlIIllIllllIlII ,IlllIllIlIlIlIlIl )).start ()
			while 1 :
				with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :yield IllIIIlIIlIIlIIll .read ()
				sleep (1 )
				if IIIlIIllIllllIlII [0 ]:break 
			with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :IIllllIllllIlIlII =IllIIIlIIlIIlIIll .read ()
			print (IIllllIllllIlIlII );yield IIllllIllllIlIlII 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IlIllIlIIIIIIIIII =len (IIllIlllIllIIIIII );IlllIllIlIlIlIlIl =[]
	for (IlllIIIIllIIlIIII ,IlIlIIIIlllIlIllI )in enumerate (IIllIlllIllIIIIII ):IIIIlIlIllIIlIIll =IIIllIlllIlllllII .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IIIllIlllIlllllII .device ,IlIllIlIIIIIIIIII ,IlllIIIIllIIlIIII ,IlIlIIIIlllIlIllI ,IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ,IlIIlIllIIIIIIlIl );print (IIIIlIlIllIIlIIll );IIIIIIllIllllIllI =Popen (IIIIlIlIllIIlIIll ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IlllIllIlIlIlIlIl .append (IIIIIIllIllllIllI )
	IIIlIIllIllllIlII =[_IlllllIlIIlIllllI ];threading .Thread (target =IIIIIIIlIlllllIII ,args =(IIIlIIllIllllIlII ,IlllIllIlIlIlIlIl )).start ()
	while 1 :
		with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :yield IllIIIlIIlIIlIIll .read ()
		sleep (1 )
		if IIIlIIllIllllIlII [0 ]:break 
	with open (IlIIIIIlllIlIIllI %(IIlIIllIIIIIIIIll ,IIIlIllIllIIIIlll ),_IlllIllIlllIllIIl )as IllIIIlIIlIIlIIll :IIllllIllllIlIlII =IllIIIlIIlIIlIIll .read ()
	print (IIllllIllllIlIlII );yield IIllllIllllIlIlII 
def IlllllIIlIllIllIl (IlIlIlIllIIIIIIII ,IlIIlIllIlIlIIIII ,IIllIIllIlllIIllI ):
	IIIllIllIIllIIIIl =''if IIllIIllIlllIIllI ==_IIIllllIIIIIIIIlI else _IIllIlIlIIIIIIIll ;IIIIllIlIllllIllI =_IIIIIlllIlllIIlll if IlIIlIllIlIlIIIII else '';IllllIIlIIlIIIIll =os .access (_IIllllIlllIIlIIll %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII ),os .F_OK );IIIIlIllIIlIlIllI =os .access (_IlIlIllIIIIlIlIII %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII ),os .F_OK )
	if not IllllIIlIIlIIIIll :print (_IIllllIlllIIlIIll %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII ),_IIIIIllIIIllIlIII )
	if not IIIIlIllIIlIlIllI :print (_IlIlIllIIIIlIlIII %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII ),_IIIIIllIIIllIlIII )
	return _IIllllIlllIIlIIll %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII )if IllllIIlIIlIIIIll else '',_IlIlIllIIIIlIlIII %(IIIllIllIIllIIIIl ,IIIIllIlIllllIllI ,IlIlIlIllIIIIIIII )if IIIIlIllIIlIlIllI else ''
def IllIlIlIIllllIlll (IIIlIIIlllIIIlIII ,IIIllllIIIIlIIIIl ,IIlllIlllIIllIIIl ):
	IIllllIIlIllIIlll =''if IIlllIlllIIllIIIl ==_IIIllllIIIIIIIIlI else _IIllIlIlIIIIIIIll 
	if IIIlIIIlllIIIlIII ==_IlIIllIlIlIIIlIII and IIlllIlllIIllIIIl ==_IIIllllIIIIIIIIlI :IIIlIIIlllIIIlIII =_IlIlIIIIlIlIllIll 
	IlIIlIIlIllIIllll ={_IIlIIlIIlIIlIIlll :[_IlIlIIIIlIlIllIll ,_IIlllIllIIlIlllll ],_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll ,_IllIllllIIIIllIll :IIIlIIIlllIIIlIII }if IIlllIlllIIllIIIl ==_IIIllllIIIIIIIIlI else {_IIlIIlIIlIIlIIlll :[_IlIlIIIIlIlIllIll ,_IIlllIllIIlIlllll ,_IlIIllIlIlIIIlIII ],_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll ,_IllIllllIIIIllIll :IIIlIIIlllIIIlIII };IlIlIIllIllllllll =_IIIIIlllIlllIIlll if IIIllllIIIIlIIIIl else '';IIllIlIlIlIIIIllI =os .access (_IIllllIlllIIlIIll %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII ),os .F_OK );IllIllIlIlllIIlIl =os .access (_IlIlIllIIIIlIlIII %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII ),os .F_OK )
	if not IIllIlIlIlIIIIllI :print (_IIllllIlllIIlIIll %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII ),_IIIIIllIIIllIlIII )
	if not IllIllIlIlllIIlIl :print (_IlIlIllIIIIlIlIII %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII ),_IIIIIllIIIllIlIII )
	return _IIllllIlllIIlIIll %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII )if IIllIlIlIlIIIIllI else '',_IlIlIllIIIIlIlIII %(IIllllIIlIllIIlll ,IlIlIIllIllllllll ,IIIlIIIlllIIIlIII )if IllIllIlIlllIIlIl else '',IlIIlIIlIllIIllll 
def IllIlllIIIIIllIlI (IlIIllIlllIlIlIll ,IIIllllIlIlIIllll ,IIIllllIIIllIllII ):
	IlIllIIIIlIllIlII ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IlIlIllllIIllIIlI ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IlIlllIIIIlllllIl =''if IIIllllIIIllIllII ==_IIIllllIIIIIIIIlI else _IIllIlIlIIIIIIIll ;IlIIllIlIIllllIII =os .access (IlIlIllllIIllIIlI %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll ),os .F_OK );IlllllIlIllIIIlII =os .access (IlIllIIIIlIllIlII %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll ),os .F_OK )
	if not IlIIllIlIIllllIII :print (IlIlIllllIIllIIlI %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll ),_IIIIIllIIIllIlIII )
	if not IlllllIlIllIIIlII :print (IlIllIIIIlIllIlII %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll ),_IIIIIllIIIllIlIII )
	if IlIIllIlllIlIlIll :return {_IlIIIIIIIllllIIII :_IlIlIlIIIllIIIIll ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },IlIlIllllIIllIIlI %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll )if IlIIllIlIIllllIII else '',IlIllIIIIlIllIlII %(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll )if IlllllIlIllIIIlII else ''
	return {_IlIIIIIIIllllIIII :_IlllllIlIIlIllllI ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll )if IlIIllIlIIllllIII else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IlIlllIIIIlllllIl ,IIIllllIlIlIIllll )if IlllllIlIllIIIlII else ''
def IlIIIIlIIllllIIIl (IIIIIIIIIlIIlllll ,IllIIlIIllIllIIII ,IIIIIllIlllIIIllI ,IIllIIlIIllIlIlIl ,IllIIIIIlIlIlllII ,IllIllIlIllIIlIlI ,IlIllllIIIIlIIllI ,IIlIllIIlllIIIllI ,IlIIllIllIIIlIIIl ,IIlIllIIlllIllllI ,IlIIIlIllIIIIlllI ,IIIlIIlIIIlIIlllI ,IIIlIlIIIIlIIlllI ,IIIIIIlIlIllIIIlI ):
	IIlIllIlIIllllllI ='\x08';IIllIlllIIlIIIIII =_IllllIIIIlllIIlIl %(IIlIIllIIIIIIIIll ,IIIIIIIIIlIIlllll );os .makedirs (IIllIlllIIlIIIIII ,exist_ok =_IlIlIlIIIllIIIIll );IllIIIIlIIIlIllIl =_IlIIIllIIlIlIIlII %IIllIlllIIlIIIIII ;IIIIllIIIIlIlllll =_IlIIIlIIlIllllIll %IIllIlllIIlIIIIII if IIIIIIlIlIllIIIlI ==_IIIllllIIIIIIIIlI else _IlIIlIlIIlIllIllI %IIllIlllIIlIIIIII 
	if IIIIIllIlllIIIllI :IllIIIlllllIlIlII ='%s/2a_f0'%IIllIlllIIlIIIIII ;IlIlIllllIllIlIlI =_IlIIlIlIIIIIIllII %IIllIlllIIlIIIIII ;IllIIllllIIllIIlI =set ([IIIIIlIIlllllIIII .split (_IIlIIIIlIIIIIllll )[0 ]for IIIIIlIIlllllIIII in os .listdir (IllIIIIlIIIlIllIl )])&set ([IIlIIIIlIlIIllIIl .split (_IIlIIIIlIIIIIllll )[0 ]for IIlIIIIlIlIIllIIl in os .listdir (IIIIllIIIIlIlllll )])&set ([IlllIIIIIIIIIIIlI .split (_IIlIIIIlIIIIIllll )[0 ]for IlllIIIIIIIIIIIlI in os .listdir (IllIIIlllllIlIlII )])&set ([IlIllIIIlIlIlIIlI .split (_IIlIIIIlIIIIIllll )[0 ]for IlIllIIIlIlIlIIlI in os .listdir (IlIlIllllIllIlIlI )])
	else :IllIIllllIIllIIlI =set ([IIIlIlIlIlIlIIlll .split (_IIlIIIIlIIIIIllll )[0 ]for IIIlIlIlIlIlIIlll in os .listdir (IllIIIIlIIIlIllIl )])&set ([IllIlIIIllIIIlIIl .split (_IIlIIIIlIIIIIllll )[0 ]for IllIlIIIllIIIlIIl in os .listdir (IIIIllIIIIlIlllll )])
	IlIIlllIllIllllII =[]
	for IllIIIIIIIlIIIlII in IllIIllllIIllIIlI :
		if IIIIIllIlllIIIllI :IlIIlllIllIllllII .append (_IllIlIlIIIlIIlllI %(IllIIIIlIIIlIllIl .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IIIIllIIIIlIlllll .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IllIIIlllllIlIlII .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IlIlIllllIllIlIlI .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IIllIIlIIllIlIlIl ))
		else :IlIIlllIllIllllII .append (_IIIlIIllllIlIIlII %(IllIIIIlIIIlIllIl .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IIIIllIIIIlIlllll .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IllIIIIIIIlIIIlII ,IIllIIlIIllIlIlIl ))
	IllIIIIlIlIlIIIlI =256 if IIIIIIlIlIllIIIlI ==_IIIllllIIIIIIIIlI else 768 
	if IIIIIllIlllIIIllI :
		for _IlllIlIlIIlIIlIIl in range (2 ):IlIIlllIllIllllII .append (_IlIllIIIllIIllIII %(IIlIIllIIIIIIIIll ,IllIIlIIllIllIIII ,IIlIIllIIIIIIIIll ,IllIIIIlIlIlIIIlI ,IIlIIllIIIIIIIIll ,IIlIIllIIIIIIIIll ,IIllIIlIIllIlIlIl ))
	else :
		for _IlllIlIlIIlIIlIIl in range (2 ):IlIIlllIllIllllII .append (_IlllIllllIIlIlIIl %(IIlIIllIIIIIIIIll ,IllIIlIIllIllIIII ,IIlIIllIIIIIIIIll ,IllIIIIlIlIlIIIlI ,IIllIIlIIllIlIlIl ))
	shuffle (IlIIlllIllIllllII )
	with open (_IIIlIlIlIlIIIllll %IIllIlllIIlIIIIII ,'w')as IllIlIllIIlllIlII :IllIlIllIIlllIlII .write (_IIIIllIIIIIIIIlII .join (IlIIlllIllIllllII ))
	print (_IIIllIllllIllIIlI );print ('use gpus:',IlIIIlIllIIIIlllI )
	if IlIIllIllIIIlIIIl =='':print ('no pretrained Generator')
	if IIlIllIIlllIllllI =='':print ('no pretrained Discriminator')
	if IlIIIlIllIIIIlllI :IIlIIIlIIIIIlllII =IIIllIlllIlllllII .python_cmd +_IllIlIllIllllIIll %(IIIIIIIIIlIIlllll ,IllIIlIIllIllIIII ,1 if IIIIIllIlllIIIllI else 0 ,IlIllllIIIIlIIllI ,IlIIIlIllIIIIlllI ,IllIllIlIllIIlIlI ,IllIIIIIlIlIlllII ,_IlllIIIlIlIlIIlIl %IlIIllIllIIIlIIIl if IlIIllIllIIIlIIIl !=''else '',_IllIIIIllllIlIllI %IIlIllIIlllIllllI if IIlIllIIlllIllllI !=''else '',1 if IIlIllIIlllIIIllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IIIlIIlIIIlIIlllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IIIlIlIIIIlIIlllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,IIIIIIlIlIllIIIlI )
	else :IIlIIIlIIIIIlllII =IIIllIlllIlllllII .python_cmd +_IIlIllIlIIllIIIII %(IIIIIIIIIlIIlllll ,IllIIlIIllIllIIII ,1 if IIIIIllIlllIIIllI else 0 ,IlIllllIIIIlIIllI ,IllIllIlIllIIlIlI ,IllIIIIIlIlIlllII ,_IlllIIIlIlIlIIlIl %IlIIllIllIIIlIIIl if IlIIllIllIIIlIIIl !=''else IIlIllIlIIllllllI ,_IllIIIIllllIlIllI %IIlIllIIlllIllllI if IIlIllIIlllIllllI !=''else IIlIllIlIIllllllI ,1 if IIlIllIIlllIIIllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IIIlIIlIIIlIIlllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IIIlIlIIIIlIIlllI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,IIIIIIlIlIllIIIlI )
	print (IIlIIIlIIIIIlllII );IllIIIlllIIllIllI =Popen (IIlIIIlIIIIIlllII ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IllIIIlllIIllIllI .wait ();return _IlIIllIIllIIllIII 
def IlIlIIllllllIIIII (IllllIIlIlIllIIIl ,IIIllllIlIlIIlIll ):
	IllIIIIIlllIIIIIl =_IllllIIIIlllIIlIl %(IIlIIllIIIIIIIIll ,IllllIIlIlIllIIIl );os .makedirs (IllIIIIIlllIIIIIl ,exist_ok =_IlIlIlIIIllIIIIll );IlIIIIIIIllIllIII =_IlIIIlIIlIllllIll %IllIIIIIlllIIIIIl if IIIllllIlIlIIlIll ==_IIIllllIIIIIIIIlI else _IlIIlIlIIlIllIllI %IllIIIIIlllIIIIIl 
	if not os .path .exists (IlIIIIIIIllIllIII ):return '请先进行特征提取!'
	IIIIllllIllIlIIll =list (os .listdir (IlIIIIIIIllIllIII ))
	if len (IIIIllllIllIlIIll )==0 :return '请先进行特征提取！'
	IIlllllIIIlIlIlII =[];IIIIIlIlIIIIIlIII =[]
	for IlIIlIlIIlIlllllI in sorted (IIIIllllIllIlIIll ):IlIlIIllIIlIIlIlI =np .load (_IIlIllIIlIlllIIll %(IlIIIIIIIllIllIII ,IlIIlIlIIlIlllllI ));IIIIIlIlIIIIIlIII .append (IlIlIIllIIlIIlIlI )
	IIIlIlIlIllIllIll =np .concatenate (IIIIIlIlIIIIIlIII ,0 );IlllIIIllIIlIIlll =np .arange (IIIlIlIlIllIllIll .shape [0 ]);np .random .shuffle (IlllIIIllIIlIIlll );IIIlIlIlIllIllIll =IIIlIlIlIllIllIll [IlllIIIllIIlIIlll ]
	if IIIlIlIlIllIllIll .shape [0 ]>2e5 :
		IIlllllIIIlIlIlII .append (_IlllIIlllIIllIIll %IIIlIlIlIllIllIll .shape [0 ]);yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII )
		try :IIIlIlIlIllIllIll =MiniBatchKMeans (n_clusters =10000 ,verbose =_IlIlIlIIIllIIIIll ,batch_size =256 *IIIllIlllIlllllII .n_cpu ,compute_labels =_IlllllIlIIlIllllI ,init ='random').fit (IIIlIlIlIllIllIll ).cluster_centers_ 
		except :IlllllIIlIlllIIll =traceback .format_exc ();print (IlllllIIlIlllIIll );IIlllllIIIlIlIlII .append (IlllllIIlIlllIIll );yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII )
	np .save (_IIllIlIlIlIIlIIII %IllIIIIIlllIIIIIl ,IIIlIlIlIllIllIll );IIIIllIlIIlIIllIl =min (int (16 *np .sqrt (IIIlIlIlIllIllIll .shape [0 ])),IIIlIlIlIllIllIll .shape [0 ]//39 );IIlllllIIIlIlIlII .append ('%s,%s'%(IIIlIlIlIllIllIll .shape ,IIIIllIlIIlIIllIl ));yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII );IIIlllIlllIllIllI =faiss .index_factory (256 if IIIllllIlIlIIlIll ==_IIIllllIIIIIIIIlI else 768 ,_IlIllIlllIIlIIllI %IIIIllIlIIlIIllIl );IIlllllIIIlIlIlII .append ('training');yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII );IIIIIlIIIIlIIIlll =faiss .extract_index_ivf (IIIlllIlllIllIllI );IIIIIlIIIIlIIIlll .nprobe =1 ;IIIlllIlllIllIllI .train (IIIlIlIlIllIllIll );faiss .write_index (IIIlllIlllIllIllI ,_IlIIIIIllIlllllIl %(IllIIIIIlllIIIIIl ,IIIIllIlIIlIIllIl ,IIIIIlIIIIlIIIlll .nprobe ,IllllIIlIlIllIIIl ,IIIllllIlIlIIlIll ));IIlllllIIIlIlIlII .append ('adding');yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII );IIlIIllIIllIIIlll =8192 
	for IIIIIIIllIIIlllll in range (0 ,IIIlIlIlIllIllIll .shape [0 ],IIlIIllIIllIIIlll ):IIIlllIlllIllIllI .add (IIIlIlIlIllIllIll [IIIIIIIllIIIlllll :IIIIIIIllIIIlllll +IIlIIllIIllIIIlll ])
	faiss .write_index (IIIlllIlllIllIllI ,_IlllIIIllIlIlIIll %(IllIIIIIlllIIIIIl ,IIIIllIlIIlIIllIl ,IIIIIlIIIIlIIIlll .nprobe ,IllllIIlIlIllIIIl ,IIIllllIlIlIIlIll ));IIlllllIIIlIlIlII .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIIIllIlIIlIIllIl ,IIIIIlIIIIlIIIlll .nprobe ,IllllIIlIlIllIIIl ,IIIllllIlIlIIlIll ));yield _IIIIllIIIIIIIIlII .join (IIlllllIIIlIlIlII )
def IIIIIIlllllIIlllI (IlIlIIIIIIIIlIIll ,IIlllllIllIllIIlI ,IlllllllIllllIIll ,IlIlIIIlIlllIIIII ,IlllIlllIIlllIlll ,IIIlllllIIIIlllII ,IIlIlIIlIlIlIIIll ,IllIllllllllllllI ,IlllllllIIIIIllll ,IIllllIIIlIIllllI ,IllIlllllllllIIII ,IIlIllIllIIIlllII ,IlIlIlllllllIlllI ,IIIllIllIIllIlIIl ,IlIlllIlIlIlIlIlI ,IlllIIlIllIIIIIII ,IlIIIIlIIlIIIlllI ,IIlllIIlIllIIllll ):
	IlIIlIIIIIllIlIIl =[]
	def IIIIllllIIIllllll (IllIIllIIIIlIIIll ):IlIIlIIIIIllIlIIl .append (IllIIllIIIIlIIIll );return _IIIIllIIIIIIIIlII .join (IlIIlIIIIIllIlIIl )
	IllIIIIIlIIllIlII =_IllllIIIIlllIIlIl %(IIlIIllIIIIIIIIll ,IlIlIIIIIIIIlIIll );IIIIIIIIIlllIIIII ='%s/preprocess.log'%IllIIIIIlIIllIlII ;IIIlIIIIIllIlllIl ='%s/extract_fl_feature.log'%IllIIIIIlIIllIlII ;IIlIIIIIIIlIIIIIl =_IlIIIllIIlIlIIlII %IllIIIIIlIIllIlII ;IlIIIIIlIllIIllll =_IlIIIlIIlIllllIll %IllIIIIIlIIllIlII if IlIIIIlIIlIIIlllI ==_IIIllllIIIIIIIIlI else _IlIIlIlIIlIllIllI %IllIIIIIlIIllIlII ;os .makedirs (IllIIIIIlIIllIlII ,exist_ok =_IlIlIlIIIllIIIIll );open (IIIIIIIIIlllIIIII ,'w').close ();IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IlIlIIIlIlllIIIII ,IlIllIIlllIIIlIlI [IIlllllIllIllIIlI ],IIIlllllIIIIlllII ,IllIIIIIlIIllIlII )+str (IIIllIlllIlllllII .noparallel );yield IIIIllllIIIllllll (IlIIIIllIllIIlllI ('step1:正在处理数据'));yield IIIIllllIIIllllll (IllIlllIIIllIIIlI );IIlllIllIIlllIlll =Popen (IllIlllIIIllIIIlI ,shell =_IlIlIlIIIllIIIIll );IIlllIllIIlllIlll .wait ()
	with open (IIIIIIIIIlllIIIII ,_IlllIllIlllIllIIl )as IlllIIIIIlIllIlll :print (IlllIIIIIlIllIlll .read ())
	open (IIIlIIIIIllIlllIl ,'w')
	if IlllllllIllllIIll :
		yield IIIIllllIIIllllll ('step2a:正在提取音高')
		if IIlIlIIlIlIlIIIll !=_IlIllIllIIlllIlII :IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +' extract_fl_print.py "%s" %s %s'%(IllIIIIIlIIllIlII ,IIIlllllIIIIlllII ,IIlIlIIlIlIlIIIll );yield IIIIllllIIIllllll (IllIlllIIIllIIIlI );IIlllIllIIlllIlll =Popen (IllIlllIIIllIIIlI ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IIlllIllIIlllIlll .wait ()
		else :
			IIlllIIlIllIIllll =IIlllIIlIllIIllll .split ('-');IlllllIIIIIIIIIIl =len (IIlllIIlIllIIllll );IllIIIlIIIlIlIIII =[]
			for (IllIIlIlllIIlllll ,IlIIIlIIIllIlllIl )in enumerate (IIlllIIlIllIIllll ):IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IlllllIIIIIIIIIIl ,IllIIlIlllIIlllll ,IlIIIlIIIllIlllIl ,IllIIIIIlIIllIlII ,IIIllIlllIlllllII .is_half );yield IIIIllllIIIllllll (IllIlllIIIllIIIlI );IIlllIllIIlllIlll =Popen (IllIlllIIIllIIIlI ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IllIIIlIIIlIlIIII .append (IIlllIllIIlllIlll )
			for IIlllIllIIlllIlll in IllIIIlIIIlIlIIII :IIlllIllIIlllIlll .wait ()
		with open (IIIlIIIIIllIlllIl ,_IlllIllIlllIllIIl )as IlllIIIIIlIllIlll :print (IlllIIIIIlIllIlll .read ())
	else :yield IIIIllllIIIllllll (IlIIIIllIllIIlllI ('step2a:无需提取音高'))
	yield IIIIllllIIIllllll (IlIIIIllIllIIlllI ('step2b:正在提取特征'));IlIIIIIlllIIlIIlI =IIIllIllIIllIlIIl .split ('-');IlllllIIIIIIIIIIl =len (IlIIIIIlllIIlIIlI );IllIIIlIIIlIlIIII =[]
	for (IllIIlIlllIIlllll ,IlIIIlIIIllIlllIl )in enumerate (IlIIIIIlllIIlIIlI ):IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IIIllIlllIlllllII .device ,IlllllIIIIIIIIIIl ,IllIIlIlllIIlllll ,IlIIIlIIIllIlllIl ,IllIIIIIlIIllIlII ,IlIIIIlIIlIIIlllI );yield IIIIllllIIIllllll (IllIlllIIIllIIIlI );IIlllIllIIlllIlll =Popen (IllIlllIIIllIIIlI ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IllIIIlIIIlIlIIII .append (IIlllIllIIlllIlll )
	for IIlllIllIIlllIlll in IllIIIlIIIlIlIIII :IIlllIllIIlllIlll .wait ()
	with open (IIIlIIIIIllIlllIl ,_IlllIllIlllIllIIl )as IlllIIIIIlIllIlll :print (IlllIIIIIlIllIlll .read ())
	yield IIIIllllIIIllllll (IlIIIIllIllIIlllI ('step3a:正在训练模型'))
	if IlllllllIllllIIll :IlllIlllIlIlIIlII ='%s/2a_f0'%IllIIIIIlIIllIlII ;IIlllIIIIlIllIllI =_IlIIlIlIIIIIIllII %IllIIIIIlIIllIlII ;IIllllllIIIlIIlII =set ([IllllIIllllIlIIIl .split (_IIlIIIIlIIIIIllll )[0 ]for IllllIIllllIlIIIl in os .listdir (IIlIIIIIIIlIIIIIl )])&set ([IIIIIlIlIlIIIlIIl .split (_IIlIIIIlIIIIIllll )[0 ]for IIIIIlIlIlIIIlIIl in os .listdir (IlIIIIIlIllIIllll )])&set ([IllIIlIIllIlIIIlI .split (_IIlIIIIlIIIIIllll )[0 ]for IllIIlIIllIlIIIlI in os .listdir (IlllIlllIlIlIIlII )])&set ([IIlIlIlllIIlllIIl .split (_IIlIIIIlIIIIIllll )[0 ]for IIlIlIlllIIlllIIl in os .listdir (IIlllIIIIlIllIllI )])
	else :IIllllllIIIlIIlII =set ([IllllIIlllllIIIII .split (_IIlIIIIlIIIIIllll )[0 ]for IllllIIlllllIIIII in os .listdir (IIlIIIIIIIlIIIIIl )])&set ([IlIlIIIIlIIIllIII .split (_IIlIIIIlIIIIIllll )[0 ]for IlIlIIIIlIIIllIII in os .listdir (IlIIIIIlIllIIllll )])
	IlIlIlIllllIIlIlI =[]
	for IIlIIlIllIIIllIIl in IIllllllIIIlIIlII :
		if IlllllllIllllIIll :IlIlIlIllllIIlIlI .append (_IllIlIlIIIlIIlllI %(IIlIIIIIIIlIIIIIl .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IlIIIIIlIllIIllll .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IlllIlllIlIlIIlII .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IIlllIIIIlIllIllI .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IlllIlllIIlllIlll ))
		else :IlIlIlIllllIIlIlI .append (_IIIlIIllllIlIIlII %(IIlIIIIIIIlIIIIIl .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IlIIIIIlIllIIllll .replace (_IIIlIlIlllIIIIlIl ,_IlllllIllllllIIlI ),IIlIIlIllIIIllIIl ,IlllIlllIIlllIlll ))
	IllIIlIIIlIIllIII =256 if IlIIIIlIIlIIIlllI ==_IIIllllIIIIIIIIlI else 768 
	if IlllllllIllllIIll :
		for _IIIIllIllIlIllIIl in range (2 ):IlIlIlIllllIIlIlI .append (_IlIllIIIllIIllIII %(IIlIIllIIIIIIIIll ,IIlllllIllIllIIlI ,IIlIIllIIIIIIIIll ,IllIIlIIIlIIllIII ,IIlIIllIIIIIIIIll ,IIlIIllIIIIIIIIll ,IlllIlllIIlllIlll ))
	else :
		for _IIIIllIllIlIllIIl in range (2 ):IlIlIlIllllIIlIlI .append (_IlllIllllIIlIlIIl %(IIlIIllIIIIIIIIll ,IIlllllIllIllIIlI ,IIlIIllIIIIIIIIll ,IllIIlIIIlIIllIII ,IlllIlllIIlllIlll ))
	shuffle (IlIlIlIllllIIlIlI )
	with open (_IIIlIlIlIlIIIllll %IllIIIIIlIIllIlII ,'w')as IlllIIIIIlIllIlll :IlllIIIIIlIllIlll .write (_IIIIllIIIIIIIIlII .join (IlIlIlIllllIIlIlI ))
	yield IIIIllllIIIllllll (_IIIllIllllIllIIlI )
	if IIIllIllIIllIlIIl :IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +_IllIlIllIllllIIll %(IlIlIIIIIIIIlIIll ,IIlllllIllIllIIlI ,1 if IlllllllIllllIIll else 0 ,IIllllIIIlIIllllI ,IIIllIllIIllIlIIl ,IlllllllIIIIIllll ,IllIllllllllllllI ,_IlllIIIlIlIlIIlIl %IIlIllIllIIIlllII if IIlIllIllIIIlllII !=''else '',_IllIIIIllllIlIllI %IlIlIlllllllIlllI if IlIlIlllllllIlllI !=''else '',1 if IllIlllllllllIIII ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IlIlllIlIlIlIlIlI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IlllIIlIllIIIIIII ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,IlIIIIlIIlIIIlllI )
	else :IllIlllIIIllIIIlI =IIIllIlllIlllllII .python_cmd +_IIlIllIlIIllIIIII %(IlIlIIIIIIIIlIIll ,IIlllllIllIllIIlI ,1 if IlllllllIllllIIll else 0 ,IIllllIIIlIIllllI ,IlllllllIIIIIllll ,IllIllllllllllllI ,_IlllIIIlIlIlIIlIl %IIlIllIllIIIlllII if IIlIllIllIIIlllII !=''else '',_IllIIIIllllIlIllI %IlIlIlllllllIlllI if IlIlIlllllllIlllI !=''else '',1 if IllIlllllllllIIII ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IlIlllIlIlIlIlIlI ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,1 if IlllIIlIllIIIIIII ==IlIIIIllIllIIlllI (_IlIIllIIIIllIllII )else 0 ,IlIIIIlIIlIIIlllI )
	yield IIIIllllIIIllllll (IllIlllIIIllIIIlI );IIlllIllIIlllIlll =Popen (IllIlllIIIllIIIlI ,shell =_IlIlIlIIIllIIIIll ,cwd =IIlIIllIIIIIIIIll );IIlllIllIIlllIlll .wait ();yield IIIIllllIIIllllll (IlIIIIllIllIIlllI (_IlIIllIIllIIllIII ));IllIIlIIIIlIIIlIl =[];IIIlIllIlIIllIIlI =list (os .listdir (IlIIIIIlIllIIllll ))
	for IIlIIlIllIIIllIIl in sorted (IIIlIllIlIIllIIlI ):IIlIlIlIlllIIIIll =np .load (_IIlIllIIlIlllIIll %(IlIIIIIlIllIIllll ,IIlIIlIllIIIllIIl ));IllIIlIIIIlIIIlIl .append (IIlIlIlIlllIIIIll )
	IIlIIIIlIIllIlllI =np .concatenate (IllIIlIIIIlIIIlIl ,0 );IlIIIllIlIIIlIIlI =np .arange (IIlIIIIlIIllIlllI .shape [0 ]);np .random .shuffle (IlIIIllIlIIIlIIlI );IIlIIIIlIIllIlllI =IIlIIIIlIIllIlllI [IlIIIllIlIIIlIIlI ]
	if IIlIIIIlIIllIlllI .shape [0 ]>2e5 :
		IIlIlIlllIIlIlIIl =_IlllIIlllIIllIIll %IIlIIIIlIIllIlllI .shape [0 ];print (IIlIlIlllIIlIlIIl );yield IIIIllllIIIllllll (IIlIlIlllIIlIlIIl )
		try :IIlIIIIlIIllIlllI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IlIlIlIIIllIIIIll ,batch_size =256 *IIIllIlllIlllllII .n_cpu ,compute_labels =_IlllllIlIIlIllllI ,init ='random').fit (IIlIIIIlIIllIlllI ).cluster_centers_ 
		except :IIlIlIlllIIlIlIIl =traceback .format_exc ();print (IIlIlIlllIIlIlIIl );yield IIIIllllIIIllllll (IIlIlIlllIIlIlIIl )
	np .save (_IIllIlIlIlIIlIIII %IllIIIIIlIIllIlII ,IIlIIIIlIIllIlllI );IIIIllllIlIllIlIl =min (int (16 *np .sqrt (IIlIIIIlIIllIlllI .shape [0 ])),IIlIIIIlIIllIlllI .shape [0 ]//39 );yield IIIIllllIIIllllll ('%s,%s'%(IIlIIIIlIIllIlllI .shape ,IIIIllllIlIllIlIl ));IIIlIIllIIlIllIII =faiss .index_factory (256 if IlIIIIlIIlIIIlllI ==_IIIllllIIIIIIIIlI else 768 ,_IlIllIlllIIlIIllI %IIIIllllIlIllIlIl );yield IIIIllllIIIllllll ('training index');IlIIIllIllllllIlI =faiss .extract_index_ivf (IIIlIIllIIlIllIII );IlIIIllIllllllIlI .nprobe =1 ;IIIlIIllIIlIllIII .train (IIlIIIIlIIllIlllI );faiss .write_index (IIIlIIllIIlIllIII ,_IlIIIIIllIlllllIl %(IllIIIIIlIIllIlII ,IIIIllllIlIllIlIl ,IlIIIllIllllllIlI .nprobe ,IlIlIIIIIIIIlIIll ,IlIIIIlIIlIIIlllI ));yield IIIIllllIIIllllll ('adding index');IlIIlIlIIllIlIIlI =8192 
	for IlIIIlIIIIlllIIlI in range (0 ,IIlIIIIlIIllIlllI .shape [0 ],IlIIlIlIIllIlIIlI ):IIIlIIllIIlIllIII .add (IIlIIIIlIIllIlllI [IlIIIlIIIIlllIIlI :IlIIIlIIIIlllIIlI +IlIIlIlIIllIlIIlI ])
	faiss .write_index (IIIlIIllIIlIllIII ,_IlllIIIllIlIlIIll %(IllIIIIIlIIllIlII ,IIIIllllIlIllIlIl ,IlIIIllIllllllIlI .nprobe ,IlIlIIIIIIIIlIIll ,IlIIIIlIIlIIIlllI ));yield IIIIllllIIIllllll ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIIIllllIlIllIlIl ,IlIIIllIllllllIlI .nprobe ,IlIlIIIIIIIIlIIll ,IlIIIIlIIlIIIlllI ));yield IIIIllllIIIllllll (IlIIIIllIllIIlllI ('全流程结束！'))
def IlIlIIllIllIlIIIl (IIlIIIIIIllIllIll ):
	IIlIIlIlIllIlIIlI ='train.log'
	if not os .path .exists (IIlIIIIIIllIllIll .replace (os .path .basename (IIlIIIIIIllIllIll ),IIlIIlIlIllIlIIlI )):return {_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },{_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },{_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
	try :
		with open (IIlIIIIIIllIllIll .replace (os .path .basename (IIlIIIIIIllIllIll ),IIlIIlIlIllIlIIlI ),_IlllIllIlllIllIIl )as IIllIIlllIIlIIIll :IlIIlIIIIlIllllIl =eval (IIllIIlllIIlIIIll .read ().strip (_IIIIllIIIIIIIIlII ).split (_IIIIllIIIIIIIIlII )[0 ].split ('\t')[-1 ]);IlIlIlIIllllIllII ,IlIIlllIIIlIlIIIl =IlIIlIIIIlIllllIl [_IIlIlIlIIIlIIIlIl ],IlIIlIIIIlIllllIl ['if_f0'];IlIIllIlIIlllIlII =_IllIllllIllIlllIl if _IIllIIlllIlIllIll in IlIIlIIIIlIllllIl and IlIIlIIIIlIllllIl [_IIllIIlllIlIllIll ]==_IllIllllIllIlllIl else _IIIllllIIIIIIIIlI ;return IlIlIlIIllllIllII ,str (IlIIlllIIIlIlIIIl ),IlIIllIlIIlllIlII 
	except :traceback .print_exc ();return {_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },{_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll },{_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
def IIIIIIlIIlIIIlIll (IllIllIIIlIllIllI ):
	if IllIllIIIlIllIllI ==_IlIllIllIIlllIlII :IlIIlIIIlIlIIlIll =_IlIlIlIIIllIIIIll 
	else :IlIIlIIIlIlIIlIll =_IlllllIlIIlIllllI 
	return {_IlIIIIIIIllllIIII :IlIIlIIIlIlIIlIll ,_IIIllIlllllIlllII :_IlIIIlIlIIlllIlll }
def IIIlIllIllIIIIIIl (IIllIIllIIIllIIII ,IlIllllIIllllIIlI ):IlIlllIIllIlllIIl ='rnd';IlIlllIIIlIIlllII ='pitchf';IIlIIIlIlllllIllI ='pitch';IIIIIlIIllllIlIlI ='phone';global IIlIlIlllllIIllll ;IIlIlIlllllIIllll =torch .load (IIllIIllIIIllIIII ,map_location =_IIllIllllIIlIIlII );IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ][-3 ]=IIlIlIlllllIIllll [_IIIllllllllIIIllI ][_IIllIIllIIIIIlIll ].shape [0 ];IlIllIlIIIlIIllIl =256 if IIlIlIlllllIIllll .get (_IIllIIlllIlIllIll ,_IIIllllIIIIIIIIlI )==_IIIllllIIIIIIIIlI else 768 ;IIlllllllIlIIIIII =torch .rand (1 ,200 ,IlIllIlIIIlIIllIl );IlllllIlIIlIlIllI =torch .tensor ([200 ]).long ();IIlIlIIlllIlIIlll =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IllIIIllIlIllIlIl =torch .rand (1 ,200 );IllIlIlIIlllIIlII =torch .LongTensor ([0 ]);IllIIIIlllllllIII =torch .rand (1 ,192 ,200 );IlIlIIIIlIIllIlIl =_IIllIllllIIlIIlII ;IllIlIIllIIIIlIIl =SynthesizerTrnMsNSFsidM (*IIlIlIlllllIIllll [_IllIlIIIIIIIIllll ],is_half =_IlllllIlIIlIllllI ,version =IIlIlIlllllIIllll .get (_IIllIIlllIlIllIll ,_IIIllllIIIIIIIIlI ));IllIlIIllIIIIlIIl .load_state_dict (IIlIlIlllllIIllll [_IIIllllllllIIIllI ],strict =_IlllllIlIIlIllllI );IlllIlIlIIlIIllIl =[IIIIIlIIllllIlIlI ,'phone_lengths',IIlIIIlIlllllIllI ,IlIlllIIIlIIlllII ,'ds',IlIlllIIllIlllIIl ];IllIIlIIIIIllllll =['audio'];torch .onnx .export (IllIlIIllIIIIlIIl ,(IIlllllllIlIIIIII .to (IlIlIIIIlIIllIlIl ),IlllllIlIIlIlIllI .to (IlIlIIIIlIIllIlIl ),IIlIlIIlllIlIIlll .to (IlIlIIIIlIIllIlIl ),IllIIIllIlIllIlIl .to (IlIlIIIIlIIllIlIl ),IllIlIlIIlllIIlII .to (IlIlIIIIlIIllIlIl ),IllIIIIlllllllIII .to (IlIlIIIIlIIllIlIl )),IlIllllIIllllIIlI ,dynamic_axes ={IIIIIlIIllllIlIlI :[1 ],IIlIIIlIlllllIllI :[1 ],IlIlllIIIlIIlllII :[1 ],IlIlllIIllIlllIIl :[2 ]},do_constant_folding =_IlllllIlIIlIllllI ,opset_version =13 ,verbose =_IlllllIlIIlIllllI ,input_names =IlllIlIlIIlIIllIl ,output_names =IllIIlIIIIIllllll );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IIIlIIIIIIlIlIIlI :
	gr .Markdown (value =IlIIIIllIllIIlllI ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IlIIIIllIllIIlllI ('模型推理')):
			with gr .Row ():IIlIlllIlIlllIIII =gr .Dropdown (label =IlIIIIllIllIIlllI ('推理音色'),choices =sorted (IIlllllIlllllIIlI ));IlIIIIlllIIlIIIll =gr .Button (IlIIIIllIllIIlllI ('刷新音色列表和索引路径'),variant =_IIIIlllIlIIIIIIlI );IlIlIlIllIIlIIIII =gr .Button (IlIIIIllIllIIlllI ('卸载音色省显存'),variant =_IIIIlllIlIIIIIIlI );IlIIlIIIlIIllllIl =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IlIIIIllIllIIlllI ('请选择说话人id'),value =0 ,visible =_IlllllIlIIlIllllI ,interactive =_IlIlIlIIIllIIIIll );IlIlIlIllIIlIIIII .click (fn =IlIIIlllIIlIIIllI ,inputs =[],outputs =[IIlIlllIlIlllIIII ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IIIllIIIlllIIIIIl =gr .Number (label =IlIIIIllIllIIlllI (_IlIIlIIIIIlIllIll ),value =0 );IlIIIIlIIIIIIllII =gr .Textbox (label =IlIIIIllIllIIlllI ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IlIllIIIIlIlIllIl =gr .Radio (label =IlIIIIllIllIIlllI (_IIIlIlIIlIllllllI ),choices =[_IIIlIIIlIIIIIIIll ,_IllllIlllllllIlll ,'crepe',_IlIllIlIlIllllllI ],value =_IIIlIIIlIIIIIIIll ,interactive =_IlIlIlIIIllIIIIll );IlIllIIIIIIIIIlll =gr .Slider (minimum =0 ,maximum =7 ,label =IlIIIIllIllIIlllI (_IllIlIIllIlIlIlIl ),value =3 ,step =1 ,interactive =_IlIlIlIIIllIIIIll )
					with gr .Column ():IIlIIIllIIlIIlIll =gr .Textbox (label =IlIIIIllIllIIlllI (_IllIIlIllllIlllIl ),value ='',interactive =_IlIlIlIIIllIIIIll );IIlIIIIIlIlIlIllI =gr .Dropdown (label =IlIIIIllIllIIlllI (_IIlIlIIlIlIIIlIII ),choices =sorted (IlIIlIIllIIIIlIIl ),interactive =_IlIlIlIIIllIIIIll );IlIIIIlllIIlIIIll .click (fn =IIllllIIIlIIIlIlI ,inputs =[],outputs =[IIlIlllIlIlllIIII ,IIlIIIIIlIlIlIllI ],api_name ='infer_refresh');IIIIIlIlIIIllIIlI =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIIIllIllIIlllI ('检索特征占比'),value =.75 ,interactive =_IlIlIlIIIllIIIIll )
					with gr .Column ():IllIllIlIIIIllIIl =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIIIIllIllIIlllI (_IllllIlIIIIllIIlI ),value =0 ,step =1 ,interactive =_IlIlIlIIIllIIIIll );IIIllllIllIllllll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIIIllIllIIlllI (_IIllIIIIlIIlIIlII ),value =.25 ,interactive =_IlIlIlIIIllIIIIll );IIIlIlIIIlIlIIlII =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIIIIllIllIIlllI (_IIllIllllllllIlII ),value =.33 ,step =.01 ,interactive =_IlIlIlIIIllIIIIll )
					IIllIIIlIIIIlIIIl =gr .File (label =IlIIIIllIllIIlllI ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IlIIIIllIlIlIIlII =gr .Button (IlIIIIllIllIIlllI ('转换'),variant =_IIIIlllIlIIIIIIlI )
					with gr .Row ():IIIllIIlIIIIIIIIl =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ));IlllIIlIIllIlIlIl =gr .Audio (label =IlIIIIllIllIIlllI ('输出音频(右下角三个点,点了可以下载)'))
					IlIIIIllIlIlIIlII .click (IlIIIlIlIllllIlIl ,[IlIIlIIIlIIllllIl ,IlIIIIlIIIIIIllII ,IIIllIIIlllIIIIIl ,IIllIIIlIIIIlIIIl ,IlIllIIIIlIlIllIl ,IIlIIIllIIlIIlIll ,IIlIIIIIlIlIlIllI ,IIIIIlIlIIIllIIlI ,IlIllIIIIIIIIIlll ,IllIllIlIIIIllIIl ,IIIllllIllIllllll ,IIIlIlIIIlIlIIlII ],[IIIllIIlIIIIIIIIl ,IlllIIlIIllIlIlIl ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IIllllllIlIIIIlII =gr .Number (label =IlIIIIllIllIIlllI (_IlIIlIIIIIlIllIll ),value =0 );IlIIlIIIIIlIlIIIl =gr .Textbox (label =IlIIIIllIllIIlllI ('指定输出文件夹'),value =_IIIIIlIIIlIlIIIll );IIllllIlIlIlIllll =gr .Radio (label =IlIIIIllIllIIlllI (_IIIlIlIIlIllllllI ),choices =[_IIIlIIIlIIIIIIIll ,_IllllIlllllllIlll ,'crepe',_IlIllIlIlIllllllI ],value =_IIIlIIIlIIIIIIIll ,interactive =_IlIlIlIIIllIIIIll );IIIlllIIIIlllIllI =gr .Slider (minimum =0 ,maximum =7 ,label =IlIIIIllIllIIlllI (_IllIlIIllIlIlIlIl ),value =3 ,step =1 ,interactive =_IlIlIlIIIllIIIIll )
					with gr .Column ():IIIlIllIlIllIIIIl =gr .Textbox (label =IlIIIIllIllIIlllI (_IllIIlIllllIlllIl ),value ='',interactive =_IlIlIlIIIllIIIIll );IIIIIIllIllIIlllI =gr .Dropdown (label =IlIIIIllIllIIlllI (_IIlIlIIlIlIIIlIII ),choices =sorted (IlIIlIIllIIIIlIIl ),interactive =_IlIlIlIIIllIIIIll );IlIIIIlllIIlIIIll .click (fn =lambda :IIllllIIIlIIIlIlI ()[1 ],inputs =[],outputs =IIIIIIllIllIIlllI ,api_name ='infer_refresh_batch');IlIIIIlllIlIlIIll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIIIllIllIIlllI ('检索特征占比'),value =1 ,interactive =_IlIlIlIIIllIIIIll )
					with gr .Column ():IIIIIlIlIllIIlIIl =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIIIIllIllIIlllI (_IllllIlIIIIllIIlI ),value =0 ,step =1 ,interactive =_IlIlIlIIIllIIIIll );IlllIlIIIIIlIlllI =gr .Slider (minimum =0 ,maximum =1 ,label =IlIIIIllIllIIlllI (_IIllIIIIlIIlIIlII ),value =1 ,interactive =_IlIlIlIIIllIIIIll );IlIlIlIIlIIIIlllI =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIIIIllIllIIlllI (_IIllIllllllllIlII ),value =.33 ,step =.01 ,interactive =_IlIlIlIIIllIIIIll )
					with gr .Column ():IIIIlIIIIlllIIlIl =gr .Textbox (label =IlIIIIllIllIIlllI ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IlIIlIlIlIlIIllII =gr .File (file_count ='multiple',label =IlIIIIllIllIIlllI (_IlIlllIIlIlIIllll ))
					with gr .Row ():IIIlIIllllIlIIlll =gr .Radio (label =IlIIIIllIllIIlllI ('导出文件格式'),choices =[_IIIIlIIIlIIllIlIl ,_IlIIlIIIIIllIlIll ,'mp3','m4a'],value =_IlIIlIIIIIllIlIll ,interactive =_IlIlIlIIIllIIIIll );IIlIIIIIllIIlllII =gr .Button (IlIIIIllIllIIlllI ('转换'),variant =_IIIIlllIlIIIIIIlI );IlIIlIlIlIIIllIlI =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ))
					IIlIIIIIllIIlllII .click (IllIIIllIllIllIll ,[IlIIlIIIlIIllllIl ,IIIIlIIIIlllIIlIl ,IlIIlIIIIIlIlIIIl ,IlIIlIlIlIlIIllII ,IIllllllIlIIIIlII ,IIllllIlIlIlIllll ,IIIlIllIlIllIIIIl ,IIIIIIllIllIIlllI ,IlIIIIlllIlIlIIll ,IIIlllIIIIlllIllI ,IIIIIlIlIllIIlIIl ,IlllIlIIIIIlIlllI ,IlIlIlIIlIIIIlllI ,IIIlIIllllIlIIlll ],[IlIIlIlIlIIIllIlI ],api_name ='infer_convert_batch')
			IIlIlllIlIlllIIII .change (fn =IIlIIllIIIIIllllI ,inputs =[IIlIlllIlIlllIIII ,IIIlIlIIIlIlIIlII ,IlIlIlIIlIIIIlllI ],outputs =[IlIIlIIIlIIllllIl ,IIIlIlIIIlIlIIlII ,IlIlIlIIlIIIIlllI ,IIlIIIIIlIlIlIllI ])
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IIIIIIlllllIIllll =gr .Textbox (label =IlIIIIllIllIIlllI ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IlIlllIlIllIIIllI =gr .File (file_count ='multiple',label =IlIIIIllIllIIlllI (_IlIlllIIlIlIIllll ))
					with gr .Column ():IIIlIIlIIIIlIllll =gr .Dropdown (label =IlIIIIllIllIIlllI ('模型'),choices =IIIlIIIIIIllIIlII );IllIlIIIllIlllIIl =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IlIlIlIIIllIIIIll ,visible =_IlllllIlIIlIllllI );IllIIlllIllllIIll =gr .Textbox (label =IlIIIIllIllIIlllI ('指定输出主人声文件夹'),value =_IIIIIlIIIlIlIIIll );IIIlllIllllllIlIl =gr .Textbox (label =IlIIIIllIllIIlllI ('指定输出非主人声文件夹'),value =_IIIIIlIIIlIlIIIll );IllllllIIIlIlIIlI =gr .Radio (label =IlIIIIllIllIIlllI ('导出文件格式'),choices =[_IIIIlIIIlIIllIlIl ,_IlIIlIIIIIllIlIll ,'mp3','m4a'],value =_IlIIlIIIIIllIlIll ,interactive =_IlIlIlIIIllIIIIll )
					IllllIIlIIIlIIIll =gr .Button (IlIIIIllIllIIlllI ('转换'),variant =_IIIIlllIlIIIIIIlI );IIlIIllIIIllIIlll =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ));IllllIIlIIIlIIIll .click (IllllllllIlIIIlll ,[IIIlIIlIIIIlIllll ,IIIIIIlllllIIllll ,IllIIlllIllllIIll ,IlIlllIlIllIIIllI ,IIIlllIllllllIlIl ,IllIlIIIllIlllIIl ,IllllllIIIlIlIIlI ],[IIlIIllIIIllIIlll ],api_name ='uvr_convert')
		with gr .TabItem (IlIIIIllIllIIlllI ('训练')):
			gr .Markdown (value =IlIIIIllIllIIlllI ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIlIIlIlIlIllIIIl =gr .Textbox (label =IlIIIIllIllIIlllI ('输入实验名'),value ='mi-test');IlIlIIlllllIllIlI =gr .Radio (label =IlIIIIllIllIIlllI ('目标采样率'),choices =[_IlIlIIIIlIlIllIll ],value =_IlIlIIIIlIlIllIll ,interactive =_IlIlIlIIIllIIIIll );IlllllIIllIIIIIII =gr .Radio (label =IlIIIIllIllIIlllI ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IlIlIlIIIllIIIIll ,_IlllllIlIIlIllllI ],value =_IlIlIlIIIllIIIIll ,interactive =_IlIlIlIIIllIIIIll );IIlIIlIlllIIIIlIl =gr .Radio (label =IlIIIIllIllIIlllI ('版本'),choices =[_IllIllllIllIlllIl ],value =_IllIllllIllIlllIl ,interactive =_IlIlIlIIIllIIIIll ,visible =_IlIlIlIIIllIIIIll );IIIlIIIlIlIlIlllI =gr .Slider (minimum =0 ,maximum =IIIllIlllIlllllII .n_cpu ,step =1 ,label =IlIIIIllIllIIlllI ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IIIllIlllIlllllII .n_cpu /1.5 )),interactive =_IlIlIlIIIllIIIIll )
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IlllllIlIIIIIlIlI =gr .Textbox (label =IlIIIIllIllIIlllI ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IlllllIIllIIIlIlI =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IlIIIIllIllIIlllI ('请指定说话人id'),value =0 ,interactive =_IlIlIlIIIllIIIIll );IIlIIIIIllIIlllII =gr .Button (IlIIIIllIllIIlllI ('处理数据'),variant =_IIIIlllIlIIIIIIlI );IIIIllIlIIllIIlIl =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ),value ='');IIlIIIIIllIIlllII .click (IIlIIIlIlllIlIIII ,[IlllllIlIIIIIlIlI ,IIlIIlIlIlIllIIIl ,IlIlIIlllllIllIlI ,IIIlIIIlIlIlIlllI ],[IIIIllIlIIllIIlIl ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IIllIlllIIlllIIll =gr .Textbox (label =IlIIIIllIllIIlllI (_IIIlIlIIlllllIIIl ),value =IIllIIlllllllllIl ,interactive =_IlIlIlIIIllIIIIll );IllIIIIllIlIIlIll =gr .Textbox (label =IlIIIIllIllIIlllI ('显卡信息'),value =IlIIIIlIIlllIllII )
					with gr .Column ():IIllIllIlIllIlIII =gr .Radio (label =IlIIIIllIllIIlllI ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IIIlIIIlIIIIIIIll ,_IllllIlllllllIlll ,'dio',_IlIllIlIlIllllllI ,_IlIllIllIIlllIlII ],value =_IlIllIllIIlllIlII ,interactive =_IlIlIlIIIllIIIIll );IlllIllIlIlIIIlll =gr .Textbox (label =IlIIIIllIllIIlllI ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IIllIIlllllllllIl ,IIllIIlllllllllIl ),interactive =_IlIlIlIIIllIIIIll ,visible =_IlIlIlIIIllIIIIll )
					IllllIIlIIIlIIIll =gr .Button (IlIIIIllIllIIlllI ('特征提取'),variant =_IIIIlllIlIIIIIIlI );IlIlIIlllllIllIll =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ),value ='',max_lines =8 );IIllIllIlIllIlIII .change (fn =IIIIIIlIIlIIIlIll ,inputs =[IIllIllIlIllIlIII ],outputs =[IlllIllIlIlIIIlll ]);IllllIIlIIIlIIIll .click (IIIIlIIllIlIlIIII ,[IIllIlllIIlllIIll ,IIIlIIIlIlIlIlllI ,IIllIllIlIllIlIII ,IlllllIIllIIIIIII ,IIlIIlIlIlIllIIIl ,IIlIIlIlllIIIIlIl ,IlllIllIlIlIIIlll ],[IlIlIIlllllIllIll ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IlIIIIllIllIIlllI ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IIIIIllIlIIIlllII =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IlIIIIllIllIIlllI ('保存频率save_every_epoch'),value =5 ,interactive =_IlIlIlIIIllIIIIll );IllIIIIIIllIIlIII =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IlIIIIllIllIIlllI ('总训练轮数total_epoch'),value =300 ,interactive =_IlIlIlIIIllIIIIll );IIIIIlIIlIIlIIlll =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IlIIIIllIllIIlllI ('每张显卡的batch_size'),value =IlllIIIlIIIllllll ,interactive =_IlIlIlIIIllIIIIll );IllllIlllllIIIlIl =gr .Radio (label =IlIIIIllIllIIlllI ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IlIIIIllIllIIlllI (_IlIIllIIIIllIllII ),IlIIIIllIllIIlllI ('否')],value =IlIIIIllIllIIlllI (_IlIIllIIIIllIllII ),interactive =_IlIlIlIIIllIIIIll );IIlIIllIlIllIIIII =gr .Radio (label =IlIIIIllIllIIlllI ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IlIIIIllIllIIlllI (_IlIIllIIIIllIllII ),IlIIIIllIllIIlllI ('否')],value =IlIIIIllIllIIlllI ('否'),interactive =_IlIlIlIIIllIIIIll );IIlIIIlIllIlIlIIl =gr .Radio (label =IlIIIIllIllIIlllI ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IlIIIIllIllIIlllI (_IlIIllIIIIllIllII ),IlIIIIllIllIIlllI ('否')],value =IlIIIIllIllIIlllI (_IlIIllIIIIllIllII ),interactive =_IlIlIlIIIllIIIIll )
				with gr .Row ():IIIllllIIlIllIIIl =gr .Textbox (label =IlIIIIllIllIIlllI ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IlIlIlIIIllIIIIll );IIIllIlIllIlllIlI =gr .Textbox (label =IlIIIIllIllIIlllI ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IlIlIlIIIllIIIIll );IlIlIIlllllIllIlI .change (IlllllIIlIllIllIl ,[IlIlIIlllllIllIlI ,IlllllIIllIIIIIII ,IIlIIlIlllIIIIlIl ],[IIIllllIIlIllIIIl ,IIIllIlIllIlllIlI ]);IIlIIlIlllIIIIlIl .change (IllIlIlIIllllIlll ,[IlIlIIlllllIllIlI ,IlllllIIllIIIIIII ,IIlIIlIlllIIIIlIl ],[IIIllllIIlIllIIIl ,IIIllIlIllIlllIlI ,IlIlIIlllllIllIlI ]);IlllllIIllIIIIIII .change (IllIlllIIIIIllIlI ,[IlllllIIllIIIIIII ,IlIlIIlllllIllIlI ,IIlIIlIlllIIIIlIl ],[IIllIllIlIllIlIII ,IIIllllIIlIllIIIl ,IIIllIlIllIlllIlI ]);IIllIIlIIlIIIlIIl =gr .Textbox (label =IlIIIIllIllIIlllI (_IIIlIlIIlllllIIIl ),value =IIllIIlllllllllIl ,interactive =_IlIlIlIIIllIIIIll );IIlIIllIIIIIllIll =gr .Button (IlIIIIllIllIIlllI ('训练模型'),variant =_IIIIlllIlIIIIIIlI );IIlIIllIllIIIIIlI =gr .Button (IlIIIIllIllIIlllI ('训练特征索引'),variant =_IIIIlllIlIIIIIIlI );IIIIIIlIIlIIIIlIl =gr .Button (IlIIIIllIllIIlllI ('一键训练'),variant =_IIIIlllIlIIIIIIlI );IlIIIIIIIIlIIIIll =gr .Textbox (label =IlIIIIllIllIIlllI (_IlIlIllIllllIllII ),value ='',max_lines =10 );IIlIIllIIIIIllIll .click (IlIIIIlIIllllIIIl ,[IIlIIlIlIlIllIIIl ,IlIlIIlllllIllIlI ,IlllllIIllIIIIIII ,IlllllIIllIIIlIlI ,IIIIIllIlIIIlllII ,IllIIIIIIllIIlIII ,IIIIIlIIlIIlIIlll ,IllllIlllllIIIlIl ,IIIllllIIlIllIIIl ,IIIllIlIllIlllIlI ,IIllIIlIIlIIIlIIl ,IIlIIllIlIllIIIII ,IIlIIIlIllIlIlIIl ,IIlIIlIlllIIIIlIl ],IlIIIIIIIIlIIIIll ,api_name ='train_start');IIlIIllIllIIIIIlI .click (IlIlIIllllllIIIII ,[IIlIIlIlIlIllIIIl ,IIlIIlIlllIIIIlIl ],IlIIIIIIIIlIIIIll );IIIIIIlIIlIIIIlIl .click (IIIIIIlllllIIlllI ,[IIlIIlIlIlIllIIIl ,IlIlIIlllllIllIlI ,IlllllIIllIIIIIII ,IlllllIlIIIIIlIlI ,IlllllIIllIIIlIlI ,IIIlIIIlIlIlIlllI ,IIllIllIlIllIlIII ,IIIIIllIlIIIlllII ,IllIIIIIIllIIlIII ,IIIIIlIIlIIlIIlll ,IllllIlllllIIIlIl ,IIIllllIIlIllIIIl ,IIIllIlIllIlllIlI ,IIllIIlIIlIIIlIIl ,IIlIIllIlIllIIIII ,IIlIIIlIllIlIlIIl ,IIlIIlIlllIIIIlIl ,IlllIllIlIlIIIlll ],IlIIIIIIIIlIIIIll ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IlllIllIlllIllIIl ,encoding ='utf8')as IIlIlIIIlIlIllIll :IIIlllllIllllIIII =IIlIlIIIlIlIllIll .read ()
				else :
					with open ('docs/faq_en.md',_IlllIllIlllIllIIl ,encoding ='utf8')as IIlIlIIIlIlIllIll :IIIlllllIllllIIII =IIlIlIIIlIlIllIll .read ()
				gr .Markdown (value =IIIlllllIllllIIII )
			except :gr .Markdown (traceback .format_exc ())
	if IIIllIlllIlllllII .iscolab :IIIlIIIIIIlIlIIlI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IIIllIlllIlllllII .listen_port ,share =_IlllllIlIIlIllllI )
	else :IIIlIIIIIIlIlIIlI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IIIllIlllIlllllII .noautoopen ,server_port =IIIllIlllIlllllII .listen_port ,quiet =_IlIlIlIIIllIIIIll )