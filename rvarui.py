_IIIIIIIllIllIlIII ='以-分隔输入使用的卡号, 例如   0-1-2   使用卡l和卡1和卡2'
_IIIllIlllllllIlIl ='也可批量输入音频文件, 二选一, 优先读文件夹'
_IlIIllIIlIIIlIllI ='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_IIIIlllllllllIIIl ='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_IIIIIIlIIllllIlll ='后处理重采样至最终采样率，0为不进行重采样'
_IlIllIllIIllIlIII ='自动检测index路径,下拉式选择(dropdown)'
_IIllIllIlllIllIII ='特征检索库文件路径,为空则使用下拉的选择结果'
_IlIIIIIllIllIlIll ='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_IlIIllllIlIlIIllI ='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_IllIIIlllIlIlIllI ='变调(整数, 半音数量, 升八度12降八度-12)'
_IIlIlIIlIIllIlIll ='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IIIlIIllllIlIIlll ='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_IIlIIllIIIlIlIIII ='IVF%s,Flat'
_IlIllllllIIIlllll ='%s/total_fea.npy'
_IllIIllllllllIlIl ='Trying doing kmeans %s shape to 10k centers.'
_IIlIIIlIIllIIIllI ='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_IIIIIlllIIlllIIII =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IIIlIIllIlIIlIlII =' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_IlIIIIIllIIIlllll ='write filelist done'
_IlIlllIIllIlIlIlI ='%s/filelist.txt'
_IIlIIIlIlllIlIllI ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_IIllIIllllIllIlIl ='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_IlIIIlllllllllIll ='%s/%s.wav|%s/%s.npy|%s'
_IIlIIllllIIIIlIIl ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_IlllIIllllIllIlIl ='%s/2b-f0nsf'
_IlIIIIlIIlIllIIlI ='%s/0_gt_wavs'
_IllllIlIlIlllllll ='emb_g.weight'
_IlIllIIllIllIlIIl ='clean_empty_cache'
_IlIIIIIIIllIIIIIl ='sample_rate'
_IIllIIlIllIIIllII ='%s->%s'
_IIlIIlIIIIllIIIIl ='.index'
_IIlllIIlllIlIIlll ='weights'
_IIIlllIllllIlIlIl ='opt'
_IIIlIIIIlIIlllIII ='rmvpe'
_IIIIllIIlIllIIIIl ='harvest'
_IlIlIllllIIIIIIlI ='%s/3_feature768'
_IIlIllIlllIIIlIll ='%s/3_feature256'
_IlIIIlllIIIIlIIII ='_v2'
_IIIlllllIIlIIIIlI ='48k'
_IlllIIlllIllllIIl ='32k'
_IIllIllIIlIIIllll ='cpu'
_IIIlIIIIIIIIIlllI ='wav'
_IlIllllIIIIlIlllI ='trained'
_IlIIIlIIIIllIllII ='logs'
_IIllIIIlllIlIlIlI ='-pd %s'
_IlllIIIIIlIlIIIlI ='-pg %s'
_IlIIIlIIllllIIlll ='choices'
_IIllllllIIIllIlIl ='weight'
_IlIllIlIlIlIIllll ='pm'
_IIlIllIllIIIlllll ='rmvpe_gpu'
_IllIllIllIIlllIIl ='%s/logs/%s'
_IIllIIlIlIIllIllI ='flac'
_IIlllIIlIlIIlIlII ='f0'
_IlIlIlIIIllllIlll ='%s/%s'
_IlllIlllIllIIllIl ='.pth'
_IIIIIlIlIlllIIIlI ='输出信息'
_IlIllIlIllIIllIlI ='not exist, will not use pretrained model'
_IllIIIIIIllIlIlll ='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth'
_IIIIIlIIllllIllll ='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth'
_IlIIlIIlIIIIIllIl ='40k'
_IlIlllIllIIIIIIlI ='value'
_IlIlIllIIllIIIlll ='v2'
_IIlllIllIlllIIllI ='version'
_IllIllllIlIIlIllI ='visible'
_IllIlIIllIlIIlIll ='primary'
_IIIIllIllIIlllIII =None 
_IIIlIIIlIIIlllIII ='\\\\'
_IlIllIIllllIIllIl ='\\'
_IllllIlIIIllIIlIl ='"'
_IlllIIlIIIIIllIll =' '
_IllIlIIIIIlIlllll ='config'
_IlIllIlllIIIIIlII ='.'
_IllIIIlIIIIllIIII ='r'
_IllIIIlIllIllIlll ='是'
_IIIIllIIllllllIlI ='update'
_IllllIllllIlIIIlI ='__type__'
_IlIllIlIllIIIlIlI ='v1'
_IIlIIIIIlllIlIIlI =False 
_IllIlIIllIIIllIIl ='\n'
_IllIlIllIIlIlllll =True 
import os ,shutil ,sys 
IllIlIlIIIIlIllIl =os .getcwd ()
sys .path .append (IllIlIlIIIIlIllIl )
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
IllIlIlIIIIlIllIl =os .getcwd ()
IllIIlIlllIIlIllI =os .path .join (IllIlIlIIIIlIllIl ,'TEMP')
shutil .rmtree (IllIIlIlllIIlIllI ,ignore_errors =_IllIlIllIIlIlllll )
shutil .rmtree ('%s/runtime/Lib/site-packages/infer_pack'%IllIlIlIIIIlIllIl ,ignore_errors =_IllIlIllIIlIlllll )
shutil .rmtree ('%s/runtime/Lib/site-packages/uvr5_pack'%IllIlIlIIIIlIllIl ,ignore_errors =_IllIlIllIIlIlllll )
os .makedirs (IllIIlIlllIIlIllI ,exist_ok =_IllIlIllIIlIlllll )
os .makedirs (os .path .join (IllIlIlIIIIlIllIl ,_IlIIIlIIIIllIllII ),exist_ok =_IllIlIllIIlIlllll )
os .makedirs (os .path .join (IllIlIlIIIIlIllIl ,_IIlllIIlllIlIIlll ),exist_ok =_IllIlIllIIlIlllll )
os .environ ['TEMP']=IllIIlIlllIIlIllI 
warnings .filterwarnings ('ignore')
torch .manual_seed (114514 )
IlllllllllIlIIlll =Config ()
IlIlllIlIIlIlIIll =I18nAuto ()
IlIlllIlIIlIlIIll .print ()
IIllIlllIIIIIIlll =torch .cuda .device_count ()
IIlIIIlIIIlIIlIIl =[]
IIllllllllIllIIll =[]
IIlllIlIIlIlIIIIl =_IIlIIIIIlllIlIIlI 
if torch .cuda .is_available ()or IIllIlllIIIIIIlll !=0 :
	for IllllIIIlIllIlIIl in range (IIllIlllIIIIIIlll ):
		IIlIlIllIlllIlIII =torch .cuda .get_device_name (IllllIIIlIllIlIIl )
		if any (IlIlIlllllIlIlIIl in IIlIlIllIlllIlIII .upper ()for IlIlIlllllIlIlIIl in ['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):IIlllIlIIlIlIIIIl =_IllIlIllIIlIlllll ;IIlIIIlIIIlIIlIIl .append ('%s\t%s'%(IllllIIIlIllIlIIl ,IIlIlIllIlllIlIII ));IIllllllllIllIIll .append (int (torch .cuda .get_device_properties (IllllIIIlIllIlIIl ).total_memory /1024 /1024 /1024 +.4 ))
if IIlllIlIIlIlIIIIl and len (IIlIIIlIIIlIIlIIl )>0 :IlIIIIIllIlIllIll =_IllIlIIllIIIllIIl .join (IIlIIIlIIIlIIlIIl );IIllIlllIIlIlllII =min (IIllllllllIllIIll )//2 
else :IlIIIIIllIlIllIll =IlIlllIlIIlIlIIll ('很遗憾您这没有能用的显卡来支持您训练');IIllIlllIIlIlllII =1 
IlllllIlIlllIllII ='-'.join ([IlIlIIIIIlIIllIlI [0 ]for IlIlIIIIIlIIllIlI in IIlIIIlIIIlIIlIIl ])
class IIllIIllIIlIIlIII (gr .Button ,gr .components .FormComponent ):
	""
	def __init__ (IIlIIlIIIlIlllIIl ,**IllllIIIIlIllIIII ):super ().__init__ (variant ='tool',**IllllIIIIlIllIIII )
	def get_block_name (IIIIlIlIIIlIlIIIl ):return 'button'
IllIIlIllIllIIlIl =_IIIIllIllIIlllIII 
def IlIllIlIIIIllIlII ():
	global IllIIlIllIllIIlIl ;IIIIlIlIlllllIIIl ,_IllIlIlIIlIllIlIl ,_IllIlIlIIlIllIlIl =checkpoint_utils .load_model_ensemble_and_task (['/kaggle/input/ax-rmf/hubert_base.pt'],suffix ='');IllIIlIllIllIIlIl =IIIIlIlIlllllIIIl [0 ];IllIIlIllIllIIlIl =IllIIlIllIllIIlIl .to (IlllllllllIlIIlll .device )
	if IlllllllllIlIIlll .is_half :IllIIlIllIllIIlIl =IllIIlIllIllIIlIl .half ()
	else :IllIIlIllIllIIlIl =IllIIlIllIllIIlIl .float ()
	IllIIlIllIllIIlIl .eval ()
IllllllIlllIlllll =_IIlllIIlllIlIIlll 
IlIllIllIlIlIllIl ='uvr5_weights'
IlIlIIIlIIlIIlIll =_IlIIIlIIIIllIllII 
IIIlllIIIIIIIIlII =[]
for IIIlIIIIIIIlIlIlI in os .listdir (IllllllIlllIlllll ):
	if IIIlIIIIIIIlIlIlI .endswith (_IlllIlllIllIIllIl ):IIIlllIIIIIIIIlII .append (IIIlIIIIIIIlIlIlI )
IIlIllllIIlIIIIII =[]
for (IlIlIIllIlIllllII ,IIIIlIIIIIllIIlIl ,IllllIlIIlllllIlI )in os .walk (IlIlIIIlIIlIIlIll ,topdown =_IIlIIIIIlllIlIIlI ):
	for IIIlIIIIIIIlIlIlI in IllllIlIIlllllIlI :
		if IIIlIIIIIIIlIlIlI .endswith (_IIlIIlIIIIllIIIIl )and _IlIllllIIIIlIlllI not in IIIlIIIIIIIlIlIlI :IIlIllllIIlIIIIII .append (_IlIlIlIIIllllIlll %(IlIlIIllIlIllllII ,IIIlIIIIIIIlIlIlI ))
IIlllIIIIlIIlIIll =[]
for IIIlIIIIIIIlIlIlI in os .listdir (IlIllIllIlIlIllIl ):
	if IIIlIIIIIIIlIlIlI .endswith (_IlllIlllIllIIllIl )or 'onnx'in IIIlIIIIIIIlIlIlI :IIlllIIIIlIIlIIll .append (IIIlIIIIIIIlIlIlI .replace (_IlllIlllIllIIllIl ,''))
IIlIllIIIIlIllllI =_IIIIllIllIIlllIII 
def IlIllIllIIlIlIIll (IllIlllIllllIIIlI ,IIIIlllIlIlIIIIII ,IIlIlllIIIIllIIIl ,IIllIIlIlIIIllIll ,IIIlIllIIIllIllII ,IIIIIIlIlllIIlllI ,IIlllIlIlIlIllllI ,IIIIIllIIlllllIII ,IlIIIIIlIlllllIlI ,IIllllIlIIIIllIlI ,IllIIIlIIlllIlIIl ,IIlIIIllllllIIlll ):
	global IllllIIlllIIlllll ,IIlIlllIIIIIlIIIl ,IlIIIlllIllIIllII ,IllIIlIllIllIIlIl ,IlIlllIlIlIIIIIll ,IIlIllIIIIlIllllI 
	if IIIIlllIlIlIIIIII is _IIIIllIllIIlllIII :return 'You need to upload an audio',_IIIIllIllIIlllIII 
	IIlIlllIIIIllIIIl =int (IIlIlllIIIIllIIIl )
	try :
		IIllllIllIlIllIlI =load_audio (IIIIlllIlIlIIIIII ,16000 );IIlIllIIIIIllIIII =np .abs (IIllllIllIlIllIlI ).max ()/.95 
		if IIlIllIIIIIllIIII >1 :IIllllIllIlIllIlI /=IIlIllIIIIIllIIII 
		IIllIIllllIlIlllI =[0 ,0 ,0 ]
		if not IllIIlIllIllIIlIl :IlIllIlIIIIllIlII ()
		IIIlllIlIIIIIllIl =IIlIllIIIIlIllllI .get (_IIlllIIlIlIIlIlII ,1 );IIIIIIlIlllIIlllI =IIIIIIlIlllIIlllI .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll ).replace (_IlIllllIIIIlIlllI ,'added')if IIIIIIlIlllIIlllI !=''else IIlllIlIlIlIllllI ;IIIIIllllIllIIlII =IlIIIlllIllIIllII .pipeline (IllIIlIllIllIIlIl ,IIlIlllIIIIIlIIIl ,IllIlllIllllIIIlI ,IIllllIllIlIllIlI ,IIIIlllIlIlIIIIII ,IIllIIllllIlIlllI ,IIlIlllIIIIllIIIl ,IIIlIllIIIllIllII ,IIIIIIlIlllIIlllI ,IIIIIllIIlllllIII ,IIIlllIlIIIIIllIl ,IlIIIIIlIlllllIlI ,IllllIIlllIIlllll ,IIllllIlIIIIllIlI ,IllIIIlIIlllIlIIl ,IlIlllIlIlIIIIIll ,IIlIIIllllllIIlll ,f0_file =IIllIIlIlIIIllIll )
		if IllllIIlllIIlllll !=IIllllIlIIIIllIlI >=16000 :IllllIIlllIIlllll =IIllllIlIIIIllIlI 
		IlIllllllllllIIlI ='Using index:%s.'%IIIIIIlIlllIIlllI if os .path .exists (IIIIIIlIlllIIlllI )else 'Index not used.';return 'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(IlIllllllllllIIlI ,IIllIIllllIlIlllI [0 ],IIllIIllllIlIlllI [1 ],IIllIIllllIlIlllI [2 ]),(IllllIIlllIIlllll ,IIIIIllllIllIIlII )
	except :IIIlllIlllIllIIII =traceback .format_exc ();print (IIIlllIlllIllIIII );return IIIlllIlllIllIIII ,(_IIIIllIllIIlllIII ,_IIIIllIllIIlllIII )
def IIlIIlllllIIlIIII (IlIIllIllllIllllI ,IlIlIlllllIllIIll ,IIlllIIllllIIIIll ,IIIIllIlIlIllIIII ,IIlllllIIllIIIlIl ,IIllIIllllllIIIll ,IlIIllIllIIlIIIll ,IIIIlllIIlIIlIIll ,IIIIIlIlIllIIIlII ,IlIIIIllIIlllIlIl ,IIlIlIllIlIIIllll ,IIllIllIlllIlIIII ,IllIIllIlIIllIllI ,IIlIllIlIIIlIlIlI ):
	try :
		IlIlIlllllIllIIll =IlIlIlllllIllIIll .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll );IIlllIIllllIIIIll =IIlllIIllllIIIIll .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll );os .makedirs (IIlllIIllllIIIIll ,exist_ok =_IllIlIllIIlIlllll )
		try :
			if IlIlIlllllIllIIll !='':IIIIllIlIlIllIIII =[os .path .join (IlIlIlllllIllIIll ,IIIlIllllllllIlll )for IIIlIllllllllIlll in os .listdir (IlIlIlllllIllIIll )]
			else :IIIIllIlIlIllIIII =[IIIIIIlIIIIlIIIIl .name for IIIIIIlIIIIlIIIIl in IIIIllIlIlIllIIII ]
		except :traceback .print_exc ();IIIIllIlIlIllIIII =[IlIIIIlllllIllIll .name for IlIIIIlllllIllIll in IIIIllIlIlIllIIII ]
		IIIlIIlllIlIlllll =[]
		for IllIllIIIllllllll in IIIIllIlIlIllIIII :
			IlllIIIlIllIIlIIl ,IllIlIIllIlIlIllI =IlIllIllIIlIlIIll (IlIIllIllllIllllI ,IllIllIIIllllllll ,IIlllllIIllIIIlIl ,_IIIIllIllIIlllIII ,IIllIIllllllIIIll ,IlIIllIllIIlIIIll ,IIIIlllIIlIIlIIll ,IIIIIlIlIllIIIlII ,IlIIIIllIIlllIlIl ,IIlIlIllIlIIIllll ,IIllIllIlllIlIIII ,IllIIllIlIIllIllI )
			if 'Success'in IlllIIIlIllIIlIIl :
				try :
					IIIllllIIIlllIlIl ,IIIlIIlllIIIIIIII =IllIlIIllIlIlIllI 
					if IIlIllIlIIIlIlIlI in [_IIIlIIIIIIIIIlllI ,_IIllIIlIlIIllIllI ]:sf .write ('%s/%s.%s'%(IIlllIIllllIIIIll ,os .path .basename (IllIllIIIllllllll ),IIlIllIlIIIlIlIlI ),IIIlIIlllIIIIIIII ,IIIllllIIIlllIlIl )
					else :
						IllIllIIIllllllll ='%s/%s.wav'%(IIlllIIllllIIIIll ,os .path .basename (IllIllIIIllllllll ));sf .write (IllIllIIIllllllll ,IIIlIIlllIIIIIIII ,IIIllllIIIlllIlIl )
						if os .path .exists (IllIllIIIllllllll ):os .system ('ffmpeg -i %s -vn %s -q:a 2 -y'%(IllIllIIIllllllll ,IllIllIIIllllllll [:-4 ]+'.%s'%IIlIllIlIIIlIlIlI ))
				except :IlllIIIlIllIIlIIl +=traceback .format_exc ()
			IIIlIIlllIlIlllll .append (_IIllIIlIllIIIllII %(os .path .basename (IllIllIIIllllllll ),IlllIIIlIllIIlIIl ));yield _IllIlIIllIIIllIIl .join (IIIlIIlllIlIlllll )
		yield _IllIlIIllIIIllIIl .join (IIIlIIlllIlIlllll )
	except :yield traceback .format_exc ()
def IIlIlIIIllIIlIIIl (IIIlIlIIlllllIlII ,IIlIllIlllIIIIllI ,IIlllllIllllllIIl ,IIIlllllIlIIIlllI ,IIIIIIllIIlIIIIll ,IlIlllIlIIlllIlll ,IllllIlIllIIlIIII ):
	IIIlllllIIIIIlIII ='streams';IlIllIIIlllIIIIIl ='onnx_dereverb_By_FoxJoy';IlIIIIlIIlIIIllII =[]
	try :
		IIlIllIlllIIIIllI =IIlIllIlllIIIIllI .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll );IIlllllIllllllIIl =IIlllllIllllllIIl .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll );IIIIIIllIIlIIIIll =IIIIIIllIIlIIIIll .strip (_IlllIIlIIIIIllIll ).strip (_IllllIlIIIllIIlIl ).strip (_IllIlIIllIIIllIIl ).strip (_IllllIlIIIllIIlIl ).strip (_IlllIIlIIIIIllIll )
		if IIIlIlIIlllllIlII ==IlIllIIIlllIIIIIl :from MDXNet import MDXNetDereverb ;IIlIlllllIlIlIlll =MDXNetDereverb (15 )
		else :IllIIlIlIIlIIIlIl =_audio_pre_ if 'DeEcho'not in IIIlIlIIlllllIlII else _audio_pre_new ;IIlIlllllIlIlIlll =IllIIlIlIIlIIIlIl (agg =int (IlIlllIlIIlllIlll ),model_path =os .path .join (IlIllIllIlIlIllIl ,IIIlIlIIlllllIlII +_IlllIlllIllIIllIl ),device =IlllllllllIlIIlll .device ,is_half =IlllllllllIlIIlll .is_half )
		if IIlIllIlllIIIIllI !='':IIIlllllIlIIIlllI =[os .path .join (IIlIllIlllIIIIllI ,IlIIlIIIlllIlIIIl )for IlIIlIIIlllIlIIIl in os .listdir (IIlIllIlllIIIIllI )]
		else :IIIlllllIlIIIlllI =[IllIIlIlIlIIlIlll .name for IllIIlIlIlIIlIlll in IIIlllllIlIIIlllI ]
		for IlIllllIIllllIlII in IIIlllllIlIIIlllI :
			IIIlllllIllllllII =os .path .join (IIlIllIlllIIIIllI ,IlIllllIIllllIlII );IlllIIIllIIIlllII =1 ;IlIlIIIlIllIlllll =0 
			try :
				IlIIlIlllIlllIlII =ffmpeg .probe (IIIlllllIllllllII ,cmd ='ffprobe')
				if IlIIlIlllIlllIlII [IIIlllllIIIIIlIII ][0 ]['channels']==2 and IlIIlIlllIlllIlII [IIIlllllIIIIIlIII ][0 ][_IlIIIIIIIllIIIIIl ]=='44100':IlllIIIllIIIlllII =0 ;IIlIlllllIlIlIlll ._path_audio_ (IIIlllllIllllllII ,IIIIIIllIIlIIIIll ,IIlllllIllllllIIl ,IllllIlIllIIlIIII );IlIlIIIlIllIlllll =1 
			except :IlllIIIllIIIlllII =1 ;traceback .print_exc ()
			if IlllIIIllIIIlllII ==1 :IlllIllIlIIlIllII ='%s/%s.reformatted.wav'%(IllIIlIlllIIlIllI ,os .path .basename (IIIlllllIllllllII ));os .system ('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(IIIlllllIllllllII ,IlllIllIlIIlIllII ));IIIlllllIllllllII =IlllIllIlIIlIllII 
			try :
				if IlIlIIIlIllIlllll ==0 :IIlIlllllIlIlIlll ._path_audio_ (IIIlllllIllllllII ,IIIIIIllIIlIIIIll ,IIlllllIllllllIIl ,IllllIlIllIIlIIII )
				IlIIIIlIIlIIIllII .append ('%s->Success'%os .path .basename (IIIlllllIllllllII ));yield _IllIlIIllIIIllIIl .join (IlIIIIlIIlIIIllII )
			except :IlIIIIlIIlIIIllII .append (_IIllIIlIllIIIllII %(os .path .basename (IIIlllllIllllllII ),traceback .format_exc ()));yield _IllIlIIllIIIllIIl .join (IlIIIIlIIlIIIllII )
	except :IlIIIIlIIlIIIllII .append (traceback .format_exc ());yield _IllIlIIllIIIllIIl .join (IlIIIIlIIlIIIllII )
	finally :
		try :
			if IIIlIlIIlllllIlII ==IlIllIIIlllIIIIIl :del IIlIlllllIlIlIlll .pred .model ;del IIlIlllllIlIlIlll .pred .model_ 
			else :del IIlIlllllIlIlIlll .model ;del IIlIlllllIlIlIlll 
		except :traceback .print_exc ()
		print (_IlIllIIllIllIlIIl )
		if torch .cuda .is_available ():torch .cuda .empty_cache ()
	yield _IllIlIIllIIIllIIl .join (IlIIIIlIIlIIIllII )
def IlIlllIlIIIIlIlll (IIllIIlIIIIIlIlll ):
	IIIIllllIIllllIlI ='';IlIlllIlIlllllllI =os .path .join (_IlIIIlIIIIllIllII ,IIllIIlIIIIIlIlll .split (_IlIllIlllIIIIIlII )[0 ],'')
	for IIllIIIIlIIIIlIII in IIlIllllIIlIIIIII :
		if IlIlllIlIlllllllI in IIllIIIIlIIIIlIII :IIIIllllIIllllIlI =IIllIIIIlIIIIlIII ;break 
	return IIIIllllIIllllIlI 
def IlIlllIlIIIlllllI (IlIIlIllllIIlIIlI ,IlllllllIIIIllllI ,IllIIlllIIIllllII ):
	global IIlIlllllIIIIIIIl ,IllllIIlllIIlllll ,IIlIlllIIIIIlIIIl ,IlIIIlllIllIIllII ,IIlIllIIIIlIllllI ,IlIlllIlIlIIIIIll 
	if IlIIlIllllIIlIIlI ==''or IlIIlIllllIIlIIlI ==[]:
		global IllIIlIllIllIIlIl 
		if IllIIlIllIllIIlIl is not _IIIIllIllIIlllIII :
			print (_IlIllIIllIllIlIIl );del IIlIlllIIIIIlIIIl ,IIlIlllllIIIIIIIl ,IlIIIlllIllIIllII ,IllIIlIllIllIIlIl ,IllllIIlllIIlllll ;IllIIlIllIllIIlIl =IIlIlllIIIIIlIIIl =IIlIlllllIIIIIIIl =IlIIIlllIllIIllII =IllIIlIllIllIIlIl =IllllIIlllIIlllll =_IIIIllIllIIlllIII 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
			IllIllllIlIIlIIlI =IIlIllIIIIlIllllI .get (_IIlllIIlIlIIlIlII ,1 );IlIlllIlIlIIIIIll =IIlIllIIIIlIllllI .get (_IIlllIllIlllIIllI ,_IlIllIlIllIIIlIlI )
			if IlIlllIlIlIIIIIll ==_IlIllIlIllIIIlIlI :
				if IllIllllIlIIlIIlI ==1 :IIlIlllIIIIIlIIIl =SynthesizerTrnMs256NSFsid (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ],is_half =IlllllllllIlIIlll .is_half )
				else :IIlIlllIIIIIlIIIl =SynthesizerTrnMs256NSFsid_nono (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ])
			elif IlIlllIlIlIIIIIll ==_IlIlIllIIllIIIlll :
				if IllIllllIlIIlIIlI ==1 :IIlIlllIIIIIlIIIl =SynthesizerTrnMs768NSFsid (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ],is_half =IlllllllllIlIIlll .is_half )
				else :IIlIlllIIIIIlIIIl =SynthesizerTrnMs768NSFsid_nono (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ])
			del IIlIlllIIIIIlIIIl ,IIlIllIIIIlIllllI 
			if torch .cuda .is_available ():torch .cuda .empty_cache ()
		return {_IllIllllIlIIlIllI :_IIlIIIIIlllIlIIlI ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
	IlIIlIlIlIIIIIIlI =_IlIlIlIIIllllIlll %(IllllllIlllIlllll ,IlIIlIllllIIlIIlI );print ('loading %s'%IlIIlIlIlIIIIIIlI );IIlIllIIIIlIllllI =torch .load (IlIIlIlIlIIIIIIlI ,map_location =_IIllIllIIlIIIllll );IllllIIlllIIlllll =IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ][-1 ];IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ][-3 ]=IIlIllIIIIlIllllI [_IIllllllIIIllIlIl ][_IllllIlIlIlllllll ].shape [0 ];IllIllllIlIIlIIlI =IIlIllIIIIlIllllI .get (_IIlllIIlIlIIlIlII ,1 )
	if IllIllllIlIIlIIlI ==0 :IlllllllIIIIllllI =IllIIlllIIIllllII ={_IllIllllIlIIlIllI :_IIlIIIIIlllIlIIlI ,_IlIlllIllIIIIIIlI :.5 ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
	else :IlllllllIIIIllllI ={_IllIllllIlIIlIllI :_IllIlIllIIlIlllll ,_IlIlllIllIIIIIIlI :IlllllllIIIIllllI ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI };IllIIlllIIIllllII ={_IllIllllIlIIlIllI :_IllIlIllIIlIlllll ,_IlIlllIllIIIIIIlI :IllIIlllIIIllllII ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
	IlIlllIlIlIIIIIll =IIlIllIIIIlIllllI .get (_IIlllIllIlllIIllI ,_IlIllIlIllIIIlIlI )
	if IlIlllIlIlIIIIIll ==_IlIllIlIllIIIlIlI :
		if IllIllllIlIIlIIlI ==1 :IIlIlllIIIIIlIIIl =SynthesizerTrnMs256NSFsid (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ],is_half =IlllllllllIlIIlll .is_half )
		else :IIlIlllIIIIIlIIIl =SynthesizerTrnMs256NSFsid_nono (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ])
	elif IlIlllIlIlIIIIIll ==_IlIlIllIIllIIIlll :
		if IllIllllIlIIlIIlI ==1 :IIlIlllIIIIIlIIIl =SynthesizerTrnMs768NSFsid (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ],is_half =IlllllllllIlIIlll .is_half )
		else :IIlIlllIIIIIlIIIl =SynthesizerTrnMs768NSFsid_nono (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ])
	del IIlIlllIIIIIlIIIl .enc_q ;print (IIlIlllIIIIIlIIIl .load_state_dict (IIlIllIIIIlIllllI [_IIllllllIIIllIlIl ],strict =_IIlIIIIIlllIlIIlI ));IIlIlllIIIIIlIIIl .eval ().to (IlllllllllIlIIlll .device )
	if IlllllllllIlIIlll .is_half :IIlIlllIIIIIlIIIl =IIlIlllIIIIIlIIIl .half ()
	else :IIlIlllIIIIIlIIIl =IIlIlllIIIIIlIIIl .float ()
	IlIIIlllIllIIllII =VC (IllllIIlllIIlllll ,IlllllllllIlIIlll );IIlIlllllIIIIIIIl =IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ][-3 ];return {_IllIllllIlIIlIllI :_IllIlIllIIlIlllll ,'maximum':IIlIlllllIIIIIIIl ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },IlllllllIIIIllllI ,IllIIlllIIIllllII ,IlIlllIlIIIIlIlll (IlIIlIllllIIlIIlI )
def IIllllllllIIlllIl ():
	IIIIlllIlIllIlIII =[]
	for IIIIlllIIIIIlIlIl in os .listdir (IllllllIlllIlllll ):
		if IIIIlllIIIIIlIlIl .endswith (_IlllIlllIllIIllIl ):IIIIlllIlIllIlIII .append (IIIIlllIIIIIlIlIl )
	IlIlIIIIIIlllIlIl =[]
	for (IllllIllIIlIlllll ,IIlIllIIlIIIlllIl ,IIlllllllllIlllII )in os .walk (IlIlIIIlIIlIIlIll ,topdown =_IIlIIIIIlllIlIIlI ):
		for IIIIlllIIIIIlIlIl in IIlllllllllIlllII :
			if IIIIlllIIIIIlIlIl .endswith (_IIlIIlIIIIllIIIIl )and _IlIllllIIIIlIlllI not in IIIIlllIIIIIlIlIl :IlIlIIIIIIlllIlIl .append (_IlIlIlIIIllllIlll %(IllllIllIIlIlllll ,IIIIlllIIIIIlIlIl ))
	return {_IlIIIlIIllllIIlll :sorted (IIIIlllIlIllIlIII ),_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },{_IlIIIlIIllllIIlll :sorted (IlIlIIIIIIlllIlIl ),_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
def IIllIIIllIlIIllll ():return {_IlIlllIllIIIIIIlI :'',_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
IlIlllIlIllIIlIll ={_IlllIIlllIllllIIl :32000 ,_IlIIlIIlIIIIIllIl :40000 ,_IIIlllllIIlIIIIlI :48000 }
def IllllIlIIIIlIIllI (IlIllIIlllIlllIll ,IlllIlIllIIlIlIII ):
	while 1 :
		if IlllIlIllIIlIlIII .poll ()is _IIIIllIllIIlllIII :sleep (.5 )
		else :break 
	IlIllIIlllIlllIll [0 ]=_IllIlIllIIlIlllll 
def IIlIIIIIIlIIllIll (IIIllllIIIllllIlI ,IIlIlllIIlIlllIll ):
	while 1 :
		IIIllllIIIlIIlIIl =1 
		for IlIllIllIlIIlIIII in IIlIlllIIlIlllIll :
			if IlIllIllIlIIlIIII .poll ()is _IIIIllIllIIlllIII :IIIllllIIIlIIlIIl =0 ;sleep (.5 );break 
		if IIIllllIIIlIIlIIl ==1 :break 
	IIIllllIIIllllIlI [0 ]=_IllIlIllIIlIlllll 
def IllIllIlIIIIIIlll (IIllIllIllIlIIIlI ,IlIllllIlIIIIIIlI ,IIlllllIIIIIIlllI ,IlIIIIIlllllIIIII ):
	IllIIllIIllIlllII ='%s/logs/%s/preprocess.log';IIlllllIIIIIIlllI =IlIlllIlIllIIlIll [IIlllllIIIIIIlllI ];os .makedirs (_IllIllIllIIlllIIl %(IllIlIlIIIIlIllIl ,IlIllllIlIIIIIIlI ),exist_ok =_IllIlIllIIlIlllll );IIlIlllIllIIIllll =open (IllIIllIIllIlllII %(IllIlIlIIIIlIllIl ,IlIllllIlIIIIIIlI ),'w');IIlIlllIllIIIllll .close ();IlIllIIlIIlIlIlll =IlllllllllIlIIlll .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(IIllIllIllIlIIIlI ,IIlllllIIIIIIlllI ,IlIIIIIlllllIIIII ,IllIlIlIIIIlIllIl ,IlIllllIlIIIIIIlI )+str (IlllllllllIlIIlll .noparallel );print (IlIllIIlIIlIlIlll );IlllIlllIlIIIllII =Popen (IlIllIIlIIlIlIlll ,shell =_IllIlIllIIlIlllll );IllIllIIIllllIIll =[_IIlIIIIIlllIlIIlI ];threading .Thread (target =IllllIlIIIIlIIllI ,args =(IllIllIIIllllIIll ,IlllIlllIlIIIllII )).start ()
	while 1 :
		with open (IllIIllIIllIlllII %(IllIlIlIIIIlIllIl ,IlIllllIlIIIIIIlI ),_IllIIIlIIIIllIIII )as IIlIlllIllIIIllll :yield IIlIlllIllIIIllll .read ()
		sleep (1 )
		if IllIllIIIllllIIll [0 ]:break 
	with open (IllIIllIIllIlllII %(IllIlIlIIIIlIllIl ,IlIllllIlIIIIIIlI ),_IllIIIlIIIIllIIII )as IIlIlllIllIIIllll :IIIllIIIlIllllIll =IIlIlllIllIIIllll .read ()
	print (IIIllIIIlIllllIll );yield IIIllIIIlIllllIll 
def IlIlIIlllllIlIllI (IIlIlllIllIIllIll ,IlIIIlIllIIIIllll ,IIlIlIlIllllIlIll ,IlIIlllllIIllIIIl ,IllIlIlllIllllllI ,IIlIIlIlllIllIlIl ,IIlllllllIIlIllII ):
	IIIlIIIlIlIIlIlII ='%s/logs/%s/extract_fl_feature.log';IIlIlllIllIIllIll =IIlIlllIllIIllIll .split ('-');os .makedirs (_IllIllIllIIlllIIl %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),exist_ok =_IllIlIllIIlIlllll );IIlIIIlllIIIlIIII =open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),'w');IIlIIIlllIIIlIIII .close ()
	if IlIIlllllIIllIIIl :
		if IIlIlIlIllllIlIll !=_IIlIllIllIIIlllll :
			IllllIIIIIlllIlll =IlllllllllIlIIlll .python_cmd +' extract_fl_print.py "%s/logs/%s" %s %s'%(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ,IlIIIlIllIIIIllll ,IIlIlIlIllllIlIll );print (IllllIIIIIlllIlll );IlllllIlIllIlIllI =Popen (IllllIIIIIlllIlll ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IlIIIIIIllIllIIll =[_IIlIIIIIlllIlIIlI ];threading .Thread (target =IllllIlIIIIlIIllI ,args =(IlIIIIIIllIllIIll ,IlllllIlIllIlIllI )).start ()
			while 1 :
				with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :yield IIlIIIlllIIIlIIII .read ()
				sleep (1 )
				if IlIIIIIIllIllIIll [0 ]:break 
			with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :IllIIlIIIlllllllI =IIlIIIlllIIIlIIII .read ()
			print (IllIIlIIIlllllllI );yield IllIIlIIIlllllllI 
		else :
			IIlllllllIIlIllII =IIlllllllIIlIllII .split ('-');IlIlIIIlllllIlIII =len (IIlllllllIIlIllII );IIlIlllIllllllllI =[]
			for (IllIIIIlllIIIIIII ,IlllIIllllllIIllI )in enumerate (IIlllllllIIlIllII ):IllllIIIIIlllIlll =IlllllllllIlIIlll .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s/logs/%s" %s '%(IlIlIIIlllllIlIII ,IllIIIIlllIIIIIII ,IlllIIllllllIIllI ,IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ,IlllllllllIlIIlll .is_half );print (IllllIIIIIlllIlll );IlllllIlIllIlIllI =Popen (IllllIIIIIlllIlll ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IIlIlllIllllllllI .append (IlllllIlIllIlIllI )
			IlIIIIIIllIllIIll =[_IIlIIIIIlllIlIIlI ];threading .Thread (target =IIlIIIIIIlIIllIll ,args =(IlIIIIIIllIllIIll ,IIlIlllIllllllllI )).start ()
			while 1 :
				with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :yield IIlIIIlllIIIlIIII .read ()
				sleep (1 )
				if IlIIIIIIllIllIIll [0 ]:break 
			with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :IllIIlIIIlllllllI =IIlIIIlllIIIlIIII .read ()
			print (IllIIlIIIlllllllI );yield IllIIlIIIlllllllI 
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';IlIlIIIlllllIlIII =len (IIlIlllIllIIllIll );IIlIlllIllllllllI =[]
	for (IllIIIIlllIIIIIII ,IlllIIllllllIIllI )in enumerate (IIlIlllIllIIllIll ):IllllIIIIIlllIlll =IlllllllllIlIIlll .python_cmd +' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(IlllllllllIlIIlll .device ,IlIlIIIlllllIlIII ,IllIIIIlllIIIIIII ,IlllIIllllllIIllI ,IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ,IIlIIlIlllIllIlIl );print (IllllIIIIIlllIlll );IlllllIlIllIlIllI =Popen (IllllIIIIIlllIlll ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IIlIlllIllllllllI .append (IlllllIlIllIlIllI )
	IlIIIIIIllIllIIll =[_IIlIIIIIlllIlIIlI ];threading .Thread (target =IIlIIIIIIlIIllIll ,args =(IlIIIIIIllIllIIll ,IIlIlllIllllllllI )).start ()
	while 1 :
		with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :yield IIlIIIlllIIIlIIII .read ()
		sleep (1 )
		if IlIIIIIIllIllIIll [0 ]:break 
	with open (IIIlIIIlIlIIlIlII %(IllIlIlIIIIlIllIl ,IllIlIlllIllllllI ),_IllIIIlIIIIllIIII )as IIlIIIlllIIIlIIII :IllIIlIIIlllllllI =IIlIIIlllIIIlIIII .read ()
	print (IllIIlIIIlllllllI );yield IllIIlIIIlllllllI 
def IllIIllIIIllIIlII (IIllIIlllIIlllIlI ,IIlIIlIllllIIlllI ,IllIllIIIIllIIIII ):
	IIIlIlIIIllIlIlIl =''if IllIllIIIIllIIIII ==_IlIllIlIllIIIlIlI else _IlIIIlllIIIIlIIII ;IIIlIlIllIllIlIIl =_IIlllIIlIlIIlIlII if IIlIIlIllllIIlllI else '';IllllllllIIlIllll =os .access (_IIIIIlIIllllIllll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI ),os .F_OK );IIIIlllIIIllIlIII =os .access (_IllIIIIIIllIlIlll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI ),os .F_OK )
	if not IllllllllIIlIllll :print (_IIIIIlIIllllIllll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI ),_IlIllIlIllIIllIlI )
	if not IIIIlllIIIllIlIII :print (_IllIIIIIIllIlIlll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI ),_IlIllIlIllIIllIlI )
	return _IIIIIlIIllllIllll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI )if IllllllllIIlIllll else '',_IllIIIIIIllIlIlll %(IIIlIlIIIllIlIlIl ,IIIlIlIllIllIlIIl ,IIllIIlllIIlllIlI )if IIIIlllIIIllIlIII else ''
def IIllIIIIIllIIIlII (IIIIllllllIllIlIl ,IIIIllIlIIllIllIl ,IllIllIIlIlIllIlI ):
	IllIllIIIIIIlIIll =''if IllIllIIlIlIllIlI ==_IlIllIlIllIIIlIlI else _IlIIIlllIIIIlIIII 
	if IIIIllllllIllIlIl ==_IlllIIlllIllllIIl and IllIllIIlIlIllIlI ==_IlIllIlIllIIIlIlI :IIIIllllllIllIlIl =_IlIIlIIlIIIIIllIl 
	IlllIIlIIIllllIlI ={_IlIIIlIIllllIIlll :[_IlIIlIIlIIIIIllIl ,_IIIlllllIIlIIIIlI ],_IllllIllllIlIIIlI :_IIIIllIIllllllIlI ,_IlIlllIllIIIIIIlI :IIIIllllllIllIlIl }if IllIllIIlIlIllIlI ==_IlIllIlIllIIIlIlI else {_IlIIIlIIllllIIlll :[_IlIIlIIlIIIIIllIl ,_IIIlllllIIlIIIIlI ,_IlllIIlllIllllIIl ],_IllllIllllIlIIIlI :_IIIIllIIllllllIlI ,_IlIlllIllIIIIIIlI :IIIIllllllIllIlIl };IIlIIlIIllIlIlIlI =_IIlllIIlIlIIlIlII if IIIIllIlIIllIllIl else '';IIIIlIIIlIlIIlIII =os .access (_IIIIIlIIllllIllll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl ),os .F_OK );IIlIIllllIlllllIl =os .access (_IllIIIIIIllIlIlll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl ),os .F_OK )
	if not IIIIlIIIlIlIIlIII :print (_IIIIIlIIllllIllll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl ),_IlIllIlIllIIllIlI )
	if not IIlIIllllIlllllIl :print (_IllIIIIIIllIlIlll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl ),_IlIllIlIllIIllIlI )
	return _IIIIIlIIllllIllll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl )if IIIIlIIIlIlIIlIII else '',_IllIIIIIIllIlIlll %(IllIllIIIIIIlIIll ,IIlIIlIIllIlIlIlI ,IIIIllllllIllIlIl )if IIlIIllllIlllllIl else '',IlllIIlIIIllllIlI 
def IllIIlIlllIlllIII (IIlIIIllIlllIIlIl ,IIlIlllllllIlIIIl ,IlIIIIIIIIlIlIIll ):
	IlIIlIllIIIIIIIlI ='/kaggle/input/ax-rmf/pretrained%s/f0D%s.pth';IIIlllIIIllIlIIll ='/kaggle/input/ax-rmf/pretrained%s/f0G%s.pth';IIlIIllllIIllIllI =''if IlIIIIIIIIlIlIIll ==_IlIllIlIllIIIlIlI else _IlIIIlllIIIIlIIII ;IllllllIIllIlllIl =os .access (IIIlllIIIllIlIIll %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl ),os .F_OK );IlllIIIIlllIIlllI =os .access (IlIIlIllIIIIIIIlI %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl ),os .F_OK )
	if not IllllllIIllIlllIl :print (IIIlllIIIllIlIIll %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl ),_IlIllIlIllIIllIlI )
	if not IlllIIIIlllIIlllI :print (IlIIlIllIIIIIIIlI %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl ),_IlIllIlIllIIllIlI )
	if IIlIIIllIlllIIlIl :return {_IllIllllIlIIlIllI :_IllIlIllIIlIlllll ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },IIIlllIIIllIlIIll %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl )if IllllllIIllIlllIl else '',IlIIlIllIIIIIIIlI %(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl )if IlllIIIIlllIIlllI else ''
	return {_IllIllllIlIIlIllI :_IIlIIIIIlllIlIIlI ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },'/kaggle/input/ax-rmf/pretrained%s/G%s.pth'%(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl )if IllllllIIllIlllIl else '','/kaggle/input/ax-rmf/pretrained%s/D%s.pth'%(IIlIIllllIIllIllI ,IIlIlllllllIlIIIl )if IlllIIIIlllIIlllI else ''
def IlIIIIIlllIlIlIII (IlIllllIIIIIIlIIl ,IIIlIIlllIIIIlIlI ,IllIlIlIIIIllIlII ,IIlllllIllllllIlI ,IlllIlIIIIIIlIlII ,IIlllIIIIIIIIlllI ,IllllIlIIlIlllIll ,IIllIlIIIIllIIIlI ,IIIIIlIlIIIlIIllI ,IIlllIlIIIlIIIlII ,IllIllllIIllIIIIl ,IIIIllIllIIllIIII ,IlIllIlIlIllIlllI ,IlIIllIIlllIllIll ):
	IIllIIIllllIIlIll ='\x08';IlIllIIIIIIlIllIl =_IllIllIllIIlllIIl %(IllIlIlIIIIlIllIl ,IlIllllIIIIIIlIIl );os .makedirs (IlIllIIIIIIlIllIl ,exist_ok =_IllIlIllIIlIlllll );IIlllIIlIIIllllIl =_IlIIIIlIIlIllIIlI %IlIllIIIIIIlIllIl ;IlIlIIIIIllIIIIll =_IIlIllIlllIIIlIll %IlIllIIIIIIlIllIl if IlIIllIIlllIllIll ==_IlIllIlIllIIIlIlI else _IlIlIllllIIIIIIlI %IlIllIIIIIIlIllIl 
	if IllIlIlIIIIllIlII :IllIIIIIIIllllIlI ='%s/2a_f0'%IlIllIIIIIIlIllIl ;IIllIIlIIIlIlIlll =_IlllIIllllIllIlIl %IlIllIIIIIIlIllIl ;IIlIIIIIllllllIlI =set ([IlIIlllIIllllIlll .split (_IlIllIlllIIIIIlII )[0 ]for IlIIlllIIllllIlll in os .listdir (IIlllIIlIIIllllIl )])&set ([IIIIIIllIIIllIIII .split (_IlIllIlllIIIIIlII )[0 ]for IIIIIIllIIIllIIII in os .listdir (IlIlIIIIIllIIIIll )])&set ([IIIIIlIlIIIIIlIlI .split (_IlIllIlllIIIIIlII )[0 ]for IIIIIlIlIIIIIlIlI in os .listdir (IllIIIIIIIllllIlI )])&set ([IllIIlIlIlIIIlIII .split (_IlIllIlllIIIIIlII )[0 ]for IllIIlIlIlIIIlIII in os .listdir (IIllIIlIIIlIlIlll )])
	else :IIlIIIIIllllllIlI =set ([IllllIIlIIlllIIIl .split (_IlIllIlllIIIIIlII )[0 ]for IllllIIlIIlllIIIl in os .listdir (IIlllIIlIIIllllIl )])&set ([IlIIIlIIIIIlIlllI .split (_IlIllIlllIIIIIlII )[0 ]for IlIIIlIIIIIlIlllI in os .listdir (IlIlIIIIIllIIIIll )])
	IIlIlIIIIIIIlIIlI =[]
	for IlllIIlllllIIlIIl in IIlIIIIIllllllIlI :
		if IllIlIlIIIIllIlII :IIlIlIIIIIIIlIIlI .append (_IIlIIllllIIIIlIIl %(IIlllIIlIIIllllIl .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IlIlIIIIIllIIIIll .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IllIIIIIIIllllIlI .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IIllIIlIIIlIlIlll .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IIlllllIllllllIlI ))
		else :IIlIlIIIIIIIlIIlI .append (_IlIIIlllllllllIll %(IIlllIIlIIIllllIl .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IlIlIIIIIllIIIIll .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IlllIIlllllIIlIIl ,IIlllllIllllllIlI ))
	IllIlIIIllIlllllI =256 if IlIIllIIlllIllIll ==_IlIllIlIllIIIlIlI else 768 
	if IllIlIlIIIIllIlII :
		for _IIIlIIIIlllllIIll in range (2 ):IIlIlIIIIIIIlIIlI .append (_IIllIIllllIllIlIl %(IllIlIlIIIIlIllIl ,IIIlIIlllIIIIlIlI ,IllIlIlIIIIlIllIl ,IllIlIIIllIlllllI ,IllIlIlIIIIlIllIl ,IllIlIlIIIIlIllIl ,IIlllllIllllllIlI ))
	else :
		for _IIIlIIIIlllllIIll in range (2 ):IIlIlIIIIIIIlIIlI .append (_IIlIIIlIlllIlIllI %(IllIlIlIIIIlIllIl ,IIIlIIlllIIIIlIlI ,IllIlIlIIIIlIllIl ,IllIlIIIllIlllllI ,IIlllllIllllllIlI ))
	shuffle (IIlIlIIIIIIIlIIlI )
	with open (_IlIlllIIllIlIlIlI %IlIllIIIIIIlIllIl ,'w')as IIllIIlIlIlIIllIl :IIllIIlIlIlIIllIl .write (_IllIlIIllIIIllIIl .join (IIlIlIIIIIIIlIIlI ))
	print (_IlIIIIIllIIIlllll );print ('use gpus:',IllIllllIIllIIIIl )
	if IIIIIlIlIIIlIIllI =='':print ('no pretrained Generator')
	if IIlllIlIIIlIIIlII =='':print ('no pretrained Discriminator')
	if IllIllllIIllIIIIl :IIllIIlIIllIIIlll =IlllllllllIlIIlll .python_cmd +_IIIlIIllIlIIlIlII %(IlIllllIIIIIIlIIl ,IIIlIIlllIIIIlIlI ,1 if IllIlIlIIIIllIlII else 0 ,IllllIlIIlIlllIll ,IllIllllIIllIIIIl ,IIlllIIIIIIIIlllI ,IlllIlIIIIIIlIlII ,_IlllIIIIIlIlIIIlI %IIIIIlIlIIIlIIllI if IIIIIlIlIIIlIIllI !=''else '',_IIllIIIlllIlIlIlI %IIlllIlIIIlIIIlII if IIlllIlIIIlIIIlII !=''else '',1 if IIllIlIIIIllIIIlI ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IIIIllIllIIllIIII ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IlIllIlIlIllIlllI ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,IlIIllIIlllIllIll )
	else :IIllIIlIIllIIIlll =IlllllllllIlIIlll .python_cmd +_IIIIIlllIIlllIIII %(IlIllllIIIIIIlIIl ,IIIlIIlllIIIIlIlI ,1 if IllIlIlIIIIllIlII else 0 ,IllllIlIIlIlllIll ,IIlllIIIIIIIIlllI ,IlllIlIIIIIIlIlII ,_IlllIIIIIlIlIIIlI %IIIIIlIlIIIlIIllI if IIIIIlIlIIIlIIllI !=''else IIllIIIllllIIlIll ,_IIllIIIlllIlIlIlI %IIlllIlIIIlIIIlII if IIlllIlIIIlIIIlII !=''else IIllIIIllllIIlIll ,1 if IIllIlIIIIllIIIlI ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IIIIllIllIIllIIII ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IlIllIlIlIllIlllI ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,IlIIllIIlllIllIll )
	print (IIllIIlIIllIIIlll );IlIIlllIIlIllIIIl =Popen (IIllIIlIIllIIIlll ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IlIIlllIIlIllIIIl .wait ();return _IIlIIIlIIllIIIllI 
def IlIIIIIIlIIlIlllI (IIlIlIlllIllIlIll ,IIIlIIIlllIIIlIll ):
	IIIIIIllIlIlIIIll =_IllIllIllIIlllIIl %(IllIlIlIIIIlIllIl ,IIlIlIlllIllIlIll );os .makedirs (IIIIIIllIlIlIIIll ,exist_ok =_IllIlIllIIlIlllll );IlIIllllIIIlIIlIl =_IIlIllIlllIIIlIll %IIIIIIllIlIlIIIll if IIIlIIIlllIIIlIll ==_IlIllIlIllIIIlIlI else _IlIlIllllIIIIIIlI %IIIIIIllIlIlIIIll 
	if not os .path .exists (IlIIllllIIIlIIlIl ):return '请先进行特征提取!'
	IIllIIIIIIlIIlIll =list (os .listdir (IlIIllllIIIlIIlIl ))
	if len (IIllIIIIIIlIIlIll )==0 :return '请先进行特征提取！'
	IllIllIIllIllIIII =[];IlllIIlIllllllIll =[]
	for IIIIIIlIlIlIIIIlI in sorted (IIllIIIIIIlIIlIll ):IIllIlIlIlIllIIll =np .load (_IlIlIlIIIllllIlll %(IlIIllllIIIlIIlIl ,IIIIIIlIlIlIIIIlI ));IlllIIlIllllllIll .append (IIllIlIlIlIllIIll )
	IlIlIIllllIIIIllI =np .concatenate (IlllIIlIllllllIll ,0 );IlIIllIIIIllIIlII =np .arange (IlIlIIllllIIIIllI .shape [0 ]);np .random .shuffle (IlIIllIIIIllIIlII );IlIlIIllllIIIIllI =IlIlIIllllIIIIllI [IlIIllIIIIllIIlII ]
	if IlIlIIllllIIIIllI .shape [0 ]>2e5 :
		IllIllIIllIllIIII .append (_IllIIllllllllIlIl %IlIlIIllllIIIIllI .shape [0 ]);yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII )
		try :IlIlIIllllIIIIllI =MiniBatchKMeans (n_clusters =10000 ,verbose =_IllIlIllIIlIlllll ,batch_size =256 *IlllllllllIlIIlll .n_cpu ,compute_labels =_IIlIIIIIlllIlIIlI ,init ='random').fit (IlIlIIllllIIIIllI ).cluster_centers_ 
		except :IIlIIIlIIlIllllll =traceback .format_exc ();print (IIlIIIlIIlIllllll );IllIllIIllIllIIII .append (IIlIIIlIIlIllllll );yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII )
	np .save (_IlIllllllIIIlllll %IIIIIIllIlIlIIIll ,IlIlIIllllIIIIllI );IlIIIllllIIlIIIIl =min (int (16 *np .sqrt (IlIlIIllllIIIIllI .shape [0 ])),IlIlIIllllIIIIllI .shape [0 ]//39 );IllIllIIllIllIIII .append ('%s,%s'%(IlIlIIllllIIIIllI .shape ,IlIIIllllIIlIIIIl ));yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII );IllIllIllIlIlllIl =faiss .index_factory (256 if IIIlIIIlllIIIlIll ==_IlIllIlIllIIIlIlI else 768 ,_IIlIIllIIIlIlIIII %IlIIIllllIIlIIIIl );IllIllIIllIllIIII .append ('training');yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII );IllIlIIIllllIIIIl =faiss .extract_index_ivf (IllIllIllIlIlllIl );IllIlIIIllllIIIIl .nprobe =1 ;IllIllIllIlIlllIl .train (IlIlIIllllIIIIllI );faiss .write_index (IllIllIllIlIlllIl ,_IIIlIIllllIlIIlll %(IIIIIIllIlIlIIIll ,IlIIIllllIIlIIIIl ,IllIlIIIllllIIIIl .nprobe ,IIlIlIlllIllIlIll ,IIIlIIIlllIIIlIll ));IllIllIIllIllIIII .append ('adding');yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII );IIlIllllIIIlllIIl =8192 
	for IlIIllIlllIlIIlII in range (0 ,IlIlIIllllIIIIllI .shape [0 ],IIlIllllIIIlllIIl ):IllIllIllIlIlllIl .add (IlIlIIllllIIIIllI [IlIIllIlllIlIIlII :IlIIllIlllIlIIlII +IIlIllllIIIlllIIl ])
	faiss .write_index (IllIllIllIlIlllIl ,_IIlIlIIlIIllIlIll %(IIIIIIllIlIlIIIll ,IlIIIllllIIlIIIIl ,IllIlIIIllllIIIIl .nprobe ,IIlIlIlllIllIlIll ,IIIlIIIlllIIIlIll ));IllIllIIllIllIIII .append ('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IlIIIllllIIlIIIIl ,IllIlIIIllllIIIIl .nprobe ,IIlIlIlllIllIlIll ,IIIlIIIlllIIIlIll ));yield _IllIlIIllIIIllIIl .join (IllIllIIllIllIIII )
def IlIIlIllIlIIIIIll (IlIlllIIIllllIIll ,IlIlllIIlIlIllIII ,IlIIlllIlIIIIIIlI ,IIllIIlIlllIlIlIl ,IIIIIlllllllIIIII ,IllIIllllIllIIIII ,IlIIIIlIlIIlIllll ,IIIIllIlllIllIIll ,IIIIIlIllllllllII ,IIllIIIIlIlIIIllI ,IlIlIlIIlllllIIll ,IlIlIlIlIIllllllI ,IlIIlIlIIIllIIlll ,IIlIIIIIIllIlllll ,IlllIIlIIlIIIIlIl ,IIllllllIIllllIll ,IIlIlllllIllllIIl ,IIIIlIIlIIIIlIIlI ):
	IllllIIllllIIIIIl =[]
	def IIlllIIIlllllIIII (IIlllllllIIllllll ):IllllIIllllIIIIIl .append (IIlllllllIIllllll );return _IllIlIIllIIIllIIl .join (IllllIIllllIIIIIl )
	IIIIlllIlIIIIllII =_IllIllIllIIlllIIl %(IllIlIlIIIIlIllIl ,IlIlllIIIllllIIll );IlIIIIllIlIIIIlIl ='%s/preprocess.log'%IIIIlllIlIIIIllII ;IIIIIllIIllIlIlII ='%s/extract_fl_feature.log'%IIIIlllIlIIIIllII ;IllllllIlIlIIllII =_IlIIIIlIIlIllIIlI %IIIIlllIlIIIIllII ;IIlIIlIlIIlIIlIIl =_IIlIllIlllIIIlIll %IIIIlllIlIIIIllII if IIlIlllllIllllIIl ==_IlIllIlIllIIIlIlI else _IlIlIllllIIIIIIlI %IIIIlllIlIIIIllII ;os .makedirs (IIIIlllIlIIIIllII ,exist_ok =_IllIlIllIIlIlllll );open (IlIIIIllIlIIIIlIl ,'w').close ();IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(IIllIIlIlllIlIlIl ,IlIlllIlIllIIlIll [IlIlllIIlIlIllIII ],IllIIllllIllIIIII ,IIIIlllIlIIIIllII )+str (IlllllllllIlIIlll .noparallel );yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll ('step1:正在处理数据'));yield IIlllIIIlllllIIII (IIIllllllIIIIlIII );IlIlIlIIllIIIIIII =Popen (IIIllllllIIIIlIII ,shell =_IllIlIllIIlIlllll );IlIlIlIIllIIIIIII .wait ()
	with open (IlIIIIllIlIIIIlIl ,_IllIIIlIIIIllIIII )as IIIlIIIlIlIIllIII :print (IIIlIIIlIlIIllIII .read ())
	open (IIIIIllIIllIlIlII ,'w')
	if IlIIlllIlIIIIIIlI :
		yield IIlllIIIlllllIIII ('step2a:正在提取音高')
		if IlIIIIlIlIIlIllll !=_IIlIllIllIIIlllll :IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +' extract_fl_print.py "%s" %s %s'%(IIIIlllIlIIIIllII ,IllIIllllIllIIIII ,IlIIIIlIlIIlIllll );yield IIlllIIIlllllIIII (IIIllllllIIIIlIII );IlIlIlIIllIIIIIII =Popen (IIIllllllIIIIlIII ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IlIlIlIIllIIIIIII .wait ()
		else :
			IIIIlIIlIIIIlIIlI =IIIIlIIlIIIIlIIlI .split ('-');IlIlIlIlIllIIlIlI =len (IIIIlIIlIIIIlIIlI );IIllllIlIllIllllI =[]
			for (IlIlIlIlIlIlIlIIl ,IlIllIllIlIlllIlI )in enumerate (IIIIlIIlIIIIlIIlI ):IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +' extract_fl_rmvpe.py %s %s %s "%s" %s '%(IlIlIlIlIllIIlIlI ,IlIlIlIlIlIlIlIIl ,IlIllIllIlIlllIlI ,IIIIlllIlIIIIllII ,IlllllllllIlIIlll .is_half );yield IIlllIIIlllllIIII (IIIllllllIIIIlIII );IlIlIlIIllIIIIIII =Popen (IIIllllllIIIIlIII ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IIllllIlIllIllllI .append (IlIlIlIIllIIIIIII )
			for IlIlIlIIllIIIIIII in IIllllIlIllIllllI :IlIlIlIIllIIIIIII .wait ()
		with open (IIIIIllIIllIlIlII ,_IllIIIlIIIIllIIII )as IIIlIIIlIlIIllIII :print (IIIlIIIlIlIIllIII .read ())
	else :yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll ('step2a:无需提取音高'))
	yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll ('step2b:正在提取特征'));IlIIlIlIlIllIlIII =IIlIIIIIIllIlllll .split ('-');IlIlIlIlIllIIlIlI =len (IlIIlIlIlIllIlIII );IIllllIlIllIllllI =[]
	for (IlIlIlIlIlIlIlIIl ,IlIllIllIlIlllIlI )in enumerate (IlIIlIlIlIllIlIII ):IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +' extract_feature_print.py %s %s %s %s "%s" %s'%(IlllllllllIlIIlll .device ,IlIlIlIlIllIIlIlI ,IlIlIlIlIlIlIlIIl ,IlIllIllIlIlllIlI ,IIIIlllIlIIIIllII ,IIlIlllllIllllIIl );yield IIlllIIIlllllIIII (IIIllllllIIIIlIII );IlIlIlIIllIIIIIII =Popen (IIIllllllIIIIlIII ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IIllllIlIllIllllI .append (IlIlIlIIllIIIIIII )
	for IlIlIlIIllIIIIIII in IIllllIlIllIllllI :IlIlIlIIllIIIIIII .wait ()
	with open (IIIIIllIIllIlIlII ,_IllIIIlIIIIllIIII )as IIIlIIIlIlIIllIII :print (IIIlIIIlIlIIllIII .read ())
	yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll ('step3a:正在训练模型'))
	if IlIIlllIlIIIIIIlI :IllIIIllIllllIllI ='%s/2a_f0'%IIIIlllIlIIIIllII ;IllIlIIIIIlllllIl =_IlllIIllllIllIlIl %IIIIlllIlIIIIllII ;IlIIlIllIIIlIlllI =set ([IlllIlllllllllllI .split (_IlIllIlllIIIIIlII )[0 ]for IlllIlllllllllllI in os .listdir (IllllllIlIlIIllII )])&set ([IlIllIIIIlllIIlII .split (_IlIllIlllIIIIIlII )[0 ]for IlIllIIIIlllIIlII in os .listdir (IIlIIlIlIIlIIlIIl )])&set ([IIlIllIIIllIIIllI .split (_IlIllIlllIIIIIlII )[0 ]for IIlIllIIIllIIIllI in os .listdir (IllIIIllIllllIllI )])&set ([IIIIllIIlIIllllll .split (_IlIllIlllIIIIIlII )[0 ]for IIIIllIIlIIllllll in os .listdir (IllIlIIIIIlllllIl )])
	else :IlIIlIllIIIlIlllI =set ([IIlIIllllIIlllIIl .split (_IlIllIlllIIIIIlII )[0 ]for IIlIIllllIIlllIIl in os .listdir (IllllllIlIlIIllII )])&set ([IIlIlIlIlIllIlIIl .split (_IlIllIlllIIIIIlII )[0 ]for IIlIlIlIlIllIlIIl in os .listdir (IIlIIlIlIIlIIlIIl )])
	IIllIlIIlllllIIIl =[]
	for IIIIIIIIllllIIlII in IlIIlIllIIIlIlllI :
		if IlIIlllIlIIIIIIlI :IIllIlIIlllllIIIl .append (_IIlIIllllIIIIlIIl %(IllllllIlIlIIllII .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IIlIIlIlIIlIIlIIl .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IllIIIllIllllIllI .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IllIlIIIIIlllllIl .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IIIIIlllllllIIIII ))
		else :IIllIlIIlllllIIIl .append (_IlIIIlllllllllIll %(IllllllIlIlIIllII .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IIlIIlIlIIlIIlIIl .replace (_IlIllIIllllIIllIl ,_IIIlIIIlIIIlllIII ),IIIIIIIIllllIIlII ,IIIIIlllllllIIIII ))
	IIIIlIlIIIlIlllIl =256 if IIlIlllllIllllIIl ==_IlIllIlIllIIIlIlI else 768 
	if IlIIlllIlIIIIIIlI :
		for _IIIIIIlIIIIlIllll in range (2 ):IIllIlIIlllllIIIl .append (_IIllIIllllIllIlIl %(IllIlIlIIIIlIllIl ,IlIlllIIlIlIllIII ,IllIlIlIIIIlIllIl ,IIIIlIlIIIlIlllIl ,IllIlIlIIIIlIllIl ,IllIlIlIIIIlIllIl ,IIIIIlllllllIIIII ))
	else :
		for _IIIIIIlIIIIlIllll in range (2 ):IIllIlIIlllllIIIl .append (_IIlIIIlIlllIlIllI %(IllIlIlIIIIlIllIl ,IlIlllIIlIlIllIII ,IllIlIlIIIIlIllIl ,IIIIlIlIIIlIlllIl ,IIIIIlllllllIIIII ))
	shuffle (IIllIlIIlllllIIIl )
	with open (_IlIlllIIllIlIlIlI %IIIIlllIlIIIIllII ,'w')as IIIlIIIlIlIIllIII :IIIlIIIlIlIIllIII .write (_IllIlIIllIIIllIIl .join (IIllIlIIlllllIIIl ))
	yield IIlllIIIlllllIIII (_IlIIIIIllIIIlllll )
	if IIlIIIIIIllIlllll :IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +_IIIlIIllIlIIlIlII %(IlIlllIIIllllIIll ,IlIlllIIlIlIllIII ,1 if IlIIlllIlIIIIIIlI else 0 ,IIllIIIIlIlIIIllI ,IIlIIIIIIllIlllll ,IIIIIlIllllllllII ,IIIIllIlllIllIIll ,_IlllIIIIIlIlIIIlI %IlIlIlIlIIllllllI if IlIlIlIlIIllllllI !=''else '',_IIllIIIlllIlIlIlI %IlIIlIlIIIllIIlll if IlIIlIlIIIllIIlll !=''else '',1 if IlIlIlIIlllllIIll ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IlllIIlIIlIIIIlIl ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IIllllllIIllllIll ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,IIlIlllllIllllIIl )
	else :IIIllllllIIIIlIII =IlllllllllIlIIlll .python_cmd +_IIIIIlllIIlllIIII %(IlIlllIIIllllIIll ,IlIlllIIlIlIllIII ,1 if IlIIlllIlIIIIIIlI else 0 ,IIllIIIIlIlIIIllI ,IIIIIlIllllllllII ,IIIIllIlllIllIIll ,_IlllIIIIIlIlIIIlI %IlIlIlIlIIllllllI if IlIlIlIlIIllllllI !=''else '',_IIllIIIlllIlIlIlI %IlIIlIlIIIllIIlll if IlIIlIlIIIllIIlll !=''else '',1 if IlIlIlIIlllllIIll ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IlllIIlIIlIIIIlIl ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,1 if IIllllllIIllllIll ==IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll )else 0 ,IIlIlllllIllllIIl )
	yield IIlllIIIlllllIIII (IIIllllllIIIIlIII );IlIlIlIIllIIIIIII =Popen (IIIllllllIIIIlIII ,shell =_IllIlIllIIlIlllll ,cwd =IllIlIlIIIIlIllIl );IlIlIlIIllIIIIIII .wait ();yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll (_IIlIIIlIIllIIIllI ));IIIIIlIIIIIIlIlII =[];IlIlIIIlllIlIIlII =list (os .listdir (IIlIIlIlIIlIIlIIl ))
	for IIIIIIIIllllIIlII in sorted (IlIlIIIlllIlIIlII ):IIIIIIIlIlIllIIlI =np .load (_IlIlIlIIIllllIlll %(IIlIIlIlIIlIIlIIl ,IIIIIIIIllllIIlII ));IIIIIlIIIIIIlIlII .append (IIIIIIIlIlIllIIlI )
	IIIllllIIllIlIIII =np .concatenate (IIIIIlIIIIIIlIlII ,0 );IlllIllIIlIIIlIlI =np .arange (IIIllllIIllIlIIII .shape [0 ]);np .random .shuffle (IlllIllIIlIIIlIlI );IIIllllIIllIlIIII =IIIllllIIllIlIIII [IlllIllIIlIIIlIlI ]
	if IIIllllIIllIlIIII .shape [0 ]>2e5 :
		IIIIIIIllllIIIIIl =_IllIIllllllllIlIl %IIIllllIIllIlIIII .shape [0 ];print (IIIIIIIllllIIIIIl );yield IIlllIIIlllllIIII (IIIIIIIllllIIIIIl )
		try :IIIllllIIllIlIIII =MiniBatchKMeans (n_clusters =10000 ,verbose =_IllIlIllIIlIlllll ,batch_size =256 *IlllllllllIlIIlll .n_cpu ,compute_labels =_IIlIIIIIlllIlIIlI ,init ='random').fit (IIIllllIIllIlIIII ).cluster_centers_ 
		except :IIIIIIIllllIIIIIl =traceback .format_exc ();print (IIIIIIIllllIIIIIl );yield IIlllIIIlllllIIII (IIIIIIIllllIIIIIl )
	np .save (_IlIllllllIIIlllll %IIIIlllIlIIIIllII ,IIIllllIIllIlIIII );IIlIlIlIIlIIIllIl =min (int (16 *np .sqrt (IIIllllIIllIlIIII .shape [0 ])),IIIllllIIllIlIIII .shape [0 ]//39 );yield IIlllIIIlllllIIII ('%s,%s'%(IIIllllIIllIlIIII .shape ,IIlIlIlIIlIIIllIl ));IlIllIlIIIlIllIIl =faiss .index_factory (256 if IIlIlllllIllllIIl ==_IlIllIlIllIIIlIlI else 768 ,_IIlIIllIIIlIlIIII %IIlIlIlIIlIIIllIl );yield IIlllIIIlllllIIII ('training index');IlIIIllIIIIIlIlIl =faiss .extract_index_ivf (IlIllIlIIIlIllIIl );IlIIIllIIIIIlIlIl .nprobe =1 ;IlIllIlIIIlIllIIl .train (IIIllllIIllIlIIII );faiss .write_index (IlIllIlIIIlIllIIl ,_IIIlIIllllIlIIlll %(IIIIlllIlIIIIllII ,IIlIlIlIIlIIIllIl ,IlIIIllIIIIIlIlIl .nprobe ,IlIlllIIIllllIIll ,IIlIlllllIllllIIl ));yield IIlllIIIlllllIIII ('adding index');IlIlIIllllIlllllI =8192 
	for IIIIlIIIlllIIIIll in range (0 ,IIIllllIIllIlIIII .shape [0 ],IlIlIIllllIlllllI ):IlIllIlIIIlIllIIl .add (IIIllllIIllIlIIII [IIIIlIIIlllIIIIll :IIIIlIIIlllIIIIll +IlIlIIllllIlllllI ])
	faiss .write_index (IlIllIlIIIlIllIIl ,_IIlIlIIlIIllIlIll %(IIIIlllIlIIIIllII ,IIlIlIlIIlIIIllIl ,IlIIIllIIIIIlIlIl .nprobe ,IlIlllIIIllllIIll ,IIlIlllllIllllIIl ));yield IIlllIIIlllllIIII ('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(IIlIlIlIIlIIIllIl ,IlIIIllIIIIIlIlIl .nprobe ,IlIlllIIIllllIIll ,IIlIlllllIllllIIl ));yield IIlllIIIlllllIIII (IlIlllIlIIlIlIIll ('全流程结束！'))
def IIIlllIIllIlIlllI (IllllllIllllIIIII ):
	IllIlIllIlIlIlIIl ='train.log'
	if not os .path .exists (IllllllIllllIIIII .replace (os .path .basename (IllllllIllllIIIII ),IllIlIllIlIlIlIIl )):return {_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },{_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },{_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
	try :
		with open (IllllllIllllIIIII .replace (os .path .basename (IllllllIllllIIIII ),IllIlIllIlIlIlIIl ),_IllIIIlIIIIllIIII )as IIllIIIlIlIIIIIll :IIlIIlIllIlllIIIl =eval (IIllIIIlIlIIIIIll .read ().strip (_IllIlIIllIIIllIIl ).split (_IllIlIIllIIIllIIl )[0 ].split ('\t')[-1 ]);IIllllllIlIlllllI ,IlIlIlIlIlIlIIllI =IIlIIlIllIlllIIIl [_IlIIIIIIIllIIIIIl ],IIlIIlIllIlllIIIl ['if_f0'];IIlllIllIIIlllllI =_IlIlIllIIllIIIlll if _IIlllIllIlllIIllI in IIlIIlIllIlllIIIl and IIlIIlIllIlllIIIl [_IIlllIllIlllIIllI ]==_IlIlIllIIllIIIlll else _IlIllIlIllIIIlIlI ;return IIllllllIlIlllllI ,str (IlIlIlIlIlIlIIllI ),IIlllIllIIIlllllI 
	except :traceback .print_exc ();return {_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },{_IllllIllllIlIIIlI :_IIIIllIIllllllIlI },{_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
def IllllIlllllllIIII (IllllIlllllIIIIIl ):
	if IllllIlllllIIIIIl ==_IIlIllIllIIIlllll :IllllIlIlIlIIIlII =_IllIlIllIIlIlllll 
	else :IllllIlIlIlIIIlII =_IIlIIIIIlllIlIIlI 
	return {_IllIllllIlIIlIllI :IllllIlIlIlIIIlII ,_IllllIllllIlIIIlI :_IIIIllIIllllllIlI }
def IIIIlIlIIIIlllIll (IllIIllIIIIllIlll ,IIIlIIllIIIlIIIlI ):IlIlIIIIlIIllllII ='rnd';IlIIlllIIllIIIIIl ='pitchf';IIIIllllIIIlIlIlI ='pitch';IlIIlllIIlIIIIlll ='phone';global IIlIllIIIIlIllllI ;IIlIllIIIIlIllllI =torch .load (IllIIllIIIIllIlll ,map_location =_IIllIllIIlIIIllll );IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ][-3 ]=IIlIllIIIIlIllllI [_IIllllllIIIllIlIl ][_IllllIlIlIlllllll ].shape [0 ];IIIIIIlllllllllll =256 if IIlIllIIIIlIllllI .get (_IIlllIllIlllIIllI ,_IlIllIlIllIIIlIlI )==_IlIllIlIllIIIlIlI else 768 ;IIIllllIllIlIlllI =torch .rand (1 ,200 ,IIIIIIlllllllllll );IlIlllIIIllIlIlIl =torch .tensor ([200 ]).long ();IlIIIllllllIIIlII =torch .randint (size =(1 ,200 ),low =5 ,high =255 );IlllIllIlllIIllII =torch .rand (1 ,200 );IIIlIlIIIIIlllllI =torch .LongTensor ([0 ]);IlIIlIIlIIIIlIIll =torch .rand (1 ,192 ,200 );IIllIllIIlllIllII =_IIllIllIIlIIIllll ;IIlIlIIllIIlIIlll =SynthesizerTrnMsNSFsidM (*IIlIllIIIIlIllllI [_IllIlIIIIIlIlllll ],is_half =_IIlIIIIIlllIlIIlI ,version =IIlIllIIIIlIllllI .get (_IIlllIllIlllIIllI ,_IlIllIlIllIIIlIlI ));IIlIlIIllIIlIIlll .load_state_dict (IIlIllIIIIlIllllI [_IIllllllIIIllIlIl ],strict =_IIlIIIIIlllIlIIlI );IIIllIlIIlIIllIlI =[IlIIlllIIlIIIIlll ,'phone_lengths',IIIIllllIIIlIlIlI ,IlIIlllIIllIIIIIl ,'ds',IlIlIIIIlIIllllII ];IllIIIllIIIlllIII =['audio'];torch .onnx .export (IIlIlIIllIIlIIlll ,(IIIllllIllIlIlllI .to (IIllIllIIlllIllII ),IlIlllIIIllIlIlIl .to (IIllIllIIlllIllII ),IlIIIllllllIIIlII .to (IIllIllIIlllIllII ),IlllIllIlllIIllII .to (IIllIllIIlllIllII ),IIIlIlIIIIIlllllI .to (IIllIllIIlllIllII ),IlIIlIIlIIIIlIIll .to (IIllIllIIlllIllII )),IIIlIIllIIIlIIIlI ,dynamic_axes ={IlIIlllIIlIIIIlll :[1 ],IIIIllllIIIlIlIlI :[1 ],IlIIlllIIllIIIIIl :[1 ],IlIlIIIIlIIllllII :[2 ]},do_constant_folding =_IIlIIIIIlllIlIIlI ,opset_version =13 ,verbose =_IIlIIIIIlllIlIIlI ,input_names =IIIllIlIIlIIllIlI ,output_names =IllIIIllIIIlllIII );return 'Finished'
with gr .Blocks (theme ='JohnSmith9982/small_and_pretty',title ='AX RVC WebUI')as IlllIIIIIIIllIllI :
	gr .Markdown (value =IlIlllIlIIlIlIIll ('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.'))
	with gr .Tabs ():
		with gr .TabItem (IlIlllIlIIlIlIIll ('模型推理')):
			with gr .Row ():IlIllIIllllIIlIIl =gr .Dropdown (label =IlIlllIlIIlIlIIll ('推理音色'),choices =sorted (IIIlllIIIIIIIIlII ));IIIlIlIIlIlIlllII =gr .Button (IlIlllIlIIlIlIIll ('刷新音色列表和索引路径'),variant =_IllIlIIllIlIIlIll );IlIIIlIIIllIlIIlI =gr .Button (IlIlllIlIIlIlIIll ('卸载音色省显存'),variant =_IllIlIIllIlIIlIll );IlIlllIlIIlllllIl =gr .Slider (minimum =0 ,maximum =2333 ,step =1 ,label =IlIlllIlIIlIlIIll ('请选择说话人id'),value =0 ,visible =_IIlIIIIIlllIlIIlI ,interactive =_IllIlIllIIlIlllll );IlIIIlIIIllIlIIlI .click (fn =IIllIIIllIlIIllll ,inputs =[],outputs =[IlIllIIllllIIlIIl ],api_name ='infer_clean')
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr .Row ():
					with gr .Column ():IlIlIIllllIIllIll =gr .Number (label =IlIlllIlIIlIlIIll (_IllIIIlllIlIlIllI ),value =0 );IlllIIlIIlIlIIlII =gr .Textbox (label =IlIlllIlIIlIlIIll ('输入待处理音频文件路径(默认是正确格式示例)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\冬之花clip1.wav');IIIIIllllIIlllIII =gr .Radio (label =IlIlllIlIIlIlIIll (_IlIIllllIlIlIIllI ),choices =[_IlIllIlIlIlIIllll ,_IIIIllIIlIllIIIIl ,'crepe',_IIIlIIIIlIIlllIII ],value =_IlIllIlIlIlIIllll ,interactive =_IllIlIllIIlIlllll );IlllIIlIlllIlllII =gr .Slider (minimum =0 ,maximum =7 ,label =IlIlllIlIIlIlIIll (_IlIIIIIllIllIlIll ),value =3 ,step =1 ,interactive =_IllIlIllIIlIlllll )
					with gr .Column ():IlIlllllIIlIIIIlI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIllIllIlllIllIII ),value ='',interactive =_IllIlIllIIlIlllll );IIIllllIlIlIIIIII =gr .Dropdown (label =IlIlllIlIIlIlIIll (_IlIllIllIIllIlIII ),choices =sorted (IIlIllllIIlIIIIII ),interactive =_IllIlIllIIlIlllll );IIIlIlIIlIlIlllII .click (fn =IIllllllllIIlllIl ,inputs =[],outputs =[IlIllIIllllIIlIIl ,IIIllllIlIlIIIIII ],api_name ='infer_refresh');IIlIIIIllIIlIIllI =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIlIIlIlIIll ('检索特征占比'),value =.75 ,interactive =_IllIlIllIIlIlllll )
					with gr .Column ():IIlllIllIllIlIIlI =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIlllIlIIlIlIIll (_IIIIIIlIIllllIlll ),value =0 ,step =1 ,interactive =_IllIlIllIIlIlllll );IIllIIlIIIllIllll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIlIIlIlIIll (_IIIIlllllllllIIIl ),value =.25 ,interactive =_IllIlIllIIlIlllll );IIIIllIlIIllIlllI =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIlllIlIIlIlIIll (_IlIIllIIlIIIlIllI ),value =.33 ,step =.01 ,interactive =_IllIlIllIIlIlllll )
					IIlIlIllllllIlllI =gr .File (label =IlIlllIlIIlIlIIll ('F0曲线文件, 可选, 一行一个音高, 代替默认Fl及升降调'));IIlIlIIIIlIllIIIl =gr .Button (IlIlllIlIIlIlIIll ('转换'),variant =_IllIlIIllIlIIlIll )
					with gr .Row ():IlllIlIIIlIlIlIIl =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ));IIlIlIIIIllIIIllI =gr .Audio (label =IlIlllIlIIlIlIIll ('输出音频(右下角三个点,点了可以下载)'))
					IIlIlIIIIlIllIIIl .click (IlIllIllIIlIlIIll ,[IlIlllIlIIlllllIl ,IlllIIlIIlIlIIlII ,IlIlIIllllIIllIll ,IIlIlIllllllIlllI ,IIIIIllllIIlllIII ,IlIlllllIIlIIIIlI ,IIIllllIlIlIIIIII ,IIlIIIIllIIlIIllI ,IlllIIlIlllIlllII ,IIlllIllIllIlIIlI ,IIllIIlIIIllIllll ,IIIIllIlIIllIlllI ],[IlllIlIIIlIlIlIIl ,IIlIlIIIIllIIIllI ],api_name ='infer_convert')
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr .Row ():
					with gr .Column ():IIIlIIllIIIIlllIl =gr .Number (label =IlIlllIlIIlIlIIll (_IllIIIlllIlIlIllI ),value =0 );IllIIlIIlIlllIIIl =gr .Textbox (label =IlIlllIlIIlIlIIll ('指定输出文件夹'),value =_IIIlllIllllIlIlIl );IIIIllIIlIlllllII =gr .Radio (label =IlIlllIlIIlIlIIll (_IlIIllllIlIlIIllI ),choices =[_IlIllIlIlIlIIllll ,_IIIIllIIlIllIIIIl ,'crepe',_IIIlIIIIlIIlllIII ],value =_IlIllIlIlIlIIllll ,interactive =_IllIlIllIIlIlllll );IlIlIIIllIlllllIl =gr .Slider (minimum =0 ,maximum =7 ,label =IlIlllIlIIlIlIIll (_IlIIIIIllIllIlIll ),value =3 ,step =1 ,interactive =_IllIlIllIIlIlllll )
					with gr .Column ():IllIlIlIIlIIIIIll =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIllIllIlllIllIII ),value ='',interactive =_IllIlIllIIlIlllll );IIlllIIlIllllIlll =gr .Dropdown (label =IlIlllIlIIlIlIIll (_IlIllIllIIllIlIII ),choices =sorted (IIlIllllIIlIIIIII ),interactive =_IllIlIllIIlIlllll );IIIlIlIIlIlIlllII .click (fn =lambda :IIllllllllIIlllIl ()[1 ],inputs =[],outputs =IIlllIIlIllllIlll ,api_name ='infer_refresh_batch');IIIIIIIlIllllIlIl =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIlIIlIlIIll ('检索特征占比'),value =1 ,interactive =_IllIlIllIIlIlllll )
					with gr .Column ():IIllIlIlIlIIIlIlI =gr .Slider (minimum =0 ,maximum =48000 ,label =IlIlllIlIIlIlIIll (_IIIIIIlIIllllIlll ),value =0 ,step =1 ,interactive =_IllIlIllIIlIlllll );IIIIIlIllIIllIIll =gr .Slider (minimum =0 ,maximum =1 ,label =IlIlllIlIIlIlIIll (_IIIIlllllllllIIIl ),value =1 ,interactive =_IllIlIllIIlIlllll );IlIIIIllllIIIlIlI =gr .Slider (minimum =0 ,maximum =.5 ,label =IlIlllIlIIlIlIIll (_IlIIllIIlIIIlIllI ),value =.33 ,step =.01 ,interactive =_IllIlIllIIlIlllll )
					with gr .Column ():IIIlIIlIlIIIlIIIl =gr .Textbox (label =IlIlllIlIIlIlIIll ('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs');IIllIllllIllIllII =gr .File (file_count ='multiple',label =IlIlllIlIIlIlIIll (_IIIllIlllllllIlIl ))
					with gr .Row ():IllllIIIllIllIIll =gr .Radio (label =IlIlllIlIIlIlIIll ('导出文件格式'),choices =[_IIIlIIIIIIIIIlllI ,_IIllIIlIlIIllIllI ,'mp3','m4a'],value =_IIllIIlIlIIllIllI ,interactive =_IllIlIllIIlIlllll );IlIlIlIIlIlllIIlI =gr .Button (IlIlllIlIIlIlIIll ('转换'),variant =_IllIlIIllIlIIlIll );IIllllIlIIllIIlIl =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ))
					IlIlIlIIlIlllIIlI .click (IIlIIlllllIIlIIII ,[IlIlllIlIIlllllIl ,IIIlIIlIlIIIlIIIl ,IllIIlIIlIlllIIIl ,IIllIllllIllIllII ,IIIlIIllIIIIlllIl ,IIIIllIIlIlllllII ,IllIlIlIIlIIIIIll ,IIlllIIlIllllIlll ,IIIIIIIlIllllIlIl ,IlIlIIIllIlllllIl ,IIllIlIlIlIIIlIlI ,IIIIIlIllIIllIIll ,IlIIIIllllIIIlIlI ,IllllIIIllIllIIll ],[IIllllIlIIllIIlIl ],api_name ='infer_convert_batch')
			IlIllIIllllIIlIIl .change (fn =IlIlllIlIIIlllllI ,inputs =[IlIllIIllllIIlIIl ,IIIIllIlIIllIlllI ,IlIIIIllllIIIlIlI ],outputs =[IlIlllIlIIlllllIl ,IIIIllIlIIllIlllI ,IlIIIIllllIIIlIlI ,IIIllllIlIlIIIIII ])
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr .Row ():
					with gr .Column ():IllllllIllIlIIlIl =gr .Textbox (label =IlIlllIlIIlIlIIll ('输入待处理音频文件夹路径'),value ='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');IlIIlllIllIIIIllI =gr .File (file_count ='multiple',label =IlIlllIlIIlIlIIll (_IIIllIlllllllIlIl ))
					with gr .Column ():IlIIIlIlIIlIIllll =gr .Dropdown (label =IlIlllIlIIlIlIIll ('模型'),choices =IIlllIIIIlIIlIIll );IllIlllIIIIlIIlIl =gr .Slider (minimum =0 ,maximum =20 ,step =1 ,label ='人声提取激进程度',value =10 ,interactive =_IllIlIllIIlIlllll ,visible =_IIlIIIIIlllIlIIlI );IllIIIlIlIllIlllI =gr .Textbox (label =IlIlllIlIIlIlIIll ('指定输出主人声文件夹'),value =_IIIlllIllllIlIlIl );IlIlIlllIllIlIlll =gr .Textbox (label =IlIlllIlIIlIlIIll ('指定输出非主人声文件夹'),value =_IIIlllIllllIlIlIl );IlIlIlIllIlllllll =gr .Radio (label =IlIlllIlIIlIlIIll ('导出文件格式'),choices =[_IIIlIIIIIIIIIlllI ,_IIllIIlIlIIllIllI ,'mp3','m4a'],value =_IIllIIlIlIIllIllI ,interactive =_IllIlIllIIlIlllll )
					IIlIlllIIlIlIlIIl =gr .Button (IlIlllIlIIlIlIIll ('转换'),variant =_IllIlIIllIlIIlIll );IlIIllIlllIIlIllI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ));IIlIlllIIlIlIlIIl .click (IIlIlIIIllIIlIIIl ,[IlIIIlIlIIlIIllll ,IllllllIllIlIIlIl ,IllIIIlIlIllIlllI ,IlIIlllIllIIIIllI ,IlIlIlllIllIlIlll ,IllIlllIIIIlIIlIl ,IlIlIlIllIlllllll ],[IlIIllIlllIIlIllI ],api_name ='uvr_convert')
		with gr .TabItem (IlIlllIlIIlIlIIll ('训练')):
			gr .Markdown (value =IlIlllIlIIlIlIIll ('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr .Row ():IIlIlllIllIlIlllI =gr .Textbox (label =IlIlllIlIIlIlIIll ('输入实验名'),value ='mi-test');IlIIlIlIlllIIIlIl =gr .Radio (label =IlIlllIlIIlIlIIll ('目标采样率'),choices =[_IlIIlIIlIIIIIllIl ],value =_IlIIlIIlIIIIIllIl ,interactive =_IllIlIllIIlIlllll );IIIlIllIIllIlIlll =gr .Radio (label =IlIlllIlIIlIlIIll ('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices =[_IllIlIllIIlIlllll ,_IIlIIIIIlllIlIIlI ],value =_IllIlIllIIlIlllll ,interactive =_IllIlIllIIlIlllll );IlIlllIIlIlIIIlII =gr .Radio (label =IlIlllIlIIlIlIIll ('版本'),choices =[_IlIlIllIIllIIIlll ],value =_IlIlIllIIllIIIlll ,interactive =_IllIlIllIIlIlllll ,visible =_IllIlIllIIlIlllll );IIlIIlIllIllllIII =gr .Slider (minimum =0 ,maximum =IlllllllllIlIIlll .n_cpu ,step =1 ,label =IlIlllIlIIlIlIIll ('提取音高和处理数据使用的CPU进程数'),value =int (np .ceil (IlllllllllIlIIlll .n_cpu /1.5 )),interactive =_IllIlIllIIlIlllll )
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr .Row ():IIIllIlllIlIlIIIl =gr .Textbox (label =IlIlllIlIIlIlIIll ('输入训练文件夹路径'),value ='/kaggle/working/dataset');IIIlIlllIlIllIIIl =gr .Slider (minimum =0 ,maximum =4 ,step =1 ,label =IlIlllIlIIlIlIIll ('请指定说话人id'),value =0 ,interactive =_IllIlIllIIlIlllll );IlIlIlIIlIlllIIlI =gr .Button (IlIlllIlIIlIlIIll ('处理数据'),variant =_IllIlIIllIlIIlIll );IllIllIlIllllIIlI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ),value ='');IlIlIlIIlIlllIIlI .click (IllIllIlIIIIIIlll ,[IIIllIlllIlIlIIIl ,IIlIlllIllIlIlllI ,IlIIlIlIlllIIIlIl ,IIlIIlIllIllllIII ],[IllIllIlIllllIIlI ],api_name ='train_preprocess')
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr .Row ():
					with gr .Column ():IlIlIIlIllIllIIlI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIIIllIllIlIII ),value =IlllllIlIlllIllII ,interactive =_IllIlIllIIlIlllll );IllIlIlIIlIIlllII =gr .Textbox (label =IlIlllIlIIlIlIIll ('显卡信息'),value =IlIIIIIllIlIllIll )
					with gr .Column ():IlllllIllIlllIIII =gr .Radio (label =IlIlllIlIIlIlIIll ('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices =[_IlIllIlIlIlIIllll ,_IIIIllIIlIllIIIIl ,'dio',_IIIlIIIIlIIlllIII ,_IIlIllIllIIIlllll ],value =_IIlIllIllIIIlllll ,interactive =_IllIlIllIIlIlllll );IIlIlllIIIlllIIII =gr .Textbox (label =IlIlllIlIIlIlIIll ('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡l上跑2个进程并在卡1上跑1个进程'),value ='%s-%s'%(IlllllIlIlllIllII ,IlllllIlIlllIllII ),interactive =_IllIlIllIIlIlllll ,visible =_IllIlIllIIlIlllll )
					IIlIlllIIlIlIlIIl =gr .Button (IlIlllIlIIlIlIIll ('特征提取'),variant =_IllIlIIllIlIIlIll );IIIllIIIIlIlIIIll =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ),value ='',max_lines =8 );IlllllIllIlllIIII .change (fn =IllllIlllllllIIII ,inputs =[IlllllIllIlllIIII ],outputs =[IIlIlllIIIlllIIII ]);IIlIlllIIlIlIlIIl .click (IlIlIIlllllIlIllI ,[IlIlIIlIllIllIIlI ,IIlIIlIllIllllIII ,IlllllIllIlllIIII ,IIIlIllIIllIlIlll ,IIlIlllIllIlIlllI ,IlIlllIIlIlIIIlII ,IIlIlllIIIlllIIII ],[IIIllIIIIlIlIIIll ],api_name ='train_extract_fl_feature')
			with gr .Group ():
				gr .Markdown (value =IlIlllIlIIlIlIIll ('step3: 填写训练设置, 开始训练模型和索引'))
				with gr .Row ():IlIIllllIlIllIIII =gr .Slider (minimum =0 ,maximum =100 ,step =1 ,label =IlIlllIlIIlIlIIll ('保存频率save_every_epoch'),value =5 ,interactive =_IllIlIllIIlIlllll );IIIlIllIlIllIllll =gr .Slider (minimum =0 ,maximum =1000 ,step =1 ,label =IlIlllIlIIlIlIIll ('总训练轮数total_epoch'),value =300 ,interactive =_IllIlIllIIlIlllll );IllIIIlllllIlllII =gr .Slider (minimum =1 ,maximum =40 ,step =1 ,label =IlIlllIlIIlIlIIll ('每张显卡的batch_size'),value =IIllIlllIIlIlllII ,interactive =_IllIlIllIIlIlllll );IlIlIllIIIlIIIIII =gr .Radio (label =IlIlllIlIIlIlIIll ('是否仅保存最新的ckpt文件以节省硬盘空间'),choices =[IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll ),IlIlllIlIIlIlIIll ('否')],value =IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll ),interactive =_IllIlIllIIlIlllll );IlllIlIIlIlllIIIl =gr .Radio (label =IlIlllIlIIlIlIIll ('是否缓存所有训练集至显存. 1lmin以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices =[IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll ),IlIlllIlIIlIlIIll ('否')],value =IlIlllIlIIlIlIIll ('否'),interactive =_IllIlIllIIlIlllll );IlIIllIIlIlllIlIl =gr .Radio (label =IlIlllIlIIlIlIIll ('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices =[IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll ),IlIlllIlIIlIlIIll ('否')],value =IlIlllIlIIlIlIIll (_IllIIIlIllIllIlll ),interactive =_IllIlIllIIlIlllll )
				with gr .Row ():IIIIIlIIIllIIIIII =gr .Textbox (label =IlIlllIlIIlIlIIll ('加载预训练底模G路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive =_IllIlIllIIlIlllll );IllIIIIllIlIIlllI =gr .Textbox (label =IlIlllIlIIlIlIIll ('加载预训练底模D路径'),value ='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive =_IllIlIllIIlIlllll );IlIIlIlIlllIIIlIl .change (IllIIllIIIllIIlII ,[IlIIlIlIlllIIIlIl ,IIIlIllIIllIlIlll ,IlIlllIIlIlIIIlII ],[IIIIIlIIIllIIIIII ,IllIIIIllIlIIlllI ]);IlIlllIIlIlIIIlII .change (IIllIIIIIllIIIlII ,[IlIIlIlIlllIIIlIl ,IIIlIllIIllIlIlll ,IlIlllIIlIlIIIlII ],[IIIIIlIIIllIIIIII ,IllIIIIllIlIIlllI ,IlIIlIlIlllIIIlIl ]);IIIlIllIIllIlIlll .change (IllIIlIlllIlllIII ,[IIIlIllIIllIlIlll ,IlIIlIlIlllIIIlIl ,IlIlllIIlIlIIIlII ],[IlllllIllIlllIIII ,IIIIIlIIIllIIIIII ,IllIIIIllIlIIlllI ]);IIlllIllIIIlllIlI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIIIllIllIlIII ),value =IlllllIlIlllIllII ,interactive =_IllIlIllIIlIlllll );IlIIlIIIlIllIlIII =gr .Button (IlIlllIlIIlIlIIll ('训练模型'),variant =_IllIlIIllIlIIlIll );IIlIIllIlIIlIllII =gr .Button (IlIlllIlIIlIlIIll ('训练特征索引'),variant =_IllIlIIllIlIIlIll );IlIIIIIlllIllllII =gr .Button (IlIlllIlIIlIlIIll ('一键训练'),variant =_IllIlIIllIlIIlIll );IlIIlIIlIIllIlIlI =gr .Textbox (label =IlIlllIlIIlIlIIll (_IIIIIlIlIlllIIIlI ),value ='',max_lines =10 );IlIIlIIIlIllIlIII .click (IlIIIIIlllIlIlIII ,[IIlIlllIllIlIlllI ,IlIIlIlIlllIIIlIl ,IIIlIllIIllIlIlll ,IIIlIlllIlIllIIIl ,IlIIllllIlIllIIII ,IIIlIllIlIllIllll ,IllIIIlllllIlllII ,IlIlIllIIIlIIIIII ,IIIIIlIIIllIIIIII ,IllIIIIllIlIIlllI ,IIlllIllIIIlllIlI ,IlllIlIIlIlllIIIl ,IlIIllIIlIlllIlIl ,IlIlllIIlIlIIIlII ],IlIIlIIlIIllIlIlI ,api_name ='train_start');IIlIIllIlIIlIllII .click (IlIIIIIIlIIlIlllI ,[IIlIlllIllIlIlllI ,IlIlllIIlIlIIIlII ],IlIIlIIlIIllIlIlI );IlIIIIIlllIllllII .click (IlIIlIllIlIIIIIll ,[IIlIlllIllIlIlllI ,IlIIlIlIlllIIIlIl ,IIIlIllIIllIlIlll ,IIIllIlllIlIlIIIl ,IIIlIlllIlIllIIIl ,IIlIIlIllIllllIII ,IlllllIllIlllIIII ,IlIIllllIlIllIIII ,IIIlIllIlIllIllll ,IllIIIlllllIlllII ,IlIlIllIIIlIIIIII ,IIIIIlIIIllIIIIII ,IllIIIIllIlIIlllI ,IIlllIllIIIlllIlI ,IlllIlIIlIlllIIIl ,IlIIllIIlIlllIlIl ,IlIlllIIlIlIIIlII ,IIlIlllIIIlllIIII ],IlIIlIIlIIllIlIlI ,api_name ='train_start_all')
			try :
				if tab_faq =='常见问题解答':
					with open ('docs/faq.md',_IllIIIlIIIIllIIII ,encoding ='utf8')as IIllIlIIlIIIlIlII :IIIlIlllIlIlllllI =IIllIlIIlIIIlIlII .read ()
				else :
					with open ('docs/faq_en.md',_IllIIIlIIIIllIIII ,encoding ='utf8')as IIllIlIIlIIIlIlII :IIIlIlllIlIlllllI =IIllIlIIlIIIlIlII .read ()
				gr .Markdown (value =IIIlIlllIlIlllllI )
			except :gr .Markdown (traceback .format_exc ())
	if IlllllllllIlIIlll .iscolab :IlllIIIIIIIllIllI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_port =IlllllllllIlIIlll .listen_port ,share =_IIlIIIIIlllIlIIlI )
	else :IlllIIIIIIIllIllI .queue (concurrency_count =511 ,max_size =1022 ).launch (server_name ='0.0.0.0',inbrowser =not IlllllllllIlIIlll .noautoopen ,server_port =IlllllllllIlIIlll .listen_port ,quiet =_IIlIIIIIlllIlIIlI )