_Af='以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2'
_Ae='也可批量输入音频文件, 二选一, 优先读文件夹'
_Ad='Default value is 1.0'
_Ac='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_Ab='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_Aa='后处理重采样至最终采样率，0为不进行重采样'
_AZ='特征检索库文件路径,为空则使用下拉的选择结果'
_AY='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_AX='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU'
_AW='变调(整数, 半音数量, 升八度12降八度-12)'
_AV='mangio-crepe-tiny'
_AU='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_AT='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_AS='IVF%s,Flat'
_AR='%s/total_fea.npy'
_AQ='Trying doing kmeans %s shape to 10k centers.'
_AP='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_AO='write filelist done'
_AN='%s/filelist.txt'
_AM='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_AL='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_AK='%s/%s.wav|%s/%s.npy|%s'
_AJ='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_AI='%s/2b-f0nsf'
_AH='%s/0_gt_wavs'
_AG='emb_g.weight'
_AF='clean_empty_cache'
_AE='sample_rate'
_AD='Whether the model has pitch guidance.'
_AC='目标采样率'
_AB='crepe_hop_length'
_AA='crepe'
_A9='harvest'
_A8='mangio-crepe'
_A7='presets'
_A6='EXTRACT-MODEL'
_A5='TRAIN-FEATURE'
_A4='TRAIN'
_A3='EXTRACT-FEATURE'
_A2='PRE-PROCESS'
_A1='INFER'
_A0='%s/3_feature768'
_z='%s/3_feature256'
_y='_v2'
_x='cpu'
_w='mp3'
_v='wav'
_u='csvdb/stop.csv'
_t='../inference-presets.json'
_s='-pd %s'
_r='-pg %s'
_q="doesn't exist, will not use pretrained model"
_p='32k'
_o='/audios/'
_n='weight'
_m='trained'
_l='%s/logs/%s'
_k='flac'
_j='f0'
_i='.index'
_h='.pth'
_g='rmvpe'
_f='pretrained%s/%sD%s.pth'
_e='pretrained%s/%sG%s.pth'
_d='48k'
_c='choices'
_b='version'
_a='%s/%s'
_Z='./logs/'
_Y='v2'
_X='w+'
_W='formanting'
_V='csvdb/formanting.csv'
_U='w'
_T='输出信息'
_S='40k'
_R='\\\\'
_Q='"'
_P=' '
_O='config'
_N='/'
_M='value'
_L='.'
_K='primary'
_J=None
_I='r'
_H='\\'
_G='v1'
_F='\n'
_E='visible'
_D=False
_C='update'
_B='__type__'
_A=True
import os,shutil,sys,json,math,signal
now_dir=os.getcwd()
sys.path.append(now_dir)
import traceback,pdb,warnings,numpy as np,torch,re
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['no_proxy']='localhost, 127.0.0.1, ::1'
import logging,threading
from random import shuffle
from subprocess import Popen
from time import sleep
import faiss,ffmpeg,gradio as gr,soundfile as sf
from config import Config
from fairseq import checkpoint_utils
from i18n import I18nAuto
from lib.infer_pack.models import SynthesizerTrnMs256NSFsid,SynthesizerTrnMs256NSFsid_nono,SynthesizerTrnMs768NSFsid,SynthesizerTrnMs768NSFsid_nono
from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM
from infer_uvr5 import _audio_pre_,_audio_pre_new
from MDXNet import MDXNetDereverb
from my_utils import load_audio,CSVutil
from train.process_ckpt import change_info,extract_small_model,merge,show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans
tmp=os.path.join(now_dir,'TEMP')
shutil.rmtree(tmp,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/infer_pack'%now_dir,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/uvr5_pack'%now_dir,ignore_errors=_A)
os.makedirs(tmp,exist_ok=_A)
os.makedirs(os.path.join(now_dir,'logs'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'audios'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'datasets'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'weights'),exist_ok=_A)
os.environ['TEMP']=tmp
warnings.filterwarnings('ignore')
torch.manual_seed(114514)
logging.getLogger('numba').setLevel(logging.WARNING)
import csv
if not os.path.isdir('csvdb/'):os.makedirs('csvdb');frmnt,stp=open(_V,_U),open(_u,_U);frmnt.close();stp.close()
global DoFormant,Quefrency,Timbre
try:DoFormant,Quefrency,Timbre=CSVutil(_V,_I,_W);DoFormant=(lambda DoFormant:_A if DoFormant.lower()=='true'else _D if DoFormant.lower()=='false'else DoFormant)(DoFormant)
except(ValueError,TypeError,IndexError):DoFormant,Quefrency,Timbre=_D,1.,1.;CSVutil(_V,_X,_W,DoFormant,Quefrency,Timbre)
config=Config()
i18n=I18nAuto()
i18n.print()
ngpu=torch.cuda.device_count()
gpu_infos=[]
mem=[]
if_gpu_ok=_D
isinterrupted=0
if torch.cuda.is_available()or ngpu!=0:
	for i in range(ngpu):
		gpu_name=torch.cuda.get_device_name(i)
		if any(value in gpu_name.upper()for value in['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):if_gpu_ok=_A;gpu_infos.append('%s\t%s'%(i,gpu_name));mem.append(int(torch.cuda.get_device_properties(i).total_memory/1024/1024/1024+.4))
if if_gpu_ok and len(gpu_infos)>0:gpu_info=_F.join(gpu_infos);default_batch_size=min(mem)//2
else:gpu_info=i18n('很遗憾您这没有能用的显卡来支持您训练');default_batch_size=1
gpus='-'.join([i[0]for i in gpu_infos])
hubert_model=_J
def load_hubert():
	global hubert_model;models,_,_=checkpoint_utils.load_model_ensemble_and_task(['/kaggle/input/ax-rmf/hubert_base.pt'],suffix='');hubert_model=models[0];hubert_model=hubert_model.to(config.device)
	if config.is_half:hubert_model=hubert_model.half()
	else:hubert_model=hubert_model.float()
	hubert_model.eval()
weight_root='weights'
weight_uvr5_root='uvr5_weights'
index_root=_Z
audio_root='audios'
names=[]
for name in os.listdir(weight_root):
	if name.endswith(_h):names.append(name)
index_paths=[]
global indexes_list
indexes_list=[]
audio_paths=[]
for(root,dirs,files)in os.walk(index_root,topdown=_D):
	for name in files:
		if name.endswith(_i)and _m not in name:index_paths.append('%s\\%s'%(root,name))
for(root,dirs,files)in os.walk(audio_root,topdown=_D):
	for name in files:audio_paths.append(_a%(root,name))
uvr5_names=[]
for name in os.listdir(weight_uvr5_root):
	if name.endswith(_h)or'onnx'in name:uvr5_names.append(name.replace(_h,''))
def check_for_name():
	if len(names)>0:return sorted(names)[0]
	else:return''
def get_index():
	if check_for_name()!='':
		chosen_model=sorted(names)[0].split(_L)[0];logs_path=_Z+chosen_model
		if os.path.exists(logs_path):
			for file in os.listdir(logs_path):
				if file.endswith(_i):return os.path.join(logs_path,file).replace(_H,_N)
			return''
		else:return''
def get_indexes():
	for(dirpath,dirnames,filenames)in os.walk(_Z):
		for filename in filenames:
			if filename.endswith(_i)and _m not in filename:indexes_list.append(os.path.join(dirpath,filename).replace(_H,_N))
	if len(indexes_list)>0:return indexes_list
	else:return''
fshift_presets_list=[]
def get_fshift_presets():
	fshift_presets_list=[]
	for(dirpath,dirnames,filenames)in os.walk('./formantshiftcfg/'):
		for filename in filenames:
			if filename.endswith('.txt'):fshift_presets_list.append(os.path.join(dirpath,filename).replace(_H,_N))
	if len(fshift_presets_list)>0:return fshift_presets_list
	else:return''
def vc_single(sid,input_audio_path0,input_audio_path1,f0_up_key,f0_file,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,crepe_hop_length):
	global tgt_sr,net_g,vc,hubert_model,version
	if input_audio_path0 is _J or input_audio_path0 is _J:return'You need to upload an audio',_J
	f0_up_key=int(f0_up_key)
	try:
		if input_audio_path0=='':audio=load_audio(input_audio_path1,16000,DoFormant,Quefrency,Timbre)
		else:audio=load_audio(input_audio_path0,16000,DoFormant,Quefrency,Timbre)
		audio_max=np.abs(audio).max()/.95
		if audio_max>1:audio/=audio_max
		times=[0,0,0]
		if not hubert_model:load_hubert()
		if_f0=cpt.get(_j,1);file_index=file_index.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P).replace(_m,'added')if file_index!=''else file_index2;audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,input_audio_path1,times,f0_up_key,f0_method,file_index,index_rate,if_f0,filter_radius,tgt_sr,resample_sr,rms_mix_rate,version,protect,crepe_hop_length,f0_file=f0_file)
		if tgt_sr!=resample_sr>=16000:tgt_sr=resample_sr
		index_info='Using index:%s.'%file_index if os.path.exists(file_index)else'Index not used.';return'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(index_info,times[0],times[1],times[2]),(tgt_sr,audio_opt)
	except:info=traceback.format_exc();print(info);return info,(_J,_J)
def vc_multi(sid,dir_path,opt_root,paths,f0_up_key,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,format1,crepe_hop_length):
	try:
		dir_path=dir_path.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P);opt_root=opt_root.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P);os.makedirs(opt_root,exist_ok=_A)
		try:
			if dir_path!='':paths=[os.path.join(dir_path,name)for name in os.listdir(dir_path)]
			else:paths=[path.name for path in paths]
		except:traceback.print_exc();paths=[path.name for path in paths]
		infos=[]
		for path in paths:
			info,opt=vc_single(sid,path,_J,f0_up_key,_J,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,crepe_hop_length)
			if'Success'in info:
				try:
					tgt_sr,audio_opt=opt
					if format1 in[_v,_k,_w,'ogg','aac']:sf.write('%s/%s.%s'%(opt_root,os.path.basename(path),format1),audio_opt,tgt_sr)
					else:
						path='%s/%s.wav'%(opt_root,os.path.basename(path));sf.write(path,audio_opt,tgt_sr)
						if os.path.exists(path):os.system('ffmpeg -i %s -vn %s -q:a 2 -y'%(path,path[:-4]+'.%s'%format1))
				except:info+=traceback.format_exc()
			infos.append('%s->%s'%(os.path.basename(path),info));yield _F.join(infos)
		yield _F.join(infos)
	except:yield traceback.format_exc()
def uvr(model_name,inp_root,save_root_vocal,paths,save_root_ins,agg,format0):
	B='streams';A='onnx_dereverb_By_FoxJoy';infos=[]
	try:
		inp_root=inp_root.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P);save_root_vocal=save_root_vocal.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P);save_root_ins=save_root_ins.strip(_P).strip(_Q).strip(_F).strip(_Q).strip(_P)
		if model_name==A:pre_fun=MDXNetDereverb(15)
		else:func=_audio_pre_ if'DeEcho'not in model_name else _audio_pre_new;pre_fun=func(agg=int(agg),model_path=os.path.join(weight_uvr5_root,model_name+_h),device=config.device,is_half=config.is_half)
		if inp_root!='':paths=[os.path.join(inp_root,name)for name in os.listdir(inp_root)]
		else:paths=[path.name for path in paths]
		for path in paths:
			inp_path=os.path.join(inp_root,path);need_reformat=1;done=0
			try:
				info=ffmpeg.probe(inp_path,cmd='ffprobe')
				if info[B][0]['channels']==2 and info[B][0][_AE]=='44100':need_reformat=0;pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0);done=1
			except:need_reformat=1;traceback.print_exc()
			if need_reformat==1:tmp_path='%s/%s.reformatted.wav'%(tmp,os.path.basename(inp_path));os.system('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(inp_path,tmp_path));inp_path=tmp_path
			try:
				if done==0:pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0)
				infos.append('%s->Success'%os.path.basename(inp_path));yield _F.join(infos)
			except:infos.append('%s->%s'%(os.path.basename(inp_path),traceback.format_exc()));yield _F.join(infos)
	except:infos.append(traceback.format_exc());yield _F.join(infos)
	finally:
		try:
			if model_name==A:del pre_fun.pred.model;del pre_fun.pred.model_
			else:del pre_fun.model;del pre_fun
		except:traceback.print_exc()
		print(_AF)
		if torch.cuda.is_available():torch.cuda.empty_cache()
	yield _F.join(infos)
def get_vc(sid,to_return_protect0,to_return_protect1):
	global n_spk,tgt_sr,net_g,vc,cpt,version
	if sid==''or sid==[]:
		global hubert_model
		if hubert_model is not _J:
			print(_AF);del net_g,n_spk,vc,hubert_model,tgt_sr;hubert_model=net_g=n_spk=vc=hubert_model=tgt_sr=_J
			if torch.cuda.is_available():torch.cuda.empty_cache()
			if_f0=cpt.get(_j,1);version=cpt.get(_b,_G)
			if version==_G:
				if if_f0==1:net_g=SynthesizerTrnMs256NSFsid(*cpt[_O],is_half=config.is_half)
				else:net_g=SynthesizerTrnMs256NSFsid_nono(*cpt[_O])
			elif version==_Y:
				if if_f0==1:net_g=SynthesizerTrnMs768NSFsid(*cpt[_O],is_half=config.is_half)
				else:net_g=SynthesizerTrnMs768NSFsid_nono(*cpt[_O])
			del net_g,cpt
			if torch.cuda.is_available():torch.cuda.empty_cache()
			cpt=_J
		return{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C}
	person=_a%(weight_root,sid);print('loading %s'%person);cpt=torch.load(person,map_location=_x);tgt_sr=cpt[_O][-1];cpt[_O][-3]=cpt[_n][_AG].shape[0];if_f0=cpt.get(_j,1)
	if if_f0==0:to_return_protect0=to_return_protect1={_E:_D,_M:.5,_B:_C}
	else:to_return_protect0={_E:_A,_M:to_return_protect0,_B:_C};to_return_protect1={_E:_A,_M:to_return_protect1,_B:_C}
	version=cpt.get(_b,_G)
	if version==_G:
		if if_f0==1:net_g=SynthesizerTrnMs256NSFsid(*cpt[_O],is_half=config.is_half)
		else:net_g=SynthesizerTrnMs256NSFsid_nono(*cpt[_O])
	elif version==_Y:
		if if_f0==1:net_g=SynthesizerTrnMs768NSFsid(*cpt[_O],is_half=config.is_half)
		else:net_g=SynthesizerTrnMs768NSFsid_nono(*cpt[_O])
	del net_g.enc_q;print(net_g.load_state_dict(cpt[_n],strict=_D));net_g.eval().to(config.device)
	if config.is_half:net_g=net_g.half()
	else:net_g=net_g.float()
	vc=VC(tgt_sr,config);n_spk=cpt[_O][-3];return{_E:_A,'maximum':n_spk,_B:_C},to_return_protect0,to_return_protect1
def change_choices():
	names=[]
	for name in os.listdir(weight_root):
		if name.endswith(_h):names.append(name)
	index_paths=[];audio_paths=[];audios_path=os.path.abspath(os.getcwd())+_o
	for(root,dirs,files)in os.walk(index_root,topdown=_D):
		for name in files:
			if name.endswith(_i)and _m not in name:index_paths.append(_a%(root,name))
	for file in os.listdir(audios_path):audio_paths.append(_a%(audio_root,file))
	return{_c:sorted(names),_B:_C},{_c:sorted(index_paths),_B:_C},{_c:sorted(audio_paths),_B:_C}
def clean():return{_M:'',_B:_C}
sr_dict={_p:32000,_S:40000,_d:48000}
def if_done(done,p):
	while 1:
		if p.poll()is _J:sleep(.5)
		else:break
	done[0]=_A
def if_done_multi(done,ps):
	while 1:
		flag=1
		for p in ps:
			if p.poll()is _J:flag=0;sleep(.5);break
		if flag==1:break
	done[0]=_A
def formant_enabled(cbox,qfrency,tmbre,frmntapply,formantpreset,formant_refresh_button):
	if cbox:DoFormant=_A;CSVutil(_V,_X,_W,DoFormant,qfrency,tmbre);return{_M:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C}
	else:DoFormant=_D;CSVutil(_V,_X,_W,DoFormant,qfrency,tmbre);return{_M:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C}
def formant_apply(qfrency,tmbre):Quefrency=qfrency;Timbre=tmbre;DoFormant=_A;CSVutil(_V,_X,_W,DoFormant,qfrency,tmbre);return{_M:Quefrency,_B:_C},{_M:Timbre,_B:_C}
def update_fshift_presets(preset,qfrency,tmbre):
	qfrency,tmbre=preset_apply(preset,qfrency,tmbre)
	if str(preset)!='':
		with open(str(preset),_I)as p:content=p.readlines();qfrency,tmbre=content[0].split(_F)[0],content[1];formant_apply(qfrency,tmbre)
	else:0
	return{_c:get_fshift_presets(),_B:_C},{_M:qfrency,_B:_C},{_M:tmbre,_B:_C}
def preprocess_dataset(trainset_dir,exp_dir,sr,n_p):
	A='%s/logs/%s/preprocess.log';sr=sr_dict[sr];os.makedirs(_l%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),_U);f.close();cmd=config.python_cmd+' trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s '%(trainset_dir,sr,n_p,now_dir,exp_dir)+str(config.noparallel);print(cmd);p=Popen(cmd,shell=_A);done=[_D];threading.Thread(target=if_done,args=(done,p)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
	print(log);yield log
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir,version19,echl):
	A='%s/logs/%s/extract_f0_feature.log';gpus=gpus.split('-');os.makedirs(_l%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),_U);f.close()
	if if_f0:
		cmd=config.python_cmd+' extract_f0_print.py %s/logs/%s %s %s %s'%(now_dir,exp_dir,n_p,f0method,echl);print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);done=[_D];threading.Thread(target=if_done,args=(done,p)).start()
		while 1:
			with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
			sleep(1)
			if done[0]:break
		with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
		print(log);yield log
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd=config.python_cmd+' extract_feature_print.py %s %s %s %s %s/logs/%s %s'%(config.device,leng,idx,n_g,now_dir,exp_dir,version19);print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	done=[_D];threading.Thread(target=if_done_multi,args=(done,ps)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
	print(log);yield log
def change_sr2(sr2,if_f0_3,version19):
	path_str=''if version19==_G else _y;f0_str=_j if if_f0_3 else'';if_pretrained_generator_exist=os.access(_e%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(_f%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(_e%(path_str,f0_str,sr2),_q)
	if not if_pretrained_discriminator_exist:print(_f%(path_str,f0_str,sr2),_q)
	return _e%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',_f%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else''
def change_version19(sr2,if_f0_3,version19):
	path_str=''if version19==_G else _y
	if sr2==_p and version19==_G:sr2=_S
	to_return_sr2={_c:[_S,_d],_B:_C,_M:sr2}if version19==_G else{_c:[_S,_d,_p],_B:_C,_M:sr2};f0_str=_j if if_f0_3 else'';if_pretrained_generator_exist=os.access(_e%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(_f%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(_e%(path_str,f0_str,sr2),_q)
	if not if_pretrained_discriminator_exist:print(_f%(path_str,f0_str,sr2),_q)
	return _e%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',_f%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else'',to_return_sr2
def change_f0(if_f0_3,sr2,version19,step2b,gpus6,gpu_info9,extraction_crepe_hop_length,but2,info2):
	C='not exist, will not use pretrained model';B='pretrained%s/f0D%s.pth';A='pretrained%s/f0G%s.pth';path_str=''if version19==_G else _y;if_pretrained_generator_exist=os.access(A%(path_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(B%(path_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(A%(path_str,sr2),C)
	if not if_pretrained_discriminator_exist:print(B%(path_str,sr2),C)
	if if_f0_3:return{_E:_A,_B:_C},A%(path_str,sr2)if if_pretrained_generator_exist else'',B%(path_str,sr2)if if_pretrained_discriminator_exist else'',{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C},{_E:_A,_B:_C}
	return{_E:_D,_B:_C},'pretrained%s/G%s.pth'%(path_str,sr2)if if_pretrained_generator_exist else'','pretrained%s/D%s.pth'%(path_str,sr2)if if_pretrained_discriminator_exist else'',{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C},{_E:_D,_B:_C}
global log_interval
def set_log_interval(exp_dir,batch_size12):
	log_interval=1;folder_path=os.path.join(exp_dir,'1_16k_wavs')
	if os.path.exists(folder_path)and os.path.isdir(folder_path):
		wav_files=[f for f in os.listdir(folder_path)if f.endswith('.wav')]
		if wav_files:
			sample_size=len(wav_files);log_interval=math.ceil(sample_size/batch_size12)
			if log_interval>1:log_interval+=1
	return log_interval
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19):
	A='\x08';CSVutil(_u,_X,_W,_D);exp_dir=_l%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);gt_wavs_dir=_AH%exp_dir;feature_dir=_z%exp_dir if version19==_G else _A0%exp_dir;log_interval=set_log_interval(exp_dir,batch_size12)
	if if_f0_3:f0_dir='%s/2a_f0'%exp_dir;f0nsf_dir=_AI%exp_dir;names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])&set([name.split(_L)[0]for name in os.listdir(f0_dir)])&set([name.split(_L)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append(_AJ%(gt_wavs_dir.replace(_H,_R),name,feature_dir.replace(_H,_R),name,f0_dir.replace(_H,_R),name,f0nsf_dir.replace(_H,_R),name,spk_id5))
		else:opt.append(_AK%(gt_wavs_dir.replace(_H,_R),name,feature_dir.replace(_H,_R),name,spk_id5))
	fea_dim=256 if version19==_G else 768
	if if_f0_3:
		for _ in range(2):opt.append(_AL%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append(_AM%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open(_AN%exp_dir,_U)as f:f.write(_F.join(opt))
	print(_AO);print('use gpus:',gpus16)
	if pretrained_G14=='':print('no pretrained Generator')
	if pretrained_D15=='':print('no pretrained Discriminator')
	if gpus16:cmd=config.python_cmd+' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s'%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,_r%pretrained_G14 if pretrained_G14!=''else'',_s%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19,log_interval)
	else:cmd=config.python_cmd+' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s'%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,_r%pretrained_G14 if pretrained_G14!=''else A,_s%pretrained_D15 if pretrained_D15!=''else A,1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19,log_interval)
	print(cmd);global p;p=Popen(cmd,shell=_A,cwd=now_dir);global PID;PID=p.pid;p.wait();return _AP,{_E:_D,_B:_C},{_E:_A,_B:_C}
def train_index(exp_dir1,version19):
	exp_dir=_l%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);feature_dir=_z%exp_dir if version19==_G else _A0%exp_dir
	if not os.path.exists(feature_dir):return'请先进行特征提取!'
	listdir_res=list(os.listdir(feature_dir))
	if len(listdir_res)==0:return'请先进行特征提取！'
	infos=[];npys=[]
	for name in sorted(listdir_res):phone=np.load(_a%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		infos.append(_AQ%big_npy.shape[0]);yield _F.join(infos)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_D,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();print(info);infos.append(info);yield _F.join(infos)
	np.save(_AR%exp_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);infos.append('%s,%s'%(big_npy.shape,n_ivf));yield _F.join(infos);index=faiss.index_factory(256 if version19==_G else 768,_AS%n_ivf);infos.append('training');yield _F.join(infos);index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,_AT%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('adding');yield _F.join(infos);batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,_AU%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('Successful Index Construction，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19));yield _F.join(infos)
def train1key(exp_dir1,sr2,if_f0_3,trainset_dir4,spk_id5,np7,f0method8,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19,echl):
	infos=[]
	def get_info_str(strr):infos.append(strr);return _F.join(infos)
	model_log_dir=_l%(now_dir,exp_dir1);preprocess_log_path='%s/preprocess.log'%model_log_dir;extract_f0_feature_log_path='%s/extract_f0_feature.log'%model_log_dir;gt_wavs_dir=_AH%model_log_dir;feature_dir=_z%model_log_dir if version19==_G else _A0%model_log_dir;os.makedirs(model_log_dir,exist_ok=_A);open(preprocess_log_path,_U).close();cmd=config.python_cmd+' trainset_preprocess_pipeline_print.py %s %s %s %s '%(trainset_dir4,sr_dict[sr2],np7,model_log_dir)+str(config.noparallel);yield get_info_str(i18n('step1:正在处理数据'));yield get_info_str(cmd);p=Popen(cmd,shell=_A);p.wait()
	with open(preprocess_log_path,_I)as f:print(f.read())
	open(extract_f0_feature_log_path,_U)
	if if_f0_3:
		yield get_info_str('step2a:正在提取音高');cmd=config.python_cmd+' extract_f0_print.py %s %s %s %s'%(model_log_dir,np7,f0method8,echl);yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait()
		with open(extract_f0_feature_log_path,_I)as f:print(f.read())
	else:yield get_info_str(i18n('step2a:无需提取音高'))
	yield get_info_str(i18n('step2b:正在提取特征'));gpus=gpus16.split('-');leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd=config.python_cmd+' extract_feature_print.py %s %s %s %s %s %s'%(config.device,leng,idx,n_g,model_log_dir,version19);yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	for p in ps:p.wait()
	with open(extract_f0_feature_log_path,_I)as f:print(f.read())
	yield get_info_str(i18n('step3a:正在训练模型'))
	if if_f0_3:f0_dir='%s/2a_f0'%model_log_dir;f0nsf_dir=_AI%model_log_dir;names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])&set([name.split(_L)[0]for name in os.listdir(f0_dir)])&set([name.split(_L)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append(_AJ%(gt_wavs_dir.replace(_H,_R),name,feature_dir.replace(_H,_R),name,f0_dir.replace(_H,_R),name,f0nsf_dir.replace(_H,_R),name,spk_id5))
		else:opt.append(_AK%(gt_wavs_dir.replace(_H,_R),name,feature_dir.replace(_H,_R),name,spk_id5))
	fea_dim=256 if version19==_G else 768
	if if_f0_3:
		for _ in range(2):opt.append(_AL%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append(_AM%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open(_AN%model_log_dir,_U)as f:f.write(_F.join(opt))
	yield get_info_str(_AO)
	if gpus16:cmd=config.python_cmd+' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,_r%pretrained_G14 if pretrained_G14!=''else'',_s%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19)
	else:cmd=config.python_cmd+' train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,_r%pretrained_G14 if pretrained_G14!=''else'',_s%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19)
	yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();yield get_info_str(i18n(_AP));npys=[];listdir_res=list(os.listdir(feature_dir))
	for name in sorted(listdir_res):phone=np.load(_a%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		info=_AQ%big_npy.shape[0];print(info);yield get_info_str(info)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_D,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();print(info);yield get_info_str(info)
	np.save(_AR%model_log_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);yield get_info_str('%s,%s'%(big_npy.shape,n_ivf));index=faiss.index_factory(256 if version19==_G else 768,_AS%n_ivf);yield get_info_str('training index');index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,_AT%(model_log_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str('adding index');batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,_AU%(model_log_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str(i18n('全流程结束！'))
def change_info_(ckpt_path):
	A='train.log'
	if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path),A)):return{_B:_C},{_B:_C},{_B:_C}
	try:
		with open(ckpt_path.replace(os.path.basename(ckpt_path),A),_I)as f:info=eval(f.read().strip(_F).split(_F)[0].split('\t')[-1]);sr,f0=info[_AE],info['if_f0'];version=_Y if _b in info and info[_b]==_Y else _G;return sr,str(f0),version
	except:traceback.print_exc();return{_B:_C},{_B:_C},{_B:_C}
def export_onnx(ModelPath,ExportedPath):D='rnd';C='pitchf';B='pitch';A='phone';cpt=torch.load(ModelPath,map_location=_x);cpt[_O][-3]=cpt[_n][_AG].shape[0];vec_channels=256 if cpt.get(_b,_G)==_G else 768;test_phone=torch.rand(1,200,vec_channels);test_phone_lengths=torch.tensor([200]).long();test_pitch=torch.randint(size=(1,200),low=5,high=255);test_pitchf=torch.rand(1,200);test_ds=torch.LongTensor([0]);test_rnd=torch.rand(1,192,200);device=_x;net_g=SynthesizerTrnMsNSFsidM(*cpt[_O],is_half=_D,version=cpt.get(_b,_G));net_g.load_state_dict(cpt[_n],strict=_D);input_names=[A,'phone_lengths',B,C,'ds',D];output_names=['audio'];torch.onnx.export(net_g,(test_phone.to(device),test_phone_lengths.to(device),test_pitch.to(device),test_pitchf.to(device),test_ds.to(device),test_rnd.to(device)),ExportedPath,dynamic_axes={A:[1],B:[1],C:[1],D:[2]},do_constant_folding=_D,opset_version=13,verbose=_D,input_names=input_names,output_names=output_names);return'Finished'
import re as regex,scipy.io.wavfile as wavfile
cli_current_page='HOME'
def cli_split_command(com):exp='(?:(?<=\\s)|^)"(.*?)"(?=\\s|$)|(\\S+)';split_array=regex.findall(exp,com);split_array=[group[0]if group[0]else group[1]for group in split_array];return split_array
def execute_generator_function(genObject):
	for _ in genObject:0
def cli_infer(com):
	A='audio-outputs';com=cli_split_command(com);model_name=com[0];source_audio_path=com[1];output_file_name=com[2];feature_index_path=com[3];f0_file=_J;speaker_id=int(com[4]);transposition=float(com[5]);f0_method=com[6];crepe_hop_length=int(com[7]);harvest_median_filter=int(com[8]);resample=int(com[9]);mix=float(com[10]);feature_ratio=float(com[11]);protection_amnt=float(com[12]);protect1=.5
	if com[14]=='False'or com[14]=='false':DoFormant=_D;Quefrency=.0;Timbre=.0;CSVutil(_V,_X,_W,DoFormant,Quefrency,Timbre)
	else:DoFormant=_A;Quefrency=float(com[15]);Timbre=float(com[16]);CSVutil(_V,_X,_W,DoFormant,Quefrency,Timbre)
	print('Mangio-RVC-Fork Infer-CLI: Starting the inference...');vc_data=get_vc(model_name,protection_amnt,protect1);print(vc_data);print('Mangio-RVC-Fork Infer-CLI: Performing inference...');conversion_data=vc_single(speaker_id,source_audio_path,source_audio_path,transposition,f0_file,f0_method,feature_index_path,feature_index_path,feature_ratio,harvest_median_filter,resample,mix,protection_amnt,crepe_hop_length)
	if'Success.'in conversion_data[0]:print('Mangio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s...'%(A,output_file_name));wavfile.write(_a%(A,output_file_name),conversion_data[1][0],conversion_data[1][1]);print('Mangio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s'%(A,output_file_name))
	else:print("Mangio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ");print(conversion_data[0])
def cli_pre_process(com):com=cli_split_command(com);model_name=com[0];trainset_directory=com[1];sample_rate=com[2];num_processes=int(com[3]);print('Mangio-RVC-Fork Pre-process: Starting...');generator=preprocess_dataset(trainset_directory,model_name,sample_rate,num_processes);execute_generator_function(generator);print('Mangio-RVC-Fork Pre-process: Finished')
def cli_extract_feature(com):com=cli_split_command(com);model_name=com[0];gpus=com[1];num_processes=int(com[2]);has_pitch_guidance=_A if int(com[3])==1 else _D;f0_method=com[4];crepe_hop_length=int(com[5]);version=com[6];print('Mangio-RVC-CLI: Extract Feature Has Pitch: '+str(has_pitch_guidance));print('Mangio-RVC-CLI: Extract Feature Version: '+str(version));print('Mangio-RVC-Fork Feature Extraction: Starting...');generator=extract_f0_feature(gpus,num_processes,f0_method,has_pitch_guidance,model_name,version,crepe_hop_length);execute_generator_function(generator);print('Mangio-RVC-Fork Feature Extraction: Finished')
def cli_train(com):com=cli_split_command(com);model_name=com[0];sample_rate=com[1];has_pitch_guidance=_A if int(com[2])==1 else _D;speaker_id=int(com[3]);save_epoch_iteration=int(com[4]);total_epoch=int(com[5]);batch_size=int(com[6]);gpu_card_slot_numbers=com[7];if_save_latest=_A if int(com[8])==1 else _D;if_cache_gpu=_A if int(com[9])==1 else _D;if_save_every_weight=_A if int(com[10])==1 else _D;version=com[11];pretrained_base='/kaggle/input/ax-rmf/pretrained/'if version==_G else'/kaggle/input/ax-rmf/pretrained_v2/';g_pretrained_path='%sf0G%s.pth'%(pretrained_base,sample_rate);d_pretrained_path='%sf0D%s.pth'%(pretrained_base,sample_rate);print('Mangio-RVC-Fork Train-CLI: Training...');click_train(model_name,sample_rate,has_pitch_guidance,speaker_id,save_epoch_iteration,total_epoch,batch_size,if_save_latest,g_pretrained_path,d_pretrained_path,gpu_card_slot_numbers,if_cache_gpu,if_save_every_weight,version)
def cli_train_feature(com):com=cli_split_command(com);model_name=com[0];version=com[1];print('Mangio-RVC-Fork Train Feature Index-CLI: Training... Please wait');generator=train_index(model_name,version);execute_generator_function(generator);print('Mangio-RVC-Fork Train Feature Index-CLI: Done!')
def cli_extract_model(com):
	com=cli_split_command(com);model_path=com[0];save_name=com[1];sample_rate=com[2];has_pitch_guidance=com[3];info=com[4];version=com[5];extract_small_model_process=extract_small_model(model_path,save_name,sample_rate,has_pitch_guidance,info,version)
	if extract_small_model_process=='Success.':print('Mangio-RVC-Fork Extract Small Model: Success!')
	else:print(str(extract_small_model_process));print('Mangio-RVC-Fork Extract Small Model: Failed!')
def preset_apply(preset,qfer,tmbr):
	if str(preset)!='':
		with open(str(preset),_I)as p:content=p.readlines();qfer,tmbr=content[0].split(_F)[0],content[1];formant_apply(qfer,tmbr)
	else:0
	return{_M:qfer,_B:_C},{_M:tmbr,_B:_C}
def print_page_details():
	if cli_current_page=='HOME':print('\n    go home            : Takes you back to home with a navigation list.\n    go infer           : Takes you to inference command execution.\n    go pre-process     : Takes you to training step.1) pre-process command execution.\n    go extract-feature : Takes you to training step.2) extract-feature command execution.\n    go train           : Takes you to training step.3) being or continue training command execution.\n    go train-feature   : Takes you to the train feature index command execution.\n    go extract-model   : Takes you to the extract small model command execution.')
	elif cli_current_page==_A1:print("\n    arg 1) model name with .pth in ./weights: mi-test.pth\n    arg 2) source audio path: myFolder\\MySource.wav\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index\n    arg 5) speaker id: 0\n    arg 6) transposition: 0\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)\n    arg 8) crepe hop length: 160\n    arg 9) harvest median filter radius: 3 (0-7)\n    arg 10) post resample rate: 0\n    arg 11) mix volume envelope: 1\n    arg 12) feature index ratio: 0.78 (0-1)\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2")
	elif cli_current_page==_A2:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Number of CPU threads to use: 8 \n\nExample: mi-test mydataset 40k 24')
	elif cli_current_page==_A3:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 3) Number of CPU threads to use: 8\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)\n    arg 6) Crepe hop length: 128\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n\nExample: mi-test 0 24 1 harvest 128 v2')
	elif cli_current_page==_A4:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Sample rate: 40k (32k, 40k, 48k)\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 4) speaker id: 0\n    arg 5) Save epoch iteration: 50\n    arg 6) Total epochs: 10000\n    arg 7) Batch size: 8\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2')
	elif cli_current_page==_A5:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test v2')
	elif cli_current_page==_A6:print('\n    arg 1) Model Path: logs/mi-test/G_168000.pth\n    arg 2) Model save name: MyModel\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) Model information: "My Model"\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2')
def change_page(page):global cli_current_page;cli_current_page=page;return 0
def execute_command(com):
	if com=='go home':return change_page('HOME')
	elif com=='go infer':return change_page(_A1)
	elif com=='go pre-process':return change_page(_A2)
	elif com=='go extract-feature':return change_page(_A3)
	elif com=='go train':return change_page(_A4)
	elif com=='go train-feature':return change_page(_A5)
	elif com=='go extract-model':return change_page(_A6)
	elif com[:3]=='go ':print("page '%s' does not exist!"%com[3:]);return 0
	if cli_current_page==_A1:cli_infer(com)
	elif cli_current_page==_A2:cli_pre_process(com)
	elif cli_current_page==_A3:cli_extract_feature(com)
	elif cli_current_page==_A4:cli_train(com)
	elif cli_current_page==_A5:cli_train_feature(com)
	elif cli_current_page==_A6:cli_extract_model(com)
def cli_navigation_loop():
	while _A:
		print("\nYou are currently in '%s':"%cli_current_page);print_page_details();command=input('%s: '%cli_current_page)
		try:execute_command(command)
		except:print(traceback.format_exc())
if config.is_cli:print('\n\nMangio-RVC-Fork v2 CLI App!\n');print('Welcome to the CLI version of RVC. Please read the documentation on https://github.com/Mangio621/Mangio-RVC-Fork (README.MD) to understand how to use this app.\n');cli_navigation_loop()
def get_presets():
	data=_J
	with open(_t,_I)as file:data=json.load(file)
	preset_names=[]
	for preset in data[_A7]:preset_names.append(preset['name'])
	return preset_names
def stepdisplay(if_save_every_weights):return{_E:if_save_every_weights,_B:_C}
def match_index(sid0):
	picked=_D;folder=sid0.split(_L)[0].split('_')[0];parent_dir=_Z+folder
	if os.path.exists(parent_dir):
		for filename in os.listdir(parent_dir.replace(_H,_N)):
			if filename.endswith(_i):
				for i in range(len(indexes_list)):
					if indexes_list[i]==os.path.join(_Z+folder,filename).replace(_H,_N):break
					elif indexes_list[i]==os.path.join(_Z+folder.lower(),filename).replace(_H,_N):parent_dir=_Z+folder.lower();break
				index_path=os.path.join(parent_dir.replace(_H,_N),filename.replace(_H,_N)).replace(_H,_N);return index_path,index_path
	else:return'',''
def stoptraining(mim):
	if int(mim)==1:
		CSVutil(_u,_X,'stop','True')
		try:os.kill(PID,signal.SIGTERM)
		except Exception as e:print(f"Couldn't click due to {e}");pass
	else:0
	return{_E:_D,_B:_C},{_E:_A,_B:_C}
def whethercrepeornah(radio):mango=_A if radio==_A8 or radio==_AV else _D;return{_E:mango,_B:_C}
with gr.Blocks(theme=gr.themes.Soft(),title='Mangio-RVC-Web 💻')as app:
	gr.HTML('<h1> The Mangio-RVC-Fork 💻 </h1>');gr.Markdown(value=i18n('本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>使用需遵守的协议-LICENSE.txt</b>.'))
	with gr.Tabs():
		with gr.TabItem(i18n('模型推理')):
			with gr.Row():sid0=gr.Dropdown(label=i18n('推理音色'),choices=sorted(names),value='');refresh_button=gr.Button(i18n('Refresh voice list, index path and audio files'),variant=_K);clean_button=gr.Button(i18n('卸载音色省显存'),variant=_K);spk_item=gr.Slider(minimum=0,maximum=2333,step=1,label=i18n('请选择说话人id'),value=0,visible=_D,interactive=_A);clean_button.click(fn=clean,inputs=[],outputs=[sid0])
			with gr.Group():
				gr.Markdown(value=i18n('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr.Row():
					with gr.Column():vc_transform0=gr.Number(label=i18n(_AW),value=0);input_audio0=gr.Textbox(label=i18n("Add audio's name to the path to the audio file to be processed (default is the correct format example) Remove the path to use an audio from the dropdown list:"),value=os.path.abspath(os.getcwd()).replace(_H,_N)+_o+'audio.wav');input_audio1=gr.Dropdown(label=i18n('Auto detect audio path and select from the dropdown:'),choices=sorted(audio_paths),value='',interactive=_A);input_audio1.change(fn=lambda:'',inputs=[],outputs=[input_audio0]);f0method0=gr.Radio(label=i18n(_AX),choices=['pm',_A9,'dio',_AA,'crepe-tiny',_A8,_AV,_g],value=_g,interactive=_A);crepe_hop_length=gr.Slider(minimum=1,maximum=512,step=1,label=i18n(_AB),value=120,interactive=_A,visible=_D);f0method0.change(fn=whethercrepeornah,inputs=[f0method0],outputs=[crepe_hop_length]);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(_AY),value=3,step=1,interactive=_A)
					with gr.Column():file_index1=gr.Textbox(label=i18n(_AZ),value='',interactive=_A);file_index2=gr.Dropdown(label="3. Path to your added.index file (if it didn't automatically find it.)",choices=get_indexes(),value=get_index(),interactive=_A,allow_custom_value=_A);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2,input_audio1]);index_rate1=gr.Slider(minimum=0,maximum=1,label=i18n('检索特征占比'),value=.75,interactive=_A)
					with gr.Column():resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(_Aa),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(_Ab),value=.25,interactive=_A);protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(_Ac),value=.33,step=.01,interactive=_A);formanting=gr.Checkbox(value=bool(DoFormant),label='[EXPERIMENTAL] Formant shift inference audio',info='Used for male to female and vice-versa conversions',interactive=_A,visible=_A);formant_preset=gr.Dropdown(value='',choices=get_fshift_presets(),label='browse presets for formanting',visible=bool(DoFormant));formant_refresh_button=gr.Button(value='🔄',visible=bool(DoFormant),variant=_K);qfrency=gr.Slider(value=Quefrency,info=_Ad,label='Quefrency for formant shifting',minimum=.0,maximum=16.,step=.1,visible=bool(DoFormant),interactive=_A);tmbre=gr.Slider(value=Timbre,info=_Ad,label='Timbre for formant shifting',minimum=.0,maximum=16.,step=.1,visible=bool(DoFormant),interactive=_A);formant_preset.change(fn=preset_apply,inputs=[formant_preset,qfrency,tmbre],outputs=[qfrency,tmbre]);frmntbut=gr.Button('Apply',variant=_K,visible=bool(DoFormant));formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button]);frmntbut.click(fn=formant_apply,inputs=[qfrency,tmbre],outputs=[qfrency,tmbre]);formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset,qfrency,tmbre],outputs=[formant_preset,qfrency,tmbre])
					f0_file=gr.File(label=i18n('F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调'));but0=gr.Button(i18n('转换'),variant=_K)
					with gr.Row():vc_output1=gr.Textbox(label=i18n(_T));vc_output2=gr.Audio(label=i18n('输出音频(右下角三个点,点了可以下载)'))
					but0.click(vc_single,[spk_item,input_audio0,input_audio1,vc_transform0,f0_file,f0method0,file_index1,file_index2,index_rate1,filter_radius0,resample_sr0,rms_mix_rate0,protect0,crepe_hop_length],[vc_output1,vc_output2])
			with gr.Group():
				gr.Markdown(value=i18n('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr.Row():
					with gr.Column():vc_transform1=gr.Number(label=i18n(_AW),value=0);opt_input=gr.Textbox(label=i18n('指定输出文件夹'),value='opt');f0method1=gr.Radio(label=i18n(_AX),choices=['pm',_A9,_AA,_g],value=_g,interactive=_A);filter_radius1=gr.Slider(minimum=0,maximum=7,label=i18n(_AY),value=3,step=1,interactive=_A)
					with gr.Column():file_index3=gr.Textbox(label=i18n(_AZ),value='',interactive=_A);file_index4=gr.Dropdown(label=i18n('自动检测index路径,下拉式选择(dropdown)'),choices=get_indexes(),value=get_index(),interactive=_A);sid0.select(fn=match_index,inputs=[sid0],outputs=[file_index2,file_index4]);refresh_button.click(fn=lambda:change_choices()[1],inputs=[],outputs=file_index4);index_rate2=gr.Slider(minimum=0,maximum=1,label=i18n('检索特征占比'),value=1,interactive=_A)
					with gr.Column():resample_sr1=gr.Slider(minimum=0,maximum=48000,label=i18n(_Aa),value=0,step=1,interactive=_A);rms_mix_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_Ab),value=1,interactive=_A);protect1=gr.Slider(minimum=0,maximum=.5,label=i18n(_Ac),value=.33,step=.01,interactive=_A)
					with gr.Column():dir_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value=os.path.abspath(os.getcwd()).replace(_H,_N)+_o);inputs=gr.File(file_count='multiple',label=i18n(_Ae))
					with gr.Row():format1=gr.Radio(label=i18n('导出文件格式'),choices=[_v,_k,_w,'m4a'],value=_k,interactive=_A);but1=gr.Button(i18n('转换'),variant=_K);vc_output3=gr.Textbox(label=i18n(_T))
					but1.click(vc_multi,[spk_item,dir_input,opt_input,inputs,vc_transform1,f0method1,file_index3,file_index4,index_rate2,filter_radius1,resample_sr1,rms_mix_rate1,protect1,format1,crepe_hop_length],[vc_output3])
			sid0.change(fn=get_vc,inputs=[sid0,protect0,protect1],outputs=[spk_item,protect0,protect1])
		with gr.TabItem(i18n('伴奏人声分离&去混响&去回声')):
			with gr.Group():
				gr.Markdown(value=i18n('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr.Row():
					with gr.Column():dir_wav_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径'),value=os.getcwd().replace(_H,_N)+_o);wav_inputs=gr.File(file_count='multiple',label=i18n(_Ae))
					with gr.Column():model_choose=gr.Dropdown(label=i18n('模型'),choices=uvr5_names);agg=gr.Slider(minimum=0,maximum=20,step=1,label='人声提取激进程度',value=10,interactive=_A,visible=_D);opt_vocal_root=gr.Textbox(label=i18n('指定输出主人声文件夹'),value='opt');opt_ins_root=gr.Textbox(label=i18n('指定输出非主人声文件夹'),value='opt');format0=gr.Radio(label=i18n('导出文件格式'),choices=[_v,_k,_w,'m4a'],value=_k,interactive=_A)
					but2=gr.Button(i18n('转换'),variant=_K);vc_output4=gr.Textbox(label=i18n(_T));but2.click(uvr,[model_choose,dir_wav_input,opt_vocal_root,wav_inputs,opt_ins_root,agg,format0],[vc_output4])
		with gr.TabItem(i18n('训练')):
			gr.Markdown(value=i18n('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. '))
			with gr.Row():exp_dir1=gr.Textbox(label=i18n('输入实验名'),value='mi-test');sr2=gr.Radio(label=i18n(_AC),choices=[_S,_d],value=_S,interactive=_A);if_f0_3=gr.Checkbox(label=_AD,value=_A,interactive=_A);version19=gr.Radio(label=i18n('版本'),choices=[_G,_Y],value=_G,interactive=_A,visible=_A);np7=gr.Slider(minimum=0,maximum=config.n_cpu,step=1,label=i18n('提取音高和处理数据使用的CPU进程数'),value=int(np.ceil(config.n_cpu/1.5)),interactive=_A)
			with gr.Group():
				gr.Markdown(value=i18n('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. '))
				with gr.Row():trainset_dir4=gr.Textbox(label=i18n('输入训练文件夹路径'),value=os.path.abspath(os.getcwd())+'\\datasets\\');spk_id5=gr.Slider(minimum=0,maximum=4,step=1,label=i18n('请指定说话人id'),value=0,interactive=_A);but1=gr.Button(i18n('处理数据'),variant=_K);info1=gr.Textbox(label=i18n(_T),value='');but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2,np7],[info1])
			with gr.Group():
				step2b=gr.Markdown(value=i18n('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)'))
				with gr.Row():
					with gr.Column():gpus6=gr.Textbox(label=i18n(_Af),value=gpus,interactive=_A);gpu_info9=gr.Textbox(label=i18n('显卡信息'),value=gpu_info)
					with gr.Column():f0method8=gr.Radio(label=i18n('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices=['pm',_A9,'dio',_AA,_A8,_g],value=_g,interactive=_A);extraction_crepe_hop_length=gr.Slider(minimum=1,maximum=512,step=1,label=i18n(_AB),value=64,interactive=_A,visible=_D);f0method8.change(fn=whethercrepeornah,inputs=[f0method8],outputs=[extraction_crepe_hop_length])
					but2=gr.Button(i18n('特征提取'),variant=_K);info2=gr.Textbox(label=i18n(_T),value='',max_lines=8,interactive=_D);but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1,version19,extraction_crepe_hop_length],[info2])
			with gr.Group():
				gr.Markdown(value=i18n('step3: 填写训练设置, 开始训练模型和索引'))
				with gr.Row():save_epoch10=gr.Slider(minimum=1,maximum=50,step=1,label=i18n('保存频率save_every_epoch'),value=5,interactive=_A,visible=_A);total_epoch11=gr.Slider(minimum=1,maximum=10000,step=1,label=i18n('总训练轮数total_epoch'),value=20,interactive=_A);batch_size12=gr.Slider(minimum=1,maximum=40,step=1,label=i18n('每张显卡的batch_size'),value=default_batch_size,interactive=_A);if_save_latest13=gr.Checkbox(label='Whether to save only the latest .ckpt file to save hard drive space',value=_A,interactive=_A);if_cache_gpu17=gr.Checkbox(label='Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement',value=_D,interactive=_A);if_save_every_weights18=gr.Checkbox(label="Save a small final model to the 'weights' folder at each save point",value=_A,interactive=_A)
				with gr.Row():pretrained_G14=gr.Textbox(lines=2,label=i18n('加载预训练底模G路径'),value='/kaggle/input/ax-rmf/pretrained/f0G40k.pth',interactive=_A);pretrained_D15=gr.Textbox(lines=2,label=i18n('加载预训练底模D路径'),value='/kaggle/input/ax-rmf/pretrained/f0D40k.pth',interactive=_A);sr2.change(change_sr2,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15]);version19.change(change_version19,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15,sr2]);if_f0_3.change(fn=change_f0,inputs=[if_f0_3,sr2,version19,step2b,gpus6,gpu_info9,extraction_crepe_hop_length,but2,info2],outputs=[f0method8,pretrained_G14,pretrained_D15,step2b,gpus6,gpu_info9,extraction_crepe_hop_length,but2,info2]);if_f0_3.change(fn=whethercrepeornah,inputs=[f0method8],outputs=[extraction_crepe_hop_length]);gpus16=gr.Textbox(label=i18n(_Af),value=gpus,interactive=_A);butstop=gr.Button('Stop Training',variant=_K,visible=_D);but3=gr.Button(i18n('训练模型'),variant=_K,visible=_A);but3.click(fn=stoptraining,inputs=[gr.Number(value=0,visible=_D)],outputs=[but3,butstop]);butstop.click(fn=stoptraining,inputs=[gr.Number(value=1,visible=_D)],outputs=[butstop,but3]);but4=gr.Button(i18n('训练特征索引'),variant=_K);info3=gr.Textbox(label=i18n(_T),value='',max_lines=10);if_save_every_weights18.change(fn=stepdisplay,inputs=[if_save_every_weights18],outputs=[save_epoch10]);but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19],[info3,butstop,but3]);but4.click(train_index,[exp_dir1,version19],info3)
		with gr.TabItem(i18n('ckpt处理')):
			with gr.Group():
				gr.Markdown(value=i18n('模型融合, 可用于测试音色融合'))
				with gr.Row():ckpt_a=gr.Textbox(label=i18n('A模型路径'),value='',interactive=_A,placeholder='Path to your model A.');ckpt_b=gr.Textbox(label=i18n('B模型路径'),value='',interactive=_A,placeholder='Path to your model B.');alpha_a=gr.Slider(minimum=0,maximum=1,label=i18n('A模型权重'),value=.5,interactive=_A)
				with gr.Row():sr_=gr.Radio(label=i18n(_AC),choices=[_S,_d],value=_S,interactive=_A);if_f0_=gr.Checkbox(label=_AD,value=_A,interactive=_A);info__=gr.Textbox(label=i18n('要置入的模型信息'),value='',max_lines=8,interactive=_A,placeholder='Model information to be placed.');name_to_save0=gr.Textbox(label=i18n('保存的模型名不带后缀'),value='',placeholder='Name for saving.',max_lines=1,interactive=_A);version_2=gr.Radio(label=i18n('模型版本型号'),choices=[_G,_Y],value=_G,interactive=_A)
				with gr.Row():but6=gr.Button(i18n('融合'),variant=_K);info4=gr.Textbox(label=i18n(_T),value='',max_lines=8)
				but6.click(merge,[ckpt_a,ckpt_b,alpha_a,sr_,if_f0_,info__,name_to_save0,version_2],info4)
			with gr.Group():
				gr.Markdown(value=i18n('修改模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr.Row():ckpt_path0=gr.Textbox(label=i18n('模型路径'),placeholder='Path to your Model.',value='',interactive=_A);info_=gr.Textbox(label=i18n('要改的模型信息'),value='',max_lines=8,interactive=_A,placeholder='Model information to be changed.');name_to_save1=gr.Textbox(label=i18n('保存的文件名, 默认空为和源文件同名'),placeholder='Either leave empty or put in the Name of the Model to be saved.',value='',max_lines=8,interactive=_A)
				with gr.Row():but7=gr.Button(i18n('修改'),variant=_K);info5=gr.Textbox(label=i18n(_T),value='',max_lines=8)
				but7.click(change_info,[ckpt_path0,info_,name_to_save1],info5)
			with gr.Group():
				gr.Markdown(value=i18n('查看模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr.Row():ckpt_path1=gr.Textbox(label=i18n('模型路径'),value='',interactive=_A,placeholder='Model path here.');but8=gr.Button(i18n('查看'),variant=_K);info6=gr.Textbox(label=i18n(_T),value='',max_lines=8)
				but8.click(show_info,[ckpt_path1],info6)
			with gr.Group():
				gr.Markdown(value=i18n('模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况'))
				with gr.Row():ckpt_path2=gr.Textbox(lines=3,label=i18n('模型路径'),value=os.path.abspath(os.getcwd()).replace(_H,_N)+'/logs/[YOUR_MODEL]/G_23333.pth',interactive=_A);save_name=gr.Textbox(label=i18n('保存名'),value='',interactive=_A,placeholder='Your filename here.');sr__=gr.Radio(label=i18n(_AC),choices=[_p,_S,_d],value=_S,interactive=_A);if_f0__=gr.Checkbox(label=_AD,value=_A,interactive=_A);version_1=gr.Radio(label=i18n('模型版本型号'),choices=[_G,_Y],value=_Y,interactive=_A);info___=gr.Textbox(label=i18n('要置入的模型信息'),value='',max_lines=8,interactive=_A,placeholder='Model info here.');but9=gr.Button(i18n('提取'),variant=_K);info7=gr.Textbox(label=i18n(_T),value='',max_lines=8);ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__,version_1])
				but9.click(extract_small_model,[ckpt_path2,save_name,sr__,if_f0__,info___,version_1],info7)
		with gr.TabItem(i18n('Onnx导出')):
			with gr.Row():ckpt_dir=gr.Textbox(label=i18n('RVC模型路径'),value='',interactive=_A,placeholder='RVC model path.')
			with gr.Row():onnx_dir=gr.Textbox(label=i18n('Onnx输出路径'),value='',interactive=_A,placeholder='Onnx model output path.')
			with gr.Row():infoOnnx=gr.Label(label='info')
			with gr.Row():butOnnx=gr.Button(i18n('导出Onnx模型'),variant=_K)
			butOnnx.click(export_onnx,[ckpt_dir,onnx_dir],infoOnnx)
		tab_faq=i18n('常见问题解答')
		with gr.TabItem(tab_faq):
			try:
				if tab_faq=='常见问题解答':
					with open('docs/faq.md',_I,encoding='utf8')as f:info=f.read()
				else:
					with open('docs/faq_en.md',_I,encoding='utf8')as f:info=f.read()
				gr.Markdown(value=info)
			except:gr.Markdown(traceback.format_exc())
	def save_preset(preset_name,sid0,vc_transform,input_audio0,input_audio1,f0method,crepe_hop_length,filter_radius,file_index1,file_index2,index_rate,resample_sr,rms_mix_rate,protect,f0_file):
		data=_J
		with open(_t,_I)as file:data=json.load(file)
		preset_json={'name':preset_name,'model':sid0,'transpose':vc_transform,'audio_file':input_audio0,'auto_audio_file':input_audio1,'f0_method':f0method,_AB:crepe_hop_length,'median_filtering':filter_radius,'feature_path':file_index1,'auto_feature_path':file_index2,'search_feature_ratio':index_rate,'resample':resample_sr,'volume_envelope':rms_mix_rate,'protect_voiceless':protect,'f0_file_path':f0_file};data[_A7].append(preset_json)
		with open(_t,_U)as file:json.dump(data,file);file.flush()
		print('Saved Preset %s into inference-presets.json!'%preset_name)
	def on_preset_changed(preset_name):
		print('Changed Preset to %s!'%preset_name);data=_J
		with open(_t,_I)as file:data=json.load(file)
		print('Searching for '+preset_name);returning_preset=_J
		for preset in data[_A7]:
			if preset['name']==preset_name:print('Found a preset');returning_preset=preset
		return()
	if config.iscolab or config.paperspace:app.queue(concurrency_count=511,max_size=1022).launch(server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A,share=_D)
	else:app.queue(concurrency_count=511,max_size=1022).launch(server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_D)