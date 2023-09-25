_A1='Replacing old dropdown file...'
_A0='%s/3_feature768'
_z='%s/3_feature256'
_y='rmvpe_gpu'
_x='sample_rate'
_w='assets/audios/audio-others'
_v='assets/audios'
_u='weight_uvr5_root'
_t='Model Only'
_s='EXTRACT-MODEL'
_r='TRAIN-FEATURE'
_q='TRAIN'
_p='EXTRACT-FEATURE'
_o='PRE-PROCESS'
_n='INFER'
_m='HOME'
_l='_v2'
_k='%s/logs/%s'
_j='MDX'
_i='.onnx'
_h='m4a'
_g='mp3'
_f='datasets'
_e='lib/csvdb/stop.csv'
_d='v2'
_c='48k'
_b='32k'
_a='VR'
_Z='flac'
_Y='wav'
_X='logs'
_W='trained'
_V='.index'
_U='.pth'
_T='40k'
_S='audios'
_R='assets'
_Q='.'
_P='w+'
_O='formanting'
_N='lib/csvdb/formanting.csv'
_M=' '
_L=None
_K='rmvpe'
_J='choices'
_I='r'
_H='v1'
_G='value'
_F='\n'
_E='visible'
_D='update'
_C='__type__'
_B=False
_A=True
import os,sys
now_dir=os.getcwd()
sys.path.append(now_dir)
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['no_proxy']='localhost, 127.0.0.1, ::1'
import logging,shutil,threading,lib.globals.globals as rvc_globals,lib.infer.infer_libs.uvr5_pack.mdx as mdx
from lib.infer.modules.uvr5.mdxprocess import get_model_list,id_to_ptm,prepare_mdx,run_mdx
import math as math,ffmpeg as ffmpeg,traceback,warnings
from random import shuffle
from subprocess import Popen
from time import sleep
import json,pathlib,fairseq
logging.getLogger('faiss').setLevel(logging.WARNING)
import faiss,gradio as gr,numpy as np,torch as torch,regex as re,soundfile as SF
SFWrite=SF.write
from dotenv import load_dotenv
from sklearn.cluster import MiniBatchKMeans
import datetime
from glob import glob1
import signal
from signal import SIGTERM
from assets.configs.config import Config
from assets.i18n.i18n import I18nAuto
from lib.infer.infer_libs.train.process_ckpt import change_info,extract_small_model,merge,show_info
from lib.infer.modules.uvr5.mdxnet import MDXNetDereverb
from lib.infer.modules.uvr5.preprocess import AudioPre,AudioPreDeEcho
from lib.infer.modules.vc.modules import VC
from lib.infer.modules.vc.utils import*
import lib.globals.globals as rvc_globals,nltk
nltk.download('punkt',quiet=_A)
import tabs.resources as resources,tabs.tts as tts,tabs.merge as mergeaudios,tabs.processing as processing
from lib.infer.infer_libs.csvutil import CSVutil
import time,csv
from shlex import quote as SQuote
from huggingface_hub import HfApi
from huggingface_hub import login
import requests,os
RQuote=lambda val:SQuote(str(val))
tmp=os.path.join(now_dir,'temp')
shutil.rmtree(tmp,ignore_errors=_A)
os.makedirs(tmp,exist_ok=_A)
def remove_invalid_chars(text):pattern=re.compile('[^\\x00-\\x7F]+');return pattern.sub('',text)
def remove_text_between_parentheses(lines,start_line,end_line):
	pattern='\\[([^\\]]*)\\]\\([^)]*\\)';processed_lines=[]
	for(line_number,line)in enumerate(lines,start=1):
		if start_line<=line_number<=end_line:modified_line=re.sub(pattern,'\\1',line);processed_lines.append(modified_line)
		else:processed_lines.append(line)
	return _F.join(processed_lines)
with open('README.md',_I,encoding='utf8')as f:inforeadme=f.read()
inforeadme=remove_text_between_parentheses(inforeadme.split(_F),6,15)
inforeadme=remove_invalid_chars(inforeadme)
inforeadme=remove_text_between_parentheses(inforeadme.split(_F),191,207)
os.makedirs(tmp,exist_ok=_A)
os.makedirs(os.path.join(now_dir,_X),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'logs/weights'),exist_ok=_A)
os.environ['temp']=tmp
warnings.filterwarnings('ignore')
torch.manual_seed(114514)
logging.getLogger('numba').setLevel(logging.WARNING)
logger=logging.getLogger(__name__)
if not os.path.isdir('lib/csvdb/'):os.makedirs('lib/csvdb');frmnt,stp=open(_N,'w'),open(_e,'w');frmnt.close();stp.close()
global DoFormant,Quefrency,Timbre
try:DoFormant,Quefrency,Timbre=CSVutil(_N,_I,_O);DoFormant=(lambda DoFormant:_A if DoFormant.lower()=='true'else _B if DoFormant.lower()=='false'else DoFormant)(DoFormant)
except(ValueError,TypeError,IndexError):DoFormant,Quefrency,Timbre=_B,1.,1.;CSVutil(_N,_P,_O,DoFormant,Quefrency,Timbre)
load_dotenv()
config=Config()
vc=VC(config)
if config.dml==_A:
	def forward_dml(ctx,x,scale):ctx.scale=scale;res=x.clone().detach();return res
	fairseq.modules.grad_multiply.GradMultiply.forward=forward_dml
i18n=I18nAuto()
i18n.print()
ngpu=torch.cuda.device_count()
gpu_infos=[]
mem=[]
if_gpu_ok=_B
isinterrupted=0
if torch.cuda.is_available()or ngpu!=0:
	for i in range(ngpu):
		gpu_name=torch.cuda.get_device_name(i)
		if any(value in gpu_name.upper()for value in['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):if_gpu_ok=_A;gpu_infos.append('%s\t%s'%(i,gpu_name));mem.append(int(torch.cuda.get_device_properties(i).total_memory/1024/1024/1024+.4))
if if_gpu_ok and len(gpu_infos)>0:gpu_info=_F.join(gpu_infos);default_batch_size=min(mem)//2
else:gpu_info='Unfortunately, there is no compatible GPU available to support your training.';default_batch_size=1
gpus='-'.join([i[0]for i in gpu_infos])
class ToolButton(gr.Button,gr.components.FormComponent):
	'Small button with single emoji as text, fits inside gradio forms'
	def __init__(self,**kwargs):super().__init__(variant='tool',**kwargs)
	def get_block_name(self):return'button'
hubert_model=_L
weight_root='/kaggle/working/AX-RVC/logs/weights'
weight_uvr5_root=os.getenv(_u)
index_root='/kaggle/working/AX-RVC/logs'
datasets_root=_f
fshift_root='lib/infer/infer_libs/formantshiftcfg'
audio_root=_v
audio_others_root=_w
sup_audioext={_Y,_g,_Z,'ogg','opus',_h,'mp4','aac','alac','wma','aiff','webm','ac3'}
names=[os.path.join(root,file)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_U,_i))]
indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_V)and _W not in name]
audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root]
audio_others_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_others_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_others_root]
uvr5_names=[name.replace(_U,'')for name in os.listdir(weight_uvr5_root)if name.endswith(_U)or'onnx'in name]
check_for_name=lambda:sorted(names)[0]if names else''
datasets=[]
for foldername in os.listdir(os.path.join(now_dir,datasets_root)):
	if _Q not in foldername:datasets.append(os.path.join(now_dir,_f,foldername))
def get_dataset():
	if len(datasets)>0:return sorted(datasets)[0]
	else:return''
def update_model_choices(select_value):
	model_ids=get_model_list();model_ids_list=list(model_ids)
	if select_value==_a:return{_J:uvr5_names,_C:_D}
	elif select_value==_j:return{_J:model_ids_list,_C:_D}
def update_dataset_list(name):
	new_datasets=[]
	for foldername in os.listdir(os.path.join(now_dir,datasets_root)):
		if _Q not in foldername:new_datasets.append(os.path.join(now_dir,_f,foldername))
	return gr.Dropdown.update(choices=new_datasets)
def get_indexes():indexes_list=[os.path.join(dirpath,filename)for(dirpath,_,filenames)in os.walk(index_root)for filename in filenames if filename.endswith(_V)and _W not in filename];return indexes_list if indexes_list else''
def get_fshift_presets():fshift_presets_list=[os.path.join(dirpath,filename)for(dirpath,_,filenames)in os.walk(fshift_root)for filename in filenames if filename.endswith('.txt')];return fshift_presets_list if fshift_presets_list else''
def uvr(model_name,inp_root,save_root_vocal,paths,save_root_ins,agg,format0,architecture):
	E='%s->Success';D='streams';C='onnx_dereverb_By_FoxJoy';B='Starting audio conversion... (This might take a moment)';A='"';infos=[]
	if architecture==_a:
		try:
			infos.append(i18n(B));inp_root=inp_root.strip(_M).strip(A).strip(_F).strip(A).strip(_M);save_root_vocal=save_root_vocal.strip(_M).strip(A).strip(_F).strip(A).strip(_M);save_root_ins=save_root_ins.strip(_M).strip(A).strip(_F).strip(A).strip(_M)
			if model_name==C:pre_fun=MDXNetDereverb(15,config.device)
			else:func=AudioPre if'DeEcho'not in model_name else AudioPreDeEcho;pre_fun=func(agg=int(agg),model_path=os.path.join(os.getenv(_u),model_name+_U),device=config.device,is_half=config.is_half)
			if inp_root!='':paths=[os.path.join(inp_root,name)for(root,_,files)in os.walk(inp_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==inp_root]
			else:paths=[path.name for path in paths]
			for path in paths:
				inp_path=os.path.join(inp_root,path);need_reformat=1;done=0
				try:
					info=ffmpeg.probe(inp_path,cmd='ffprobe')
					if info[D][0]['channels']==2 and info[D][0][_x]=='44100':need_reformat=0;pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0);done=1
				except:need_reformat=1;traceback.print_exc()
				if need_reformat==1:tmp_path='%s/%s.reformatted.wav'%(os.path.join(os.environ['tmp']),os.path.basename(inp_path));os.system('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(inp_path,tmp_path));inp_path=tmp_path
				try:
					if done==0:pre_fun.path_audio(inp_path,save_root_ins,save_root_vocal,format0)
					infos.append(E%os.path.basename(inp_path));yield _F.join(infos)
				except:
					try:
						if done==0:pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0)
						infos.append(E%os.path.basename(inp_path));yield _F.join(infos)
					except:infos.append('%s->%s'%(os.path.basename(inp_path),traceback.format_exc()));yield _F.join(infos)
		except:infos.append(traceback.format_exc());yield _F.join(infos)
		finally:
			try:
				if model_name==C:del pre_fun.pred.model;del pre_fun.pred.model_
				else:del pre_fun.model;del pre_fun
			except:traceback.print_exc()
			if torch.cuda.is_available():torch.cuda.empty_cache();logger.info('Executed torch.cuda.empty_cache()')
		yield _F.join(infos)
	elif architecture==_j:
		try:
			infos.append(i18n(B));yield _F.join(infos);inp_root,save_root_vocal,save_root_ins=[x.strip(_M).strip(A).strip(_F).strip(A).strip(_M)for x in[inp_root,save_root_vocal,save_root_ins]]
			if inp_root!='':paths=[os.path.join(inp_root,name)for(root,_,files)in os.walk(inp_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==inp_root]
			else:paths=[path.name for path in paths]
			print(paths);invert=_A;denoise=_A;use_custom_parameter=_A;dim_f=3072;dim_t=256;n_fft=7680;use_custom_compensation=_A;compensation=1.025;suffix='Vocals_custom';suffix_invert='Instrumental_custom';print_settings=_A;onnx=id_to_ptm(model_name);compensation=compensation if use_custom_compensation or use_custom_parameter else _L;mdx_model=prepare_mdx(onnx,use_custom_parameter,dim_f,dim_t,n_fft,compensation=compensation)
			for path in paths:suffix_naming=suffix if use_custom_parameter else _L;diff_suffix_naming=suffix_invert if use_custom_parameter else _L;run_mdx(onnx,mdx_model,path,format0,diff=invert,suffix=suffix_naming,diff_suffix=diff_suffix_naming,denoise=denoise)
			if print_settings:
				print();print('[MDX-Net_Colab settings used]');print(f"Model used: {onnx}");print(f"Model MD5: {mdx.MDX.get_hash(onnx)}");print(f"Model parameters:");print(f"    -dim_f: {mdx_model.dim_f}");print(f"    -dim_t: {mdx_model.dim_t}");print(f"    -n_fft: {mdx_model.n_fft}");print(f"    -compensation: {mdx_model.compensation}");print();print('[Input file]');print('filename(s): ')
				for filename in paths:print(f"    -{filename}");infos.append(f"{os.path.basename(filename)}->Success");yield _F.join(infos)
		except:infos.append(traceback.format_exc());yield _F.join(infos)
		finally:
			try:del mdx_model
			except:traceback.print_exc()
			print('clean_empty_cache')
			if torch.cuda.is_available():torch.cuda.empty_cache()
def change_choices():names=[os.path.join(root,file)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_U,_i))];indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_V)and _W not in name];audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root];print(names);print(indexes_list);print(audio_paths);return{_J:sorted(names),_C:_D},{_J:sorted(indexes_list),_C:_D},{_J:sorted(audio_paths),_C:_D}
def change_choices2():names=[os.path.join(root,file)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_U,_i))];indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_V)and _W not in name];return{_J:sorted(names),_C:_D},{_J:sorted(indexes_list),_C:_D}
def clean():return{_G:'',_C:_D}
def export_onnx():from lib.infer.modules.onnx.export import export_onnx as eo;eo()
sr_dict={_b:32000,_T:40000,_c:48000}
def if_done(done,p):
	while 1:
		if p.poll()is _L:sleep(.5)
		else:break
	done[0]=_A
def if_done_multi(done,ps):
	while 1:
		flag=1
		for p in ps:
			if p.poll()is _L:flag=0;sleep(.5);break
		if flag==1:break
	done[0]=_A
def formant_enabled(cbox,qfrency,tmbre,frmntapply,formantpreset,formant_refresh_button):
	if cbox:DoFormant=_A;CSVutil(_N,_P,_O,DoFormant,qfrency,tmbre);return{_G:_A,_C:_D},{_E:_A,_C:_D},{_E:_A,_C:_D},{_E:_A,_C:_D},{_E:_A,_C:_D},{_E:_A,_C:_D}
	else:DoFormant=_B;CSVutil(_N,_P,_O,DoFormant,qfrency,tmbre);return{_G:_B,_C:_D},{_E:_B,_C:_D},{_E:_B,_C:_D},{_E:_B,_C:_D},{_E:_B,_C:_D},{_E:_B,_C:_D},{_E:_B,_C:_D}
def formant_apply(qfrency,tmbre):Quefrency=qfrency;Timbre=tmbre;DoFormant=_A;CSVutil(_N,_P,_O,DoFormant,qfrency,tmbre);return{_G:Quefrency,_C:_D},{_G:Timbre,_C:_D}
def update_fshift_presets(preset,qfrency,tmbre):
	if preset:
		with open(preset,_I)as p:content=p.readlines();qfrency,tmbre=content[0].strip(),content[1]
		formant_apply(qfrency,tmbre)
	else:qfrency,tmbre=preset_apply(preset,qfrency,tmbre)
	return{_J:get_fshift_presets(),_C:_D},{_G:qfrency,_C:_D},{_G:tmbre,_C:_D}
def preprocess_dataset(trainset_dir,exp_dir,sr,n_p,dataset_path):
	A='%s/logs/%s/preprocess.log'
	if not dataset_path.strip()=='':trainset_dir=dataset_path
	sr=sr_dict[sr];os.makedirs(_k%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close();per=3. if config.is_half else 3.7;cmd='"%s" lib/infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f'%(config.python_cmd,trainset_dir,sr,n_p,now_dir,exp_dir,config.noparallel,per);logger.info(cmd);p=Popen(cmd,shell=_A);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
	logger.info(log);yield log
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir,version19,echl):
	A='%s/logs/%s/extract_f0_feature.log';gpus_rmvpe=gpus;gpus=gpus.split('-');os.makedirs(_k%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close()
	if if_f0:
		if f0method!=_y:cmd='"%s" lib/infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s %s'%(config.python_cmd,now_dir,exp_dir,n_p,f0method,RQuote(echl));logger.info(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
		elif gpus_rmvpe!='-':
			gpus_rmvpe=gpus_rmvpe.split('-');leng=len(gpus_rmvpe);ps=[]
			for(idx,n_g)in enumerate(gpus_rmvpe):cmd='"%s" lib/infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '%(config.python_cmd,leng,idx,n_g,now_dir,exp_dir,config.is_half);logger.info(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
			done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
		else:cmd=config.python_cmd+' lib/infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '%(now_dir,exp_dir);logger.info(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();done=[_A]
		while 1:
			with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
			sleep(1)
			if done[0]:break
		with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
		logger.info(log);yield log
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd='"%s" lib/infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(config.python_cmd,config.device,leng,idx,n_g,now_dir,exp_dir,version19);logger.info(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_I)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_I)as f:log=f.read()
	logger.info(log);yield log
def get_pretrained_models(path_str,f0_str,sr2):
	B='/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth';A='/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth';if_pretrained_generator_exist=os.access(A%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(B%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:logger.warn('/kaggle/input/ax-rmf/pretrained%s/%sG%s.pth not exist, will not use pretrained model',path_str,f0_str,sr2)
	if not if_pretrained_discriminator_exist:logger.warn('/kaggle/input/ax-rmf/pretrained%s/%sD%s.pth not exist, will not use pretrained model',path_str,f0_str,sr2)
	return A%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',B%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else''
def change_sr2(sr2,if_f0_3,version19):path_str=''if version19==_H else _l;f0_str='f0'if if_f0_3 else'';return get_pretrained_models(path_str,f0_str,sr2)
def change_version19(sr2,if_f0_3,version19):
	path_str=''if version19==_H else _l
	if sr2==_b and version19==_H:sr2=_T
	to_return_sr2={_J:[_T,_c],_C:_D,_G:sr2}if version19==_H else{_J:[_T,_c,_b],_C:_D,_G:sr2};f0_str='f0'if if_f0_3 else'';return*get_pretrained_models(path_str,f0_str,sr2),to_return_sr2
def change_f0(if_f0_3,sr2,version19):path_str=''if version19==_H else _l;return{_E:if_f0_3,_C:_D},*get_pretrained_models(path_str,'f0',sr2)
global log_interval
def set_log_interval(exp_dir,batch_size12):
	log_interval=1;folder_path=os.path.join(exp_dir,'1_16k_wavs')
	if os.path.isdir(folder_path):
		wav_files_num=len(glob1(folder_path,'*.wav'))
		if wav_files_num>0:
			log_interval=math.ceil(wav_files_num/batch_size12)
			if log_interval>1:log_interval+=1
	return log_interval
global PID,PROCESS
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19):
	D='-pd %s';C='-pg %s';B='\\\\';A='\\';CSVutil(_e,_P,_O,_B);exp_dir=_k%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);gt_wavs_dir='%s/0_gt_wavs'%exp_dir;feature_dir=_z%exp_dir if version19==_H else _A0%exp_dir
	if if_f0_3:f0_dir='%s/2a_f0'%exp_dir;f0nsf_dir='%s/2b-f0nsf'%exp_dir;names=set([name.split(_Q)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_Q)[0]for name in os.listdir(feature_dir)])&set([name.split(_Q)[0]for name in os.listdir(f0_dir)])&set([name.split(_Q)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_Q)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_Q)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append('%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'%(gt_wavs_dir.replace(A,B),name,feature_dir.replace(A,B),name,f0_dir.replace(A,B),name,f0nsf_dir.replace(A,B),name,spk_id5))
		else:opt.append('%s/%s.wav|%s/%s.npy|%s'%(gt_wavs_dir.replace(A,B),name,feature_dir.replace(A,B),name,spk_id5))
	fea_dim=256 if version19==_H else 768
	if if_f0_3:
		for _ in range(2):opt.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open('%s/filelist.txt'%exp_dir,'w')as f:f.write(_F.join(opt))
	logger.debug('Write filelist done');logger.info('Use gpus: %s',str(gpus16))
	if pretrained_G14=='':logger.info('No pretrained Generator')
	if pretrained_D15=='':logger.info('No pretrained Discriminator')
	if version19==_H or sr2==_T:config_path='v1/%s.json'%sr2
	else:config_path='v2/%s.json'%sr2
	config_save_path=os.path.join(exp_dir,'config.json')
	if not pathlib.Path(config_save_path).exists():
		with open(config_save_path,'w',encoding='utf-8')as f:json.dump(config.json_config[config_path],f,ensure_ascii=_B,indent=4,sort_keys=_A);f.write(_F)
	if gpus16:cmd='"%s" lib/infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,C%pretrained_G14 if pretrained_G14!=''else'',D%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19)
	else:cmd='"%s" lib/infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,C%pretrained_G14 if pretrained_G14!=''else'',D%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==_A else 0,1 if if_cache_gpu17==_A else 0,1 if if_save_every_weights18==_A else 0,version19)
	logger.info(cmd);global p;p=Popen(cmd,shell=_A,cwd=now_dir);global PID;PID=p.pid;p.wait();return i18n('Training is done, check train.log'),{_E:_B,_C:_D},{_E:_A,_C:_D}
def train_index(exp_dir1,version19):
	exp_dir=os.path.join(now_dir,_X,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);feature_dir=_z%exp_dir if version19==_H else _A0%exp_dir
	if not os.path.exists(feature_dir):return'Please do the feature extraction first'
	listdir_res=list(os.listdir(feature_dir))
	if len(listdir_res)==0:return'Please perform the feature extraction first'
	infos=[];npys=[]
	for name in sorted(listdir_res):phone=np.load('%s/%s'%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		infos.append('Trying doing kmeans %s shape to 10k centers.'%big_npy.shape[0]);yield _F.join(infos)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_B,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();logger.info(info);infos.append(info);yield _F.join(infos)
	np.save('%s/total_fea.npy'%exp_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);infos.append('%s,%s'%(big_npy.shape,n_ivf));yield _F.join(infos);index=faiss.index_factory(256 if version19==_H else 768,'IVF%s,Flat'%n_ivf);infos.append('training');yield _F.join(infos);index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,'%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('adding');yield _F.join(infos);batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,'%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('Successful Index Construction，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19));yield _F.join(infos)
def change_info_(ckpt_path):
	B='version';A='train.log'
	if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path),A)):return{_C:_D},{_C:_D},{_C:_D}
	try:
		with open(ckpt_path.replace(os.path.basename(ckpt_path),A),_I)as f:info=eval(f.read().strip(_F).split(_F)[0].split('\t')[-1]);sr,f0=info[_x],info['if_f0'];version=_d if B in info and info[B]==_d else _H;return sr,str(f0),version
	except:traceback.print_exc();return{_C:_D},{_C:_D},{_C:_D}
F0GPUVisible=config.dml==_B
import re as regex,scipy.io.wavfile as wavfile
cli_current_page=_m
def cli_split_command(com):exp='(?:(?<=\\s)|^)"(.*?)"(?=\\s|$)|(\\S+)';split_array=regex.findall(exp,com);split_array=[group[0]if group[0]else group[1]for group in split_array];return split_array
def execute_generator_function(genObject):
	for _ in genObject:0
def cli_infer(com):
	A='audio-outputs';com=cli_split_command(com);model_name=com[0];source_audio_path=com[1];output_file_name=com[2];feature_index_path=com[3];f0_file=_L;speaker_id=int(com[4]);transposition=float(com[5]);f0_method=com[6];crepe_hop_length=int(com[7]);harvest_median_filter=int(com[8]);resample=int(com[9]);mix=float(com[10]);feature_ratio=float(com[11]);protection_amnt=float(com[12]);protect1=.5
	if com[14]=='False'or com[14]=='false':DoFormant=_B;Quefrency=.0;Timbre=.0;CSVutil(_N,_P,_O,DoFormant,Quefrency,Timbre)
	else:DoFormant=_A;Quefrency=float(com[15]);Timbre=float(com[16]);CSVutil(_N,_P,_O,DoFormant,Quefrency,Timbre)
	print('Applio-RVC-Fork Infer-CLI: Starting the inference...');vc_data=vc.get_vc(model_name,protection_amnt,protect1);print(vc_data);print('Applio-RVC-Fork Infer-CLI: Performing inference...');conversion_data=vc.vc_single(speaker_id,source_audio_path,transposition,f0_file,f0_method,feature_index_path,feature_index_path,feature_ratio,harvest_median_filter,resample,mix,protection_amnt,crepe_hop_length)
	if'Success.'in conversion_data[0]:print('Applio-RVC-Fork Infer-CLI: Inference succeeded. Writing to %s/%s...'%(_R,_S,A,output_file_name));wavfile.write('%s/%s'%(_R,_S,A,output_file_name),conversion_data[1][0],conversion_data[1][1]);print('Applio-RVC-Fork Infer-CLI: Finished! Saved output to %s/%s'%(_R,_S,A,output_file_name))
	else:print("Applio-RVC-Fork Infer-CLI: Inference failed. Here's the traceback: ");print(conversion_data[0])
def cli_pre_process(com):com=cli_split_command(com);model_name=com[0];trainset_directory=com[1];sample_rate=com[2];num_processes=int(com[3]);print('Applio-RVC-Fork Pre-process: Starting...');generator=preprocess_dataset(trainset_directory,model_name,sample_rate,num_processes);execute_generator_function(generator);print('Applio-RVC-Fork Pre-process: Finished')
def cli_extract_feature(com):com=cli_split_command(com);model_name=com[0];gpus=com[1];num_processes=int(com[2]);has_pitch_guidance=_A if int(com[3])==1 else _B;f0_method=com[4];crepe_hop_length=int(com[5]);version=com[6];print('Applio-RVC-CLI: Extract Feature Has Pitch: '+str(has_pitch_guidance));print('Applio-RVC-CLI: Extract Feature Version: '+str(version));print('Applio-RVC-Fork Feature Extraction: Starting...');generator=extract_f0_feature(gpus,num_processes,f0_method,has_pitch_guidance,model_name,version,crepe_hop_length);execute_generator_function(generator);print('Applio-RVC-Fork Feature Extraction: Finished')
def cli_train(com):com=cli_split_command(com);model_name=com[0];sample_rate=com[1];has_pitch_guidance=_A if int(com[2])==1 else _B;speaker_id=int(com[3]);save_epoch_iteration=int(com[4]);total_epoch=int(com[5]);batch_size=int(com[6]);gpu_card_slot_numbers=com[7];if_save_latest=_A if int(com[8])==1 else _B;if_cache_gpu=_A if int(com[9])==1 else _B;if_save_every_weight=_A if int(com[10])==1 else _B;version=com[11];pretrained_base='/kaggle/input/ax-rmf/pretrained/'if version==_H else'/kaggle/input/ax-rmf/pretrained_v2/';g_pretrained_path='%sf0G%s.pth'%(pretrained_base,sample_rate);d_pretrained_path='%sf0D%s.pth'%(pretrained_base,sample_rate);print('Applio-RVC-Fork Train-CLI: Training...');click_train(model_name,sample_rate,has_pitch_guidance,speaker_id,save_epoch_iteration,total_epoch,batch_size,if_save_latest,g_pretrained_path,d_pretrained_path,gpu_card_slot_numbers,if_cache_gpu,if_save_every_weight,version)
def cli_train_feature(com):com=cli_split_command(com);model_name=com[0];version=com[1];print('Applio-RVC-Fork Train Feature Index-CLI: Training... Please wait');generator=train_index(model_name,version);execute_generator_function(generator);print('Applio-RVC-Fork Train Feature Index-CLI: Done!')
def cli_extract_model(com):
	com=cli_split_command(com);model_path=com[0];save_name=com[1];sample_rate=com[2];has_pitch_guidance=com[3];info=com[4];version=com[5];extract_small_model_process=extract_small_model(model_path,save_name,sample_rate,has_pitch_guidance,info,version)
	if extract_small_model_process=='Success.':print('Applio-RVC-Fork Extract Small Model: Success!')
	else:print(str(extract_small_model_process));print('Applio-RVC-Fork Extract Small Model: Failed!')
def preset_apply(preset,qfer,tmbr):
	if str(preset)!='':
		with open(str(preset),_I)as p:content=p.readlines();qfer,tmbr=content[0].split(_F)[0],content[1];formant_apply(qfer,tmbr)
	else:0
	return{_G:qfer,_C:_D},{_G:tmbr,_C:_D}
def print_page_details():
	if cli_current_page==_m:print('\n    go home            : Takes you back to home with a navigation list.\n    go infer           : Takes you to inference command execution.\n    go pre-process     : Takes you to training step.1) pre-process command execution.\n    go extract-feature : Takes you to training step.2) extract-feature command execution.\n    go train           : Takes you to training step.3) being or continue training command execution.\n    go train-feature   : Takes you to the train feature index command execution.\n    go extract-model   : Takes you to the extract small model command execution.')
	elif cli_current_page==_n:print("\n    arg 1) model name with .pth in ./weights: mi-test.pth\n    arg 2) source audio path: myFolder\\MySource.wav\n    arg 3) output file name to be placed in './audio-outputs': MyTest.wav\n    arg 4) feature index file path: logs/mi-test/added_IVF3042_Flat_nprobe_1.index\n    arg 5) speaker id: 0\n    arg 6) transposition: 0\n    arg 7) f0 method: harvest (pm, harvest, crepe, crepe-tiny, hybrid[x,x,x,x], mangio-crepe, mangio-crepe-tiny, rmvpe)\n    arg 8) crepe hop length: 160\n    arg 9) harvest median filter radius: 3 (0-7)\n    arg 10) post resample rate: 0\n    arg 11) mix volume envelope: 1\n    arg 12) feature index ratio: 0.78 (0-1)\n    arg 13) Voiceless Consonant Protection (Less Artifact): 0.33 (Smaller number = more protection. 0.50 means Dont Use.)\n    arg 14) Whether to formant shift the inference audio before conversion: False (if set to false, you can ignore setting the quefrency and timbre values for formanting)\n    arg 15)* Quefrency for formanting: 8.0 (no need to set if arg14 is False/false)\n    arg 16)* Timbre for formanting: 1.2 (no need to set if arg14 is False/false) \n\nExample: mi-test.pth saudio/Sidney.wav myTest.wav logs/mi-test/added_index.index 0 -2 harvest 160 3 0 1 0.95 0.33 0.45 True 8.0 1.2")
	elif cli_current_page==_o:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Trainset directory: mydataset (or) E:\\my-data-set\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Number of CPU threads to use: 8 \n\nExample: mi-test mydataset 40k 24')
	elif cli_current_page==_p:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 3) Number of CPU threads to use: 8\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) f0 Method: harvest (pm, harvest, dio, crepe)\n    arg 6) Crepe hop length: 128\n    arg 7) Version for pre-trained models: v2 (use either v1 or v2)\n\nExample: mi-test 0 24 1 harvest 128 v2')
	elif cli_current_page==_q:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Sample rate: 40k (32k, 40k, 48k)\n    arg 3) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 4) speaker id: 0\n    arg 5) Save epoch iteration: 50\n    arg 6) Total epochs: 10000\n    arg 7) Batch size: 8\n    arg 8) Gpu card slot: 0 (0-1-2 if using 3 GPUs)\n    arg 9) Save only the latest checkpoint: 0 (0 for no, 1 for yes)\n    arg 10) Whether to cache training set to vram: 0 (0 for no, 1 for yes)\n    arg 11) Save extracted small model every generation?: 0 (0 for no, 1 for yes)\n    arg 12) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test 40k 1 0 50 10000 8 0 0 0 0 v2')
	elif cli_current_page==_r:print('\n    arg 1) Model folder name in ./logs: mi-test\n    arg 2) Model architecture version: v2 (use either v1 or v2)\n\nExample: mi-test v2')
	elif cli_current_page==_s:print('\n    arg 1) Model Path: logs/mi-test/G_168000.pth\n    arg 2) Model save name: MyModel\n    arg 3) Sample rate: 40k (32k, 40k, 48k)\n    arg 4) Has Pitch Guidance?: 1 (0 for no, 1 for yes)\n    arg 5) Model information: "My Model"\n    arg 6) Model architecture version: v2 (use either v1 or v2)\n\nExample: logs/mi-test/G_168000.pth MyModel 40k 1 "Created by Cole Mangio" v2')
def change_page(page):global cli_current_page;cli_current_page=page;return 0
def execute_command(com):
	if com=='go home':return change_page(_m)
	elif com=='go infer':return change_page(_n)
	elif com=='go pre-process':return change_page(_o)
	elif com=='go extract-feature':return change_page(_p)
	elif com=='go train':return change_page(_q)
	elif com=='go train-feature':return change_page(_r)
	elif com=='go extract-model':return change_page(_s)
	elif com[:3]=='go ':print("page '%s' does not exist!"%com[3:]);return 0
	if cli_current_page==_n:cli_infer(com)
	elif cli_current_page==_o:cli_pre_process(com)
	elif cli_current_page==_p:cli_extract_feature(com)
	elif cli_current_page==_q:cli_train(com)
	elif cli_current_page==_r:cli_train_feature(com)
	elif cli_current_page==_s:cli_extract_model(com)
def cli_navigation_loop():
	while _A:
		print("\nYou are currently in '%s':"%cli_current_page);print_page_details();command=input('%s: '%cli_current_page)
		try:execute_command(command)
		except:print(traceback.format_exc())
if config.is_cli:print('\n\nApplio-RVC-Fork CLI\n');print('Welcome to the CLI version of RVC. Please read the documentation on README.MD to understand how to use this app.\n');cli_navigation_loop()
def switch_pitch_controls(f0method0):
	is_visible=f0method0!=_K
	if rvc_globals.NotesOrHertz:return{_E:_B,_C:_D},{_E:is_visible,_C:_D},{_E:_B,_C:_D},{_E:is_visible,_C:_D}
	else:return{_E:is_visible,_C:_D},{_E:_B,_C:_D},{_E:is_visible,_C:_D},{_E:_B,_C:_D}
def match_index(sid0):
	sid0strip=re.sub('\\.pth|\\.onnx$','',sid0);sid0name=os.path.split(sid0strip)[-1]
	if re.match('.+_e\\d+_s\\d+$',sid0name):base_model_name=sid0name.rsplit('_',2)[0]
	else:base_model_name=sid0name
	sid_directory=os.path.join(_X,base_model_name);directories_to_search=[sid_directory]if os.path.exists(sid_directory)else[];directories_to_search.append(_X);matching_index_files=[]
	for directory in directories_to_search:
		for filename in os.listdir(directory):
			if filename.endswith(_V)and _W not in filename:
				name_match=any(name.lower()in filename.lower()for name in[sid0name,base_model_name]);folder_match=directory==sid_directory
				if name_match or folder_match:
					index_path=os.path.join(directory,filename)
					if index_path in indexes_list:matching_index_files.append((index_path,os.path.getsize(index_path),_M not in filename))
	if matching_index_files:matching_index_files.sort(key=lambda x:(-x[2],-x[1]));best_match_index_path=matching_index_files[0][0];return best_match_index_path,best_match_index_path
	return'',''
def stoptraining(mim):
	if int(mim)==1:
		CSVutil(_e,_P,'stop','True')
		try:os.kill(PID,signal.SIGTERM)
		except Exception as e:print(f"Couldn't click due to {e}");pass
	else:0
	return{_E:_B,_C:_D},{_E:_A,_C:_D}
weights_dir='weights/'
def note_to_hz(note_name):SEMITONES={'C':-9,'C#':-8,'D':-7,'D#':-6,'E':-5,'F':-4,'F#':-3,'G':-2,'G#':-1,'A':0,'A#':1,'B':2};pitch_class,octave=note_name[:-1],int(note_name[-1]);semitone=SEMITONES[pitch_class];note_number=12*(octave-4)+semitone;frequency=44e1*(2.**(1./12))**note_number;return frequency
def save_to_wav(record_button):
	if record_button is _L:0
	else:path_to_file=record_button;new_name=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.wav';target_path=os.path.join(_R,_S,os.path.basename(new_name));shutil.move(path_to_file,target_path);return target_path
def save_to_wav2_edited(dropbox):
	if dropbox is _L:0
	else:
		file_path=dropbox.name;target_path=os.path.join(_R,_S,os.path.basename(file_path))
		if os.path.exists(target_path):os.remove(target_path);print(_A1)
		shutil.move(file_path,target_path)
def save_to_wav2(dropbox):
	file_path=dropbox.name;target_path=os.path.join(_R,_S,os.path.basename(file_path))
	if os.path.exists(target_path):os.remove(target_path);print(_A1)
	shutil.move(file_path,target_path);return target_path
from assets.themes.black import Applio
def start_upload_to_huggingface(hgf_token_gr,hgf_name_gr,hgf_repo_gr,model_name_gr,zip_name_gr,what_upload_gr):
	login(token=hgf_token_gr,add_to_git_credential=_A,new_session=_A);hug_file_path='/kaggle/working/AX-RVC/hugupload';hug_file_name=f"{zip_name_gr}.zip"
	if what_upload_gr==_t:os.system(f"cp /kaggle/working/AX-RVC/logs/weights/{model_name_gr}.pth {hug_file_path}");os.system(f"cp /kaggle/working/AX-RVC/logs/{model_name_gr}/added*.index {hug_file_path}");time.sleep(5);os.system(f"zip -r /kaggle/working/AX-RVC/hugupload/{hug_file_name} /kaggle/working/AX-RVC/hugupload/{model_name_gr}.pth /kaggle/working/AX-RVC/hugupload/added*.index");api=HfApi(token=hgf_token_gr);api.upload_file(path_or_fileobj=f"{hug_file_path}/{hug_file_name}",path_in_repo=hug_file_name,repo_id=f"{hgf_name_gr}/{hgf_repo_gr}",repo_type='model');time.sleep(5);os.system(f"rm -rf /kaggle/working/AX-RVC/hugupload/{hug_file_name}");os.system(f"rm -rf /kaggle/working/AX-RVC/hugupload/{model_name_gr}.pth");os.system(f"rm -rf /kaggle/working/AX-RVC/hugupload/added*.index");return'Succesful upload Model to Hugging Face'
mi_applio=Applio()
def GradioSetup():
	f='Edge-tts';e="Provide the GPU index(es) separated by '-', like 0-1-2 for using GPUs 0, 1, and 2:";d='Export file format:';c='You can also input audio files in batches. Choose one of the two options. Priority is given to reading from the folder.';b='multiple';a='Default value is 1.0';Z='Search feature ratio:';Y='Protect voiceless consonants and breath sounds to prevent artifacts such as tearing in electronic music. Set to 0.5 to disable. Decrease the value to increase protection, but it may reduce indexing accuracy:';X='Use the volume envelope of the input to replace or mix with the volume envelope of the output. The closer the ratio is to 1, the more the output envelope is used:';W='Resample the output audio in post-processing to the final sample rate. Set to 0 for no resampling:';V='Feature search database file path:';U='Max pitch:';T='Min pitch:';S='If >=3: apply median filtering to the harvested pitch results. The value represents the filter radius and can reduce breathiness.';R='Enable autotune';Q='rmvpe+';P='crepe-tiny';O='Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):';N='Auto-detect index path and select from the dropdown:';M='Refresh';L='Mangio-Crepe Hop Length (Only applies to mangio-crepe): Hop length refers to the time it takes for the speaker to jump to a dramatic pitch. Lower hop lengths take more time to infer but are more pitch accurate.';K='crepe';J='dio';I='harvest';H='pm';G='Select the pitch extraction algorithm:';F='Convert';E='Advanced Settings';D='mangio-crepe-tiny';C='mangio-crepe';B='Output information:';A='primary';default_weight=names[0]if names else''
	with gr.Blocks(title='🔊 AX-RVC',theme=gr.themes.Base(primary_hue='blue',neutral_hue='zinc'))as app:
		gr.HTML('<h1> 🍏 AX-RVC </h1>')
		with gr.Tabs():
			with gr.TabItem(i18n('Model Inference')):
				with gr.Row():sid0=gr.Dropdown(label=i18n('Inferencing voice:'),choices=sorted(names),value=default_weight);refresh_button=gr.Button(i18n(M),variant=A);clean_button=gr.Button(i18n('Unload voice to save GPU memory'),variant=A);clean_button.click(fn=lambda:{_G:'',_C:_D},inputs=[],outputs=[sid0])
				with gr.TabItem(i18n('Single')):
					with gr.Row():spk_item=gr.Slider(minimum=0,maximum=2333,step=1,label=i18n('Select Speaker/Singer ID:'),value=0,visible=_B,interactive=_A)
					with gr.Row():
						with gr.Column():dropbox=gr.File(label=i18n('Drag your audio here:'));record_button=gr.Audio(source='microphone',label=i18n('Or record an audio:'),type='filepath')
						best_match_index_path1,_=match_index(sid0.value)
						with gr.Column():
							file_index2=gr.Dropdown(label=i18n(N),choices=get_indexes(),value=best_match_index_path1,interactive=_A,allow_custom_value=_A)
							with gr.Column():input_audio1=gr.Dropdown(label=i18n('Auto detect audio path and select from the dropdown:'),choices=sorted(audio_paths),value='',interactive=_A);vc_transform0=gr.Number(label=i18n(O),value=0)
							dropbox.upload(fn=save_to_wav2,inputs=[dropbox],outputs=[input_audio1]);record_button.change(fn=save_to_wav,inputs=[record_button],outputs=[input_audio1]);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2,input_audio1])
					advanced_settings_checkbox=gr.Checkbox(value=_B,label=i18n(E),interactive=_A)
					with gr.Column(visible=_B)as advanced_settings:
						with gr.Row(label=i18n(E),open=_B):
							with gr.Column():f0method0=gr.Radio(label=i18n(G),choices=[H,I,J,K,P,C,D,_K,Q],value=Q,interactive=_A);f0_autotune=gr.Checkbox(label=R,interactive=_A);crepe_hop_length=gr.Slider(minimum=1,maximum=512,step=1,label=i18n(L),value=120,interactive=_A,visible=_B);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(S),value=3,step=1,interactive=_A);minpitch_slider=gr.Slider(label=i18n(T),info=i18n('Specify minimal pitch for inference [HZ]'),step=.1,minimum=1,scale=0,value=50,maximum=16000,interactive=_A,visible=not rvc_globals.NotesOrHertz and f0method0.value!=_K);minpitch_txtbox=gr.Textbox(label=i18n(T),info=i18n('Specify minimal pitch for inference [NOTE][OCTAVE]'),placeholder='C5',visible=rvc_globals.NotesOrHertz and f0method0.value!=_K,interactive=_A);maxpitch_slider=gr.Slider(label=i18n(U),info=i18n('Specify max pitch for inference [HZ]'),step=.1,minimum=1,scale=0,value=1100,maximum=16000,interactive=_A,visible=not rvc_globals.NotesOrHertz and f0method0.value!=_K);maxpitch_txtbox=gr.Textbox(label=i18n(U),info=i18n('Specify max pitch for inference [NOTE][OCTAVE]'),placeholder='C6',visible=rvc_globals.NotesOrHertz and f0method0.value!=_K,interactive=_A);file_index1=gr.Textbox(label=i18n(V),value='',interactive=_A);f0_file=gr.File(label=i18n('F0 curve file (optional). One pitch per line. Replaces the default F0 and pitch modulation:'))
							f0method0.change(fn=lambda radio:{_E:radio in[C,D],_C:_D},inputs=[f0method0],outputs=[crepe_hop_length]);f0method0.change(fn=switch_pitch_controls,inputs=[f0method0],outputs=[minpitch_slider,minpitch_txtbox,maxpitch_slider,maxpitch_txtbox])
							with gr.Column():resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(W),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(X),value=.25,interactive=_A);protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(Y),value=.33,step=.01,interactive=_A);index_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(Z),value=.75,interactive=_A);formanting=gr.Checkbox(value=bool(DoFormant),label=i18n('Formant shift inference audio'),info=i18n('Used for male to female and vice-versa conversions'),interactive=_A,visible=_A);formant_preset=gr.Dropdown(value='',choices=get_fshift_presets(),label=i18n('Browse presets for formanting'),info=i18n('Presets are located in formantshiftcfg/ folder'),visible=bool(DoFormant));formant_refresh_button=gr.Button(value='🔄',visible=bool(DoFormant),variant=A);qfrency=gr.Slider(value=Quefrency,info=i18n(a),label=i18n('Quefrency for formant shifting'),minimum=.0,maximum=16.,step=.1,visible=bool(DoFormant),interactive=_A);tmbre=gr.Slider(value=Timbre,info=i18n(a),label=i18n('Timbre for formant shifting'),minimum=.0,maximum=16.,step=.1,visible=bool(DoFormant),interactive=_A);frmntbut=gr.Button('Apply',variant=A,visible=bool(DoFormant))
							formant_preset.change(fn=preset_apply,inputs=[formant_preset,qfrency,tmbre],outputs=[qfrency,tmbre]);formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button]);frmntbut.click(fn=formant_apply,inputs=[qfrency,tmbre],outputs=[qfrency,tmbre]);formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset,qfrency,tmbre],outputs=[formant_preset,qfrency,tmbre])
					def toggle_advanced_settings(checkbox):return{_E:checkbox,_C:_D}
					advanced_settings_checkbox.change(fn=toggle_advanced_settings,inputs=[advanced_settings_checkbox],outputs=[advanced_settings]);but0=gr.Button(i18n(F),variant=A).style(full_width=_A)
					with gr.Row():vc_output1=gr.Textbox(label=i18n(B));vc_output2=gr.Audio(label=i18n('Export audio (click on the three dots in the lower right corner to download)'))
					with gr.Group():
						with gr.Row():but0.click(vc.vc_single,[spk_item,input_audio1,vc_transform0,f0_file,f0method0,file_index1,file_index2,index_rate1,filter_radius0,resample_sr0,rms_mix_rate0,protect0,crepe_hop_length,minpitch_slider,minpitch_txtbox,maxpitch_slider,maxpitch_txtbox,f0_autotune],[vc_output1,vc_output2])
				with gr.TabItem(i18n('Batch')):
					with gr.Row():
						with gr.Column():vc_transform1=gr.Number(label=i18n(O),value=0);opt_input=gr.Textbox(label=i18n('Specify output folder:'),value='opt')
						with gr.Column():file_index4=gr.Dropdown(label=i18n(N),choices=get_indexes(),value=best_match_index_path1,interactive=_A);dir_input=gr.Textbox(label=i18n('Enter the path of the audio folder to be processed (copy it from the address bar of the file manager):'),value=os.path.join(now_dir,_R,_S));sid0.select(fn=match_index,inputs=[sid0],outputs=[file_index2,file_index4]);refresh_button.click(fn=lambda:change_choices()[1],inputs=[],outputs=file_index4)
						with gr.Column():inputs=gr.File(file_count=b,label=i18n(c))
					with gr.Row():
						with gr.Column():
							advanced_settings_batch_checkbox=gr.Checkbox(value=_B,label=i18n(E),interactive=_A)
							with gr.Row(visible=_B)as advanced_settings_batch:
								with gr.Row(label=i18n(E),open=_B):
									with gr.Column():file_index3=gr.Textbox(label=i18n(V),value='',interactive=_A);f0method1=gr.Radio(label=i18n(G),choices=[H,I,J,K,P,C,D,_K],value=_K,interactive=_A);format1=gr.Radio(label=i18n(d),choices=[_Y,_Z,_g,_h],value=_Y,interactive=_A)
								with gr.Column():resample_sr1=gr.Slider(minimum=0,maximum=48000,label=i18n(W),value=0,step=1,interactive=_A);rms_mix_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(X),value=1,interactive=_A);protect1=gr.Slider(minimum=0,maximum=.5,label=i18n(Y),value=.33,step=.01,interactive=_A);filter_radius1=gr.Slider(minimum=0,maximum=7,label=i18n(S),value=3,step=1,interactive=_A);index_rate2=gr.Slider(minimum=0,maximum=1,label=i18n(Z),value=.75,interactive=_A);f0_autotune=gr.Checkbox(label=R,interactive=_A);crepe_hop_length=gr.Slider(minimum=1,maximum=512,step=1,label=i18n(L),value=120,interactive=_A,visible=_B)
							but1=gr.Button(i18n(F),variant=A);vc_output3=gr.Textbox(label=i18n(B));but1.click(vc.vc_multi,[spk_item,dir_input,opt_input,inputs,vc_transform1,f0method1,file_index3,file_index4,index_rate2,filter_radius1,resample_sr1,rms_mix_rate1,protect1,format1,crepe_hop_length,minpitch_slider,minpitch_txtbox,maxpitch_slider,maxpitch_txtbox,f0_autotune],[vc_output3])
					sid0.change(fn=vc.get_vc,inputs=[sid0,protect0,protect1],outputs=[spk_item,protect0,protect1])
					if not sid0.value=='':spk_item,protect0,protect1=vc.get_vc(sid0.value,protect0,protect1)
					def toggle_advanced_settings_batch(checkbox):return{_E:checkbox,_C:_D}
					advanced_settings_batch_checkbox.change(fn=toggle_advanced_settings_batch,inputs=[advanced_settings_batch_checkbox],outputs=[advanced_settings_batch])
			with gr.TabItem(i18n('Train')):
				with gr.Accordion(label=i18n('Step 1: Processing data')):
					with gr.Row():exp_dir1=gr.Textbox(label=i18n('Enter the model name:'),value=i18n('Model_Name'));sr2=gr.Radio(label=i18n('Target sample rate:'),choices=[_T,_c,_b],value=_T,interactive=_A);if_f0_3=gr.Checkbox(label=i18n('Whether the model has pitch guidance.'),value=_A,interactive=_A);version19=gr.Radio(label=i18n('Version:'),choices=[_d],value=_d,interactive=_A,visible=_A);np7=gr.Slider(minimum=1,maximum=config.n_cpu,step=1,label=i18n('Number of CPU processes:'),value=config.n_cpu,interactive=_A)
				with gr.Accordion(label=i18n('Step 2: Skipping pitch extraction')):
					with gr.Row():
						with gr.Column():trainset_dir4=gr.Dropdown(choices=sorted(datasets),label=i18n('Select your dataset:'),value=get_dataset());btn_update_dataset_list=gr.Button(i18n('Update list'),variant=A)
						with gr.Column():dataset_path=gr.Textbox(label=i18n('Or add your dataset path:'),interactive=_A)
						spk_id5=gr.Slider(minimum=0,maximum=4,step=1,label=i18n('Specify the model ID:'),value=0,interactive=_A);btn_update_dataset_list.click(resources.update_dataset_list,[spk_id5],trainset_dir4);but1=gr.Button(i18n('Process data'),variant=A);info1=gr.Textbox(label=i18n(B),value='');but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2,np7,dataset_path],[info1])
				with gr.Accordion(label=i18n('Step 3: Extracting features')):
					with gr.Row():
						with gr.Column():gpus6=gr.Textbox(label=i18n(e),value=gpus,interactive=_A);gpu_info9=gr.Textbox(label=i18n('GPU Information:'),value=gpu_info,visible=F0GPUVisible)
						with gr.Column():f0method8=gr.Radio(label=i18n(G),choices=[H,I,J,K,C,_K,_y],value=_K,interactive=_A);extraction_crepe_hop_length=gr.Slider(minimum=1,maximum=512,step=1,label=i18n(L),value=64,interactive=_A,visible=_B);f0method8.change(fn=lambda radio:{_E:radio in[C,D],_C:_D},inputs=[f0method8],outputs=[extraction_crepe_hop_length])
						but2=gr.Button(i18n('Feature extraction'),variant=A);info2=gr.Textbox(label=i18n(B),value='',max_lines=8,interactive=_B);but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1,version19,extraction_crepe_hop_length],[info2])
				with gr.Row():
					with gr.Accordion(label=i18n('Step 4: Model training started')):
						with gr.Row():save_epoch10=gr.Slider(minimum=1,maximum=100,step=1,label=i18n('Save frequency:'),value=10,interactive=_A,visible=_A);total_epoch11=gr.Slider(minimum=1,maximum=10000,step=2,label=i18n('Training epochs:'),value=750,interactive=_A);batch_size12=gr.Slider(minimum=1,maximum=50,step=1,label=i18n('Batch size per GPU:'),value=default_batch_size,interactive=_A)
						with gr.Row():if_save_latest13=gr.Checkbox(label=i18n('Whether to save only the latest .ckpt file to save hard drive space'),value=_A,interactive=_A);if_cache_gpu17=gr.Checkbox(label=i18n('Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training'),value=_B,interactive=_A);if_save_every_weights18=gr.Checkbox(label=i18n("Save a small final model to the 'weights' folder at each save point"),value=_A,interactive=_A)
						with gr.Row():
							pretrained_G14=gr.Textbox(lines=4,label=i18n('Load pre-trained base model G path:'),value='/kaggle/input/ax-rmf/pretrained_v2/f0G40k.pth',interactive=_A);pretrained_D15=gr.Textbox(lines=4,label=i18n('Load pre-trained base model D path:'),value='/kaggle/input/ax-rmf/pretrained_v2/f0D40k.pth',interactive=_A);gpus16=gr.Textbox(label=i18n(e),value=gpus,interactive=_A);sr2.change(change_sr2,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15]);version19.change(change_version19,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15,sr2]);if_f0_3.change(fn=change_f0,inputs=[if_f0_3,sr2,version19],outputs=[f0method8,pretrained_G14,pretrained_D15]);if_f0_3.change(fn=lambda radio:{_E:radio in[C,D],_C:_D},inputs=[f0method8],outputs=[extraction_crepe_hop_length]);butstop=gr.Button(i18n('Stop training'),variant=A,visible=_B);but3=gr.Button(i18n('Train model'),variant=A,visible=_A);but3.click(fn=stoptraining,inputs=[gr.Number(value=0,visible=_B)],outputs=[but3,butstop]);butstop.click(fn=stoptraining,inputs=[gr.Number(value=1,visible=_B)],outputs=[but3,butstop])
							with gr.Column():info3=gr.Textbox(label=i18n(B),value='',max_lines=4);save_action=gr.Dropdown(label=i18n('Save type'),choices=[i18n('Save all'),i18n('Save D and G'),i18n('Save voice')],value=i18n('Choose the method'),interactive=_A);but7=gr.Button(i18n('Save model'),variant=A);but4=gr.Button(i18n('Train feature index'),variant=A)
							if_save_every_weights18.change(fn=lambda if_save_every_weights:{_E:if_save_every_weights,_C:_D},inputs=[if_save_every_weights18],outputs=[save_epoch10])
						but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19],[info3,butstop,but3]);but4.click(train_index,[exp_dir1,version19],info3);but7.click(resources.save_model,[exp_dir1,save_action],info3)
			with gr.TabItem(i18n('UVR5')):
				with gr.Row():
					with gr.Column():model_select=gr.Radio(label=i18n('Model Architecture:'),choices=[_a,_j],value=_a,interactive=_A);dir_wav_input=gr.Textbox(label=i18n('Enter the path of the audio folder to be processed:'),value=os.path.join(now_dir,_R,_S));wav_inputs=gr.File(file_count=b,label=i18n(c))
					with gr.Column():model_choose=gr.Dropdown(label=i18n('Model:'),choices=uvr5_names);agg=gr.Slider(minimum=0,maximum=20,step=1,label='Vocal Extraction Aggressive',value=10,interactive=_A,visible=_B);opt_vocal_root=gr.Textbox(label=i18n('Specify the output folder for vocals:'),value=_v);opt_ins_root=gr.Textbox(label=i18n('Specify the output folder for accompaniment:'),value=_w);format0=gr.Radio(label=i18n(d),choices=[_Y,_Z,_g,_h],value=_Z,interactive=_A)
					model_select.change(fn=update_model_choices,inputs=model_select,outputs=model_choose);but2=gr.Button(i18n(F),variant=A);vc_output4=gr.Textbox(label=i18n(B));but2.click(uvr,[model_choose,dir_wav_input,opt_vocal_root,wav_inputs,opt_ins_root,agg,format0,model_select],[vc_output4])
			with gr.TabItem(i18n('TTS')):
				with gr.Column():text_test=gr.Textbox(label=i18n('Text:'),placeholder=i18n('Enter the text you want to convert to voice...'),lines=6)
				with gr.Row():
					with gr.Column():tts_methods_voice=[f,'Bark-tts'];ttsmethod_test=gr.Dropdown(tts_methods_voice,value=f,label=i18n('TTS Method:'),visible=_A);tts_test=gr.Dropdown(tts.set_edge_voice,label=i18n('TTS Model:'),visible=_A);ttsmethod_test.change(fn=tts.update_tts_methods_voice,inputs=ttsmethod_test,outputs=tts_test)
					with gr.Column():model_voice_path07=gr.Dropdown(label=i18n('RVC Model:'),choices=sorted(names),value=default_weight);best_match_index_path1,_=match_index(model_voice_path07.value);file_index2_07=gr.Dropdown(label=i18n('Select the .index file:'),choices=get_indexes(),value=best_match_index_path1,interactive=_A,allow_custom_value=_A)
				with gr.Row():refresh_button_=gr.Button(i18n(M),variant=A);refresh_button_.click(fn=change_choices2,inputs=[],outputs=[model_voice_path07,file_index2_07])
				with gr.Row():original_ttsvoice=gr.Audio(label=i18n('Audio TTS:'));ttsvoice=gr.Audio(label=i18n('Audio RVC:'))
				with gr.Row():button_test=gr.Button(i18n(F),variant=A)
				button_test.click(tts.use_tts,inputs=[text_test,tts_test,model_voice_path07,file_index2_07,vc_transform0,f0method8,index_rate1,crepe_hop_length,f0_autotune,ttsmethod_test],outputs=[ttsvoice,original_ttsvoice])
			with gr.TabItem('HuggingFace 🤗'):
				with gr.Row():
					with gr.Column():hgf_token_gr=gr.Textbox(label='Enter HuggingFace Write Token:');hgf_name_gr=gr.Textbox(label='Enter HuggingFace Username:');hgf_repo_gr=gr.Textbox(label='Enter HuggingFace Model-Repo name:')
					with gr.Column():model_name_gr=gr.Textbox(label='Trained model name:');zip_name_gr=gr.Textbox(label='Name of Zip file:');what_upload_gr=gr.Radio(label='Upload files:',choices=[_t,'Model Log Folder'],value=_t,interactive=_A,visible=_A)
				with gr.Row():uploadbut1=gr.Button('Start upload',variant=A);uploadinfo1=gr.Textbox(label=B,value='');uploadbut1.click(start_upload_to_huggingface,[hgf_token_gr,hgf_name_gr,hgf_repo_gr,model_name_gr,zip_name_gr,what_upload_gr],[uploadinfo1])
			with gr.TabItem(i18n('Resources')):resources.download_model();resources.download_backup();resources.download_dataset(trainset_dir4);resources.download_audio();resources.youtube_separator()
			with gr.TabItem(i18n('Extra')):
				gr.Markdown(value=i18n('This section contains some extra utilities that often may be in experimental phases'))
				with gr.TabItem(i18n('Merge Audios')):mergeaudios.merge_audios()
				with gr.TabItem(i18n('Processing')):processing.processing_()
			with gr.TabItem(i18n('Settings')):
				with gr.Row():
					with gr.Column():gr.Markdown(value=i18n('Pitch settings'));noteshertz=gr.Checkbox(label=i18n('Whether to use note names instead of their hertz value. E.G. [C5, D6] instead of [523.25, 1174.66]Hz'),value=rvc_globals.NotesOrHertz,interactive=_A)
			noteshertz.change(fn=lambda nhertz:rvc_globals.__setattr__('NotesOrHertz',nhertz),inputs=[noteshertz],outputs=[]);noteshertz.change(fn=switch_pitch_controls,inputs=[f0method0],outputs=[minpitch_slider,minpitch_txtbox,maxpitch_slider,maxpitch_txtbox])
			with gr.TabItem(i18n('Readme')):gr.Markdown(value=inforeadme)
		return app
def GradioRun(app):
	B='./assets/images/icon.png';A='0.0.0.0';share_gradio_link=config.iscolab or config.paperspace;concurrency_count=511;max_size=1022
	if config.iscolab or config.paperspace:app.queue(concurrency_count=concurrency_count,max_size=max_size).launch(server_name=A,inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A,favicon_path=B,share=_B)
	else:app.queue(concurrency_count=concurrency_count,max_size=max_size).launch(server_name=A,inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A,favicon_path=B,share=_B)
if __name__=='__main__':
	if os.name=='nt':logger.info(i18n('Any ConnectionResetErrors post-conversion are irrelevant and purely visual; they can be ignored.\n'))
	app=GradioSetup();GradioRun(app)