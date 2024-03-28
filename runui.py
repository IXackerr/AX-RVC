_AO='/kaggle/input/ax-rmd/pretrained_v2'
_AN='以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2'
_AM='也可批量输入音频文件, 二选一, 优先读文件夹'
_AL='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_AK='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_AJ='后处理重采样至最终采样率，0为不进行重采样'
_AI='自动检测index路径,下拉式选择(dropdown)'
_AH='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_AG='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_AF='变调(整数, 半音数量, 升八度12降八度-12)'
_AE='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'
_AD='%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'
_AC='IVF%s,Flat'
_AB='%s/total_fea.npy'
_AA='Trying doing kmeans %s shape to 10k centers.'
_A9='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_A8=' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_A7=' train_nsf_sim_cache_sid_load_pretrain.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'
_A6='write filelist done'
_A5='%s/filelist.txt'
_A4='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'
_A3='%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'
_A2='%s/%s.wav|%s/%s.npy|%s'
_A1='%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'
_A0='%s/2b-f0nsf'
_z='%s/0_gt_wavs'
_y='emb_g.weight'
_x='sample_rate'
_w='%s->%s'
_v='.index'
_u='opt'
_t='harvest'
_s='%s/3_feature768'
_r='%s/3_feature256'
_q='_v2'
_p='32k'
_o='cpu'
_n='trained'
_m='m4a'
_l='mp3'
_k='rmvpe'
_j='pm'
_i='-pd %s'
_h='-pg %s'
_g='48k'
_f='weight'
_e='wav'
_d='rmvpe_gpu'
_c='%s/logs/%s'
_b='输出信息'
_a='not exist, will not use pretrained model'
_Z='/kaggle/input/ax-rmd/pretrained%s/%sD%s.pth'
_Y='/kaggle/input/ax-rmd/pretrained%s/%sG%s.pth'
_X='40k'
_W='version'
_V='.pth'
_U='flac'
_T='v2'
_S='f0'
_R='primary'
_Q='value'
_P='config'
_O='visible'
_N='\\\\'
_M='\\'
_L='r'
_K='"'
_J=' '
_I=None
_H='.'
_G='是'
_F='update'
_E='__type__'
_D='v1'
_C='\n'
_B=False
_A=True
import os,shutil,sys
now_dir=os.getcwd()
sys.path.append(now_dir)
import traceback,pdb,warnings,numpy as np,torch
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
from lib.audio import load_audio
from lib.train.process_ckpt import change_info,extract_small_model,merge,show_info
from vc_infer_pipeline import VC
from sklearn.cluster import MiniBatchKMeans
logging.getLogger('numba').setLevel(logging.WARNING)
now_dir=os.getcwd()
tmp=os.path.join(now_dir,'TEMP')
shutil.rmtree(tmp,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/infer_pack'%now_dir,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/uvr5_pack'%now_dir,ignore_errors=_A)
os.makedirs(tmp,exist_ok=_A)
os.makedirs(os.path.join(now_dir,'logs'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'weights'),exist_ok=_A)
os.environ['TEMP']=tmp
warnings.filterwarnings('ignore')
torch.manual_seed(114514)
config=Config()
i18n=I18nAuto()
i18n.print()
ngpu=torch.cuda.device_count()
gpu_infos=[]
mem=[]
if_gpu_ok=_B
audio_root='/kaggle/working/AX-RVC/assets/audios'
weight_root='/kaggle/working/AX-RVC/weights'
sup_audioext={_e,_l,_U,'ogg','opus',_m,'mp4','aac','alac','wma','aiff','webm','ac3'}
audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root]
if torch.cuda.is_available()or ngpu!=0:
	for i in range(ngpu):
		gpu_name=torch.cuda.get_device_name(i)
		if any(value in gpu_name.upper()for value in['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):if_gpu_ok=_A;gpu_infos.append('%s\t%s'%(i,gpu_name));mem.append(int(torch.cuda.get_device_properties(i).total_memory/1024/1024/1024+.4))
if if_gpu_ok and len(gpu_infos)>0:gpu_info=_C.join(gpu_infos);default_batch_size=min(mem)//2
else:gpu_info=i18n('很遗憾您这没有能用的显卡来支持您训练');default_batch_size=1
gpus='-'.join([i[0]for i in gpu_infos])
class ToolButton(gr.Button,gr.components.FormComponent):
	'Small button with single emoji as text, fits inside gradio forms'
	def __init__(self,**kwargs):super().__init__(variant='tool',**kwargs)
	def get_block_name(self):return'button'
hubert_model=_I
def load_hubert():
	global hubert_model;models,_,_=checkpoint_utils.load_model_ensemble_and_task(['/kaggle/input/ax-rmd/hubert_base.pt'],suffix='');hubert_model=models[0];hubert_model=hubert_model.to(config.device)
	if config.is_half:hubert_model=hubert_model.half()
	else:hubert_model=hubert_model.float()
	hubert_model.eval()
weight_uvr5_root='uvr5_weights'
index_root='/kaggle/working/AX-RVC/logs'
names=[os.path.join(root,file)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_V,'.onnx'))]
indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_v)and _n not in name]
audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root]
uvr5_names=[]
for name in os.listdir(weight_uvr5_root):
	if name.endswith(_V)or'onnx'in name:uvr5_names.append(name.replace(_V,''))
cpt=_I
def vc_single(sid,input_audio_path,f0_up_key,f0_file,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect):
	global tgt_sr,net_g,vc,hubert_model,version,cpt
	if input_audio_path is _I:return'You need to upload an audio',_I
	f0_up_key=int(f0_up_key)
	try:
		audio=load_audio(input_audio_path,16000);audio_max=np.abs(audio).max()/.95
		if audio_max>1:audio/=audio_max
		times=[0,0,0]
		if not hubert_model:load_hubert()
		if_f0=cpt.get(_S,1);file_index=file_index.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J).replace(_n,'added')if file_index!=''else file_index2;audio_opt=vc.pipeline(hubert_model,net_g,sid,audio,input_audio_path,times,f0_up_key,f0_method,file_index,index_rate,if_f0,filter_radius,tgt_sr,resample_sr,rms_mix_rate,version,protect,f0_file=f0_file)
		if tgt_sr!=resample_sr>=16000:tgt_sr=resample_sr
		index_info='Using index:%s.'%file_index if os.path.exists(file_index)else'Index not used.';return'Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss'%(index_info,times[0],times[1],times[2]),(tgt_sr,audio_opt)
	except:info=traceback.format_exc();print(info);return info,(_I,_I)
def vc_multi(sid,dir_path,opt_root,paths,f0_up_key,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect,format1):
	try:
		dir_path=dir_path.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J);opt_root=opt_root.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J);os.makedirs(opt_root,exist_ok=_A)
		try:
			if dir_path!='':paths=[os.path.join(dir_path,name)for name in os.listdir(dir_path)]
			else:paths=[path.name for path in paths]
		except:traceback.print_exc();paths=[path.name for path in paths]
		infos=[]
		for path in paths:
			info,opt=vc_single(sid,path,f0_up_key,_I,f0_method,file_index,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect)
			if'Success'in info:
				try:
					tgt_sr,audio_opt=opt
					if format1 in[_e,_U]:sf.write('%s/%s.%s'%(opt_root,os.path.basename(path),format1),audio_opt,tgt_sr)
					else:
						path='%s/%s.wav'%(opt_root,os.path.basename(path));sf.write(path,audio_opt,tgt_sr)
						if os.path.exists(path):os.system('ffmpeg -i %s -vn %s -q:a 2 -y'%(path,path[:-4]+'.%s'%format1))
				except:info+=traceback.format_exc()
			infos.append(_w%(os.path.basename(path),info));yield _C.join(infos)
		yield _C.join(infos)
	except:yield traceback.format_exc()
def uvr(model_name,inp_root,save_root_vocal,paths,save_root_ins,agg,format0):
	B='streams';A='onnx_dereverb_By_FoxJoy';infos=[]
	try:
		inp_root=inp_root.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J);save_root_vocal=save_root_vocal.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J);save_root_ins=save_root_ins.strip(_J).strip(_K).strip(_C).strip(_K).strip(_J)
		if model_name==A:from MDXNet import MDXNetDereverb;pre_fun=MDXNetDereverb(15)
		else:func=_audio_pre_ if'DeEcho'not in model_name else _audio_pre_new;pre_fun=func(agg=int(agg),model_path=os.path.join(weight_uvr5_root,model_name+_V),device=config.device,is_half=config.is_half)
		if inp_root!='':paths=[os.path.join(inp_root,name)for name in os.listdir(inp_root)]
		else:paths=[path.name for path in paths]
		for path in paths:
			inp_path=os.path.join(inp_root,path);need_reformat=1;done=0
			try:
				info=ffmpeg.probe(inp_path,cmd='ffprobe')
				if info[B][0]['channels']==2 and info[B][0][_x]=='44100':need_reformat=0;pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0);done=1
			except:need_reformat=1;traceback.print_exc()
			if need_reformat==1:tmp_path='%s/%s.reformatted.wav'%(tmp,os.path.basename(inp_path));os.system('ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y'%(inp_path,tmp_path));inp_path=tmp_path
			try:
				if done==0:pre_fun._path_audio_(inp_path,save_root_ins,save_root_vocal,format0)
				infos.append('%s->Success'%os.path.basename(inp_path));yield _C.join(infos)
			except:infos.append(_w%(os.path.basename(inp_path),traceback.format_exc()));yield _C.join(infos)
	except:infos.append(traceback.format_exc());yield _C.join(infos)
	finally:
		try:
			if model_name==A:del pre_fun.pred.model;del pre_fun.pred.model_
			else:del pre_fun.model;del pre_fun
		except:traceback.print_exc()
		print('clean_empty_cache')
		if torch.cuda.is_available():torch.cuda.empty_cache()
	yield _C.join(infos)
def get_index_path_from_model(sid):
	sel_index_path='';name=os.path.join('logs',sid.split(_H)[0],'')
	for f in indexes_list:
		if name in f:sel_index_path=f;break
	return sel_index_path
def get_vc(self,sid,*to_return_protect):
	A='maximum';logger.info('Get sid: '+sid);to_return_protect0={_O:self.if_f0!=0,_Q:to_return_protect[0]if self.if_f0!=0 and to_return_protect else .5,_E:_F};to_return_protect1={_O:self.if_f0!=0,_Q:to_return_protect[1]if self.if_f0!=0 and to_return_protect else .33,_E:_F}
	if sid==''or sid==[]:
		if self.hubert_model is not _I:
			logger.info('Clean model cache');del(self.net_g,self.n_spk,self.vc,self.hubert_model,self.tgt_sr);self.hubert_model=self.net_g=self.n_spk=self.vc=self.hubert_model=self.tgt_sr=_I
			if torch.cuda.is_available():torch.cuda.empty_cache()
			self.if_f0=self.cpt.get(_S,1);self.version=self.cpt.get(_W,_D)
			if self.version==_D:
				if self.if_f0==1:self.net_g=SynthesizerTrnMs256NSFsid(*self.cpt[_P],is_half=self.config.is_half)
				else:self.net_g=SynthesizerTrnMs256NSFsid_nono(*self.cpt[_P])
			elif self.version==_T:
				if self.if_f0==1:self.net_g=SynthesizerTrnMs768NSFsid(*self.cpt[_P],is_half=self.config.is_half)
				else:self.net_g=SynthesizerTrnMs768NSFsid_nono(*self.cpt[_P])
			del self.net_g,self.cpt
			if torch.cuda.is_available():torch.cuda.empty_cache()
		return{_O:_B,_E:_F},{_O:_A,_Q:to_return_protect0,_E:_F},{_O:_A,_Q:to_return_protect1,_E:_F},'',''
	person=f"{sid}";logger.info(f"Loading...");self.cpt=torch.load(person,map_location=_o);self.tgt_sr=self.cpt[_P][-1];self.cpt[_P][-3]=self.cpt[_f][_y].shape[0];self.if_f0=self.cpt.get(_S,1);self.version=self.cpt.get(_W,_D);synthesizer_class={(_D,1):SynthesizerTrnMs256NSFsid,(_D,0):SynthesizerTrnMs256NSFsid_nono,(_T,1):SynthesizerTrnMs768NSFsid,(_T,0):SynthesizerTrnMs768NSFsid_nono};self.net_g=synthesizer_class.get((self.version,self.if_f0),SynthesizerTrnMs256NSFsid)(*self.cpt[_P],is_half=self.config.is_half);del self.net_g.enc_q;self.net_g.load_state_dict(self.cpt[_f],strict=_B);self.net_g.eval().to(self.config.device)
	if self.config.is_half:self.net_g=self.net_g.half()
	else:self.net_g=self.net_g.float()
	self.pipeline=Pipeline(self.tgt_sr,self.config);n_spk=self.cpt[_P][-3];index={_Q:get_index_path_from_model(sid),_E:_F};logger.info('Select index: '+index[_Q]);return({_O:_B,A:n_spk,_E:_F},to_return_protect0,to_return_protect1)if to_return_protect else{_O:_B,A:n_spk,_E:_F}
def change_choices():names=[os.path.join(root,file)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_V,'.onnx'))];indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_v)and _n not in name];audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root];return gr.Dropdown.update(choices=sorted(names)),gr.Dropdown.update(choices=sorted(indexes_list)),gr.Dropdown.update(choices=sorted(audio_paths))
def clean():return{_Q:'',_E:_F}
sr_dict={_p:32000,_X:40000,_g:48000}
def if_done(done,p):
	while 1:
		if p.poll()is _I:sleep(.5)
		else:break
	done[0]=_A
def if_done_multi(done,ps):
	while 1:
		flag=1
		for p in ps:
			if p.poll()is _I:flag=0;sleep(.5);break
		if flag==1:break
	done[0]=_A
def preprocess_dataset(trainset_dir,exp_dir,sr,n_p):
	A='%s/logs/%s/preprocess.log';sr=sr_dict[sr];os.makedirs(_c%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close();cmd=config.python_cmd+' trainset_preprocess_pipeline_print.py "%s" %s %s "%s/logs/%s" '%(trainset_dir,sr,n_p,now_dir,exp_dir)+str(config.noparallel);print(cmd);p=Popen(cmd,shell=_A);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_L)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_L)as f:log=f.read()
	print(log);yield log
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir,version19,gpus_rmvpe):
	A='%s/logs/%s/extract_f0_feature.log';gpus=gpus.split('-');os.makedirs(_c%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close()
	if if_f0:
		if f0method!=_d:
			cmd=config.python_cmd+' extract_f0_print.py "%s/logs/%s" %s %s'%(now_dir,exp_dir,n_p,f0method);print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
			while 1:
				with open(A%(now_dir,exp_dir),_L)as f:yield f.read()
				sleep(1)
				if done[0]:break
			with open(A%(now_dir,exp_dir),_L)as f:log=f.read()
			print(log);yield log
		else:
			gpus_rmvpe=gpus_rmvpe.split('-');leng=len(gpus_rmvpe);ps=[]
			for(idx,n_g)in enumerate(gpus_rmvpe):cmd=config.python_cmd+' extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '%(leng,idx,n_g,now_dir,exp_dir,config.is_half);print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
			done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
			while 1:
				with open(A%(now_dir,exp_dir),_L)as f:yield f.read()
				sleep(1)
				if done[0]:break
			with open(A%(now_dir,exp_dir),_L)as f:log=f.read()
			print(log);yield log
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd=config.python_cmd+' extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(config.device,leng,idx,n_g,now_dir,exp_dir,version19);print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_L)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_L)as f:log=f.read()
	print(log);yield log
def change_sr2(sr2,if_f0_3,version19):
	path_str=''if version19==_D else _q;f0_str=_S if if_f0_3 else'';if_pretrained_generator_exist=os.access(_Y%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(_Z%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(_Y%(path_str,f0_str,sr2),_a)
	if not if_pretrained_discriminator_exist:print(_Z%(path_str,f0_str,sr2),_a)
	return _Y%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',_Z%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else''
def change_version19(sr2,if_f0_3,version19):
	A='choices';path_str=''if version19==_D else _q
	if sr2==_p and version19==_D:sr2=_X
	to_return_sr2={A:[_X,_g],_E:_F,_Q:sr2}if version19==_D else{A:[_X,_g,_p],_E:_F,_Q:sr2};f0_str=_S if if_f0_3 else'';if_pretrained_generator_exist=os.access(_Y%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(_Z%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(_Y%(path_str,f0_str,sr2),_a)
	if not if_pretrained_discriminator_exist:print(_Z%(path_str,f0_str,sr2),_a)
	return _Y%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',_Z%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else'',to_return_sr2
def change_f0(if_f0_3,sr2,version19):
	B='/kaggle/input/ax-rmd/pretrained%s/f0D%s.pth';A='/kaggle/input/ax-rmd/pretrained%s/f0G%s.pth';path_str=''if version19==_D else _q;if_pretrained_generator_exist=os.access(A%(path_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(B%(path_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:print(A%(path_str,sr2),_a)
	if not if_pretrained_discriminator_exist:print(B%(path_str,sr2),_a)
	if if_f0_3:return{_O:_A,_E:_F},A%(path_str,sr2)if if_pretrained_generator_exist else'',B%(path_str,sr2)if if_pretrained_discriminator_exist else''
	return{_O:_B,_E:_F},'/kaggle/input/ax-rmd/pretrained%s/G%s.pth'%(path_str,sr2)if if_pretrained_generator_exist else'','/kaggle/input/ax-rmd/pretrained%s/D%s.pth'%(path_str,sr2)if if_pretrained_discriminator_exist else''
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19):
	A='\x08';exp_dir=_c%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);gt_wavs_dir=_z%exp_dir;feature_dir=_r%exp_dir if version19==_D else _s%exp_dir
	if if_f0_3:f0_dir='%s/2a_f0'%exp_dir;f0nsf_dir=_A0%exp_dir;names=set([name.split(_H)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_H)[0]for name in os.listdir(feature_dir)])&set([name.split(_H)[0]for name in os.listdir(f0_dir)])&set([name.split(_H)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_H)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_H)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append(_A1%(gt_wavs_dir.replace(_M,_N),name,feature_dir.replace(_M,_N),name,f0_dir.replace(_M,_N),name,f0nsf_dir.replace(_M,_N),name,spk_id5))
		else:opt.append(_A2%(gt_wavs_dir.replace(_M,_N),name,feature_dir.replace(_M,_N),name,spk_id5))
	fea_dim=256 if version19==_D else 768
	if if_f0_3:
		for _ in range(2):opt.append(_A3%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append(_A4%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open(_A5%exp_dir,'w')as f:f.write(_C.join(opt))
	print(_A6);print('use gpus:',gpus16)
	if pretrained_G14=='':print('no pretrained Generator')
	if pretrained_D15=='':print('no pretrained Discriminator')
	if gpus16:cmd=config.python_cmd+_A7%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,_h%pretrained_G14 if pretrained_G14!=''else'',_i%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==i18n(_G)else 0,1 if if_cache_gpu17==i18n(_G)else 0,1 if if_save_every_weights18==i18n(_G)else 0,version19)
	else:cmd=config.python_cmd+_A8%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,_h%pretrained_G14 if pretrained_G14!=''else A,_i%pretrained_D15 if pretrained_D15!=''else A,1 if if_save_latest13==i18n(_G)else 0,1 if if_cache_gpu17==i18n(_G)else 0,1 if if_save_every_weights18==i18n(_G)else 0,version19)
	print(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();return _A9
def train_index(exp_dir1,version19):
	exp_dir=_c%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);feature_dir=_r%exp_dir if version19==_D else _s%exp_dir
	if not os.path.exists(feature_dir):return'请先进行特征提取!'
	listdir_res=list(os.listdir(feature_dir))
	if len(listdir_res)==0:return'请先进行特征提取！'
	infos=[];npys=[]
	for name in sorted(listdir_res):phone=np.load('%s/%s'%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		infos.append(_AA%big_npy.shape[0]);yield _C.join(infos)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_B,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();print(info);infos.append(info);yield _C.join(infos)
	np.save(_AB%exp_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);infos.append('%s,%s'%(big_npy.shape,n_ivf));yield _C.join(infos);index=faiss.index_factory(256 if version19==_D else 768,_AC%n_ivf);infos.append('training');yield _C.join(infos);index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,_AD%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('adding');yield _C.join(infos);batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,_AE%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19));yield _C.join(infos)
def train1key(exp_dir1,sr2,if_f0_3,trainset_dir4,spk_id5,np7,f0method8,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19,gpus_rmvpe):
	infos=[]
	def get_info_str(strr):infos.append(strr);return _C.join(infos)
	model_log_dir=_c%(now_dir,exp_dir1);preprocess_log_path='%s/preprocess.log'%model_log_dir;extract_f0_feature_log_path='%s/extract_f0_feature.log'%model_log_dir;gt_wavs_dir=_z%model_log_dir;feature_dir=_r%model_log_dir if version19==_D else _s%model_log_dir;os.makedirs(model_log_dir,exist_ok=_A);open(preprocess_log_path,'w').close();cmd=config.python_cmd+' trainset_preprocess_pipeline_print.py "%s" %s %s "%s" '%(trainset_dir4,sr_dict[sr2],np7,model_log_dir)+str(config.noparallel);yield get_info_str(i18n('step1:正在处理数据'));yield get_info_str(cmd);p=Popen(cmd,shell=_A);p.wait()
	with open(preprocess_log_path,_L)as f:print(f.read())
	open(extract_f0_feature_log_path,'w')
	if if_f0_3:
		yield get_info_str('step2a:正在提取音高')
		if f0method8!=_d:cmd=config.python_cmd+' extract_f0_print.py "%s" %s %s'%(model_log_dir,np7,f0method8);yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait()
		else:
			gpus_rmvpe=gpus_rmvpe.split('-');leng=len(gpus_rmvpe);ps=[]
			for(idx,n_g)in enumerate(gpus_rmvpe):cmd=config.python_cmd+' extract_f0_rmvpe.py %s %s %s "%s" %s '%(leng,idx,n_g,model_log_dir,config.is_half);yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
			for p in ps:p.wait()
		with open(extract_f0_feature_log_path,_L)as f:print(f.read())
	else:yield get_info_str(i18n('step2a:无需提取音高'))
	yield get_info_str(i18n('step2b:正在提取特征'));gpus=gpus16.split('-');leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd=config.python_cmd+' extract_feature_print.py %s %s %s %s "%s" %s'%(config.device,leng,idx,n_g,model_log_dir,version19);yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	for p in ps:p.wait()
	with open(extract_f0_feature_log_path,_L)as f:print(f.read())
	yield get_info_str(i18n('step3a:正在训练模型'))
	if if_f0_3:f0_dir='%s/2a_f0'%model_log_dir;f0nsf_dir=_A0%model_log_dir;names=set([name.split(_H)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_H)[0]for name in os.listdir(feature_dir)])&set([name.split(_H)[0]for name in os.listdir(f0_dir)])&set([name.split(_H)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_H)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_H)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append(_A1%(gt_wavs_dir.replace(_M,_N),name,feature_dir.replace(_M,_N),name,f0_dir.replace(_M,_N),name,f0nsf_dir.replace(_M,_N),name,spk_id5))
		else:opt.append(_A2%(gt_wavs_dir.replace(_M,_N),name,feature_dir.replace(_M,_N),name,spk_id5))
	fea_dim=256 if version19==_D else 768
	if if_f0_3:
		for _ in range(2):opt.append(_A3%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append(_A4%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open(_A5%model_log_dir,'w')as f:f.write(_C.join(opt))
	yield get_info_str(_A6)
	if gpus16:cmd=config.python_cmd+_A7%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,_h%pretrained_G14 if pretrained_G14!=''else'',_i%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==i18n(_G)else 0,1 if if_cache_gpu17==i18n(_G)else 0,1 if if_save_every_weights18==i18n(_G)else 0,version19)
	else:cmd=config.python_cmd+_A8%(exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,_h%pretrained_G14 if pretrained_G14!=''else'',_i%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==i18n(_G)else 0,1 if if_cache_gpu17==i18n(_G)else 0,1 if if_save_every_weights18==i18n(_G)else 0,version19)
	yield get_info_str(cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();yield get_info_str(i18n(_A9));npys=[];listdir_res=list(os.listdir(feature_dir))
	for name in sorted(listdir_res):phone=np.load('%s/%s'%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		info=_AA%big_npy.shape[0];print(info);yield get_info_str(info)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_B,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();print(info);yield get_info_str(info)
	np.save(_AB%model_log_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);yield get_info_str('%s,%s'%(big_npy.shape,n_ivf));index=faiss.index_factory(256 if version19==_D else 768,_AC%n_ivf);yield get_info_str('training index');index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,_AD%(model_log_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str('adding index');batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,_AE%(model_log_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str('成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19));yield get_info_str(i18n('全流程结束！'))
def change_info_(ckpt_path):
	A='train.log'
	if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path),A)):return{_E:_F},{_E:_F},{_E:_F}
	try:
		with open(ckpt_path.replace(os.path.basename(ckpt_path),A),_L)as f:info=eval(f.read().strip(_C).split(_C)[0].split('\t')[-1]);sr,f0=info[_x],info['if_f0'];version=_T if _W in info and info[_W]==_T else _D;return sr,str(f0),version
	except:traceback.print_exc();return{_E:_F},{_E:_F},{_E:_F}
def change_f0_method(f0method8):
	if f0method8==_d:visible=_A
	else:visible=_B
	return{_O:visible,_E:_F}
def save_to_wav(record_button):
	if record_button is _I:0
	else:path_to_file=record_button;new_name=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.wav';target_path=os.path.join('assets','audios',os.path.basename(new_name));shutil.move(path_to_file,target_path);return target_path
def save_to_wav2(dropbox):
	file_path=dropbox.name;target_path=os.path.join('assets','audios',os.path.basename(file_path))
	if os.path.exists(target_path):os.remove(target_path);print('Replacing old dropdown file...')
	shutil.move(file_path,target_path);return target_path
def export_onnx(ModelPath,ExportedPath):D='rnd';C='pitchf';B='pitch';A='phone';global cpt;cpt=torch.load(ModelPath,map_location=_o);cpt[_P][-3]=cpt[_f][_y].shape[0];vec_channels=256 if cpt.get(_W,_D)==_D else 768;test_phone=torch.rand(1,200,vec_channels);test_phone_lengths=torch.tensor([200]).long();test_pitch=torch.randint(size=(1,200),low=5,high=255);test_pitchf=torch.rand(1,200);test_ds=torch.LongTensor([0]);test_rnd=torch.rand(1,192,200);device=_o;net_g=SynthesizerTrnMsNSFsidM(*cpt[_P],is_half=_B,version=cpt.get(_W,_D));net_g.load_state_dict(cpt[_f],strict=_B);input_names=[A,'phone_lengths',B,C,'ds',D];output_names=['audio'];torch.onnx.export(net_g,(test_phone.to(device),test_phone_lengths.to(device),test_pitch.to(device),test_pitchf.to(device),test_ds.to(device),test_rnd.to(device)),ExportedPath,dynamic_axes={A:[1],B:[1],C:[1],D:[2]},do_constant_folding=_B,opset_version=13,verbose=_B,input_names=input_names,output_names=output_names);return'Finished'
with gr.Blocks(title='🔊 AX-RVC UI',theme=gr.themes.Base(primary_hue='sky',neutral_hue='zinc'))as app:
	gr.HTML('<h1> 🍏 AX-RVC </h1><h3>Custom RVC UI created with 💙</h3>')
	with gr.Tabs():
		with gr.TabItem(i18n('模型推理')):
			with gr.Row():sid0=gr.Dropdown(label=i18n('推理音色'),choices=sorted(names));refresh_button=gr.Button(i18n('刷新音色列表和索引路径'),variant=_R);clean_button=gr.Button(i18n('卸载音色省显存'),variant=_R);spk_item=gr.Slider(minimum=0,maximum=2333,step=1,label=i18n('请选择说话人id'),value=0,visible=_B,interactive=_A);clean_button.click(fn=clean,inputs=[],outputs=[sid0],api_name='infer_clean')
			with gr.TabItem(i18n('Single')):
				gr.Markdown(value=i18n('男转女推荐+12key, 女转男推荐-12key, 如果音域爆炸导致音色失真也可以自己调整到合适音域. '))
				with gr.Row():
					with gr.Column():dropbox=gr.File(label=i18n('Drag your audio here:'));record_button=gr.Audio(source='microphone',label=i18n('Or record an audio:'),type='filepath')
					with gr.Column():vc_transform0=gr.Number(label=i18n(_AF),value=0);f0method0=gr.Radio(label=i18n(_AG),choices=[_j,_t,'crepe',_k],value=_k,interactive=_A);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(_AH),value=3,step=1,interactive=_A)
					with gr.Column():
						file_index2=gr.Dropdown(label=i18n(_AI),choices=sorted(indexes_list),interactive=_A)
						with gr.Column():input_audio1=gr.Dropdown(label=i18n('Auto detect audio path and select from the dropdown:'),choices=sorted(audio_paths),value='',interactive=_A);vc_transform0=gr.Number(label=i18n('Transpose (integer, number of semitones, raise by an octave: 12, lower by an octave: -12):'),value=0);dropbox.upload(fn=save_to_wav2,inputs=[dropbox],outputs=[input_audio1]);record_button.change(fn=save_to_wav,inputs=[record_button],outputs=[input_audio1]);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2,input_audio1],api_name='infer_refresh')
					with gr.Column():resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(_AJ),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(_AK),value=.25,interactive=_A);protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(_AL),value=.33,step=.01,interactive=_A)
					f0_file=gr.File(label=i18n('F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调'));but0=gr.Button(i18n('转换'),variant=_R)
					with gr.Row():vc_output1=gr.Textbox(label=i18n(_b));vc_output2=gr.Audio(label=i18n('输出音频(右下角三个点,点了可以下载)'))
					but0.click(vc_single,[spk_item,input_audio1,vc_transform0,f0_file,f0method0,file_index2,filter_radius0,resample_sr0,rms_mix_rate0,protect0],[vc_output1,vc_output2],api_name='infer_convert')
			with gr.Group():
				gr.Markdown(value=i18n('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr.Row():
					with gr.Column():vc_transform1=gr.Number(label=i18n(_AF),value=0);opt_input=gr.Textbox(label=i18n('指定输出文件夹'),value=_u);f0method1=gr.Radio(label=i18n(_AG),choices=[_j,_t,'crepe',_k],value=_j,interactive=_A);filter_radius1=gr.Slider(minimum=0,maximum=7,label=i18n(_AH),value=3,step=1,interactive=_A)
					with gr.Column():file_index3=gr.Textbox(label=i18n('特征检索库文件路径,为空则使用下拉的选择结果'),value='',interactive=_A);file_index4=gr.Dropdown(label=i18n(_AI),choices=sorted(indexes_list),interactive=_A);refresh_button.click(fn=lambda:change_choices()[1],inputs=[],outputs=file_index4,api_name='infer_refresh_batch');index_rate2=gr.Slider(minimum=0,maximum=1,label=i18n('检索特征占比'),value=1,interactive=_A)
					with gr.Column():resample_sr1=gr.Slider(minimum=0,maximum=48000,label=i18n(_AJ),value=0,step=1,interactive=_A);rms_mix_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_AK),value=1,interactive=_A);protect1=gr.Slider(minimum=0,maximum=.5,label=i18n(_AL),value=.33,step=.01,interactive=_A)
					with gr.Column():dir_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),value='E:\\codes\\py39\\test-20230416b\\todo-songs');inputs=gr.File(file_count='multiple',label=i18n(_AM))
					with gr.Row():format1=gr.Radio(label=i18n('导出文件格式'),choices=[_e,_U,_l,_m],value=_U,interactive=_A);but1=gr.Button(i18n('转换'),variant=_R);vc_output3=gr.Textbox(label=i18n(_b))
					but1.click(vc_multi,[spk_item,dir_input,opt_input,inputs,vc_transform1,f0method1,file_index3,file_index4,index_rate2,filter_radius1,resample_sr1,rms_mix_rate1,protect1,format1],[vc_output3],api_name='infer_convert_batch')
			sid0.change(fn=get_vc,inputs=[sid0,protect0,protect1],outputs=[spk_item,protect0,protect1,file_index2])
			with gr.Group():
				gr.Markdown(value=i18n('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr.Row():
					with gr.Column():dir_wav_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径'),value='E:\\codes\\py39\\test-20230416b\\todo-songs\\todo-songs');wav_inputs=gr.File(file_count='multiple',label=i18n(_AM))
					with gr.Column():model_choose=gr.Dropdown(label=i18n('模型'),choices=uvr5_names);agg=gr.Slider(minimum=0,maximum=20,step=1,label='人声提取激进程度',value=10,interactive=_A,visible=_B);opt_vocal_root=gr.Textbox(label=i18n('指定输出主人声文件夹'),value=_u);opt_ins_root=gr.Textbox(label=i18n('指定输出非主人声文件夹'),value=_u);format0=gr.Radio(label=i18n('导出文件格式'),choices=[_e,_U,_l,_m],value=_U,interactive=_A)
					but2=gr.Button(i18n('转换'),variant=_R);vc_output4=gr.Textbox(label=i18n(_b));but2.click(uvr,[model_choose,dir_wav_input,opt_vocal_root,wav_inputs,opt_ins_root,agg,format0],[vc_output4],api_name='uvr_convert')
		with gr.TabItem(i18n('训练')):
			with gr.Accordion(i18n('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. ')):
				with gr.Row():exp_dir1=gr.Textbox(label=i18n('输入实验名'),value='mi-test');sr2=gr.Radio(label=i18n('目标采样率'),choices=[_X,_g],value=_X,interactive=_A);if_f0_3=gr.Radio(label=i18n('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices=[_A,_B],value=_A,interactive=_A);version19=gr.Radio(label=i18n('版本'),choices=[_T],value=_T,interactive=_A,visible=_A);np7=gr.Slider(minimum=0,maximum=config.n_cpu,step=1,label=i18n('提取音高和处理数据使用的CPU进程数'),value=int(np.ceil(config.n_cpu/1.5)),interactive=_A)
			with gr.Accordion(i18n('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. ')):
				with gr.Row():trainset_dir4=gr.Textbox(label=i18n('输入训练文件夹路径'),value='/kaggle/working/dataset');spk_id5=gr.Slider(minimum=0,maximum=4,step=1,label=i18n('请指定说话人id'),value=0,interactive=_A);but1=gr.Button(i18n('处理数据'),variant=_R);info1=gr.Textbox(label=i18n(_b),value='');but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2,np7],[info1],api_name='train_preprocess')
			with gr.Accordion(i18n('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)')):
				with gr.Row():
					with gr.Column():gpus6=gr.Textbox(label=i18n(_AN),value=gpus,interactive=_A);gpu_info9=gr.Textbox(label=i18n('显卡信息'),value=gpu_info)
					with gr.Column():f0method8=gr.Radio(label=i18n('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢'),choices=[_j,_t,'dio',_k,_d],value=_d,interactive=_A);gpus_rmvpe=gr.Textbox(label=i18n('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程'),value='%s-%s'%(gpus,gpus),interactive=_A,visible=_A)
					but2=gr.Button(i18n('特征提取'),variant=_R);info2=gr.Textbox(label=i18n(_b),value='',max_lines=8);f0method8.change(fn=change_f0_method,inputs=[f0method8],outputs=[gpus_rmvpe]);but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1,version19,gpus_rmvpe],[info2],api_name='train_extract_f0_feature')
			with gr.Accordion(i18n('step3: 填写训练设置, 开始训练模型和索引')):
				with gr.Row():save_epoch10=gr.Slider(minimum=0,maximum=100,step=1,label=i18n('保存频率save_every_epoch'),value=5,interactive=_A);total_epoch11=gr.Slider(minimum=0,maximum=1000,step=1,label=i18n('总训练轮数total_epoch'),value=300,interactive=_A);batch_size12=gr.Slider(minimum=1,maximum=40,step=1,label=i18n('每张显卡的batch_size'),value=default_batch_size,interactive=_A);if_save_latest13=gr.Radio(label=i18n('是否仅保存最新的ckpt文件以节省硬盘空间'),choices=[i18n(_G),i18n('否')],value=i18n(_G),interactive=_A);if_cache_gpu17=gr.Radio(label=i18n('是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices=[i18n(_G),i18n('否')],value=i18n('否'),interactive=_A);if_save_every_weights18=gr.Radio(label=i18n('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices=[i18n(_G),i18n('否')],value=i18n(_G),interactive=_A);file_dict={f:os.path.join(_AO,f)for f in os.listdir(_AO)};file_dict={k:v for(k,v)in file_dict.items()if k.endswith(_V)};file_dict_g={k:v for(k,v)in file_dict.items()if'G'in k and _S in k};file_dict_d={k:v for(k,v)in file_dict.items()if'D'in k and _S in k}
				with gr.Row():pretrained_G14=gr.Dropdown(label=i18n('加载预训练底模G路径'),choices=list(file_dict_g.values()),value=file_dict_g['f0G32k.pth'],interactive=_A);pretrained_D15=gr.Dropdown(label=i18n('加载预训练底模D路径'),choices=list(file_dict_d.values()),value=file_dict_d['f0D32k.pth'],interactive=_A);sr2.change(change_sr2,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15]);version19.change(change_version19,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15,sr2]);if_f0_3.change(change_f0,[if_f0_3,sr2,version19],[f0method8,pretrained_G14,pretrained_D15]);gpus16=gr.Textbox(label=i18n(_AN),value=gpus,interactive=_A);but3=gr.Button(i18n('训练模型'),variant=_R);but4=gr.Button(i18n('训练特征索引'),variant=_R);info3=gr.Textbox(label=i18n(_b),value='',max_lines=10);but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19],info3,api_name='train_start');but4.click(train_index,[exp_dir1,version19],info3)
	if config.iscolab:app.queue(concurrency_count=511,max_size=1022).launch(server_port=config.listen_port,share=_B)
	else:app.queue(concurrency_count=511,max_size=1022).launch(server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_B,share=_B)