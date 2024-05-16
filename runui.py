_AA='Output information:'
_A9='Name of Zip file:'
_A8='Enter HuggingFace Model-Repo name:'
_A7='Enter HuggingFace Username:'
_A6='Enter HuggingFace Write Token:'
_A5='/kaggle/input/ax-rmd/pretrained_v2'
_A4='以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2'
_A3='也可批量输入音频文件, 二选一, 优先读文件夹'
_A2='特征检索库文件路径,为空则使用下拉的选择结果'
_A1='>=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音'
_A0='保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果'
_z='输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络'
_y='后处理重采样至最终采样率，0为不进行重采样'
_x='选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU'
_w='自动检测index路径,下拉式选择(dropdown)'
_v='变调(整数, 半音数量, 升八度12降八度-12)'
_u='Model Log Folder'
_t='/root/hugupload'
_s='audios'
_r='assets'
_q='训练结束, 您可查看控制台训练日志或实验文件夹下的train.log'
_p='%s/3_feature768'
_o='%s/3_feature256'
_n='trained'
_m='.index'
_l='assets/weights'
_k='模型路径'
_j='目标采样率'
_i='opt'
_h='/kaggle/working/AX-RVC/'
_g='Model Only'
_f='visible'
_e='_v2'
_d='%s/logs/%s'
_c='value'
_b='m4a'
_a='mp3'
_Z='datasets'
_Y='rmvpe_gpu'
_X='32k'
_W='flac'
_V='wav'
_U='harvest'
_T='pm'
_S='f0'
_R=None
_Q='.pth'
_P='Execute: '
_O='48k'
_N='rmvpe'
_M='v2'
_L='.'
_K='输出信息'
_J='r'
_I='40k'
_H='是'
_G='update'
_F='__type__'
_E='\n'
_D='v1'
_C='primary'
_B=False
_A=True
import os,sys
from dotenv import load_dotenv
now_dir=os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from infer.modules.uvr5.modules import uvr
from infer.lib.train.process_ckpt import change_info,extract_small_model,merge,show_info
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch,platform,numpy as np,gradio as gr,faiss,fairseq,pathlib,json
from time import sleep
from subprocess import Popen
import subprocess
from random import shuffle
import warnings,traceback,threading,shutil,logging,datetime,tabs.resources as resources
from huggingface_hub import HfApi
from huggingface_hub import login
from huggingface_hub import hf_hub_download
import time,zipfile,requests
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logger=logging.getLogger(__name__)
tmp=os.path.join(now_dir,'TEMP')
shutil.rmtree(tmp,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/infer_pack'%now_dir,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/uvr5_pack'%now_dir,ignore_errors=_A)
os.makedirs(tmp,exist_ok=_A)
os.makedirs(os.path.join(now_dir,'logs'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,_l),exist_ok=_A)
os.environ['TEMP']=tmp
warnings.filterwarnings('ignore')
torch.manual_seed(114514)
config=Config()
vc=VC(config)
if config.dml==_A:
	def forward_dml(ctx,x,scale):ctx.scale=scale;res=x.clone().detach();return res
	fairseq.modules.grad_multiply.GradMultiply.forward=forward_dml
i18n=I18nAuto()
logger.info(i18n)
ngpu=torch.cuda.device_count()
gpu_infos=[]
mem=[]
if_gpu_ok=_B
if torch.cuda.is_available()or ngpu!=0:
	for i in range(ngpu):
		gpu_name=torch.cuda.get_device_name(i)
		if any(value in gpu_name.upper()for value in['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN','4060','L','6000']):if_gpu_ok=_A;gpu_infos.append('%s\t%s'%(i,gpu_name));mem.append(int(torch.cuda.get_device_properties(i).total_memory/1024/1024/1024+.4))
if if_gpu_ok and len(gpu_infos)>0:gpu_info=_E.join(gpu_infos);default_batch_size=min(mem)//2
else:gpu_info=i18n('很遗憾您这没有能用的显卡来支持您训练');default_batch_size=1
gpus='-'.join([i[0]for i in gpu_infos])
weight_root=_l
weight_uvr5_root=os.getenv('weight_uvr5_root')
index_root='logs'
audio_root='assets/audios'
outside_index_root=os.getenv('outside_index_root')
datasets_root=_Z
sup_audioext={_V,_a,_W,'ogg','opus',_b,'mp4','aac','alac','wma','aiff','webm','ac3'}
names=[os.path.relpath(os.path.join(root,file),weight_root)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_Q,'.onnx'))]
index_paths=[]
audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root]
datasets=[]
for foldername in os.listdir(os.path.join(now_dir,datasets_root)):
	if _L not in foldername:datasets.append(os.path.join(now_dir,_Z,foldername))
def lookup_indices(index_root):
	global index_paths
	for(root,dirs,files)in os.walk(index_root,topdown=_B):
		for name in files:
			if name.endswith(_m)and _n not in name:index_paths.append('%s/%s'%(root,name))
lookup_indices(index_root)
lookup_indices(outside_index_root)
uvr5_names=[]
for name in os.listdir(weight_uvr5_root):
	if name.endswith(_Q)or'onnx'in name:uvr5_names.append(name.replace(_Q,''))
def change_choices():names=[os.path.relpath(os.path.join(root,file),weight_root)for(root,_,files)in os.walk(weight_root)for file in files if file.endswith((_Q,'.onnx'))];indexes_list=[os.path.join(root,name)for(root,_,files)in os.walk(index_root,topdown=_B)for name in files if name.endswith(_m)and _n not in name];audio_paths=[os.path.join(root,name)for(root,_,files)in os.walk(audio_root,topdown=_B)for name in files if name.endswith(tuple(sup_audioext))and root==audio_root];return gr.Dropdown(choices=sorted(names)),gr.Dropdown(choices=sorted(indexes_list)),gr.Dropdown(choices=sorted(audio_paths))
def clean():return{_c:'',_F:_G}
def export_onnx(ModelPath,ExportedPath):from infer.modules.onnx.export import export_onnx as eo;eo(ModelPath,ExportedPath)
sr_dict={_X:32000,_I:40000,_O:48000}
def if_done(done,p):
	while 1:
		if p.poll()is _R:sleep(.5)
		else:break
	done[0]=_A
def if_done_multi(done,ps):
	while 1:
		flag=1
		for p in ps:
			if p.poll()is _R:flag=0;sleep(.5);break
		if flag==1:break
	done[0]=_A
def preprocess_dataset(trainset_dir,exp_dir,sr,n_p,dataset_path):
	A='%s/logs/%s/preprocess.log'
	if not dataset_path.strip()=='':trainset_dir=dataset_path
	sr=sr_dict[sr];os.makedirs(_d%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close();cmd='"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f'%(config.python_cmd,trainset_dir,sr,n_p,now_dir,exp_dir,config.noparallel,config.preprocess_per);logger.info(_P+cmd);p=Popen(cmd,shell=_A);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_J)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_J)as f:log=f.read()
	logger.info(log);yield log
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir,version19,gpus_rmvpe):
	A='%s/logs/%s/extract_f0_feature.log';gpus=gpus.split('-');os.makedirs(_d%(now_dir,exp_dir),exist_ok=_A);f=open(A%(now_dir,exp_dir),'w');f.close()
	if if_f0:
		if f0method!=_Y:cmd='"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'%(config.python_cmd,now_dir,exp_dir,n_p,f0method);logger.info(_P+cmd);p=Popen(cmd,shell=_A,cwd=now_dir);done=[_B];threading.Thread(target=if_done,args=(done,p)).start()
		elif gpus_rmvpe!='-':
			gpus_rmvpe=gpus_rmvpe.split('-');leng=len(gpus_rmvpe);ps=[]
			for(idx,n_g)in enumerate(gpus_rmvpe):cmd='"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '%(config.python_cmd,leng,idx,n_g,now_dir,exp_dir,config.is_half);logger.info(_P+cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
			done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
		else:cmd=config.python_cmd+' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '%(now_dir,exp_dir);logger.info(_P+cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();done=[_A]
		while 1:
			with open(A%(now_dir,exp_dir),_J)as f:yield f.read()
			sleep(1)
			if done[0]:break
		with open(A%(now_dir,exp_dir),_J)as f:log=f.read()
		logger.info(log);yield log
	'\n    n_part=int(sys.argv[1])\n    i_part=int(sys.argv[2])\n    i_gpu=sys.argv[3]\n    exp_dir=sys.argv[4]\n    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)\n    ';leng=len(gpus);ps=[]
	for(idx,n_g)in enumerate(gpus):cmd='"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s %s'%(config.python_cmd,config.device,leng,idx,n_g,now_dir,exp_dir,version19,config.is_half);logger.info(_P+cmd);p=Popen(cmd,shell=_A,cwd=now_dir);ps.append(p)
	done=[_B];threading.Thread(target=if_done_multi,args=(done,ps)).start()
	while 1:
		with open(A%(now_dir,exp_dir),_J)as f:yield f.read()
		sleep(1)
		if done[0]:break
	with open(A%(now_dir,exp_dir),_J)as f:log=f.read()
	logger.info(log);yield log
def get_pretrained_models(path_str,f0_str,sr2):
	B='/kaggle/input/ax-rmd/pretrained%s/%sD%s.pth';A='/kaggle/input/ax-rmd/pretrained%s/%sG%s.pth';if_pretrained_generator_exist=os.access(A%(path_str,f0_str,sr2),os.F_OK);if_pretrained_discriminator_exist=os.access(B%(path_str,f0_str,sr2),os.F_OK)
	if not if_pretrained_generator_exist:logger.warning('/kaggle/input/ax-rmd/pretrained%s/%sG%s.pth not exist, will not use pretrained model',path_str,f0_str,sr2)
	if not if_pretrained_discriminator_exist:logger.warning('/kaggle/input/ax-rmd/pretrained%s/%sD%s.pth not exist, will not use pretrained model',path_str,f0_str,sr2)
	return A%(path_str,f0_str,sr2)if if_pretrained_generator_exist else'',B%(path_str,f0_str,sr2)if if_pretrained_discriminator_exist else''
def change_sr2(sr2,if_f0_3,version19):path_str=''if version19==_D else _e;f0_str=_S if if_f0_3 else'';return get_pretrained_models(path_str,f0_str,sr2)
def change_version19(sr2,if_f0_3,version19):
	A='choices';path_str=''if version19==_D else _e
	if sr2==_X and version19==_D:sr2=_I
	to_return_sr2={A:[_I,_O],_F:_G,_c:sr2}if version19==_D else{A:[_I,_O,_X],_F:_G,_c:sr2};f0_str=_S if if_f0_3 else'';return*get_pretrained_models(path_str,f0_str,sr2),to_return_sr2
def change_f0(if_f0_3,sr2,version19):path_str=''if version19==_D else _e;return{_f:if_f0_3,_F:_G},{_f:if_f0_3,_F:_G},*get_pretrained_models(path_str,_S if if_f0_3==_A else'',sr2)
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19):
	D='-pd %s';C='-pg %s';B='\\\\';A='\\';exp_dir=_d%(now_dir,exp_dir1);os.makedirs(exp_dir,exist_ok=_A);gt_wavs_dir='%s/0_gt_wavs'%exp_dir;feature_dir=_o%exp_dir if version19==_D else _p%exp_dir
	if if_f0_3:f0_dir='%s/2a_f0'%exp_dir;f0nsf_dir='%s/2b-f0nsf'%exp_dir;names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])&set([name.split(_L)[0]for name in os.listdir(f0_dir)])&set([name.split(_L)[0]for name in os.listdir(f0nsf_dir)])
	else:names=set([name.split(_L)[0]for name in os.listdir(gt_wavs_dir)])&set([name.split(_L)[0]for name in os.listdir(feature_dir)])
	opt=[]
	for name in names:
		if if_f0_3:opt.append('%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'%(gt_wavs_dir.replace(A,B),name,feature_dir.replace(A,B),name,f0_dir.replace(A,B),name,f0nsf_dir.replace(A,B),name,spk_id5))
		else:opt.append('%s/%s.wav|%s/%s.npy|%s'%(gt_wavs_dir.replace(A,B),name,feature_dir.replace(A,B),name,spk_id5))
	fea_dim=256 if version19==_D else 768
	if if_f0_3:
		for _ in range(2):opt.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'%(now_dir,sr2,now_dir,fea_dim,now_dir,now_dir,spk_id5))
	else:
		for _ in range(2):opt.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'%(now_dir,sr2,now_dir,fea_dim,spk_id5))
	shuffle(opt)
	with open('%s/filelist.txt'%exp_dir,'w')as f:f.write(_E.join(opt))
	logger.debug('Write filelist done');logger.info('Use gpus: %s',str(gpus16))
	if pretrained_G14=='':logger.info('No pretrained Generator')
	if pretrained_D15=='':logger.info('No pretrained Discriminator')
	if version19==_D or sr2==_I:config_path='v1/%s.json'%sr2
	else:config_path='v2/%s.json'%sr2
	config_save_path=os.path.join(exp_dir,'config.json')
	if not pathlib.Path(config_save_path).exists():
		with open(config_save_path,'w',encoding='utf-8')as f:json.dump(config.json_config[config_path],f,ensure_ascii=_B,indent=4,sort_keys=_A);f.write(_E)
	if gpus16:cmd='"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,gpus16,total_epoch11,save_epoch10,C%pretrained_G14 if pretrained_G14!=''else'',D%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==i18n(_H)else 0,1 if if_cache_gpu17==i18n(_H)else 0,1 if if_save_every_weights18==i18n(_H)else 0,version19)
	else:cmd='"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,exp_dir1,sr2,1 if if_f0_3 else 0,batch_size12,total_epoch11,save_epoch10,C%pretrained_G14 if pretrained_G14!=''else'',D%pretrained_D15 if pretrained_D15!=''else'',1 if if_save_latest13==i18n(_H)else 0,1 if if_cache_gpu17==i18n(_H)else 0,1 if if_save_every_weights18==i18n(_H)else 0,version19)
	logger.info(_P+cmd);p=Popen(cmd,shell=_A,cwd=now_dir);p.wait();return _q
def train_index(exp_dir1,version19):
	A='%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index';exp_dir='logs/%s'%exp_dir1;os.makedirs(exp_dir,exist_ok=_A);feature_dir=_o%exp_dir if version19==_D else _p%exp_dir
	if not os.path.exists(feature_dir):return'请先进行特征提取!'
	listdir_res=list(os.listdir(feature_dir))
	if len(listdir_res)==0:return'请先进行特征提取！'
	infos=[];npys=[]
	for name in sorted(listdir_res):phone=np.load('%s/%s'%(feature_dir,name));npys.append(phone)
	big_npy=np.concatenate(npys,0);big_npy_idx=np.arange(big_npy.shape[0]);np.random.shuffle(big_npy_idx);big_npy=big_npy[big_npy_idx]
	if big_npy.shape[0]>2e5:
		infos.append('Trying doing kmeans %s shape to 10k centers.'%big_npy.shape[0]);yield _E.join(infos)
		try:big_npy=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_B,init='random').fit(big_npy).cluster_centers_
		except:info=traceback.format_exc();logger.info(info);infos.append(info);yield _E.join(infos)
	np.save('%s/total_fea.npy'%exp_dir,big_npy);n_ivf=min(int(16*np.sqrt(big_npy.shape[0])),big_npy.shape[0]//39);infos.append('%s,%s'%(big_npy.shape,n_ivf));yield _E.join(infos);index=faiss.index_factory(256 if version19==_D else 768,'IVF%s,Flat'%n_ivf);infos.append('training');yield _E.join(infos);index_ivf=faiss.extract_index_ivf(index);index_ivf.nprobe=1;index.train(big_npy);faiss.write_index(index,'%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('adding');yield _E.join(infos);batch_size_add=8192
	for i in range(0,big_npy.shape[0],batch_size_add):index.add(big_npy[i:i+batch_size_add])
	faiss.write_index(index,A%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('成功构建索引 added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(n_ivf,index_ivf.nprobe,exp_dir1,version19))
	try:link=os.link if platform.system()=='Windows'else os.symlink;link(A%(exp_dir,n_ivf,index_ivf.nprobe,exp_dir1,version19),'%s/%s_IVF%s_Flat_nprobe_%s_%s_%s.index'%(outside_index_root,exp_dir1,n_ivf,index_ivf.nprobe,exp_dir1,version19));infos.append('链接索引到外部-%s'%outside_index_root)
	except:infos.append('链接索引到外部-%s失败'%outside_index_root)
	yield _E.join(infos)
def train1key(exp_dir1,sr2,if_f0_3,trainset_dir4,spk_id5,np7,f0method8,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19,gpus_rmvpe):
	infos=[]
	def get_info_str(strr):infos.append(strr);return _E.join(infos)
	yield get_info_str(i18n('step1:正在处理数据'));[get_info_str(_)for _ in preprocess_dataset(trainset_dir4,exp_dir1,sr2,np7)];yield get_info_str(i18n('step2:正在提取音高&正在提取特征'));[get_info_str(_)for _ in extract_f0_feature(gpus16,np7,f0method8,if_f0_3,exp_dir1,version19,gpus_rmvpe)];yield get_info_str(i18n('step3a:正在训练模型'));click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19);yield get_info_str(i18n(_q));[get_info_str(_)for _ in train_index(exp_dir1,version19)];yield get_info_str(i18n('全流程结束！'))
def change_info_(ckpt_path):
	B='version';A='train.log'
	if not os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path),A)):return{_F:_G},{_F:_G},{_F:_G}
	try:
		with open(ckpt_path.replace(os.path.basename(ckpt_path),A),_J)as f:info=eval(f.read().strip(_E).split(_E)[0].split('\t')[-1]);sr,f0=info['sample_rate'],info['if_f0'];version=_M if B in info and info[B]==_M else _D;return sr,str(f0),version
	except:traceback.print_exc();return{_F:_G},{_F:_G},{_F:_G}
F0GPUVisible=config.dml==_B
def change_f0_method(f0method8):
	if f0method8==_Y:visible=F0GPUVisible
	else:visible=_B
	return{_f:visible,_F:_G}
def save_to_wav(record_button):
	if record_button is _R:0
	else:path_to_file=record_button;new_name=datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+'.wav';target_path=os.path.join(_r,_s,os.path.basename(new_name));shutil.move(path_to_file,target_path);return target_path
def save_to_wav2(dropbox):
	file_path=dropbox.name;target_path=os.path.join(_r,_s,os.path.basename(file_path))
	if os.path.exists(target_path):os.remove(target_path);print('Replacing old dropdown file...')
	shutil.move(file_path,target_path);return target_path
def get_dataset():
	if len(datasets)>0:return sorted(datasets)[0]
	else:return''
def update_dataset_list(name):
	new_datasets=[]
	for foldername in os.listdir(os.path.join(now_dir,datasets_root)):
		if _L not in foldername:new_datasets.append(os.path.join(now_dir,_Z,foldername))
	return gr.Dropdown(choices=new_datasets)
def start_upload_to_huggingface(hgf_token_gr,hgf_name_gr,hgf_repo_gr,model_name_gr,zip_name_gr,what_upload_gr):
	A='model';login(token=hgf_token_gr,add_to_git_credential=_A,new_session=_A);hug_file_path=_t;hug_file_name=f"{zip_name_gr}.zip";os.system(f"rm -rf {hug_file_path}")
	if what_upload_gr==_g:
		if not os.path.exists(hug_file_path):os.makedirs(hug_file_path)
		os.system(f"cp /kaggle/working/AX-RVC/assets/weights/{model_name_gr}.pth {hug_file_path}");os.system(f"cp /kaggle/working/AX-RVC/logs/{model_name_gr}/added*.index {hug_file_path}");time.sleep(2);os.chdir(hug_file_path);os.system(f"zip -r {hug_file_path}/{hug_file_name} {model_name_gr}.pth added*.index");os.chdir(_h);api=HfApi(token=hgf_token_gr);api.upload_file(path_or_fileobj=f"{hug_file_path}/{hug_file_name}",path_in_repo=hug_file_name,repo_id=f"{hgf_name_gr}/{hgf_repo_gr}",repo_type=A);os.system(f"rm -rf /root/hugupload/{hug_file_name}");os.system(f"rm -rf /root/hugupload/{model_name_gr}.pth");os.system(f"rm -rf /root/hugupload/added*.index");return'Succesful upload Model to Hugging Face'
	if what_upload_gr==_u:
		if not os.path.exists(hug_file_path):os.makedirs(hug_file_path)
		hug_file_name=f"{zip_name_gr}_logs.zip";os.system(f"cp -r /kaggle/working/AX-RVC/logs/{model_name_gr} {hug_file_path}");time.sleep(2);os.chdir(hug_file_path);os.system(f"zip -r {hug_file_path}/{hug_file_name} {model_name_gr}");os.chdir(_h);api=HfApi(token=hgf_token_gr);api.upload_file(path_or_fileobj=f"{hug_file_path}/{hug_file_name}",path_in_repo=hug_file_name,repo_id=f"{hgf_name_gr}/{hgf_repo_gr}",repo_type=A);time.sleep(2);os.system(f"rm -rf /root/hugupload/{hug_file_name}");os.system(f"rm -rf /root/hugupload/{model_name_gr}");return'Succesful upload Logs to Hugging Face'
def start_download_from_huggingface(hgf_token_gr_d,hgf_name_gr_d,hgf_repo_gr_d,zip_name_gr_d):
	hug_file_path=_t;hug_file_name=f"{zip_name_gr_d}.zip";hug_repo_id=f"{hgf_name_gr_d}/{hgf_repo_gr_d}";destination_folder='/kaggle/working/AX-RVC/logs'
	if not os.path.exists(hug_file_path):os.makedirs(hug_file_path)
	os.chdir(hug_file_path);hf_hub_download(repo_id=hug_repo_id,filename=hug_file_name,token=hgf_token_gr_d,local_dir=hug_file_path);time.sleep(2)
	with zipfile.ZipFile(f"{hug_file_path}/{hug_file_name}",_J)as zip_ref:zip_ref.extractall(destination_folder)
	os.chdir(_h);time.sleep(2);os.system(f"rm -rf /root/hugupload/{hug_file_name}");return'Succesful download Logs from Hugging Face'
def show_restart_screen():app.close();gr.HTML('\n        <div style="\n            position: fixed;\n            top: 0;\n            left: 0;\n            width: 100%;\n            height: 100%;\n            background-color: black;\n            color: white;\n            display: flex;\n            justify-content: center;\n            align-items: center;\n            font-size: 48px;\n        ">\n            RESTART\n        </div>\n        ')
def update_restart_app():subprocess.run(['git','pull']);show_restart_screen();os.execl(sys.executable,sys.executable,*sys.argv)
with gr.Blocks(title='💙 AX-RVC WebUI 💎',theme=gr.themes.Base(primary_hue='sky',neutral_hue='zinc'))as app:
	gr.Markdown('## 💙 AX-RVC WebUI')
	with gr.Tabs():
		with gr.TabItem(i18n('模型推理')):
			with gr.Row():
				sid0=gr.Dropdown(label=i18n('推理音色'),choices=sorted(names))
				with gr.Column():refresh_button=gr.Button(i18n('刷新音色列表和索引路径'),variant=_C);clean_button=gr.Button(i18n('卸载音色省显存'),variant=_C)
				spk_item=gr.Slider(minimum=0,maximum=2333,step=1,label=i18n('请选择说话人id'),value=0,visible=_B,interactive=_A);clean_button.click(fn=clean,inputs=[],outputs=[sid0],api_name='infer_clean')
			with gr.TabItem(i18n('单次推理')):
				with gr.Row():
					with gr.Column():dropbox=gr.File(label=i18n('Drag your audio here:'));record_button=gr.Audio(sources='microphone',label=i18n('Or record an audio:'),type='filepath')
					with gr.Column():vc_transform0=gr.Number(label=i18n(_v),value=0);input_audio0=gr.Dropdown(label=i18n('输入待处理音频文件路径(默认是正确格式示例)'),choices=sorted(audio_paths),value='',interactive=_A);file_index2=gr.Dropdown(label=i18n(_w),choices=sorted(index_paths),interactive=_A);dropbox.upload(fn=save_to_wav2,inputs=[dropbox],outputs=[input_audio0],show_progress='full');record_button.change(fn=save_to_wav,inputs=[record_button],outputs=[input_audio0]);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2,input_audio0],api_name='infer_refresh')
				with gr.Accordion(i18n('Advanced Settings'),open=_B):
					with gr.Row():
						with gr.Column():f0method0=gr.Radio(label=i18n(_x),choices=[_T,_U,'crepe',_N]if config.dml==_B else[_T,_U,_N],value=_N,interactive=_A);resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(_y),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(_z),value=.25,interactive=_A)
						with gr.Column():protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(_A0),value=.33,step=.01,interactive=_A);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(_A1),value=3,step=1,interactive=_A);index_rate1=gr.Slider(minimum=0,maximum=1,label=i18n('检索特征占比'),value=.75,interactive=_A);f0_file=gr.File(label=i18n('F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调'),visible=_B);file_index1=gr.Textbox(label=i18n(_A2),placeholder='C:\\Users\\Desktop\\model_example.index',interactive=_A,visible=_B)
				but0=gr.Button(i18n('转换'),variant=_C)
				with gr.Group():
					with gr.Column():
						with gr.Row():vc_output1=gr.Textbox(label=i18n(_K));vc_output2=gr.Audio(label=i18n('输出音频(右下角三个点,点了可以下载)'))
						but0.click(vc.vc_single,[spk_item,input_audio0,vc_transform0,f0_file,f0method0,file_index1,file_index2,index_rate1,filter_radius0,resample_sr0,rms_mix_rate0,protect0],[vc_output1,vc_output2],api_name='infer_convert')
			with gr.TabItem(i18n('批量推理')):
				gr.Markdown(value=i18n('批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. '))
				with gr.Row():
					with gr.Column():vc_transform1=gr.Number(label=i18n(_v),value=0);opt_input=gr.Textbox(label=i18n('指定输出文件夹'),value=_i);file_index3=gr.Textbox(label=i18n(_A2),value='',interactive=_A);file_index4=gr.Dropdown(label=i18n(_w),choices=sorted(index_paths),interactive=_A);f0method1=gr.Radio(label=i18n(_x),choices=[_T,_U,'crepe',_N]if config.dml==_B else[_T,_U,_N],value=_N,interactive=_A);format1=gr.Radio(label=i18n('导出文件格式'),choices=[_V,_W,_a,_b],value=_V,interactive=_A);refresh_button.click(fn=lambda:change_choices()[1],inputs=[],outputs=file_index4,api_name='infer_refresh_batch')
					with gr.Column():resample_sr1=gr.Slider(minimum=0,maximum=48000,label=i18n(_y),value=0,step=1,interactive=_A);rms_mix_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_z),value=1,interactive=_A);protect1=gr.Slider(minimum=0,maximum=.5,label=i18n(_A0),value=.33,step=.01,interactive=_A);filter_radius1=gr.Slider(minimum=0,maximum=7,label=i18n(_A1),value=3,step=1,interactive=_A);index_rate2=gr.Slider(minimum=0,maximum=1,label=i18n('检索特征占比'),value=1,interactive=_A)
				with gr.Row():dir_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)'),placeholder='C:\\Users\\Desktop\\input_vocal_dir');inputs=gr.File(file_count='multiple',label=i18n(_A3))
				with gr.Row():but1=gr.Button(i18n('转换'),variant=_C);vc_output3=gr.Textbox(label=i18n(_K));but1.click(vc.vc_multi,[spk_item,dir_input,opt_input,inputs,vc_transform1,f0method1,file_index3,file_index4,index_rate2,filter_radius1,resample_sr1,rms_mix_rate1,protect1,format1],[vc_output3],api_name='infer_convert_batch')
				sid0.change(fn=vc.get_vc,inputs=[sid0,protect0,protect1],outputs=[spk_item,protect0,protect1,file_index2,file_index4],api_name='infer_change_voice')
		with gr.TabItem(i18n('伴奏人声分离&去混响&去回声')):
			with gr.Group():
				gr.Markdown(value=i18n('人声伴奏分离批量处理， 使用UVR5模型。 <br>合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。 <br>模型分为三类： <br>1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点； <br>2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型； <br> 3、去混响、去延迟模型（by FoxJoy）：<br>\u2003\u2003(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；<br>&emsp;(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。<br>去混响/去延迟，附：<br>1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；<br>2、MDX-Net-Dereverb模型挺慢的；<br>3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。'))
				with gr.Row():
					with gr.Column():dir_wav_input=gr.Textbox(label=i18n('输入待处理音频文件夹路径'),placeholder='C:\\Users\\Desktop\\todo-songs');wav_inputs=gr.File(file_count='multiple',label=i18n(_A3))
					with gr.Column():model_choose=gr.Dropdown(label=i18n('模型'),choices=uvr5_names);agg=gr.Slider(minimum=0,maximum=20,step=1,label='人声提取激进程度',value=10,interactive=_A,visible=_B);opt_vocal_root=gr.Textbox(label=i18n('指定输出主人声文件夹'),value=_i);opt_ins_root=gr.Textbox(label=i18n('指定输出非主人声文件夹'),value=_i);format0=gr.Radio(label=i18n('导出文件格式'),choices=[_V,_W,_a,_b],value=_W,interactive=_A)
					but2=gr.Button(i18n('转换'),variant=_C);vc_output4=gr.Textbox(label=i18n(_K));but2.click(uvr,[model_choose,dir_wav_input,opt_vocal_root,wav_inputs,opt_ins_root,agg,format0],[vc_output4],api_name='uvr_convert')
		with gr.TabItem(i18n('训练')):
			with gr.Accordion(i18n('step1: 填写实验配置. 实验数据放在logs下, 每个实验一个文件夹, 需手工输入实验名路径, 内含实验配置, 日志, 训练得到的模型文件. ')):
				with gr.Row():exp_dir1=gr.Textbox(label=i18n('输入实验名'),value='mi-test');sr2=gr.Radio(label=i18n(_j),choices=[_I,_O],value=_I,interactive=_A);if_f0_3=gr.Radio(label=i18n('模型是否带音高指导(唱歌一定要, 语音可以不要)'),choices=[_A,_B],value=_A,interactive=_A);version19=gr.Radio(label=i18n('版本'),choices=[_M],value=_M,interactive=_A,visible=_A);np7=gr.Slider(minimum=0,maximum=config.n_cpu,step=1,label=i18n('提取音高和处理数据使用的CPU进程数'),value=int(np.ceil(config.n_cpu/1.5)),interactive=_A)
			with gr.Accordion(i18n('step2a: 自动遍历训练文件夹下所有可解码成音频的文件并进行切片归一化, 在实验目录下生成2个wav文件夹; 暂时只支持单人训练. ')):
				with gr.Row():
					with gr.Column():trainset_dir4=gr.Dropdown(choices=sorted(datasets),label=i18n('Select your dataset:'),value=get_dataset());dataset_path=gr.Textbox(label=i18n('输入训练文件夹路径'),placeholder=i18n('E:\\语音音频+标注\\米津玄师\\src'),value='');btn_update_dataset_list=gr.Button(i18n('Update list'),variant=_C)
					spk_id5=gr.Slider(minimum=0,maximum=4,step=1,label=i18n('请指定说话人id'),value=0,interactive=_A);but1=gr.Button(i18n('处理数据'),variant=_C);info1=gr.Textbox(label=i18n(_K),value='');btn_update_dataset_list.click(resources.update_dataset_list,[spk_id5],trainset_dir4);but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2,np7,dataset_path],[info1],api_name='train_preprocess')
			with gr.Accordion(i18n('step2b: 使用CPU提取音高(如果模型带音高), 使用GPU提取特征(选择卡号)')):
				with gr.Row():
					with gr.Column():gpus6=gr.Textbox(label=i18n(_A4),value=gpus,interactive=_A,visible=F0GPUVisible);gpu_info9=gr.Textbox(label=i18n('显卡信息'),value=gpu_info,visible=F0GPUVisible)
					with gr.Column():f0method8=gr.Radio(label=i18n('选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU'),choices=[_T,_U,'dio',_N,_Y],value=_Y,interactive=_A);gpus_rmvpe=gr.Textbox(label=i18n('rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程'),value='%s-%s'%(gpus,gpus),interactive=_A,visible=F0GPUVisible)
					but2=gr.Button(i18n('特征提取'),variant=_C);info2=gr.Textbox(label=i18n(_K),value='',max_lines=8);f0method8.change(fn=change_f0_method,inputs=[f0method8],outputs=[gpus_rmvpe]);but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1,version19,gpus_rmvpe],[info2],api_name='train_extract_f0_feature')
			with gr.Accordion(i18n('step3: 填写训练设置, 开始训练模型和索引')):
				with gr.Row():save_epoch10=gr.Slider(minimum=1,maximum=500,step=1,label=i18n('保存频率save_every_epoch'),value=25,interactive=_A);total_epoch11=gr.Slider(minimum=2,maximum=10000,step=1,label=i18n('总训练轮数total_epoch'),value=200,interactive=_A);batch_size12=gr.Slider(minimum=1,maximum=40,step=1,label=i18n('每张显卡的batch_size'),value=default_batch_size,interactive=_A);if_save_latest13=gr.Radio(label=i18n('是否仅保存最新的ckpt文件以节省硬盘空间'),choices=[i18n(_H),i18n('否')],value=i18n(_H),interactive=_A);if_cache_gpu17=gr.Radio(label=i18n('是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速'),choices=[i18n(_H),i18n('否')],value=i18n('否'),interactive=_A);if_save_every_weights18=gr.Radio(label=i18n('是否在每次保存时间点将最终小模型保存至weights文件夹'),choices=[i18n(_H),i18n('否')],value=i18n('否'),interactive=_A)
				with gr.Row():file_dict={f:os.path.join(_A5,f)for f in os.listdir(_A5)};file_dict={k:v for(k,v)in file_dict.items()if k.endswith(_Q)};file_dict_g={k:v for(k,v)in file_dict.items()if'G'in k and _S in k};file_dict_d={k:v for(k,v)in file_dict.items()if'D'in k and _S in k};pretrained_G14=gr.Dropdown(label=i18n('加载预训练底模G路径'),choices=list(file_dict_g.values()),value=file_dict_g['f0G32k.pth'],interactive=_A);pretrained_D15=gr.Dropdown(label=i18n('加载预训练底模D路径'),choices=list(file_dict_d.values()),value=file_dict_d['f0D32k.pth'],interactive=_A);sr2.change(change_sr2,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15]);version19.change(change_version19,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15,sr2]);if_f0_3.change(change_f0,[if_f0_3,sr2,version19],[f0method8,gpus_rmvpe,pretrained_G14,pretrained_D15]);gpus16=gr.Textbox(label=i18n(_A4),value=gpus,interactive=_A);but3=gr.Button(i18n('训练模型'),variant=_C);but4=gr.Button(i18n('训练特征索引'),variant=_C);info3=gr.Textbox(label=i18n(_K),value='',max_lines=10);but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19],info3,api_name='train_start');but4.click(train_index,[exp_dir1,version19],info3)
		with gr.TabItem('HuggingFace 🤗'):
			with gr.Accordion(label=i18n('Upload Model and Logs')):
				with gr.Row():
					with gr.Column():hgf_token_gr=gr.Textbox(label=_A6);hgf_name_gr=gr.Textbox(label=_A7);hgf_repo_gr=gr.Textbox(label=_A8)
					with gr.Column():model_name_gr=gr.Textbox(label='Trained model name:');zip_name_gr=gr.Textbox(label=_A9);what_upload_gr=gr.Radio(label='Upload files:',choices=[_g,_u],value=_g,interactive=_A,visible=_A)
				with gr.Row():uploadbut1=gr.Button('Start upload',variant=_C);uploadinfo1=gr.Textbox(label=_AA,value='');uploadbut1.click(start_upload_to_huggingface,[hgf_token_gr,hgf_name_gr,hgf_repo_gr,model_name_gr,zip_name_gr,what_upload_gr],[uploadinfo1])
			with gr.Accordion(label=i18n('Download Logs files for training')):
				with gr.Row():
					with gr.Column():hgf_token_gr_d=gr.Textbox(label=_A6);hgf_name_gr_d=gr.Textbox(label=_A7)
					with gr.Column():hgf_repo_gr_d=gr.Textbox(label=_A8);zip_name_gr_d=gr.Textbox(label=_A9)
				with gr.Row():downloadlogsbut1=gr.Button('Start download',variant=_C);downloadlogsinfo1=gr.Textbox(label=_AA,value='');downloadlogsbut1.click(start_download_from_huggingface,[hgf_token_gr_d,hgf_name_gr_d,hgf_repo_gr_d,zip_name_gr_d],[downloadlogsinfo1])
		with gr.TabItem(i18n('Resources')):resources.download_model();resources.download_backup();resources.download_dataset(trainset_dir4);resources.download_audio()
		with gr.TabItem(i18n('ckpt处理')):
			with gr.Group():
				gr.Markdown(value=i18n('模型融合, 可用于测试音色融合'))
				with gr.Row():ckpt_a=gr.Textbox(label=i18n('A模型路径'),value='',interactive=_A);ckpt_b=gr.Textbox(label=i18n('B模型路径'),value='',interactive=_A);alpha_a=gr.Slider(minimum=0,maximum=1,label=i18n('A模型权重'),value=.5,interactive=_A)
				with gr.Row():sr_=gr.Radio(label=i18n(_j),choices=[_I,_O],value=_I,interactive=_A);if_f0_=gr.Radio(label=i18n('模型是否带音高指导'),choices=[i18n(_H),i18n('否')],value=i18n(_H),interactive=_A);info__=gr.Textbox(label=i18n('要置入的模型信息'),value='',max_lines=8,interactive=_A);name_to_save0=gr.Textbox(label=i18n('保存的模型名不带后缀'),value='',max_lines=1,interactive=_A);version_2=gr.Radio(label=i18n('模型版本型号'),choices=[_D,_M],value=_D,interactive=_A)
				with gr.Row():but6=gr.Button(i18n('融合'),variant=_C);info4=gr.Textbox(label=i18n(_K),value='',max_lines=8)
				but6.click(merge,[ckpt_a,ckpt_b,alpha_a,sr_,if_f0_,info__,name_to_save0,version_2],info4,api_name='ckpt_merge')
			with gr.Group():
				gr.Markdown(value=i18n('修改模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr.Row():ckpt_path0=gr.Textbox(label=i18n(_k),value='',interactive=_A);info_=gr.Textbox(label=i18n('要改的模型信息'),value='',max_lines=8,interactive=_A);name_to_save1=gr.Textbox(label=i18n('保存的文件名, 默认空为和源文件同名'),value='',max_lines=8,interactive=_A)
				with gr.Row():but7=gr.Button(i18n('修改'),variant=_C);info5=gr.Textbox(label=i18n(_K),value='',max_lines=8)
				but7.click(change_info,[ckpt_path0,info_,name_to_save1],info5,api_name='ckpt_modify')
			with gr.Group():
				gr.Markdown(value=i18n('查看模型信息(仅支持weights文件夹下提取的小模型文件)'))
				with gr.Row():ckpt_path1=gr.Textbox(label=i18n(_k),value='',interactive=_A);but8=gr.Button(i18n('查看'),variant=_C);info6=gr.Textbox(label=i18n(_K),value='',max_lines=8)
				but8.click(show_info,[ckpt_path1],info6,api_name='ckpt_show')
			with gr.Group():
				gr.Markdown(value=i18n('模型提取(输入logs文件夹下大文件模型路径),适用于训一半不想训了模型没有自动提取保存小文件模型,或者想测试中间模型的情况'))
				with gr.Row():ckpt_path2=gr.Textbox(label=i18n(_k),value='E:\\codes\\py39\\logs\\mi-test_f0_48k\\G_23333.pth',interactive=_A);save_name=gr.Textbox(label=i18n('保存名'),value='',interactive=_A);sr__=gr.Radio(label=i18n(_j),choices=[_X,_I,_O],value=_I,interactive=_A);if_f0__=gr.Radio(label=i18n('模型是否带音高指导,1是0否'),choices=['1','0'],value='1',interactive=_A);version_1=gr.Radio(label=i18n('模型版本型号'),choices=[_M],value=_M,interactive=_A);info___=gr.Textbox(label=i18n('要置入的模型信息'),value='',max_lines=8,interactive=_A);but9=gr.Button(i18n('提取'),variant=_C);info7=gr.Textbox(label=i18n(_K),value='',max_lines=8);ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__,version_1])
				but9.click(extract_small_model,[ckpt_path2,save_name,sr__,if_f0__,info___,version_1],info7,api_name='ckpt_extract')
		with gr.TabItem(i18n('Onnx导出')):
			with gr.Row():ckpt_dir=gr.Textbox(label=i18n('RVC模型路径'),value='',interactive=_A)
			with gr.Row():onnx_dir=gr.Textbox(label=i18n('Onnx输出路径'),value='',interactive=_A)
			with gr.Row():infoOnnx=gr.Label(label='info')
			with gr.Row():butOnnx=gr.Button(i18n('导出Onnx模型'),variant=_C)
			butOnnx.click(export_onnx,[ckpt_dir,onnx_dir],infoOnnx,api_name='export_onnx')
		tab_faq=i18n('常见问题解答')
		with gr.TabItem(tab_faq):
			try:
				if tab_faq=='常见问题解答':
					with open('docs/cn/faq.md',_J,encoding='utf8')as f:info=f.read()
				else:
					with open('docs/en/faq_en.md',_J,encoding='utf8')as f:info=f.read()
				gr.Markdown(value=info)
			except:gr.Markdown(traceback.format_exc())
		with gr.TabItem(i18n('System Commands')):
			with gr.Row():butGitRestart=gr.Button(i18n('Git pull + Restart'),variant=_C)
			butGitRestart.click(update_restart_app,_R,_R,api_name='restart_ui')
	if config.iscolab or config.paperspace:app.queue(max_size=1022).launch(max_threads=511,server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A,share=_B)
	else:app.queue(max_size=1022).launch(max_threads=511,server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A)