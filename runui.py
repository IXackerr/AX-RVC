_v='/kaggle/input/ax-rmd/pretrained_v2'
_u='Convert'
_t='infer_refresh'
_s='F0 curve file [optional]'
_r='Advanced Settings'
_q='Pitch: 0 from man to man (or woman to woman); 12 from man to woman and -12 from woman to man.'
_p='Inference'
_o='Console'
_n='%s/3_feature768'
_m='%s/3_feature256'
_l='spectrogram.png'
_k='trained'
_j='Index Ratio'
_i='>=3 apply median filter to the harvested pitch results'
_h='Protect clear consonants and breathing sounds, preventing electro-acoustic tearing and other artifacts, 0.5 does not open'
_g='0=Input source volume, 1=Normalized Output'
_f='Resampling, 0=none'
_e='crepe'
_d='Pitch Extraction, rmvpe is best'
_c='%userprofile%\\Desktop\\models\\model_example.index'
_b='Path of index'
_a='visible'
_Z='_v2'
_Y='%s/logs/%s'
_X='value'
_W='%s/%s'
_V='Output'
_U='Auto-detect index path'
_T='rmvpe_gpu'
_S='48k'
_R='choices'
_Q='filepath'
_P='f0'
_O='32k'
_N='.index'
_M='harvest'
_L='40k'
_K='.pth'
_J='r'
_I='update'
_H='__type__'
_G='primary'
_F='\n'
_E='rmvpe'
_D='v1'
_C='是'
_B=False
_A=True
import os,sys
from dotenv import load_dotenv
import requests,zipfile
now_dir=os.getcwd()
sys.path.append(now_dir)
load_dotenv()
from infer.modules.vc.modules import VC
from i18n.i18n import I18nAuto
from configs.config import Config
from sklearn.cluster import MiniBatchKMeans
import torch,numpy as np,gradio as gr,faiss,fairseq,pathlib,json
from pydub import AudioSegment
from time import sleep
from subprocess import Popen
from random import shuffle
import warnings,traceback,threading,shutil,logging,matplotlib.pyplot as plt,soundfile as sf
from dotenv import load_dotenv
import edge_tts,asyncio
from infer.modules.vc.ilariatts import tts_order_voice
language_dict=tts_order_voice
ilariavoices=list(language_dict.keys())
now_dir=os.getcwd()
sys.path.append(now_dir)
load_dotenv()
logging.getLogger('numba').setLevel(logging.WARNING)
logger=logging.getLogger(__name__)
tmp=os.path.join(now_dir,'TEMP')
shutil.rmtree(tmp,ignore_errors=_A)
shutil.rmtree('%s/runtime/Lib/site-packages/infer_pack'%now_dir,ignore_errors=_A)
os.makedirs(tmp,exist_ok=_A)
os.makedirs(os.path.join(now_dir,'logs'),exist_ok=_A)
os.makedirs(os.path.join(now_dir,'models/pth'),exist_ok=_A)
os.environ['TEMP']=tmp
warnings.filterwarnings('ignore')
torch.manual_seed(114514)
config=Config()
vc=VC(config)
if config.dml:
	def forward_dml(ctx,x,scale):ctx.scale=scale;A=x.clone().detach();return A
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
		if any(A in gpu_name.upper()for A in['10','16','20','30','40','A2','A3','A4','P4','A50','500','A60','70','80','90','M4','T4','TITAN']):if_gpu_ok=_A;gpu_infos.append('%s\t%s'%(i,gpu_name));mem.append(int(torch.cuda.get_device_properties(i).total_memory/1024/1024/1024+.4))
if if_gpu_ok and len(gpu_infos)>0:gpu_info=_F.join(gpu_infos);default_batch_size=min(mem)//2
else:gpu_info=i18n("Your GPU doesn't work for training");default_batch_size=1
gpus='-'.join([A[0]for A in gpu_infos])
class ToolButton(gr.Button,gr.components.FormComponent):
	def __init__(B,**A):super().__init__(variant='tool',**A)
	def get_block_name(A):return'button'
weight_root=os.getenv('weight_root')
index_root=os.getenv('index_root')
weight_root='./models/pth'
index_root='./models/index'
names=[]
for name in os.listdir(weight_root):
	if name.endswith(_K):names.append(name)
index_paths=[]
for(root,dirs,files)in os.walk(index_root,topdown=_B):
	for name in files:
		if name.endswith(_N)and _k not in name:index_paths.append(_W%(root,name))
def generate_spectrogram(audio_data,sample_rate,file_name):plt.clf();plt.specgram(audio_data,Fs=sample_rate/1,NFFT=4096,sides='onesided',cmap='Reds_r',scale_by_freq=_A,scale='dB',mode='magnitude',window=np.hanning(4096));plt.title(file_name);plt.savefig(_l)
def get_audio_info(audio_file):
	B=audio_file;C,G=sf.read(B)
	if len(C.shape)>1:C=np.mean(C,axis=1)
	generate_spectrogram(C,G,os.path.basename(B));A=sf.info(B);E={'PCM_16':16,'FLOAT':32}.get(A.subtype,0);H,D=divmod(A.duration,60);D,F=divmod(D,1);F*=1000;I=A.samplerate*E/1000;J,L=os.path.splitext(os.path.basename(B));K=f"""
    | Information | Value |
    | :---: | :---: |
    | File Name | {J} |
    | Duration | {int(H)} minutes - {int(D)} seconds - {int(F)} milliseconds |
    | Bitrate | {I} kbp/s |
    | Audio Channels | {A.channels} |
    | Samples per second | {A.samplerate} Hz |
    | Bit per second | {A.samplerate*A.channels*E} bit/s |
    """;return K,_l
def change_choices():
	B=[]
	for A in os.listdir(weight_root):
		if A.endswith(_K):B.append(A)
	C=[]
	for(D,F,E)in os.walk(index_root,topdown=_B):
		for A in E:
			if A.endswith(_N)and _k not in A:C.append(_W%(D,A))
	return{_R:sorted(B),_H:_I},{_R:sorted(C),_H:_I}
def tts_and_convert(ttsvoice,text,spk_item,vc_transform,f0_file,f0method,file_index1,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect):A='./TEMP/temp_ilariatts.mp3';B=language_dict[ttsvoice];asyncio.run(edge_tts.Communicate(text,B).save(A));C=A;vc_output1.update('Text converted successfully!');return vc.vc_single(spk_item,C,vc_transform,f0_file,f0method,file_index1,file_index2,index_rate,filter_radius,resample_sr,rms_mix_rate,protect)
def import_files(file):
	G='./models/index/';F='./models/pth/';B=file
	if B is not None:
		C=B.name
		if C.endswith('.zip'):
			with zipfile.ZipFile(B.name,_J)as H:
				D='./TEMP';H.extractall(D)
				for(E,J,I)in os.walk(D):
					for B in I:
						if B.endswith(_K):
							A=F+B
							if not os.path.exists(A):shutil.move(os.path.join(E,B),A)
							else:print(f"File {A} already exists. Skipping.")
						elif B.endswith(_N):
							A=G+B
							if not os.path.exists(A):shutil.move(os.path.join(E,B),A)
							else:print(f"File {A} already exists. Skipping.")
				shutil.rmtree(D)
			return'Zip file has been successfully extracted.'
		elif C.endswith(_K):
			A=F+os.path.basename(B.name)
			if not os.path.exists(A):os.rename(B.name,A)
			else:print(f"File {A} already exists. Skipping.")
			return'PTH file has been successfully imported.'
		elif C.endswith(_N):
			A=G+os.path.basename(B.name)
			if not os.path.exists(A):os.rename(B.name,A)
			else:print(f"File {A} already exists. Skipping.")
			return'Index file has been successfully imported.'
		else:return'Unsupported file type.'
	else:return'No file has been uploaded.'
def import_button_click(file):return import_files(file)
def get_audio_duration(audio_file_path):A=sf.info(audio_file_path);B=A.duration/60;return B
def clean():return{_X:'',_H:_I}
def get_training_info(audio_file):
	C='Ov2';A='Normal';B=get_audio_duration(audio_file);D={(0,2):(150,C),(2,3):(200,C),(3,5):(250,C),(5,10):(300,A),(10,25):(500,A),(25,45):(700,A),(45,60):(1000,A)};E=round(B,2)
	for((F,G),(H,I))in D.items():
		if F<=B<G:return f"For an audio of {E} minutes, use {H} epochs and {I} pretrain."
	if B>=60:return'Datasets over 1 hour can result easily in overtraining; consider trimming down your dataset.'
	return'The audio duration does not meet the minimum requirement for training.'
sr_dict={_O:32000,_L:40000,_S:48000}
def if_done(done,p):
	while 1:
		if p.poll()is None:sleep(.5)
		else:break
	done[0]=_A
def on_button_click(audio_file_path):return get_training_info(audio_file_path)
def download_from_url(url,model):
	K='./unzips';G='unzips';F='zips';D=model;A=url
	if A=='':return'URL cannot be left empty.'
	if D=='':return'You need to name your model. For example: Ilaria'
	A=A.strip();L=[F,G]
	for H in L:
		if os.path.exists(H):shutil.rmtree(H)
	os.makedirs(F,exist_ok=_A);os.makedirs(G,exist_ok=_A);M=D+'.zip';E='./zips/'+M
	try:
		if'drive.google.com'in A:subprocess.run(['gdown',A,'--fuzzy','-O',E])
		elif'mega.nz'in A:N=Mega();N.download_url(A,'./zips')
		else:
			I=requests.get(A);I.raise_for_status()
			with open(E,'wb')as B:B.write(I.content)
		shutil.unpack_archive(E,K,'zip')
		for(O,Q,P)in os.walk(K):
			for B in P:
				J=os.path.join(O,B)
				if B.endswith(_N):os.makedirs(f"./models/index",exist_ok=_A);shutil.copy2(J,f"./models/index/{D}.index")
				elif'G_'not in B and'D_'not in B and B.endswith(_K):os.makedirs(f"./models/pth",exist_ok=_A);shutil.copy(J,f"./models/pth/{D}.pth")
		shutil.rmtree(F);shutil.rmtree(G);return'Model downloaded, you can go back to the inference page!'
	except subprocess.CalledProcessError as C:return f"ERROR - Download failed (gdown): {str(C)}"
	except requests.exceptions.RequestException as C:return f"ERROR - Download failed (requests): {str(C)}"
	except Exception as C:return f"ERROR - The test failed: {str(C)}"
def if_done_multi(done,ps):
	while 1:
		A=1
		for B in ps:
			if B.poll()is None:A=0;sleep(.5);break
		if A==1:break
	done[0]=_A
def preprocess_dataset(trainset_dir,exp_dir,sr,n_p):
	C='%s/logs/%s/preprocess.log';A=exp_dir;sr=sr_dict[sr];os.makedirs(_Y%(now_dir,A),exist_ok=_A);B=open(C%(now_dir,A),'w');B.close();G=3. if config.is_half else 3.7;D='"%s" infer/modules/train/preprocess.py "%s" %s %s "%s/logs/%s" %s %.1f'%(config.python_cmd,trainset_dir,sr,n_p,now_dir,A,config.noparallel,G);logger.info(D);H=Popen(D,shell=_A);E=[_B];threading.Thread(target=if_done,args=(E,H)).start()
	while 1:
		with open(C%(now_dir,A),_J)as B:yield B.read()
		sleep(1)
		if E[0]:break
	with open(C%(now_dir,A),_J)as B:F=B.read()
	logger.info(F);yield F
def extract_f0_feature(gpus,n_p,f0method,if_f0,exp_dir,version19,gpus_rmvpe):
	N=f0method;J=gpus;I='%s/logs/%s/extract_f0_feature.log';F=gpus_rmvpe;B=exp_dir;J=J.split('-');os.makedirs(_Y%(now_dir,B),exist_ok=_A);C=open(I%(now_dir,B),'w');C.close()
	if if_f0:
		if N!=_T:A='"%s" infer/modules/train/extract/extract_f0_print.py "%s/logs/%s" %s %s'%(config.python_cmd,now_dir,B,n_p,N);logger.info(A);E=Popen(A,shell=_A,cwd=now_dir);D=[_B];threading.Thread(target=if_done,args=(D,E)).start()
		elif F!='-':
			F=F.split('-');K=len(F);G=[]
			for(L,M)in enumerate(F):A='"%s" infer/modules/train/extract/extract_f0_rmvpe.py %s %s %s "%s/logs/%s" %s '%(config.python_cmd,K,L,M,now_dir,B,config.is_half);logger.info(A);E=Popen(A,shell=_A,cwd=now_dir);G.append(E)
			D=[_B];threading.Thread(target=if_done_multi,args=(D,G)).start()
		else:A=config.python_cmd+' infer/modules/train/extract/extract_f0_rmvpe_dml.py "%s/logs/%s" '%(now_dir,B);logger.info(A);E=Popen(A,shell=_A,cwd=now_dir);E.wait();D=[_A]
		while 1:
			with open(I%(now_dir,B),_J)as C:yield C.read()
			sleep(1)
			if D[0]:break
		with open(I%(now_dir,B),_J)as C:H=C.read()
		logger.info(H);yield H
	K=len(J);G=[]
	for(L,M)in enumerate(J):A='"%s" infer/modules/train/extract_feature_print.py %s %s %s %s "%s/logs/%s" %s'%(config.python_cmd,config.device,K,L,M,now_dir,B,version19);logger.info(A);E=Popen(A,shell=_A,cwd=now_dir);G.append(E)
	D=[_B];threading.Thread(target=if_done_multi,args=(D,G)).start()
	while 1:
		with open(I%(now_dir,B),_J)as C:yield C.read()
		sleep(1)
		if D[0]:break
	with open(I%(now_dir,B),_J)as C:H=C.read()
	logger.info(H);yield H
def get_pretrained_models(path_str,f0_str,sr2):
	G='/kaggle/input/ax-rmd/pretrained%s/%sD%s.pth';F='/kaggle/input/ax-rmd/pretrained%s/%sG%s.pth';C=sr2;B=f0_str;A=path_str;D=os.access(F%(A,B,C),os.F_OK);E=os.access(G%(A,B,C),os.F_OK)
	if not D:logger.warning('/kaggle/input/ax-rmd/pretrained%s/%sG%s.pth not exist, will not use pretrained model',A,B,C)
	if not E:logger.warning('/kaggle/input/ax-rmd/pretrained%s/%sD%s.pth not exist, will not use pretrained model',A,B,C)
	return F%(A,B,C)if D else'',G%(A,B,C)if E else''
def change_sr2(sr2,if_f0_3,version19):A=''if version19==_D else _Z;B=_P if if_f0_3 else'';return get_pretrained_models(A,B,sr2)
def change_version19(sr2,if_f0_3,version19):
	B=version19;A=sr2;C=''if B==_D else _Z
	if A==_O and B==_D:A=_L
	D={_R:[_L,_S],_H:_I,_X:A}if B==_D else{_R:[_O,_L,_S],_H:_I,_X:A};E=_P if if_f0_3 else'';return*get_pretrained_models(C,E,A),D
def change_f0(if_f0_3,sr2,version19):A=if_f0_3;B=''if version19==_D else _Z;return{_a:A,_H:_I},{_a:A,_H:_I},*get_pretrained_models(B,_P if A is _A else'',sr2)
def click_train(exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19):
	d='-pd %s';c='-pg %s';X=if_save_every_weights18;W=if_cache_gpu17;V=if_save_latest13;U=batch_size12;T=total_epoch11;S=save_epoch10;Q=gpus16;P=exp_dir1;L=spk_id5;K=version19;J=pretrained_D15;I=pretrained_G14;H=if_f0_3;G='\\\\';F='\\';E='.';B=sr2;global f0_dir,f0nsf_dir;A=_Y%(now_dir,P);os.makedirs(A,exist_ok=_A);M='%s/0_gt_wavs'%A;N=_m%A if K==_D else _n%A
	if H:f0_dir='%s/2a_f0'%A;f0nsf_dir='%s/2b-f0nsf'%A;Y=set([A.split(E)[0]for A in os.listdir(M)])&set([A.split(E)[0]for A in os.listdir(N)])&set([A.split(E)[0]for A in os.listdir(f0_dir)])&set([A.split(E)[0]for A in os.listdir(f0nsf_dir)])
	else:Y=set([A.split(E)[0]for A in os.listdir(M)])&set([A.split(E)[0]for A in os.listdir(N)])
	C=[]
	for D in Y:
		if H:C.append('%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s'%(M.replace(F,G),D,N.replace(F,G),D,f0_dir.replace(F,G),D,f0nsf_dir.replace(F,G),D,L))
		else:C.append('%s/%s.wav|%s/%s.npy|%s'%(M.replace(F,G),D,N.replace(F,G),D,L))
	Z=256 if K==_D else 768
	if H:
		for e in range(2):C.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s'%(now_dir,B,now_dir,Z,now_dir,now_dir,L))
	else:
		for e in range(2):C.append('%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s'%(now_dir,B,now_dir,Z,L))
	shuffle(C)
	with open('%s/filelist.txt'%A,'w')as O:O.write(_F.join(C))
	logger.debug('Write filelist done');logger.info('Use gpus: %s',str(Q))
	if I=='':logger.info('No pretrained Generator')
	if J=='':logger.info('No pretrained Discriminator')
	if K==_D or B==_L:a='v1/%s.json'%B
	else:a='v2/%s.json'%B
	b=os.path.join(A,'config.json')
	if not pathlib.Path(b).exists():
		with open(b,'w',encoding='utf-8')as O:json.dump(config.json_config[a],O,ensure_ascii=_B,indent=4,sort_keys=_A);O.write(_F)
	if Q:R='"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,P,B,1 if H else 0,U,Q,T,S,c%I if I!=''else'',d%J if J!=''else'',1 if V==i18n(_C)else 0,1 if W==i18n(_C)else 0,1 if X==i18n(_C)else 0,K)
	else:R='"%s" infer/modules/train/train.py -e "%s" -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s'%(config.python_cmd,P,B,1 if H else 0,U,T,S,c%I if I!=''else'',d%J if J!=''else'',1 if V==i18n(_C)else 0,1 if W==i18n(_C)else 0,1 if X==i18n(_C)else 0,K)
	logger.info(R);f=Popen(R,shell=_A,cwd=now_dir);f.wait();return'You can view console or train.log'
def train_index(exp_dir1,version19):
	G=exp_dir1;D=version19;C='logs/%s'%G;os.makedirs(C,exist_ok=_A);I=_m%C if D==_D else _n%C
	if not os.path.exists(I):return'Please perform Feature Extraction First!'
	J=list(os.listdir(I))
	if len(J)==0:return'Please perform Feature Extraction First！'
	B=[];K=[]
	for P in sorted(J):Q=np.load(_W%(I,P));K.append(Q)
	A=np.concatenate(K,0);L=np.arange(A.shape[0]);np.random.shuffle(L);A=A[L]
	if A.shape[0]>2e5:
		B.append('Trying doing kmeans %s shape to 10k centers.'%A.shape[0]);yield _F.join(B)
		try:A=MiniBatchKMeans(n_clusters=10000,verbose=_A,batch_size=256*config.n_cpu,compute_labels=_B,init='random').fit(A).cluster_centers_
		except:M=traceback.format_exc();logger.info(M);B.append(M);yield _F.join(B)
	np.save('%s/total_fea.npy'%C,A);E=min(int(16*np.sqrt(A.shape[0])),A.shape[0]//39);B.append('%s,%s'%(A.shape,E));yield _F.join(B);F=faiss.index_factory(256 if D==_D else 768,'IVF%s,Flat'%E);B.append('training');yield _F.join(B);H=faiss.extract_index_ivf(F);H.nprobe=1;F.train(A);faiss.write_index(F,'%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index'%(C,E,H.nprobe,G,D));B.append('adding');yield _F.join(B);N=8192
	for O in range(0,A.shape[0],N):F.add(A[O:O+N])
	faiss.write_index(F,'%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(C,E,H.nprobe,G,D));B.append('Success，added_IVF%s_Flat_nprobe_%s_%s_%s.index'%(E,H.nprobe,G,D));yield _F.join(B)
F0GPUVisible=config.dml is _B
def change_f0_method(f0method8):
	if f0method8==_T:A=F0GPUVisible
	else:A=_B
	return{_a:A,_H:_I}
vc_output1=gr.Textbox(label=i18n(_o))
vc_output2=gr.Audio(label=i18n('Audio output'))
with gr.Blocks(title='Ilaria RVC 💖')as app:
	gr.Markdown('<h1>  Ilaria RVC 💖   </h1>');gr.Markdown(value=i18n('Made with 💖 by Ilaria | Support her on [Ko-Fi](https://ko-fi.com/ilariaowo)'))
	with gr.Tabs():
		with gr.TabItem(i18n(_p)):
			with gr.Row():
				sid0=gr.Dropdown(label=i18n('Voice'),choices=sorted(names));sid1=sid0
				with gr.Column():refresh_button=gr.Button(i18n('Refresh'),variant=_G);clean_button=gr.Button(i18n('Unload Voice from VRAM'),variant=_G)
				spk_item=gr.Slider(minimum=0,maximum=2333,step=1,label=i18n('Speaker ID (Auto-Detected)'),value=0,visible=_A,interactive=_B);clean_button.click(fn=clean,inputs=[],outputs=[sid0],api_name='infer_clean')
			with gr.TabItem(i18n(_p)):
				with gr.Group():
					with gr.Row():
						with gr.Column():
							with gr.Accordion('Audio input',open=_A):input_audio0=gr.Audio(label=i18n('Upload Audio file'),type=_Q);record_button=gr.Audio(source='microphone',label='Or you can use your microphone!',type=_Q);record_button.change(fn=lambda x:x,inputs=[record_button],outputs=[input_audio0]);file_index1=gr.Textbox(label=i18n(_b),placeholder=_c,interactive=_A,visible=_B);file_index2=gr.Textbox(label=i18n(_U),choices=sorted(index_paths),interactive=_A,visible=_B)
						with gr.Column():
							vc_transform0=gr.inputs.Slider(label=i18n(_q),minimum=-12,maximum=12,default=0,step=1)
							with gr.Accordion(_r,open=_B,visible=_B):
								with gr.Column():f0method0=gr.Radio(label=i18n(_d),choices=[_M,_e,_E]if config.dml is _B else[_M,_E],value=_E,interactive=_A);resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(_f),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(_g),value=.25,interactive=_A);protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(_h),value=.33,step=.01,interactive=_A);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(_i),value=3,step=1,interactive=_A);index_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_j),value=.4,interactive=_A);f0_file=gr.File(label=i18n(_s),visible=_B);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2],api_name=_t);file_index1=gr.Textbox(label=i18n(_b),placeholder=_c,interactive=_A);file_index2=gr.Dropdown(label=i18n(_U),choices=sorted(index_paths),interactive=_A)
							with gr.Accordion('IlariaTTS',open=_A):
								with gr.Column():ilariaid=gr.Dropdown(label='Voice:',choices=ilariavoices,interactive=_A,value='English-Jenny (Female)');ilariatext=gr.Textbox(label='Input your Text',interactive=_A,value='This is a test.');ilariatts_button=gr.Button(value='Speak and Convert');ilariatts_button.click(tts_and_convert,[ilariaid,ilariatext,spk_item,vc_transform0,f0_file,f0method0,file_index1,file_index2,index_rate1,filter_radius0,resample_sr0,rms_mix_rate0,protect0],[vc_output1,vc_output2])
							with gr.Accordion(_r,open=_B,visible=_A):
								with gr.Column():f0method0=gr.Radio(label=i18n(_d),choices=[_M,_e,_E]if config.dml is _B else[_M,_E],value=_E,interactive=_A);resample_sr0=gr.Slider(minimum=0,maximum=48000,label=i18n(_f),value=0,step=1,interactive=_A);rms_mix_rate0=gr.Slider(minimum=0,maximum=1,label=i18n(_g),value=.25,interactive=_A);protect0=gr.Slider(minimum=0,maximum=.5,label=i18n(_h),value=.33,step=.01,interactive=_A);filter_radius0=gr.Slider(minimum=0,maximum=7,label=i18n(_i),value=3,step=1,interactive=_A);index_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_j),value=.4,interactive=_A);f0_file=gr.File(label=i18n(_s),visible=_B);refresh_button.click(fn=change_choices,inputs=[],outputs=[sid0,file_index2],api_name=_t);file_index1=gr.Textbox(label=i18n(_b),placeholder=_c,interactive=_A);file_index2=gr.Dropdown(label=i18n(_U),choices=sorted(index_paths),interactive=_A)
				with gr.Group():
					with gr.Column():
						but0=gr.Button(i18n(_u),variant=_G)
						with gr.Row():vc_output1.render();vc_output2.render()
						but0.click(vc.vc_single,[spk_item,input_audio0,vc_transform0,f0_file,f0method0,file_index1,file_index2,index_rate1,filter_radius0,resample_sr0,rms_mix_rate0,protect0],[vc_output1,vc_output2],api_name='infer_convert')
			with gr.TabItem('Download Voice Models'):
				with gr.Row():url=gr.Textbox(label='Huggingface Link:')
				with gr.Row():model=gr.Textbox(label='Name of the model (without spaces):');download_button=gr.Button('Download')
				with gr.Row():status_bar=gr.Textbox(label='Download Status')
				download_button.click(fn=download_from_url,inputs=[url,model],outputs=[status_bar])
			with gr.TabItem('Import Models'):file_upload=gr.File(label='Upload a .zip file containing a .pth and .index file');import_button=gr.Button('Import');import_status=gr.Textbox(label='Import Status');import_button.click(fn=import_button_click,inputs=file_upload,outputs=import_status)
			with gr.TabItem(i18n('Batch Inference')):
				gr.Markdown(value=i18n('Batch Conversion'))
				with gr.Row():
					with gr.Column():vc_transform1=gr.Number(label=i18n(_q),value=0);opt_input=gr.Textbox(label=i18n(_V),value='InferOutput');file_index3=gr.Textbox(label=i18n('Path to index'),value='',interactive=_A);file_index4=gr.Dropdown(label=i18n(_U),choices=sorted(index_paths),interactive=_A);f0method1=gr.Radio(label=i18n(_d),choices=[_M,_e,_E]if config.dml is _B else[_M,_E],value=_E,interactive=_A);format1=gr.Radio(label=i18n('Export Format'),choices=['flac','wav','mp3','m4a'],value='flac',interactive=_A);refresh_button.click(fn=lambda:change_choices()[1],inputs=[],outputs=file_index4,api_name='infer_refresh_batch')
					with gr.Column():resample_sr1=gr.Slider(minimum=0,maximum=48000,label=i18n(_f),value=0,step=1,interactive=_A);rms_mix_rate1=gr.Slider(minimum=0,maximum=1,label=i18n(_g),value=.25,interactive=_A);protect1=gr.Slider(minimum=0,maximum=.5,label=i18n(_h),value=.33,step=.01,interactive=_A);filter_radius1=gr.Slider(minimum=0,maximum=7,label=i18n(_i),value=3,step=1,interactive=_A);index_rate2=gr.Slider(minimum=0,maximum=1,label=i18n(_j),value=.4,interactive=_A)
				with gr.Row():dir_input=gr.Textbox(label=i18n('Enter the path to the audio folder to be processed'),placeholder='%userprofile%\\Desktop\\covers');inputs=gr.File(file_count='multiple',label=i18n('Audio files can also be imported in batch'))
				with gr.Row():but1=gr.Button(i18n(_u),variant=_G);vc_output3=gr.Textbox(label=i18n(_o));but1.click(vc.vc_multi,[spk_item,dir_input,opt_input,inputs,vc_transform1,f0method1,file_index3,file_index4,index_rate2,filter_radius1,resample_sr1,rms_mix_rate1,protect1,format1],[vc_output3],api_name='infer_convert_batch')
		with gr.TabItem(i18n('Train')):
			gr.Markdown(value=i18n(''))
			with gr.Row():exp_dir1=gr.Textbox(label=i18n('Model Name'),value='test-model');sr2=gr.Radio(label=i18n('Sample Rate'),choices=[_O,_L,_S],value=_O,interactive=_A);if_f0_3=gr.Radio(label=i18n('Pitch Guidance'),choices=[_A,_B],value=_A,interactive=_A);version19=gr.Radio(label=i18n('Version 2 only here'),choices=['v2'],value='v2',interactive=_B,visible=_B);np7=gr.Slider(minimum=0,maximum=config.n_cpu,step=1,label=i18n('CPU Threads'),value=int(np.ceil(config.n_cpu/2.5)),interactive=_A)
			with gr.Group():
				gr.Markdown(value=i18n(''))
				with gr.Row():trainset_dir4=gr.Textbox(label=i18n('Path to Dataset'),value='dataset');spk_id5=gr.Slider(minimum=0,maximum=4,step=1,label=i18n('Speaker ID'),value=0,interactive=_A);but1=gr.Button(i18n('Process Data'),variant=_G);info1=gr.Textbox(label=i18n(_V),value='');but1.click(preprocess_dataset,[trainset_dir4,exp_dir1,sr2,np7],[info1],api_name='train_preprocess')
			with gr.Group():
				gr.Markdown(value=i18n(''))
				with gr.Row():
					with gr.Column():gpus6=gr.Textbox(label=i18n('GPU ID (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)'),value=gpus,interactive=_A,visible=F0GPUVisible);gpu_info9=gr.Textbox(label=i18n('GPU Model'),value=gpu_info,visible=F0GPUVisible)
					with gr.Column():f0method8=gr.Radio(label=i18n('Feature Extraction Method'),choices=[_E,_T],value=_T,interactive=_A);gpus_rmvpe=gr.Textbox(label=i18n('rmvpe_gpu will use your GPU instead of the CPU for the feature extraction'),value='%s-%s'%(gpus,gpus),interactive=_A,visible=F0GPUVisible)
					but2=gr.Button(i18n('Feature Extraction'),variant=_G);info2=gr.Textbox(label=i18n(_V),value='',max_lines=8);f0method8.change(fn=change_f0_method,inputs=[f0method8],outputs=[gpus_rmvpe]);but2.click(extract_f0_feature,[gpus6,np7,f0method8,if_f0_3,exp_dir1,version19,gpus_rmvpe],[info2],api_name='train_extract_f0_feature')
			with gr.Group():
				gr.Markdown(value=i18n(''))
				with gr.Row():save_epoch10=gr.Slider(minimum=1,maximum=250,step=1,label=i18n('Save frequency'),value=50,interactive=_A);total_epoch11=gr.Slider(minimum=2,maximum=10000,step=1,label=i18n('Total Epochs'),value=300,interactive=_A);batch_size12=gr.Slider(minimum=1,maximum=16,step=1,label=i18n('Batch Size'),value=default_batch_size,interactive=_A);if_save_latest13=gr.Radio(label=i18n('Save last ckpt as final Model'),choices=[i18n(_C),i18n('否')],value=i18n(_C),interactive=_A);if_cache_gpu17=gr.Radio(label=i18n('Cache data to GPU (Only for datasets under 8 minutes)'),choices=[i18n(_C),i18n('否')],value=i18n('否'),interactive=_A);if_save_every_weights18=gr.Radio(label=i18n('Create model with save frequency'),choices=[i18n(_C),i18n('否')],value=i18n(_C),interactive=_A)
				file_dict={A:os.path.join(_v,A)for A in os.listdir(_v)};file_dict={A:B for(A,B)in file_dict.items()if A.endswith(_K)};file_dict_g={A:B for(A,B)in file_dict.items()if'G'in A and _P in A};file_dict_d={A:B for(A,B)in file_dict.items()if'D'in A and _P in A}
			with gr.Row():pretrained_G14=gr.Dropdown(label=i18n('Pretrained G'),choices=list(file_dict_g.values()),value=file_dict_g['f0G32k.pth'],interactive=_A);pretrained_D15=gr.Dropdown(label=i18n('Pretrained D'),choices=list(file_dict_d.values()),value=file_dict_d['f0D32k.pth'],interactive=_A);sr2.change(change_sr2,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15]);version19.change(change_version19,[sr2,if_f0_3,version19],[pretrained_G14,pretrained_D15,sr2]);if_f0_3.change(change_f0,[if_f0_3,sr2,version19],[f0method8,gpus_rmvpe,pretrained_G14,pretrained_D15]);gpus16=gr.Textbox(label=i18n('Enter cards to be used (Leave 0 if you have only one GPU, use 0-1 for multiple GPus)'),value=gpus if gpus!=''else'0',interactive=_A);but3=gr.Button(i18n('Train Model'),variant=_G);but4=gr.Button(i18n('Train Index'),variant=_G);info3=gr.Textbox(label=i18n(_V),value='',max_lines=10);but3.click(click_train,[exp_dir1,sr2,if_f0_3,spk_id5,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16,if_cache_gpu17,if_save_every_weights18,version19],info3,api_name='train_start');but4.click(train_index,[exp_dir1,version19],info3)
		with gr.TabItem(i18n('Extra')):
			with gr.Accordion('Model Info',open=_B):
				with gr.Column():sid1=gr.Dropdown(label=i18n('Voice Model'),choices=sorted(names));modelload_out=gr.Textbox(label='Model Metadata');get_model_info_button=gr.Button(i18n('Get Model Info'));get_model_info_button.click(fn=vc.get_vc,inputs=[sid1,protect0,protect1],outputs=[spk_item,protect0,protect1,file_index2,file_index4,modelload_out])
			with gr.Accordion('Audio Analyser',open=_B):
				with gr.Column():audio_input=gr.Audio(type=_Q);get_info_button=gr.Button(value=i18n('Get information about the audio'),variant=_G)
				with gr.Column():
					with gr.Row():
						with gr.Column():gr.Markdown(value=i18n('Information about the audio file'),visible=_A);output_markdown=gr.Markdown(value=i18n('Waiting for information...'),visible=_A)
						image_output=gr.Image(type=_Q,interactive=_B)
				get_info_button.click(fn=get_audio_info,inputs=[audio_input],outputs=[output_markdown,image_output])
			with gr.Accordion('Training Helper',open=_B):
				with gr.Column():audio_input=gr.Audio(type=_Q,label='Upload your audio file');gr.Text('Please note that these results are approximate and intended to provide a general idea for beginners.');training_info_output=gr.Textbox(label='Training Information');get_info_button=gr.Button('Get Training Info');get_info_button.click(fn=on_button_click,inputs=[audio_input],outputs=[training_info_output])
			with gr.Accordion('Credits',open=_B):gr.Markdown('\n                ## All the amazing people who worked on this!\n                \n                ### Developers\n                \n                - **Ilaria**: Founder, Lead Developer\n                - **Yui**: Training feature\n                - **GDR-**: Inference feature\n                - **Poopmaster**: Model downloader, Model importer\n                - **kitlemonfoot**: Ilaria TTS implementation\n                - **eddycrack864**: UVR5 implementation\n                \n                ### Beta Tester\n                \n                - **Charlotte**: Beta Tester\n                - **RME**: Beta Tester\n                - **Delik**: Beta Tester\n                \n                ### Pretrains Makers\n\n                - **simplcup**: Ov2Super\n                - **mustar22**: RIN_E3\n                - **mustar22**: Snowie\n                \n                ### Other\n                \n                - **yumereborn**: Ilaira RVC image\n                                \n                ### **In loving memory of JLabDX** 🕊️\n                ')
			sid0.change(fn=vc.get_vc,inputs=[sid0,protect0,protect1],outputs=[spk_item,protect0,protect1,file_index2,file_index4,modelload_out],api_name='infer_change_voice')
		with gr.TabItem(i18n('')):gr.Markdown('\n                ![ilaria](https://i.ytimg.com/vi/5PWqt2Wg-us/maxresdefault.jpg)\n            ')
	if config.iscolab:app.queue(concurrency_count=511,max_size=1022).launch(share=_A)
	else:app.queue(concurrency_count=511,max_size=1022).launch(server_name='0.0.0.0',inbrowser=not config.noautoopen,server_port=config.listen_port,quiet=_A)