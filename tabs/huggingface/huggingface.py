# huggingface.py

import os
import time
import zipfile
from huggingface_hub import HfApi, login, hf_hub_download
from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

def start_upload_to_huggingface(token, username, repo, model_name, model_epochs, model_steps, zip_preffix, upload_type):
    try:
        login(token=token, add_to_git_credential=True, new_session=True)
        
        hug_file_path = os.path.join(os.getcwd(), "hugupload")
        os.makedirs(hug_file_path, exist_ok=True)
        zip_name = ""
        
        if upload_type == "Model Only":
            # Copy model files
            os.system(f'cp logs/{model_name}/{model_name}_{model_epochs}e_{model_steps}s.pth {hug_file_path}')
            os.system(f'cp logs/{model_name}/added*.index {hug_file_path}')
            
            # Create zip
            if zip_preffix != "" or zip_preffix is not None:
                zip_name = f'{zip_preffix}_{model_name}_{model_epochs}e_{model_steps}s.zip'
            else:
                zip_name = f'{model_name}_{model_epochs}e_{model_steps}s.zip'
            os.chdir(hug_file_path)
            os.system(f'zip -r {zip_file} {model_name}.pth added*.index')
            
        else:
            # Copy full logs folder
            os.system(f'cp -r logs/{model_name} {hug_file_path}')
            if zip_preffix != "" or zip_preffix is not None:
                zip_name = f'LOGS_{zip_preffix}_{model_name}.zip'
            else:
                zip_name = f'LOGS_{model_name}.zip'
            os.chdir(hug_file_path)
            os.system(f'zip -r {zip_file} {model_name}')
        # Upload to HuggingFace
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=zip_file,
            path_in_repo=zip_file,
            repo_id=f"{username}/{repo}",
            repo_type="model"
        )

        # Cleanup
        os.chdir("..")
        os.system(f'rm -rf {hug_file_path}/*')
        
        return i18n("Successfully uploaded to HuggingFace!")
    
    except Exception as e:
        return f"Error: {str(e)}"

def start_download_from_huggingface(token, username, repo, zip_name):
    try:
        hug_file_path = os.path.join(os.getcwd(), "hugupload") 
        os.makedirs(hug_file_path, exist_ok=True)
        
        zip_file = f'{zip_name}.zip'
        repo_id = f"{username}/{repo}"
        
        # Download from HuggingFace
        hf_hub_download(
            repo_id=repo_id,
            filename=zip_file,
            token=token,
            local_dir=hug_file_path
        )
        
        # Extract files
        with zipfile.ZipFile(os.path.join(hug_file_path, zip_file), 'r') as zip_ref:
            zip_ref.extractall("logs")
        
        # Cleanup    
        os.system(f'rm -rf {hug_file_path}/*')
        
        return i18n("Successfully downloaded from HuggingFace!")
        
    except Exception as e:
        return f"Error: {str(e)}"

def huggingface_tab():
    import gradio as gr
    
    # Create tabs for Upload and Download
    with gr.Tabs():
        # Upload tab
        with gr.Tab(i18n("Upload Model")):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        hgf_token_gr = gr.Textbox(
                            label=i18n("Enter HuggingFace Write Token:"),
                            placeholder="Enter your HuggingFace write token here", 
                            type="password"
                        )
                        hgf_name_gr = gr.Textbox(
                            label=i18n("Enter HuggingFace Username:"),
                            placeholder="Your HuggingFace username"
                        )
                        hgf_repo_gr = gr.Textbox(
                            label=i18n("Enter HuggingFace Model-Repo name:"),
                            placeholder="Repository name"
                        )
                    with gr.Column():
                        model_name_gr = gr.Textbox(
                            label=i18n("Trained model name:"),
                            info=i18n("Name of your trained model (without extension)")
                        )
                        model_epochs_gr = gr.Textbox(
                            label=i18n("Number of epochs:"),
                            info=i18n("Number of epochs the model was trained")
                        )
                        model_steps_gr = gr.Textbox(
                            label=i18n("Number of steps:"),
                            info=i18n("Number of steps wich the model doed")
                        )
                        zip_name_gr = gr.Textbox(
                            label=i18n("Name of Zip Preffix file:"),
                            info=i18n("Preffix for zip file to be uploaded")
                        )
                        what_upload_gr = gr.Radio(
                            label=i18n("Upload files:"),
                            choices=["Model Only", "Model Log Folder"],
                            value="Model Only",
                            info=i18n("Choose what to upload to HuggingFace"),
                            interactive=True,
                        )
                with gr.Row():
                    uploadbut1 = gr.Button(i18n("Start upload"), variant="primary")
                    uploadinfo1 = gr.Textbox(
                        label=i18n("Output information:"),
                        placeholder=i18n("Upload status will appear here"),
                        interactive=False
                    )

        # Download tab  
        with gr.Tab(i18n("Download Model")):
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        hgf_token_gr_d = gr.Textbox(
                            label=i18n("Enter HuggingFace Write Token:"),
                            placeholder="Enter your HuggingFace token here",
                            type="password"
                        )
                        hgf_name_gr_d = gr.Textbox(
                            label=i18n("Enter HuggingFace Username:"),
                            placeholder="HuggingFace username"
                        )
                    with gr.Column():
                        hgf_repo_gr_d = gr.Textbox(
                            label=i18n("Enter HuggingFace Model-Repo name:"),
                            placeholder="Repository name"
                        )
                        zip_name_gr_d = gr.Textbox(
                            label=i18n("Name of Zip file:"),
                            info=i18n("Name of the zip file to download")
                        )
                with gr.Row():
                    downloadlogsbut1 = gr.Button(i18n("Start download"), variant="primary")
                    downloadlogsinfo1 = gr.Textbox(
                        label=i18n("Output information:"),
                        placeholder=i18n("Download status will appear here"),
                        interactive=False
                    )

    # Connect buttons to functions
    uploadbut1.click(
        start_upload_to_huggingface,
        inputs=[hgf_token_gr, hgf_name_gr, hgf_repo_gr, model_name_gr, model_epochs_gr, model_steps_gr, zip_name_gr, what_upload_gr],
        outputs=[uploadinfo1]
    )
    
    downloadlogsbut1.click(
        start_download_from_huggingface,
        inputs=[hgf_token_gr_d, hgf_name_gr_d, hgf_repo_gr_d, zip_name_gr_d],
        outputs=[downloadlogsinfo1]
    )