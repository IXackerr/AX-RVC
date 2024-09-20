import os
import sys
import json
import shutil
import requests
import tempfile
import subprocess
import gradio as gr
import pandas as pd
from rvc.lib.tools import gdown

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script
from rvc.lib.utils import format_title

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )
    else:
        file_name = format_title(os.path.basename(dropbox))
        if ".pth" in dropbox:
            model_name = format_title(file_name.split(".pth")[0])
        else:
            if "v2" not in dropbox:
                model_name = format_title(
                    file_name.split("_nprobe_1_")[1].split("_v1")[0]
                )
            else:
                model_name = format_title(
                    file_name.split("_nprobe_1_")[1].split("_v2")[0]
                )
        model_path = os.path.join(now_dir, "logs", model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if os.path.exists(os.path.join(model_path, file_name)):
            os.remove(os.path.join(model_path, file_name))
        shutil.move(dropbox, os.path.join(model_path, file_name))
        print(f"{file_name} saved in {model_path}")
        gr.Info(f"{file_name} saved in {model_path}")
    return None

def download_from_url(url):
    file_path = find_folder_parent(now_dir, "assets")
    print(file_path)
    zips_path = os.path.join(file_path, "assets", "zips")
    print(zips_path)
    os.makedirs(zips_path, exist_ok=True)
    if url != "":
        if "drive.google.com" in url:
            if "file/d/" in url:
                file_id = url.split("file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                return None

            if file_id:
                os.chdir(zips_path)
                try:
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        quiet=True,
                        fuzzy=True,
                    )
                except Exception as error:
                    error_message = str(
                        f"An error occurred downloading the file: {error}"
                    )
                    if (
                        "Too many users have viewed or downloaded this file recently"
                        in error_message
                    ):
                        os.chdir(now_dir)
                        return "too much use"
                    elif (
                        "Cannot retrieve the public link of the file." in error_message
                    ):
                        os.chdir(now_dir)
                        return "private link"
                    else:
                        print(error_message)
                        os.chdir(now_dir)
                        return None
        elif "disk.yandex.ru" in url:
            base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
            public_key = url
            final_url = base_url + urlencode(dict(public_key=public_key))
            response = requests.get(final_url)
            download_url = response.json()["href"]
            download_response = requests.get(download_url)

            if download_response.status_code == 200:
                filename = parse_qs(urlparse(unquote(download_url)).query).get(
                    "filename", [""]
                )[0]
                if filename:
                    os.chdir(zips_path)
                    with open(filename, "wb") as f:
                        f.write(download_response.content)
            else:
                print("Failed to get filename from URL.")
                return None

        elif "pixeldrain.com" in url:
            try:
                file_id = url.split("pixeldrain.com/u/")[1]
                os.chdir(zips_path)
                print(file_id)
                response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
                if response.status_code == 200:
                    file_name = (
                        response.headers.get("Content-Disposition")
                        .split("filename=")[-1]
                        .strip('";')
                    )
                    os.makedirs(zips_path, exist_ok=True)
                    with open(os.path.join(zips_path, file_name), "wb") as newfile:
                        newfile.write(response.content)
                        os.chdir(file_path)
                        return "downloaded"
                else:
                    os.chdir(file_path)
                    return None
            except Exception as error:
                print(f"An error occurred downloading the file: {error}")
                os.chdir(file_path)
                return None

        elif "cdn.discordapp.com" in url:
            file = requests.get(url)
            os.chdir(zips_path)
            if file.status_code == 200:
                name = url.split("/")
                with open(os.path.join(name[-1]), "wb") as newfile:
                    newfile.write(file.content)
            else:
                return None
        elif "/blob/" in url or "/resolve/" in url:
            os.chdir(zips_path)
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")

            response = requests.get(url, stream=True)
            if response.status_code == 200:
                content_disposition = six.moves.urllib_parse.unquote(
                    response.headers["Content-Disposition"]
                )
                m = re.search(r'filename="([^"]+)"', content_disposition)
                file_name = m.groups()[0]
                file_name = file_name.replace(os.path.sep, "_")
                total_size_in_bytes = int(response.headers.get("content-length", 0))
                block_size = 1024
                progress_bar_length = 50
                progress = 0

                with open(os.path.join(zips_path, file_name), "wb") as file:
                    for data in response.iter_content(block_size):
                        file.write(data)
                        progress += len(data)
                        progress_percent = int((progress / total_size_in_bytes) * 100)
                        num_dots = int(
                            (progress / total_size_in_bytes) * progress_bar_length
                        )
                        progress_bar = (
                            "["
                            + "." * num_dots
                            + " " * (progress_bar_length - num_dots)
                            + "]"
                        )
                        print(
                            f"{progress_percent}% {progress_bar} {progress}/{total_size_in_bytes}  ",
                            end="\r",
                        )
                        if progress_percent == 100:
                            print("\n")

            else:
                os.chdir(now_dir)
                return None
        elif "/tree/main" in url:
            os.chdir(zips_path)
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            temp_url = ""
            for link in soup.find_all("a", href=True):
                if link["href"].endswith(".zip"):
                    temp_url = link["href"]
                    break
            if temp_url:
                url = temp_url
                url = url.replace("blob", "resolve")
                if "huggingface.co" not in url:
                    url = "https://huggingface.co" + url

                    wget.download(url)
            else:
                os.chdir(now_dir)
                return None
        elif "applio.org" in url:
            parts = url.split("/")
            id_with_query = parts[-1]
            id_parts = id_with_query.split("?")
            id_number = id_parts[0]

            url = "https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models"
            headers = {
                "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqdGZxempmZGltZ3B2cHdoemx2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUxNjczODgsImV4cCI6MjAxMDc0MzM4OH0.7z5WMIbjR99c2Ooc0ma7B_FyGq10G8X-alkCYTkKR10"
            }

            params = {"id": f"eq.{id_number}"}
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                json_response = response.json()
                print(json_response)
                if json_response:
                    link = json_response[0]["link"]
                    verify = download_from_url(link)
                    if verify == "downloaded":
                        return "downloaded"
                    else:
                        return None
            else:
                return None
        else:
            try:
                os.chdir(zips_path)
                wget.download(url)
            except Exception as error:
                os.chdir(now_dir)
                print(f"An error occurred downloading the file: {error}")
                return None

        for currentPath, _, zipFiles in os.walk(zips_path):
            for Files in zipFiles:
                filePart = Files.split(".")
                extensionFile = filePart[len(filePart) - 1]
                filePart.pop()
                nameFile = "_".join(filePart)
                realPath = os.path.join(currentPath, Files)
                os.rename(realPath, nameFile + "." + extensionFile)

        os.chdir(now_dir)
        return "downloaded"

    os.chdir(now_dir)
    return None

def load_dowloaded_dataset(url):
    parent_path = find_folder_parent(now_dir, "assets")
    infos = []
    try:
        zips_path = os.path.join(parent_path, "assets", "zips")
        unzips_path = os.path.join(parent_path, "assets", "unzips")
        datasets_path = os.path.join(parent_path, "datasets")
        audio_extenions = [
            "wav",
            "mp3",
            "flac",
            "ogg",
            "opus",
            "m4a",
            "mp4",
            "aac",
            "alac",
            "wma",
            "aiff",
            "webm",
            "ac3",
        ]

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        if not os.path.exists(datasets_path):
            os.mkdir(datasets_path)

        os.mkdir(zips_path)
        os.mkdir(unzips_path)

        download_file = download_from_url(url)

        if not download_file:
            print(i18n("An error occurred downloading"))
            infos.append(i18n("An error occurred downloading"))
            yield "\n".join(infos)
            raise Exception(i18n("An error occurred downloading"))
        elif download_file == "downloaded":
            print(i18n("It has been downloaded successfully."))
            infos.append(i18n("It has been downloaded successfully."))
            yield "\n".join(infos)
        elif download_file == "too much use":
            raise Exception(
                i18n("Too many users have recently viewed or downloaded this file")
            )
        elif download_file == "private link":
            raise Exception(i18n("Cannot get file from this private link"))

        zip_path = os.listdir(zips_path)
        foldername = ""
        for file in zip_path:
            if file.endswith(".zip"):
                file_path = os.path.join(zips_path, file)
                print("....")
                foldername = file.replace(".zip", "").replace(" ", "").replace("-", "_")
                print("Dataset Path:", foldername)
                infos.append("Dataset Path:", foldername)
                dataset_path = os.path.join(datasets_path, foldername)
                print(i18n("Proceeding with the extraction..."))
                infos.append(i18n("Proceeding with the extraction..."))
                yield "\n".join(infos)
                shutil.unpack_archive(file_path, unzips_path, "zip")
                if os.path.exists(dataset_path):
                    shutil.rmtree(dataset_path)

                os.mkdir(dataset_path)

                for root, subfolders, songs in os.walk(unzips_path):
                    for song in songs:
                        song_path = os.path.join(root, song)
                        if song.endswith(tuple(audio_extenions)):
                            formatted_song_name = format_title(
                                os.path.splitext(song)[0]
                            )
                            extension = os.path.splitext(song)[1]
                            new_song_path = os.path.join(
                                dataset_path, f"{formatted_song_name}{extension}"
                            )
                            shutil.move(song_path, new_song_path)
            else:
                print(i18n("Unzip error."))
                infos.append(i18n("Unzip error."))
                yield "\n".join(infos)

        if os.path.exists(zips_path):
            shutil.rmtree(zips_path)
        if os.path.exists(unzips_path):
            shutil.rmtree(unzips_path)

        print(i18n("The Dataset has been loaded successfully."))
        infos.append(i18n("The Dataset has been loaded successfully."))
        yield "\n".join(infos)
    except Exception as e:
        os.chdir(parent_path)
        if "too much use" in str(e):
            print(i18n("Too many users have recently viewed or downloaded this file"))
            yield i18n("Too many users have recently viewed or downloaded this file")
        elif "private link" in str(e):
            print(i18n("Cannot get file from this private link"))
            yield i18n("Cannot get file from this private link")
        else:
            print(e)
            yield i18n("An error occurred downloading")
    finally:
        os.chdir(parent_path)

def update_dataset_list(name):
    new_datasets = []
    file_path = find_folder_parent(now_dir, "assets")
    for foldername in os.listdir("./datasets"):
        if "." not in foldername:
            new_datasets.append(
                os.path.join(
                    file_path, "datasets", foldername
                )
            )
    return gr.Dropdown(choices=new_datasets)

def search_models(name):
    url = f"https://cjtfqzjfdimgpvpwhzlv.supabase.co/rest/v1/models?name=ilike.%25{name}%25&order=created_at.desc&limit=15"
    headers = {
        "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNqdGZxempmZGltZ3B2cHdoemx2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2OTUxNjczODgsImV4cCI6MjAxMDc0MzM4OH0.7z5WMIbjR99c2Ooc0ma7B_FyGq10G8X-alkCYTkKR10"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    if len(data) == 0:
        gr.Info(i18n("We couldn't find models by that name."))
        return None
    else:
        df = pd.DataFrame(data)[["name", "link", "epochs", "type"]]
        df["link"] = df["link"].apply(
            lambda x: f'<a href="{x}" target="_blank">{x}</a>'
        )
        return df


json_url = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json"


def fetch_pretrained_data():
    pretraineds_custom_path = os.path.join(
        "rvc", "models", "pretraineds", "pretraineds_custom"
    )
    os.makedirs(pretraineds_custom_path, exist_ok=True)
    try:
        with open(
            os.path.join(pretraineds_custom_path, json_url.split("/")[-1]), "r"
        ) as f:
            data = json.load(f)
    except:
        try:
            response = requests.get(json_url)
            response.raise_for_status()
            data = response.json()
            with open(
                os.path.join(pretraineds_custom_path, json_url.split("/")[-1]),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    separators=(",", ": "),
                    ensure_ascii=False,
                )
        except:
            data = {
                "Titan": {
                    "32k": {"D": "null", "G": "null"},
                },
            }
    return data


def get_pretrained_list():
    data = fetch_pretrained_data()
    return list(data.keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    return list(data[model].keys())


def get_file_size(url):
    response = requests.head(url)
    return int(response.headers.get("content-length", 0))


def download_file(url, destination_path, progress_bar):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    response = requests.get(url, stream=True)
    block_size = 1024
    with open(destination_path, "wb") as file:
        for data in response.iter_content(block_size):
            file.write(data)
            progress_bar.update(len(data))


def download_pretrained_model(model, sample_rate):
    data = fetch_pretrained_data()
    paths = data[model][sample_rate]
    pretraineds_custom_path = os.path.join(
        "rvc", "models", "pretraineds", "pretraineds_custom"
    )
    os.makedirs(pretraineds_custom_path, exist_ok=True)

    d_url = f"https://huggingface.co/{paths['D']}"
    g_url = f"https://huggingface.co/{paths['G']}"

    total_size = get_file_size(d_url) + get_file_size(g_url)

    gr.Info("Downloading pretrained model...")

    with tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading files"
    ) as progress_bar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    download_file,
                    d_url,
                    os.path.join(pretraineds_custom_path, os.path.basename(paths["D"])),
                    progress_bar,
                ),
                executor.submit(
                    download_file,
                    g_url,
                    os.path.join(pretraineds_custom_path, os.path.basename(paths["G"])),
                    progress_bar,
                ),
            ]
            for future in futures:
                future.result()

    gr.Info("Pretrained model downloaded successfully!")
    print("Pretrained model downloaded successfully!")


def update_sample_rate_dropdown(model):
    return {
        "choices": get_pretrained_sample_rates(model),
        "value": get_pretrained_sample_rates(model)[0],
        "__type__": "update",
    }

def find_folder_parent(search_dir, folder_name):
    for dirpath, dirnames, filenames in os.walk(search_dir):
        if folder_name in dirnames:
            return os.path.abspath(dirpath)
    return None

def download_tab():
    with gr.Column():
        gr.Markdown(value=i18n("## Download Model"))
        model_link = gr.Textbox(
            label=i18n("Model Link"),
            placeholder=i18n("Introduce the model link"),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(i18n("Download Model"))
        model_download_button.click(
            fn=run_download_script,
            inputs=[model_link],
            outputs=[model_download_output_info],
        )
        gr.Markdown(value=i18n("## Drop files"))
        dropbox = gr.File(
            label=i18n(
                "Drag your .pth file and .index file into this space. Drag one and then the other."
            ),
            type="filepath",
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
        gr.Markdown(value=i18n("## Search Model"))
        search_name = gr.Textbox(
            label=i18n("Model Name"),
            placeholder=i18n("Introduce the model name to search."),
            interactive=True,
        )
        search_table = gr.Dataframe(datatype="markdown")
        search = gr.Button(i18n("Search"))
        search.click(
            fn=search_models,
            inputs=[search_name],
            outputs=[search_table],
        )
        search_name.submit(search_models, [search_name], search_table)
        gr.Markdown(value=i18n("## Download Dataset"))
        gr.Markdown(
            value=i18n(
                "Download the dataset with the audios in a compatible format (.wav/.flac) to train your model."
            )
        )
        with gr.Row():
            dataset_url = gr.Textbox(label=i18n("Url:"))
        with gr.Row():
            load_dataset_status_bar = gr.Textbox(label=i18n("Status:"))
        with gr.Row():
            load_dataset_button = gr.Button(i18n("Download"))
            load_dataset_button.click(
                fn=load_dowloaded_dataset,
                inputs=[dataset_url],
                outputs=[load_dataset_status_bar],
            )
            load_dataset_status_bar.change(update_dataset_list, dataset_url)
        # gr.Markdown(value=i18n("## Download Pretrained Models"))
        # pretrained_model = gr.Dropdown(
        #     label=i18n("Pretrained"),
        #     info=i18n("Select the pretrained model you want to download."),
        #     choices=get_pretrained_list(),
        #     value="Titan",
        #     interactive=True,
        # )
        # pretrained_sample_rate = gr.Dropdown(
        #     label=i18n("Sampling Rate"),
        #     info=i18n("And select the sampling rate."),
        #     choices=get_pretrained_sample_rates(pretrained_model.value),
        #     value="40k",
        #     interactive=True,
        #     allow_custom_value=True,
        # )
        # pretrained_model.change(
        #     update_sample_rate_dropdown,
        #     inputs=[pretrained_model],
        #     outputs=[pretrained_sample_rate],
        # )
        # download_pretrained = gr.Button(i18n("Download"))
        # download_pretrained.click(
        #     fn=download_pretrained_model,
        #     inputs=[pretrained_model, pretrained_sample_rate],
        #     outputs=[],
        # )
