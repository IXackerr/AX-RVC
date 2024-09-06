# 🍏 Applio x Xackerr-RVC-Fork
AX-RVC is a user-friendly fork of Mangio-RVC-Fork/RVC, designed to provide an intuitive interface, especially for newcomers, and work on kaggle.
You need AX RVC Shell for run this script on kaggle

<p align="center">A simple, high-quality voice conversion tool, focused on ease of use and performance.</p>

# use dataset for kaggle
https://kaggle.com/datasets/aaa74fd62e95ad662b09255b9ef1b716829e139836caeab1db2bf6d3da534162
## 🎯 Improvements of Applio Over RVC
### f0 Inference Algorithm Overhaul
- Applio features a comprehensive overhaul of the f0 inference algorithm, including:
  - Addition of the pyworld dio f0 method.
  - Alternative method for calculating crepe f0.
  - Introduction of the torchcrepe crepe-tiny model.
  - Customizable crepe_hop_length for the crepe algorithm via both the web GUI and CLI.

## Introduction

Applio is a powerful voice conversion tool focused on simplicity, quality, and performance. Whether you're an artist, developer, or researcher, Applio offers a straightforward platform for high-quality voice transformations. Its flexible design allows for customization through plugins and configurations, catering to a wide range of projects.

## Getting Started

### 1. Installation

Run the installation script based on your operating system:

- **Windows:** Double-click `run-install.bat`.
- **Linux/macOS:** Execute `run-install.sh`.

### 2. Running Applio

Start Applio using:

- **Windows:** Double-click `run-applio.bat`.
- **Linux/macOS:** Run `run-applio.sh`.

This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:

- **Windows:** Run `run-tensorboard.bat`.
- **Linux/macOS:** Run `run-tensorboard.sh`.

For more detailed instructions, visit the [documentation](https://docs.applio.org).

## Commercial Usage

For commercial use, follow the [MIT license](./LICENSE) and contact us at support@applio.org to ensure ethical use. The use of Applio-generated audio files must comply with applicable copyrights. Consider supporting Applio’s development [through a donation](https://ko-fi.com/iahispano).

## References

Applio is made possible thanks to these projects and their references:

- [gradio-screen-recorder](https://huggingface.co/spaces/gstaff/gradio-screen-recorder) by gstaff
- [rvc-cli](https://github.com/blaisewf/rvc-cli) by blaisewf

### Contributors

<a href="https://github.com/IAHispano/Applio/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=IAHispano/Applio" />
</a>
