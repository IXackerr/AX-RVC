#!/bin/bash
echo -e "\033]0;Applio - Installer\007"
clear
echo " :::"
echo " :::                       _ _ "
echo " :::     /\               | (_) "
echo " :::    /  \   _ __  _ __ | |_  ___ "
echo " :::   / /\ \ | '_ \| '_ \| | |/ _ \ "
echo " :::  / ____ \| |_) | |_) | | | (_) | "
echo " ::: /_/    \_\ .__/| .__/|_|_|\___/ "
echo " :::          | |   | | "
echo " :::          |_|   |_| "
echo " ::: "
echo " ::: "
  if ! command -v git &> /dev/null; then
  echo "Please install git before run install_Applio.sh"
  exit 1
  fi
  if ! command -v python3.9 &> /dev/null; then
  echo "Please install python3.9 before run install_Applio.sh"
  exit 1
  fi
# Clone the repo for make this script usable with curl
git clone https://github.com/IAHispano/Applio-RVC-Fork
cd Applio-RVC-Fork
# It just works with python3.9 so
chmod +x stftpitchshift
chmod +x *.sh
# maybe is needed idk
chmod +x ./lib/infer/infer_libs/stftpitchshift
python3.9 -m ensurepip
clear
menu() {
  while true; do
  clear
echo
echo "Only recommended for experienced users:"
echo "[1] Nvidia graphics cards"
echo "[2] AMD / Intel graphics cards"
echo "[3] I have already installed the dependencies"
echo
read -p "Select the option according to your GPU: " choice

case $choice in
    1)
        echo
        python3.9 -m pip install -r assets/requirements/requirements.txt
        echo
        python3.9 -m pip uninstall torch torchvision torchaudio -y
        echo
        python3.9 -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
        echo
        finish
        ;;
    2)
        echo
        python3.9 -m pip install -r assets/requirements/requirements.txt
        python3.9 -m pip install -r assets/requirements/requirements-dml.txt
        echo
        finish
        ;;
    3)
        finish
        ;;
    *)
        echo "Invalid option. Please enter a number from 1 to 3."
        echo ""
        read -p "Press Enter to access the main menu..."
        ;;
esac
done
}

# Finish installation
finish() {
  clear
  echo "Applio has been successfully downloaded, run the file go-applio.sh to run the web interface!"
  exit 0
}
# Loop to the main menu
menu
