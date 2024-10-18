cd setup
echo huggingface for downloading large file
source install_hf.sh
echo install requirements
pip install -r requirements.txt # removed sudo for sol
pip install --upgrade numpy==1.26.4
pip install --upgrade wandb==0.17.0
pip install --upgrade torch==2.1.2
pip install --upgrade transformers==4.37.0
pip install --upgrade torchvision==0.16.2
pip install --upgrade datasets==3.0.0