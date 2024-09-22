curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash # removed sudo for sol
apt-get install git-lfs # removed sudo for sol
git lfs install
# huggingface-cli lfs-enable-largefiles .

git config --global credential.helper cache
python -m pip install huggingface_hub