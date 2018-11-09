# Execute inside docker if run_and_time.sh complains about torchaudio.load(path)
# typename = '' error
cd /tmp
rm -rf audio
git clone https://github.com/pytorch/audio.git
cd audio
python setup.py install
