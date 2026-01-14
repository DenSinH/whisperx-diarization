pip install torch~=2.8.0 torchaudio~=2.8.0 --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
rem check if install is successful
python -c "import torch; print(torch.rand(2,3).cuda())"
