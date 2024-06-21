pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --force-reinstall
rem check if install is successful
python -c "import torch; print(torch.rand(2,3).cuda())"
