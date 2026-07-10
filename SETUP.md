# Environment Setup

This project runs on two different machines with different install paths.
requirements.txt covers shared, platform-independent packages only.
GPU-specific ML libraries (torch/torchvision) are installed separately per platform, below.

## Windows + AMD GPU (ROCm via WSL2) — Omid's machine

Training runs for the base/masking comparison experiments must happen here (or on the Cluster), for hardware consistency across experiments.

1. Enable SVM Mode in BIOS (Advanced -> CPU Configuration)
2. Install AMD Adrenalin driver with WSL2 support (amd.com/support)
3. `wsl --install -d Ubuntu-24.04`
4. Install Windows SDK, component "Windows SDK for Desktop C++ amd64 Apps"
5. Inside WSL:
    sudo amdgpu-install --usecase=wsl,rocm --no-dkms -y
6. Build librocdxg (WSL-to-GPU bridge):
git clone https://github.com/ROCm/librocdxg.git
cd librocdxg && mkdir build && cd build
export win_sdk='/mnt/c/Program Files (x86)/Windows 
Kits/10/Include/<YOUR_SDK_VERSION>'
cmake .. -DWIN_SDK="${win_sdk}/shared"
make && sudo make install && sudo ldconfig
   (Check your real SDK version folder first: `ls "/mnt/c/Program Files (x86)/Windows Kits/10/Include/"`)
7. Verify: `rocminfo` should list your GPU as an Agent.
8. Install matched ML wheels (must be exact matched versions, not PyPI):
pip install torch-2.9.1+rocm7.2.0.lw.git7e1940d4-cp312-cp312-linux_x86_64.whl 
torchvision-0.24.0+rocm7.2.0.gitb919bd0c-cp312-cp312-linux_x86_64.whl 
triton-3.5.1+rocm7.2.0.gita272dfa8-cp312-cp312-linux_x86_64.whl
   Download from: https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2/
9. Install remaining shared packages: `pip install -r requirements.txt`
10. Verify: `python3 -c "import torch; print(torch.cuda.is_available())"` → should print True

## Iman's machine (dev/debugging only, not comparison runs)

1. `python3 -m venv venv && source venv/bin/activate`
2. `pip install torch torchvision` (standard PyPI build, uses MPS backend automatically)
3. `pip install -r requirements.txt`

Note: this machine is for code development and small-scale
debugging only. All controlled comparison experiments run on Omid's machine