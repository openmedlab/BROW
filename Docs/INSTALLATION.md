Installation Guide
===========
After setting up anaconda, you can create and activate a `BROW` conda environment using the provided environment definition `environment.yaml`:
```bash
conda env create -f environment.yaml
conda activate BROW
```
Or you can install the enviroment step by step:

```bash
conda create -n BROW python=3.9.16
conda activate BROW
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 -c pytorch
```
Then install the required packages for pre-training:
```bash
pip install pandas 
pip install timm
pip install opencv-python
pip install openslide-python
pip install matplotlib
pip install h5py
```
So far the enviroment is ready for model pre-training.

To reproduce the results of slide-level subtyping tasks, some extra packages are needed:
```bash
pip install scipy 
pip install scikit-learn
pip install tensorboard 
pip install future
```
Please note that the package smooth-topk is installed by:
```bash
git clone https://github.com/oval-group/smooth-topk.git
cd smooth-topk
python setup.py install
```
