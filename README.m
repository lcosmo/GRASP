# Setup your environment
Create a conda environment
```
conda create -n grasp python=3.9
```
Install pyTorch:
```
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install other required packages:
```
pip install -r requirements.txt 
```

# Generate graphs with existing checkpoints

You simply need to run:
```
python generate_graph.py
```
Inside the `generate_graph.py` file, you can set the checkpoints to be used. 10 sample files are saved on the `results` directory.
