<h1 align='center'> Graph Rhythm Network </h1>

### Requirements
Main dependencies (with python >= 3.7):<br />
torch==1.9.0<br />
torch-cluster==1.5.9<br />
torch-geometric==2.0.3<br />
torch-scatter==2.0.9<br />
torch-sparse==0.6.12<br />
torch-spline-conv==1.2.1<br />

Commands to install all the dependencies in a new conda environment <br />
*(python 3.7 and cuda 10.2 -- for other cuda versions change accordingly)*
```
conda create --name grn python=3.7
conda activate grn

pip install torch==1.9.0

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric
pip install scipy
pip install numpy
```
