# EchoNet_cardiac_mri
Cardiac magnetic resonance (CMR) can provide unique tissue characterization including late gadolinium enhancement (LGE), T1 and T2 mapping, and extracellular volume (ECV) which are associated with tissue fibrosis, infiltration, and inflammation. However, CMR imaging is less accessible and has a complicated process to perform.
On the other side, echocardiography is the most common modality for assessing cardiac structure and function. 

In our project, we hypothesized that deep learning applied to echocardiography could predict CMR-based measurements (Wall-motion abnormalities, Myocardial Scar, native T1 time, T2 time, ECV fraction) using a large-dataset contaims more than 1400 patients who have both CMR and Echo within 30days.


**Presentation:** [European Society of Cardiology 2024 (London)](https://www.escardio.org/Congresses-Events/ESC-Congress)

**Preprint:** [Using Deep learning to Predict Cardiovascular Magnetic Resonance Findings from Echocardiography Videos Short title: Deep learning prediction of CMR from echocardiography](https://pubmed.ncbi.nlm.nih.gov/38699330/)

Yuki Sahashi, MD, MSc; Milos Vukadinovic, BS; Grant Duffy, BS; Debiao Li, PhD; Susan Cheng, MD; Daniel S. Berman, MD David Ouyang MD Alan C. Kwan MD

![EchoNet-cardiac_mri Pipeline](https://github.com/echonet/cardiac_mri/blob/main/Figure_README.png)


### Prerequisites for the project

1. Python: 3.10.12
2. PyTorch pytorch==2.2.0
3. Other dependencies listed in `requirements.txt`

### Installation
First, clone the repository and install the required packages:

## Quickstart for inference

```sh
mkdir cardiac_mri
cd cardiac_mri 
git clone https://github.com/echonet/cardiac_mri.git
pip install -r requirements.txt
```

We used [R2plus1D model](https://arxiv.org/abs/1711.11248) for echocadriography video. 
In R2+1D model, the architecture decomposes all 3D convolutions into 2D spatial convolutions followed by temporal convolutions to incorporate both spatial as well as temporal information while minimizing model size.

All you need to prepare is 
- Echocardiography Dataset (We used 112*112 AVI video, A4c/A2c/PLAX echocardiography views) 
(Note: our datasets were de-identified and electrocardiogram and respirometer tracings were masked.)
- Disease LABEL file (csv) that contain `Study_Unique_ID`, `Video_Unique_ID`, `frames`, `LABEL` (LABEL will be  0/1 in binarized outcome and numerical value if regression task)

We released model weights and inference code for binarized outcomes (only Wall motion abnormalities and Scar).


```sh
python predict.py  --dataset YOUR_112*112_EchoDataset  --label ['Wall_Motion', 'Scar'] --view ['A4C', 'A2C', 'PLAX']
#python predict.py --dataset /workspace/data/a2c_avi --label Wall_Motion --view A2C
```

Following running this code, you will get `Prediction_<VIEW>_<LABEL>.csv`. 
This prediction.csv has the outputs (preds) and filename and labels (From Disease LABEL file that you have). 
We used glm model in R to synthesize these outputs (A4c / A2c/ Plax) to calculate global AUROC.

Fin.
