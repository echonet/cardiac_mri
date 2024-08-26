import numpy as np
import pandas as pd
import os
import torch
from lightning_utilities.core.imports import compare_version
from torch.utils.data import DataLoader
from utils import sensivity_specifity_cutoff, sigmoid, EchoDataset,get_frame_count
import glob
from torchvision.models.video import r2plus1d_18
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, auc, roc_curve

# In the manuscript, we trained and inferenced 1. Wall motion abnormalities, 2. Scar detection, 3. Native T1, 4. T2, 5. ECV fraction.
# Among all, we released model with relatively higher prediction performance (AUC) (Wall motion abnormalities and Scar detection).

with torch.no_grad():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Predict script for Cardiac MRI Prediction From Echocardiography.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--label", type=str, required=True, choices = ['Wall_Motion', 'Scar'] ,help="Label for prediction from echo dataset.")
    parser.add_argument("--view", type=str, required=True, choices = ['A4C', 'A2C', 'PLAX'], help="Path to the output directory.")
    
    args = parser.parse_args()   
    
    #Weight label setting    
    if args.label == "Wall_Motion" and args.view == "A4C":
        weights_path = "/workspace/YOUR PATH/model_weights/A4C_WallMotoion.pt"    
    elif args.label == "Wall_Motion" and args.view == "A2C":
        weights_path = "/workspace/YOUR PATH/model_weights/A2C_WallMotoion.pt"
    elif args.label == "Wall_Motion" and args.view == "PLAX":
        weights_path = "/workspace/YOUR PATH/model_weights/PLAX_WallMotoion.pt"
    elif args.label == "Scar" and args.view == "A4C":
        weights_path = "/workspace/YOUR PATH/model_weights/A4C_Scar.pt"
    elif args.label == "Scar" and args.view == "A2C":
        weights_path = "/workspace/YOUR PATH/model_weights/A2C_Scar.pt"
    elif args.label == "Scar" and args.view == "PLAX":
        weights_path = "/workspace/YOUR PATH/model_weights/PLAX_Scar.pt"
    
    data_path = args.dataset    #update the manifest file when needed
    video_files = glob.glob(os.path.join(data_path, "*.avi"))
    manifest = pd.DataFrame({"filename": video_files})
    manifest["split"] = "test"
    manifest['filename']= manifest['filename'].apply(lambda x: os.path.basename(x))
    #If your dataset have a video with less than 32 frames, please remove it from the manifest file.
    manifest['frames']=  manifest["filename"].apply(lambda x: get_frame_count(os.path.join(args.dataset, f"{x}")))
    manifest = manifest[manifest['frames'] > 31]
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.csv")
    print(f"Manifest file was updated and saved to {manifest_path}")
    manifest.to_csv(manifest_path, index = False)
    
    #--------------------------------------------------
    print("Prediction LABEL: ", args.label)
   
    test_ds = EchoDataset(
        split="test",
        data_path=data_path,
        manifest_path=manifest_path,
        resize_res = (112, 112), #Input size 112x112 avi video
        )

    test_dl = DataLoader(
        test_ds, 
        num_workers=8,  
        batch_size=10, 
        drop_last=False, shuffle=False,
        )
    
    pretrained_weights = torch.load(weights_path)
    new_state_dict = {}
    for k, v in pretrained_weights.items():
        new_key = k[2:] if k.startswith('m.') else k
        new_state_dict[new_key] = v
        
    backbone = r2plus1d_18(num_classes=1)
    backbone.load_state_dict(new_state_dict, strict=True)
    backbone = backbone.to(device).eval()

    filenames = []
    predictions = []
    
    for batch in tqdm(test_dl):
        preds = backbone(batch["primary_input"].to(device))
        filenames.extend(batch["filename"])
        predictions.extend(preds.detach().cpu().squeeze(dim = 1))
    
    df_preds = pd.DataFrame({'filename': filenames, 'preds': predictions})
    manifest = manifest.merge(df_preds, on="filename", how="inner").drop_duplicates('filename')
    manifest.preds = manifest.preds.apply(sigmoid)
    
    manifest.to_csv(
        Path(os.path.dirname(os.path.abspath(__file__)))
        / Path(f"Prediction_{args.view}_{args.label}.csv"),
        index=False,
    )
    
    print(f"Predict Cardiac MRI Parameters -{args.label}- was done for {args.view}. \nSee Output csv and Calculate AUC using your label")
    
    
#SAMPLE SCRIPT
#python predict.py  --dataset YOUR_112*112_EchoDataset  --label ['Wall_Motion', 'Scar'] --view ['A4C', 'A2C', 'PLAX']