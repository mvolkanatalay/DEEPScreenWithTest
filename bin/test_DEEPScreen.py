import os
import sys
import cv2
import json
import torch
import random
import warnings
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn

from torch.autograd import Variable
from operator import itemgetter
from torch.utils.data import Dataset

# required for image generation
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions
from rdkit.Chem.Draw import rdMolDraw2D
#import cairosvg


# these are local from DEEPScreen
from evaluation_metrics import prec_rec_f1_acc_mcc, get_list_of_scores
from models import CNNModel1 
from data_processing import get_act_inact_list_for_a_target


project_file_path = "{}DEEPScreen".format(os.getcwd().split("DEEPScreen")[0])
training_files_path = "{}/training_files".format(project_file_path)
result_files_path = "{}/result_files".format(project_file_path)
trained_models_path = "{}/trained_models".format(project_file_path)
test_files_path = "{}/test_files".format(project_file_path)


# file path and name for test set, target protein chembl identifier, saved (trained) model filename
# chembl representations file 
CHEMBL_CHEMICAL_REPRESENTATIONS_FILENAME = "{}/chembl_32_chemreps.txt".format(training_files_path)
TARGET_ID = "CHEMBL210"  # we need target_id for TEST_DATASET_PATH
TEST_DATASET_PATH = "{}/{}/".format(test_files_path, TARGET_ID)


MODEL_FILENAME = "{}/my_chembl210_training/CHEMBL210_best_val-CHEMBL210-CNNModel1-256-128-0.01-64-0.25-100-my_chembl210_training-state_dict.pth".format(trained_models_path)
# model_filename can be input from user
INPUT_COMPOUND_FILENAME = "CHEMBL210_compounds.tsv"
TEST_JSON_FILENAME = "CHEMBL210_test.json"


# class for test dataset
class DEEPScreenTestDataset(Dataset):
    
# initialize the test set with the compound CHEMBL identifiers and their classes (labels-1 or 0)
    def __init__(self, target_id):
        self.target_id = target_id
        self.test_dataset_path = TEST_DATASET_PATH
        self.test_folds = json.load(open(os.path.join(self.test_dataset_path, TEST_JSON_FILENAME)))

        self.compid_list = [compid_label[0] for compid_label in self.test_folds["test"]]
        self.label_list = [compid_label[1] for compid_label in self.test_folds["test"]]
        print(self.compid_list)
        print(self.label_list) 
    

    def __len__(self):
        return len(self.compid_list)

# for a given compund extract its (previously generated) image
    def __getitem__(self, index):

        comp_id = self.compid_list[index]
        img_path = os.path.join(self.test_dataset_path, "imgs", "{}.png".format(comp_id))
        img_arr = cv2.imread(img_path)
        
        img_arr = np.array(img_arr) / 255.0
        img_arr = img_arr.transpose((2, 0, 1))
        label = self.label_list[index]

        return img_arr, label, comp_id
    




#  test set is extracted for a target protein 
def read_testdataset(): 
    
    target_id = TARGET_ID
    test_dataset = DEEPScreenTestDataset(target_id)
    test_loader = torch.utils.data.DataLoader(test_dataset)
    
    return(test_loader)



# getting predictions from a trained model by retrieving images of CHEMBL identifiers
def calculate_val_test_loss(model, criterion, data_loader, device):
    
    total_count = 0
    total_loss = 0.0
    all_comp_ids = []
    all_labels = []
    all_predictions = []

    # prediction for all compounds in test set through a for loop
    for i, data in enumerate(data_loader):
        img_arrs, labels, comp_ids = data
        img_arrs, labels = torch.tensor(img_arrs).type(torch.FloatTensor).to(device), torch.tensor(labels).to(device)
        total_count += len(comp_ids)
        y_pred = model(img_arrs).to(device)
#        loss = criterion(y_pred.squeeze(), labels)
#        total_loss += float(loss.item())  # is total loss necessary?
        all_comp_ids.extend(list(comp_ids))
        _, preds = torch.max(y_pred, 1)
        all_labels.extend(list(labels))
        all_predictions.extend(list(preds))


    return total_loss, total_count, all_comp_ids, all_labels, all_predictions


# test performance values if labels are available
def print_preformance_socres_predictions(test_perf_dict, all_test_comp_ids, all_test_labels, test_predictions):
    
    score_list = get_list_of_scores()
    
    for scr in score_list:
        print("Test {}:\t{}".format(scr, test_perf_dict[scr]))
    
    str_test_predictions = "CompoundID\tLabel\tPred\n"
    for ind in range(len(all_test_comp_ids)):
        str_test_predictions += "{}\t{}\t{}\n".format(all_test_comp_ids[ind],
                                                          all_test_labels[ind],
                                                          test_predictions[ind])
    print(str_test_predictions)



# compound image generation function
IMG_SIZE = 200

def generate_image_from_smiles(comp_id, smiles):
    # this function may write the image to a separate file 
    
    mol = Chem.MolFromSmiles(smiles)
    
    d = rdMolDraw2D.MolDraw2DCairo(IMG_SIZE, IMG_SIZE)
    d.drawOptions().bondLineWidth = 1
    d.DrawMolecule(mol)
    d.FinishDrawing()
    d.WriteDrawingText(os.path.join("{}/imgs/{}.png".format(TEST_DATASET_PATH, comp_id)))



# generate compound images from CHEMBL identifiers listed in a json file
# output images are saved in the same directory as the code
def generate_images_json_file(input_json_filename): 
    # chembl_32_chemreps.txt contains all compounds with their canonical smiles, inchi etc representations
    chem_reps_df = pd.read_csv(CHEMBL_CHEMICAL_REPRESENTATIONS_FILENAME, sep='\t')

    # each compound to be screened (tested) is in the TEST_JSON_FILENAME whose path is geven by TEST_DATASET_PATH 
    test_folds = json.load(open(input_json_filename))
    compound_id_list = [compid_label[0] for compid_label in test_folds["test"]]

    #search for the compound with comp_id inside dataframe, extract its smiles representation and generate compound image
    for comp_id in compound_id_list:
        try:
            current_df = chem_reps_df[chem_reps_df["chembl_id"] == comp_id]
            compound_smiles = current_df["canonical_smiles"].iloc[0]
            generate_image_from_smiles(comp_id, compound_smiles)
        except:
            pass  




# model hyperparameter values
fully_layer_1 = 256
fully_layer_2 = 128
drop_rate = 0.25

# main function for testing
# retrieve the trained model from file (state dictionary)
# load the test data (CHEMBL identifiers and labels are loaded)
# use the model to get predictions for the test data (model input: compound images and images are retrieved at this point)
# calculate the performance metric values wrt to the predictions and actual labels 
def test_DEEPScreen(target_id, filepath_model, test_filename):

    # retrieve the trained model from the filepath_model
    criterion = nn.CrossEntropyLoss()
    device = "cpu"
    model = CNNModel1(fully_layer_1, fully_layer_2, drop_rate).to(device)
    model.load_state_dict(torch.load(filepath_model))
    model.eval()
#    print(model)
    
    
    act_list, inact_list = get_act_inact_list_for_a_target(target_id, test_filename)
#    print("actives", act_list)
#    print("inactives", inact_list)

    # chembl_32_chemreps.txt contains all compounds with their canonical smiles, inchi etc representations
    chem_reps_df = pd.read_csv(CHEMBL_CHEMICAL_REPRESENTATIONS_FILENAME, sep='\t')

    # get the input test data from user file and make it ready for processing
    test_dictionary = dict()
    test_dictionary["test"] = []
    for comp_id in act_list:
        try:
            current_df = chem_reps_df[chem_reps_df["chembl_id"] == comp_id]
            compound_smiles = current_df["canonical_smiles"].iloc[0]
            generate_image_from_smiles(comp_id, compound_smiles)
            test_dictionary["test"].append([comp_id, 1])
        except:
            pass        
    with open(os.path.join(TEST_DATASET_PATH, TEST_JSON_FILENAME) , 'w') as fp:
        json.dump(test_dictionary, fp)


    # get the test data
    test_loader = read_testdataset()
    
    # use the model to get predictions for the test dataset
    with torch.no_grad():  # torch.set_grad_enabled(False):
        total_test_loss, total_test_count, all_test_comp_ids, all_test_labels, test_predictions = calculate_val_test_loss(model, criterion, test_loader, device)
#    print(test_predictions)
    
    # calculate the performance metric values wrt to the predictions and actual labels
    test_perf_dict = prec_rec_f1_acc_mcc(all_test_labels, test_predictions)
    print_preformance_socres_predictions(test_perf_dict, all_test_comp_ids, all_test_labels, test_predictions)

