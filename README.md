# TargetDIA

Generate a DIA hybrid spectral library combining an initial DIA library and a targeted virtual library for DIA data processing.

### Content

1. [Requirements](#1)

2. [Pipeline](#2)
    + 2.1 [Prepare an initial DIA library](#2.1)

    + 2.2 [Re-train the deep learning models](#2.2)

    + 2.3 [Generate a targeted virtual library](#2.3)

    + 2.4 [Generate a DIA hybrid library](#2.4)

    + 2.5 [Search DIA data with the hybrid library](#2.5)

3. [License](#3)

4. [Publication](#4)

5. [Contact](#5)

## 1. <span id="1">Requirements</span>

+ Latest version of anaconda is recommended.

+ A NVIDIA GPU with CUDA (10.0) and cuDNN (7.5) is needed.
   * PyTorch 0.4.1 from conda has CUDA support up to 9.0, but it also works in CUDA 10.0 for DeepRT required functions (many new cards only supports CUDA 10+, like RTX2080)

+ The main packages we used in the whole project are:

    1. Python 3.7.3

    2. TensorFlow-gpu 1.13.1 (TensorFlow 2.x is not recommended)

    3. PyTorch 0.4.1 (PyTorch 1.x is not recommended)

+ [pDeep2](https://github.com/pFindStudio/pDeep/tree/master/pDeep2) and [DeepRT](https://github.com/horsepurve/DeepRTplus) are two models for the generation of a targeted virtual library.

+ [Spectronaut](https://biognosys.com/shop/spectronaut) v12 is needed.

    + v11 is not recommended because it doesn't support dDIA and some functions for spectral library is not as good as v12.

## 2. <span id="2">Pipeline</span>

### 2.1 <span id="2.1">Prepare an initial DIA library</span>

To make more accurate prediction based on the in-house experimental data, the existing (pre-trained) models should be refined. Here we use an initial DIA spectral library to do this.

1. Statistics for the whole initial DIA library, contains:

    + Maximum and minimum length of stripped peptides, including their distribution.
    
    + Miss cleavage distribution of stripped peptides.
    
    + Charge distribution of peptide precursors.
    
    + Oxidation on methionine of modified peptides.

    ```python
    # The library statistics can be reported by spectronaut directly
    # To plot customized chart, some basic info can be read as below
      
    from scripts.utils.spectronaut import SpectronautLibrary as SNLib
    
    lib_path = '/path/of/library'
    sn = SNLib()
    sn.set_library(lib_path)
    lib_basic_info = sn.library_basic_info()  # A dict stores number of protein groups, modpep, stripped pep
    pep_length = sn.strippep_length_distrib()  # A pd series with key-value pair as 'length: number'
    mc = sn.strippep_miss_cleavage_distrib()  # Miss cleavage. A pd series with key-value pair as 'mc: number'
    charge = sn.prec_charge_distrib()  # A pd series with key-value pair as 'charge: number'
    ```

2. Filter the initial library to retain high-quality data for further use.
   
    + Peptide length ranging from 7 to 30 is an universal set but a much stricter rule can also be used for peptide length ranging from 7 to a maximum length which covers about 90% or 95% of the whole peptides.
    
    + The number of fragments in each PSM should be more than or equal to 6. Notice that the fragment number only counts fragment with no neutral loss.
    
    + The charge of peptide precursors should have a limitation such as 1-5 or 2-4.

    ```shell script
    # use python ./scripts/filter_library.py --help to get the help text
    # An example usage:
    python ./scripts/filter_library.py -l /library/path -o /output/path -min 7 -max 30 -p 6 -c 1,2,3,4,5
    ```

### 2.2 <span id="2.2">Re-train the deep learning models</span>

1. Split the filtered initial library to training and test dataset. (split based on a certain protein set or randomly split at a certain ratio)

    + Split the library based on the targeted protein family.

        ```shell script
        # use python ./scripts/generate_dataset.py --help to get the help text
        # An example usage:
        python ./scripts/generate_dataset.py -l /library/path -o /output/dir/ -t target/file/path
        
        # Notice that output dir is not a certern file path
        # target file is a pure text file contains targeted protein accession in one single column with no title
        ```

    + Split the library with a certain ratio.

        + For pDeep2, PSMs are needed. Then the library can be split by peptide precursors.

        + For DeepRT, modified peptides are needed. The library can be split by modified peptides.

        ```shell script
        # use python ./scripts/generate_dataset.py --help to get the help text
        # An example usage:
        python ./scripts/generate_dataset.py -l /library/path -o /output/dir/ -r 8:2
        
        # Notice that output dir is not a certern file path
        # -r means the split ratio of train and test
        ```

2. Fine-tune the pre-trained models.

    + Follow procedures in the repositories of [pDeep2](https://github.com/pFindStudio/pDeep/tree/master/pDeep2) and [DeepRT](https://github.com/horsepurve/DeepRTplus).
    
    + Or use a modified config file-based fine-tuning/prediction way stored in this repository
    
      + First, run two .py files in model workspace (`models_workspace`) to generate config files in JSON format. Before runing them, filling-in some values of the .py files for both pdeep2 and deeprt_plus is optional, since the config files can be edited later.
    
      + Then, check the generated JSON config files
    
      + For pDeep2 (a subset of config)
    
        ```json
        {
            "TrainsetFolder": "/path/to/just/generated/data/folder",  // The folder stores pDeep2 training data. This can be directed to the folder generated in the last step
            "ModelOutFolder": "/store/new/model/param/in/this/folder/",  // Where to store the generated pDeep2 model. Notice that this need a folder to store plural files but not a file path
            "ModelOutName": "A_fine_tuned_model_param",  // Define the generated model name
            "PretrainedModelPath": "/path/to/pDeep2/pretrained_models/pretrain-180921-modloss/or_others",  // Use this model param as the pre-trained model param. Notice this should be a pure base name with no suffix. All pre-trained model params are stored in pretrained_models/model-180921-modloss in pDeep2 model folder
            "TensorboardFolder": "/store/tensorboard/file/in/this/folder/"  // Folder for tensorboard
        }
        ```
    
      + For DeepRTPlus
    
        ```json
        {
            "PATH_PretrainModel": "/path/to/deeprt_plus/param/dia_all_epo20_dim24_conv8/dia_all_epo20_dim24_conv8_filled.pt-or_others",  // Define the path of pre-trained DeepRT model param. Three params are stored in param/dia_all_epo20_dim24_conv[8,10,12]
            "Conv_Train": 8,  // Define the conv kernal size. This should be same as the pre-trained model param used
            "PATH_TrainSet": "/path/to/just/generated/data",  // Define the training data file for DeepRT
            "PATH_TrainResult": "/path/of/training/data/result",  // The output path of training result. This will contain the input RTs final predicted RTs for further analysis
            "PATH_SavePrefix": "/path/to/store/generated/model/params",  // A folder to store the generated DeepRT model params
            "PATH_TestSet": "/path/to/test/data",  // File for model test
            "PATH_Log": "/path/to/store/log/file"  // Where to store the training log
        }
        ```
    
      + Switch `task` to `training` for `deeprt.py` and `pdeep2.py` in scripts
    
      + Fillin the `param_path` for `deeprt.py` and `pdeep2.py` and run `python deeprt.py` / `python pdeep2.py`

### 2.3 <span id="2.3">Generate a targeted virtual library</span>


Use of a targeted virtual library in DIA data processing allows much increased sensitivity of mapping the targeted protein family members without compromising the FDR control.

1. Prepare fasta file for the targeted protein family members.

2. Theoretical enzyme digestion (In-silico digestion).

    + Set the maximum and minimum lengths.
    
    + Set the maximum number of miss cleavage.

3. Predict ion intensity and retention time.

    + To determine whether the oxidation on methionine is needed.

    + For ion intensity, the charge state of peptide precursors is needed. Here, charge 2 and 3 for one peptide is recommended.

    ```shell script
    # Step 2 and step 3 will be done in one step
    # use python ./scripts/input_from_fasta.py --help to get the help text
       
    # An example usage:
    python ./scripts/input_from_fasta.py -f /fasta/path -o /output/dir/ -min 7 -max 30 -mc 2 -c 2,3 -oxi 0
    
    # Notice that output dir is not a certern file path
    # -min and -max are minimum and maximum peptide length respectively
    # -mc is an integer for max miss cleavage number
    # -c is charge state of precursor which is integer with ',' as a delimiter
    # -oxi is the oxidation on methionine and this can be 0 means no oxidation or 1 means less than or equal to 1
    ```

    + Predict the input files generated above with fine-tuned models.

      + Follow procedures in the repositories of [pDeep2](https://github.com/pFindStudio/pDeep/tree/master/pDeep2) and [DeepRT](https://github.com/horsepurve/DeepRTplus).

      + Or use a modified config file-based fine-tuning/prediction way stored in this repository

        + Similar to the training step, some values in the config files should be edited

        + For pDeep2

          ```json
          {
              "ModelForPredFolder": "/stored/new/model/param",  // The folder contains model param used for prediction. This can be set to the same value as "ModelOutFolder" for convenience
              "ModelForPredName": "A_fine_tuned_model_param",  // Use which model stored in "ModelForPredFolder". This can be set to the same value as "ModelOutName"
              "PredInputPath": "/path/to/just/generated/input/file",  // The path of prediction input file
              "PredOutPath": "/path/to/output/prediction/result"  // Where to output the prediction result file
          }
          ```

        + For DeepRTPlus

          ```json
          {
              "PATH_TrainSet": "/path/to/just/generated/data",  // Define the training data file for DeepRT. In the design of DeepRT, a train set is not only for training, but also as a dictionary to get tokens (used aa). Though in this modified version, to keep the consensus among multi uses, AA_List is defined as a key in config file, the original loading step is not modified, so this path should also be defined
              "PATH_TestSet": "/path/of/prediction/input/file",  // Different with the training step, this value becomes the prediction input now
              "PATH_Pred_Output": "/path/to/output/prediction/result",  // Where to store the prediction result file
              "PATH_R1_Model": "",  // A model param with 8 conv kernal size. Notice, if only this one model param is defined, the prediction will be performed with this model param only. If the following two are also defined, the prediction will be based on all three model params and ensemble will be performed (an average of prediction results)
              "PATH_R2_Model": "",  // A model param with 10 conv kernal size
              "PATH_R3_Model": "",  // A model param with 12 conv kernal size
              "Conv_R1_Model": 8,
              "Conv_R2_Model": 10,
              "Conv_R3_Model": 12
          }
          ```

        + Switch `task` to `predict` for `deeprt.py` and `pdeep2.py` in scripts

        + Fillin the `param_path` for `deeprt.py` and `pdeep2.py` and run `python deeprt.py` / `python pdeep2.py`

4. Combine ion intensity and retention time to generate a virtual library.

    ```shell script
    # use python ./scripts/merge_library.py --help to get the help text
    # An example usage:
    python ./scripts/merge_library.py -ion /pdeep/prediction/path -rt /deeprt/prediction/path -o /output/path
    ```

### 2.4 <span id="2.4">Generate a DIA hybrid library</span>

+ A hybrid library is generated by combining the initial DIA library and the targeted virtual library.

    ```shell script
    # use python ./scripts/merge_library.py --help to get the help text
    # An example usage:
    python ./scripts/merge_library.py -dia /dia/library/path -vir /virtual/library/path -o /output/path
    ```

### 2.5 <span id="2.5">Search DIA data with the hybrid library</span>

+ First search the experimental DIA dataset with the hybrid library using Spectronaut in default settings.

+ BGS Factory Report (default) under Normal Report is exported.

+ For peptide and protein identification and quantification, a customized Peptide Quant under Run Pivot Report is needed. The typically selected columns are EG.PrecursorId, PEP.StrippedSequence, PG.Cscore, PG.ProteinAccessions, PG.Qvalue, and EG.TotalQuantity (Settings).

+ When necessary, filter the customized report file at a higher stringency (e.g. Cscore >0.9) to reduce error rates in protein/peptide IDs with the hybrid library.

## 3 <span id="3">License</span>

For the two predictors:

  + pDeep2 is under a three-clause BSD license. we obey the license and added the license into the LICENSE file in this repository.

  + DeepRT is under an MIT license. We obey the license and added the license into the LICENSE file in this repository.

This pipeline (i.e. TargetDIA) is under a BSD 3-Clause License. Everyone please feel free to use it.

## 4 <span id="4">Publication</span>

Ronghui Lou, Pan Tang, Kang Ding, Shanshan Li, Cuiping Tian, Yunxia Li, Suwen Zhao, Yaoyang Zhang, Wenqing Shui. 
[Hybrid Spectral Library Combining DIA-MS Data and a Targeted Virtual Library Substantially Deepens the Proteome Coverage](https://www.sciencedirect.com/science/article/pii/S2589004220300870). 
iScience, Volume 23, Issue 3, 27 March 2020, 100903, [doi: 10.1016/j.isci.2020.100903](https://doi.org/10.1016/j.isci.2020.100903).

## 5 <span id="5">Contact</span>

+ Any issues or suggestions about this pipeline, please create an issue in this repository directly and we will reply once we notice that.

+ For other questions, please contact to shuiwq@shanghaitech.edu.cn.