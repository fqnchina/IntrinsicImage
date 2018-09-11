# Revisiting Deep Intrinsic Image Decompositions
This is the implementation of CVPR 2018 Oral paper *"Revisiting Deep Intrinsic Image Decompositions"* by Qingnan Fan *et al.*.

Check my [homepage](<https://fqnchina.github.io/>) for the paper, supplemental material, slides and the other info.

Compilation
----

Our codes are implemented in Torch framework. Some self-defined layers are included in the '**compile**' folder. Before testing or training the models, you need to install the latest torch framework and compile the ***.lua** under the nn module, the ***.cu** and ***.h** under the cunn module, and **adam_state.lua** under optim module.

To be specific, the lua files have to be put in **./torch/extra/nn/** module directory, and editing **init.lua** file to include the corresponding file. Finally run the command under nn directory, **luarocks make ./rocks/nn-scm-1.rockspec**. Then nn module will be independently compiled. 

Similarly for the cuda files, they need to be put into **./torch/extra/cunn/lib/THCUNN/** module directory, then editing the **./torch/extra/cunn/lib/THCUNN/generic/THCUNN.h** to include the corresponding files, and finally run the command **luarocks make ./rocks/cunn-scm-1.rockspec** under **./torch/extra/cunn** folder to compile them. 

Accordingly, the adam lua file has to be put in **./torch/pkg/optim**, edit the **init.lua** file and then run **luarocks make optim-1.0.5-0.rockspec**.

Data Generation
----

The training data for IIW can be directly obtained from their official [project page](<https://labelmaterial.s3.amazonaws.com/release/iiw-dataset-release-0.zip>). The training data for the guidance network of IIW data is L1smooth results, which is shared from [this link](<https://drive.google.com/file/d/0B_bZKF86bTHLb0NrNm1NRTZJUUk/view?usp=sharing>). Note the IIW label is transfered by us into txt format for training, we include these IIW txt label files into [this link](<https://drive.google.com/open?id=1d5GcCnl9H2k_anJftRRm11xqckrwVCH2>). As for MIT and MPI-Sintel dataset, whose data sources may be inconvinient to obtain, we also include them in the last link. 

Run **gen_trainData_MPI.m** to crop and flip the MPI images to generate our training data. The input images to our network have to cropped to image size of even number as we downsample the intermediate feature maps by half.

The sparse label of IIW dataset is in json file format originally, we extract the useful information and transfer them to txt file. You can either transfer the labels using **transferIIW.m** or directly use the transfered files in our released dataset.

The training and testing file lists are included into the .txt files in the **dataset** folder.

Training
----

The trained models are all in the **netfiles** folder.

To train the models from scratch, run the files in the **training** folder. 

Since we levearge two-fold cross validation for the MPI dataset, training on both the train and test split of MPI data is required.

Evaluation
----
To test our trained model, run **test_*.lua** files, while to evaluate the numerical errors, run **compute_*_error** files.

Cite
----

You can use our codes for research purpose only. And please cite our paper when you use our codes.
```
@article{fan2018revisiting,
  title={Revisiting Deep Intrinsic Image Decompositions},
  author={Fan, Qingnan and Yang, Jiaolong and Hua, Gang and Chen, Baoquan and Wipf, David},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  year={2018}
}
```
Contact
-------

If you find any bugs or have any ideas of optimizing these codes, please contact me via fqnchina [at] gmail [dot] com


