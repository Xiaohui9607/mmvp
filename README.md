# A Framework for Multisensory Foresight for Embodied Agents

**Abstract**
>Predicting future sensory states is crucial for learning agents such as robots, drones, and autonomous vehicles. In this paper, we couple multiple sensory modalities with exploratory actions and propose a predictive neural network architecture to address this problem. Most existing approaches rely on large, manually annotated datasets, or only use visual data as a single modality. In contrast, the unsupervised method presented here uses multi-modal perceptions for predicting future visual frames. As a result, the proposed model is more comprehensive and can better capture the spatio-temporal dynamics of the environment, leading to more accurate visual frame prediction. The other novelty of our framework is the use of sub-networks dedicated to anticipating future haptic, audio, and tactile signals. The framework was tested and validated with a dataset containing 4 sensory modalities (vision, haptic, audio, and tactile) on a humanoid robot performing 9 behaviors multiple times on a large set of objects. While the visual information is the dominant modality, utilizing the additional non-visual modalities improves the accuracy of predictions.


## Environment Setup
pip install -r requirements.txt

## Dataset Preparation
Description: https://www.eecs.tufts.edu/~ramtin/pages/2014/CY101Dataset.html

Download: https://tufts.app.box.com/v/DeepMultiSensoryDataset

Preparation:
```bash
$ python ./data/make_data.py \
        --data_dir path-to-downloaded-data-directory \ 
        --out_dir path-to-output-data-directory \
```


## Usage
```bash
$ python ./main.py  \
        --data_dir path-to-data\ # directory containing data
        --channels 3 \ # channels of input
        --height 64 \ # height of image
        --width 64 \ # width of image
        --output_dir path-to-checkpoint-dir \ # directory for model weight
        --pretrained_model path-to-checkpoint \ # filepath of a pretrained model to initialize from
        --sequence_length 10 \ # sequence length + context frames
        --context_frames 4 \ # the number of ground truth frames to pass in at start
        --model CDNA \ # model architecture to use - CDNA | DNA | STP
        --num_masks 10 \ # number of masks, usually 1 for DNA, 10 for CDNA, STP
        --device cuda  \ # device cuda | cpu
        --cdna_kern_size 5 \ # CDNA_KERN_SIZE
        --haptic_layer 16 \ # HAPTIC_LAYER
        --use_haptic \ # Whether or not to give the haptic to the model
        --behavior_layer 9 \ # number of chosen behaviors
        --use_behavior \ # Whether or not to give the behavior to the model
        --audio_layer 16 \ # AUDIO_LAYER
        --use_audio \ # Whether or not to give the audio to the model
        --vibro_layer 16 \ # VIBRO_LAYER
        --use_vibro \ # Whether or not to give the vibro to the model
        --aux \ # Whether or not to employ auxiliary tasks during training 
        --print_interval 100 \ # iterations to output loss
        --schedsamp_k  400 \ # The k hyperparameter for scheduled sampling, -1 for no scheduled sampling
        --batch_size 32 \ # batch size for training
        --learning_rate 0.001 \ # the base learning rate
        --epochs 30 \ # total training epochs
```
## Architecture
<img src="Figures/model_architecture_figures/ModelArch.png" alt="drawing" width="600px"/> 
<p>The architecture of the proposed model, which consists of 4 feature encoders (left) and prediction heads (right) for 4 modalities, and 1 fusion module (middle) for merging representations of different modalities.</p>

<img src="Figures/model_architecture_figures/Vis_feat.png" alt="drawing" width="300px"/> <img src="Figures/model_architecture_figures/Vis_pred.png" alt="drawing" width="300px"/> 
<p>Pipeline of The Visual Prediction Module: the architecture of visual feature extractor (left), the architecture of visual prediction network (right).</p>

## Results

### Dataset Visualization

<img src="Figures/dataset_visualization/haptic_drop_can_coke.png" alt="drawing" width="250px"/> <img src="Figures/dataset_visualization/audio_spectrogram_drop_can_coke.png" alt="drawing" width="250px"/> <img src="Figures/dataset_visualization/vibro_drop_can_coke.png" alt="drawing" width="250px"/>
<p>Visualization of haptic (left), audio (middle) and vibrotactile (right) modalities when the robot drops a bottle</p>

### Training the Network with All Behaviors

#### Illustrative Example

<img src="Figures/all_pred_lift_behavior.png" alt="drawing" width="400px"/> <img src="Figures/all_pred_push_behavior.png" alt="drawing" width="400px"/>
<p>Sharpness of predicted images, when the robot arm perform lift (left) and push (right) behaviors.

#### Quantitative Reconstruction Performance

<img src="Figures/all_with_std.png" alt="drawing" width="600px"/> 
<p>Ablation study on sensory input</p>

<img src="Figures/ablation_on_behavior.png" alt="drawing" width="600px"/>
<p>Ablation study on behavior input</p>

<img src="Figures/tables/table 1.png" alt="drawing" width="600px"/>


#### Training the Network with Individual Behavior

<img src="Figures/separate_behavior_group_bar.png" alt="drawing" width="600px"/>
<p>Investigating the performance of different combinations of modalities per individual behavior</p>

#### Predicting Future Frames of Auxiliary Modalities

<img src="Figures/tables/table 2.png" alt="drawing" width="600px"/>

