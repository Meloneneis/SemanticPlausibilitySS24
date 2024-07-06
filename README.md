# SemanticPlausibilitySS24

## Setting Up Project
The following commands will set up the project with a conda interpreter.
  ```
  git clone https://github.com/Meloneneis/SemanticPlausibilitySS24.git
  cd SemanticPlausibilitySS24
  conda create -n plausibility python=3.12 -y
  conda activate plausibility
  pip install -r requirements.txt -y
  ```

The project was tested on python version 3.12 with a RTX 3070 Ti (mobile) and 8GB VRAM

## Dataset Analysis
To run the dataset analysis, run the following command:
  ```
  python dataset_analysis.py
  ```

## Model Implementation
### Running the whole pipeline
The following steps are needed to fully reproduce the complete pipeline:
1. Go to model_implementation.py and change the following run parameters to these values:
```
skip_synthesizing_labels = False
preload_significance_values = False
skip_training_classifier = False

model_checkpoint = 'OpenAssistant/reward-model-deberta-v3-base'
version = 1
```
2. Run the model_implementation.py via:
````
python model_implementation.py
````
3. Go to synthesized_training.py and change the following run parameters to these values:
```
checkpoint = 'OpenAssistant/reward-model-deberta-v3-base'
skip_training = False  
version = 1  
evaluate_on_test_set = False
```
4. Run the synthesized_training.py via:
````
python synthesized_training.py
````
5. Go to model_implementation.py and change the following run parameters to these values:
```
skip_synthesizing_labels = False
preload_significance_values = False
skip_training_classifier = False

model_checkpoint = 'synthesized_model_v1'
version = 2
```
6. Run the model_implementation.py via:
````
python model_implementation.py
````
7. Go to synthesized_training.py and change the following run parameters to these values:
```
checkpoint = 'OpenAssistant/reward-model-deberta-v3-base'
skip_training = False  
version = 2
evaluate_on_test_set = True
```
8. Run the synthesized_training.py via:
````
python synthesized_training.py
````

Note that the whole pipeline took a couple of hours on the specified GPU.

### Skipping Parts of the Pipeline
Parts of the pipeline can be skipped via the skip/preload parameters in both synthesized_training.py and 
model_implementation.py. 

To skip model training, one has to download the models from here and extract the models in the models directory

For example, set the following run parameters to only evaluate the final model:
````
checkpoint = 'models/synthesized_model_v2'
skip_training = True  
evaluate_on_test_set = True
````