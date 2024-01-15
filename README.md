<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_kie/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">infer_mmlab_kie</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_mmlab_kie">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_mmlab_kie">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_mmlab_kie/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_mmlab_kie.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Run KIE (Key Information Extraction) algorithms from MMLAB framework. This algorithm will be applied after text detection and text recognition. You can use ***infer_mmlab_text_detection*** and ***infer_mmlab_text_recognition*** from Ikomia HUB for this task.

Models will come from MMLAB's model zoo if custom training is disabled. If not, you can choose to load your model trained with algorithm *train_mmlab_kie* from Ikomia HUB.

![Example image](https://raw.githubusercontent.com/Ikomia-hub/infer_mmlab_kie/main/icons/result.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithms...
# for text detection
det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)
# for text recognition
rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)
# for kie
kie = wf.add_task(name="infer_mmlab_kie", auto_connect=True)

# Run on your image
wf.run_on(url="https://github.com/open-mmlab/mmocr/blob/main/demo/demo_kie.jpeg?raw=true")

# Get results
original_image_output = kie.get_output(0)
text_detection_output = kie.get_output(1)

# Display results
display(original_image_output.get_image_with_graphics(text_detection_output))
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithms...
# for text detection
det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)
# for text recognition
rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)
# for kie
kie = wf.add_task(name="infer_mmlab_kie", auto_connect=True)

# Set parameters
kie.set_parameters({
    'model_name': 'sdmgr', 
    'cfg': 'sdmgr_unet16_60e_wildreceipt'})

# Run on your image
wf.run_on(url="https://github.com/open-mmlab/mmocr/blob/main/demo/demo_kie.jpeg?raw=true")

# Get results
original_image_output = kie.get_output(0)
text_detection_output = kie.get_output(1)
```
- **model_name** (str, default="satrn"): model name. 
- **cfg** (str, default="satrn_shallow-small_5e_st_mj"): name of the model configuration file.
- **config_file** (str, optional): path to model config file (only if *custom_training=True*). The file is generated at the end of a custom training. Use algorithm ***train_mmlab_text_recognition*** from Ikomia HUB to train custom model.
- **model_weight_file** (str, optional): path to model weights file (.pt) (only if *custom_training=True*). The file is generated at the end of a custom training.
- **dict_file** (str, default="dicts/english_digits_symbols.txt"): characters dictionary. Set it when you use a custom train.
- **class_file** (str, default="wildreceipt/class_list.txt"): Class list. Set it when you use a custom train.
- **merge_box** (bool, default=True): Merge text boxes before running KIE algorithm. 
- **max_x_dist** (int, default=-1): Used if **merge_box** is True. Text boxes closer (on x-axis) than this value are merged. If **max_x_dist** est lower than 0, it will automatically calculate this value based on the mean height of all text boxes in the input. It will first perform a merging with a maximum distance equal to a quarter of mean height, joining boxes with '', then perform a second merging with maximum distance equal to mean height, joining boxes with ' '.
- **min_y_overlap_ratio** (float, default=0.6): Used if **merge_box** is True. Text boxes can be merged if they overlap on y-axis more than this ratio.

MMLab framework offers multiple models. To ease the choice of couple (model_name/cfg), you can call the function *get_model_zoo()* to get a list of possible values.

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add kie algorithm
kie = wf.add_task(name="infer_mmlab_kie", auto_connect=True)

# Get list of possible models (model_name, model_config)
print(kie.get_model_zoo())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithms...
# for text detection
det = wf.add_task(name="infer_mmlab_text_detection", auto_connect=True)
# for text recognition
rec = wf.add_task(name="infer_mmlab_text_recognition", auto_connect=True)
# for kie
kie = wf.add_task(name="infer_mmlab_kie", auto_connect=True)

# Run on your image
wf.run_on(url="https://github.com/open-mmlab/mmocr/blob/main/demo/demo_kie.jpeg?raw=true")

# Iterate over outputs
for output in kie.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```

MMLab text recognition algorithm generates 2 outputs:

1. Forwarded original image (CImageIO)
2. Text detection output (CTextIO)