# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import copy
import os
from tempfile import NamedTemporaryFile
import random
import yaml

import torch

from ikomia import core, dataprocess
from ikomia.utils import strtobool

from mmengine import Config
from mmocr.apis.inferencers import KIEInferencer
from mmocr.utils import register_all_modules

from infer_mmlab_kie.utils import polygon2bbox, get_classes, stitch_boxes_into_lines


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferMmlabKieParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.update = False
        self.model_name = "sdmgr"
        self.cfg = "sdmgr_novisual_60e_wildreceipt.py"
        self.config_file = ""
        self.model_weight_file = ""
        self.dict = ""
        self.class_file = ""
        self.merge_box = True
        self.max_x_dist = 1
        self.min_y_overlap_ratio = 0.6

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = strtobool(param_map["update"])
        self.model_name = param_map["model_name"]
        self.cfg = param_map["cfg"]
        self.config_file = param_map["config_file"]
        self.model_weight_file = param_map["model_weight_file"]
        self.dict = param_map["dict"]
        self.class_file = param_map["class_file"]
        self.merge_box = strtobool(param_map["merge_box"])
        self.max_x_dist = int(param_map["max_x_dist"])
        self.min_y_overlap_ratio = float(param_map["min_y_overlap_ratio"])

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "update": str(self.update),
            "model_name": self.model_name,
            "cfg": self.cfg,
            "config_file": self.config_file,
            "model_weight_file": self.model_weight_file,
            "dict": self.dict,
            "class_file": self.class_file,
            "merge_box": str(self.merge_box),
            "max_x_dist": str(self.max_x_dist),
            "min_y_overlap_ratio": str(self.min_y_overlap_ratio)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabKie(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.remove_input(1)
        self.add_input(dataprocess.CTextIO())
        self.add_output(dataprocess.CTextIO())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.kie_dataset = None
        self.classes = None
        self.colors = None

        register_all_modules()

        # Create parameters class.custom
        if param is None:
            self.set_param_object(InferMmlabKieParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    @staticmethod
    def get_model_zoo():
        configs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie")
        available_pairs = []
        for model_name in os.listdir(configs_folder):
            if model_name.startswith('_'):
                continue
            yaml_file = os.path.join(configs_folder, model_name, "metafile.yml")
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)
                    if 'Models' in models_list:
                        models_list = models_list['Models']
                    if not isinstance(models_list, list):
                        continue
                for model_dict in models_list:
                    available_pairs.append({"model_name": model_name, "cfg": os.path.basename(model_dict["Name"])})
        return available_pairs

    @staticmethod
    def get_absolute_paths(param):
        model_name = param.model_name
        model_config = param.cfg
        if param.model_weight_file == "":
            yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie", model_name,
                                     "metafile.yml")

            if model_config.endswith('.py'):
                model_config = model_config[:-3]
            if os.path.isfile(yaml_file):
                with open(yaml_file, "r") as f:
                    models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

                available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"],
                                                           'ckpt': model_dict["Weights"]}
                                      for model_dict in models_list}
                if model_config in available_cfg_ckpt:
                    cfg_file = available_cfg_ckpt[model_config]['cfg']
                    ckpt_file = available_cfg_ckpt[model_config]['ckpt']
                    cfg_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg_file)
                else:
                    raise Exception(
                        f"{model_config} does not exist for {model_name}. Available configs for are {', '.join(list(available_cfg_ckpt.keys()))}")
            else:
                raise Exception(f"Model name {model_name} does not exist.")

            return cfg_file, ckpt_file
        else:
            if param.config_file == "":
                raise Exception("If model_weight_file is set you must also set config_file (absolute path of the config file that goes with model weight)")
            return param.config_file, param.model_weight_file

    def _load_model(self):
        param = self.get_param_object()
        # Set cache dir in the algorithm folder to simplify deployment
        old_torch_hub = torch.hub.get_dir()
        torch.hub.set_dir(os.path.join(os.path.dirname(__file__), "models"))

        print("Loading KIE model...")
        cfg, ckpt = self.get_absolute_paths(param)
        cfg = Config.fromfile(filename=cfg)

        if os.path.isfile(param.class_file):
            self.classes = get_classes(param.class_file)
        else:
            print("Class file ({}) can't be opened, defaulting to Wildreceipt one's".format(param.class_file))
            class_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildreceipt", "class_list.txt")
            self.classes = get_classes(class_file)

        if os.path.isfile(param.dict):
            cfg.model.dictionary.dict_file = param.dict
        else:
            print("Dict file ({}) can't be opened, defaulting to the one in the config file".format(param.dict))

        # Config object cannot be used to instantiate models, so we write it into a temporary file
        temp = NamedTemporaryFile(suffix='.py', delete=False)
        cfg.dump(temp.name)
        cfg = temp.name

        self.model = KIEInferencer(cfg, ckpt, self.device)

        random.seed(0)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]

        param.update = False
        # Reset torch cache dir for next algorithms in the workflow
        torch.hub.set_dir(old_torch_hub)
        print("Model loaded!")

    def init_long_process(self):
        self._load_model()

    def run(self):
        # Core function of your process
        # Call begin_task_run for initializationyour config file
        self.begin_task_run()
        param = self.get_param_object()

        # Get inputs
        img_input = self.get_input(0)
        text_input = self.get_input(1)
        input_items = text_input.get_text_fields()

        # Get outputs
        output = self.get_output(1)

        # Load models into memory
        if param.update:
            self._load_model()

        src_img = img_input.get_image()
        if src_img is not None and input_items is not None:
            annotations = []
            summed_h = 0

            for item in input_items:
                record = {}
                pts = []

                for p in item.polygon:
                    pts.append(p.x)
                    pts.append(p.y)

                x1, y1, x2, y2 = polygon2bbox(pts)
                record['bbox'] = [x1, y1, x2, y2]
                summed_h += y2 - y1
                record['text'] = item.text
                annotations.append(record)

            # merging boxes if needed
            if param.merge_box:
                # mean height of detected boxes gives a rough approximation of the width of an escaping character
                mean_h = int(summed_h / len(input_items))
                if param.max_x_dist < 0:
                    annotations = stitch_boxes_into_lines(annotations, max_x_dist=mean_h/4,
                                                          min_y_overlap_ratio=param.min_y_overlap_ratio, sep='')
                    annotations = stitch_boxes_into_lines(annotations, max_x_dist=mean_h,
                                                          min_y_overlap_ratio=param.min_y_overlap_ratio, sep=' ')
                else:
                    annotations = stitch_boxes_into_lines(annotations, max_x_dist=param.max_x_dist,
                                                          min_y_overlap_ratio=param.min_y_overlap_ratio, sep=' ')
            out = self.model({'img': src_img, 'instances': annotations})
            self.visualize_kie(out, output, annotations)

        self.forward_input_image(0, 0)
        # Call end_task_run to finalize process
        self.end_task_run()

    def visualize_kie(self, results, output, annotations):
        labels = results['predictions'][0]['labels']
        scores = results['predictions'][0]['scores']
        texts = [anno['text'] for anno in annotations]
        boxes = [anno['bbox'] for anno in annotations]

        for i, (label, score, text, box) in enumerate(zip(labels, scores, texts, boxes)):
            cls = self.classes[label]
            x, y, x2, y2 = box
            x, y, w, h = float(x), float(y), float(x2 - x), float(y2 - y)
            output.add_text_field(i, cls, text, score, x, y, w, h, self.colors[label])

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabKieFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_kie"
        self.info.short_description = "Inference for MMOCR from MMLAB KIE models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Text"
        self.info.version = "2.1.0"
        self.info.max_python_version = "3.9"
        self.info.max_python_version = "3.11"
        self.info.min_ikomia_version = "0.15.0"
        self.info.icon_path = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.original_repository = "https://github.com/open-mmlab/mmocr"

        self.info.repository = "https://github.com/Ikomia-hub/infer_mmlab_kie"
        # Keywords used for search
        self.info.keywords = "infer, key, information, extraction, kie, mmlab, sdmgr"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "OCR"
        self.info.hardware_config.min_cpu = 4
        self.info.hardware_config.min_ram = 8
        self.info.hardware_config.gpu_required = False
        self.info.hardware_config.min_vram = 6

    def create(self, param=None):
        # Create process object
        return InferMmlabKie(self.info.name, param)
