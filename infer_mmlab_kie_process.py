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
from ikomia import core, dataprocess
import copy
# Your imports below
from mmengine import Config
import torch
from infer_mmlab_kie.utils import polygon2bbox, get_classes, stitch_boxes_into_lines
import os
import distutils
import json
from mmocr.apis.inferencers import KIEInferencer
from tempfile import NamedTemporaryFile
from mmocr.utils import register_all_modules
import random


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
        self.weights = "https://download.openmmlab.com/mmocr/kie/sdmgr/" \
                       "sdmgr_novisual_60e_wildreceipt_20210405-07bc26ad.pth"
        self.custom_training = False
        self.dict = ""
        self.class_file = ""
        self.merge_box = True
        self.max_x_dist = 1
        self.min_y_overlap_ratio = 0.6
        self.custom_cfg = ""
        self.custom_weights = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.update = distutils.util.strtobool(param_map["update"])
        self.model_name = param_map["model_name"]

        self.cfg = param_map["cfg"]
        self.weights = param_map["weights"]
        self.custom_training = distutils.util.strtobool(param_map["custom_training"])
        self.dict = param_map["dict"]
        self.class_file = param_map["class_file"]
        self.merge_box = distutils.util.strtobool(param_map["merge_box"])
        self.max_x_dist = int(param_map["max_x_dist"])
        self.min_y_overlap_ratio = float(param_map["min_y_overlap_ratio"])
        self.custom_cfg = param_map["custom_cfg"]
        self.custom_weights = param_map["custom_weights"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["update"] = str(self.update)
        param_map["model_name"] = self.model_name
        param_map["cfg"] = self.cfg
        param_map["weights"] = self.weights
        param_map["custom_training"] = str(self.custom_training)
        param_map["dict"] = self.dict
        param_map["class_file"] = self.class_file
        param_map["merge_box"] = str(self.merge_box)
        param_map["max_x_dist"] = str(self.max_x_dist)
        param_map["min_y_overlap_ratio"] = str(self.min_y_overlap_ratio)
        param_map["custom_cfg"] = self.custom_cfg
        param_map["custom_weights"] = self.custom_weights
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferMmlabKie(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        self.addOutput(dataprocess.CObjectDetectionIO())
        self.addOutput(dataprocess.DataDictIO())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.kie_dataset = None
        self.classes = None
        self.colors = None

        register_all_modules()
        # Create parameters class
        if param is None:
            self.setParam(InferMmlabKieParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        param = self.getParam()

        # Get inputs
        input = self.getInput(0)
        graphics_input = self.getInput(1)
        input_items = graphics_input.getItems()

        # Get outputs
        output = self.getOutput(1)
        dict_output = self.getOutput(2)

        # Load models into memory
        if self.model is None or param.update:
            print("Loading KIE model...")
            if not param.custom_training:
                cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie",
                                   param.model_name, param.cfg)
                cfg = Config.fromfile(filename=cfg)
                class_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildreceipt",
                                          "class_list.txt")
                dict_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildreceipt",
                                         "dict.txt")
                cfg.model.dictionary.dict_file = dict_file
                ckpt = param.weights
                self.classes = get_classes(class_file)
            else:
                cfg = Config.fromfile(filename=param.custom_cfg)
                ckpt = param.custom_weights
                if os.path.isfile(param.class_file):
                    self.classes = get_classes(param.class_file)
                else:
                    print("Class file ({}) can't be opened, defaulting to Wildreceipt one's".format(param.class_file))
                    class_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wildreceipt",
                                              "class_list.txt")
                    self.classes = get_classes(class_file)
                if os.path.isfile(param.dict):
                    cfg.model.dictionary.dict_file = param.dict
                else:
                    print("Dict file ({}) can't be opened, defaulting to the one in the config file".format(param.dict))

            if 'visualizer' in cfg:
                cfg.pop('visualizer')

            # Config object cannot be used to instantiate models, so we write it into a temporary file
            temp = NamedTemporaryFile(suffix='.py')
            cfg.dump(temp.name)

            self.model = KIEInferencer(temp.name, ckpt, self.device)

            random.seed(0)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]

            param.update = False
            print("Model loaded!")

        srcImg = input.getImage()

        if srcImg is not None and input_items is not None:
            annotations = []
            summed_h = 0
            for item in input_items:
                record = {}
                pts = []
                for p in item.points:
                    pts.append(p.x)
                    pts.append(p.y)
                x1, y1, x2, y2 = polygon2bbox(pts)
                record['bbox'] = [x1, y1, x2, y2]
                summed_h += y2 - y1
                record['text'] = item.getCategory()
                annotations.append(record)

            # merging boxes if needed
            if param.merge_box:
                # mean height of detected boxes gives a rough approximation of the width of an escaping character
                mean_h = int(summed_h / len(input_items))
                annotations = stitch_boxes_into_lines(annotations, max_x_dist=mean_h/4,
                                                      min_y_overlap_ratio=param.min_y_overlap_ratio, sep='')
                annotations = stitch_boxes_into_lines(annotations, max_x_dist=mean_h,
                                                      min_y_overlap_ratio=param.min_y_overlap_ratio, sep=' ')

            out = self.model({'img': srcImg, 'instances': annotations})
            self.visualize_kie(out, output, dict_output, annotations)

        self.forwardInputImage(0, 0)
        # Call endTaskRun to finalize process
        self.endTaskRun()

    def visualize_kie(self, results, output, dict_output, annotations):
        labels = results['labels']
        scores = results['scores']
        texts = [anno['text'] for anno in annotations]
        boxes = [anno['bbox'] for anno in annotations]
        out_dict = []
        for i, (label, score, text, box) in enumerate(zip(labels, scores, texts, boxes)):
            cls = self.classes[label]
            x, y, x2, y2 = box
            x, y, w, h = float(x), float(y), float(x2 - x), float(y2 - y)
            output.addObject(i, cls, score, x, y, w, h,
                             self.colors[label])
            out_dict.append({'text': text, 'cls': cls, 'score': score, 'bbox': [x, y, x, h]})
        dict_output.fromJson(json.dumps(out_dict))

# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferMmlabKieFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_mmlab_kie"
        self.info.shortDescription = "Inference for MMOCR from MMLAB kie models"
        self.info.description = "If custom training is disabled, models will come from MMLAB's model zoo." \
                                "Else, you can also choose to load a model you trained yourself with our plugin " \
                                "train_mmlab_kie. In this case make sure you give to the plugin" \
                                "a config file (.py) and a model file (.pth). Both of these files are produced " \
                                "by the train plugin."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        self.info.iconPath = "icons/mmlab.png"
        self.info.authors = "Kuang, Zhanghui and Sun, Hongbin and Li, Zhizhong and Yue, Xiaoyu and Lin," \
                            " Tsui Hin and Chen, Jianyong and Wei, Huaqiang and Zhu, Yiqin and Gao, Tong and Zhang," \
                            " Wenwei and Chen, Kai and Zhang, Wayne and Lin, Dahua"
        self.info.article = "MMOCR:  A Comprehensive Toolbox for Text Detection, Recognition and Understanding"
        self.info.journal = "Arxiv"
        self.info.year = 2021
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://mmocr.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmocr"
        # Keywords used for search
        self.info.keywords = "infer, key, information, extraction, kie, mmlab, sdmgr"

    def create(self, param=None):
        # Create process object
        return InferMmlabKie(self.info.name, param)
