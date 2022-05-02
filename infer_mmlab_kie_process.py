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
from mmcv import Config
from mmocr.apis.inference import disable_text_recog_aug_test, init_detector, model_inference
import torch
from infer_mmlab_kie.utils import polygon2bbox, get_classes, stitch_boxes_into_lines
import os
from mmocr.datasets.pipelines.ocr_transforms import ResizeOCR
from mmocr.datasets.kie_dataset import KIEDataset
import distutils
from mmocr.utils.model import revert_sync_batchnorm
import numpy as np
from datetime import datetime
import cv2
import re


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
        self.merge_box = False
        self.max_x_dist = 10
        self.min_y_overlap_ratio = 0.8
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
        self.addOutput(dataprocess.CGraphicsOutput())
        self.addOutput(dataprocess.CNumericIO())
        self.addOutput(dataprocess.CImageIO())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.kie_dataset = None
        self.classes = None
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
        numeric_output = self.getOutput(2)
        numeric_output.clearData()
        numeric_output.setOutputType(dataprocess.NumericOutputType.TABLE)
        visual_output = self.getOutput(3)
        visual_img = visual_output.getImage()

        # Load models into memory
        if self.model is None or param.update:
            print("Loading KIE model...")
            if not param.custom_training:
                cfg = Config.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "kie",
                                                   param.model_name, param.cfg))
                ckpt = param.weights
                dict_file = os.path.join(os.path.dirname(__file__), "wildreceipt/dict.txt")
                self.kie_dataset = KIEDataset(dict_file=dict_file)
                cfg.model.class_list = os.path.join(os.path.dirname(__file__), "wildreceipt/class_list.txt")
                device = torch.device(self.device)

            else:
                cfg = Config.fromfile(param.custom_cfg)
                ckpt = param.custom_weights
                device = torch.device(self.device)
                dict_file = param.dict
                self.kie_dataset = KIEDataset(dict_file=dict_file)
                cfg.model.class_list = param.class_file
            self.model = init_detector(cfg, ckpt, device=device)
            self.model = revert_sync_batchnorm(self.model)

            self.classes = get_classes(cfg.model.class_list)
            param.update = False
            print("Model loaded!")

        srcImg = input.getImage()

        if srcImg is not None and input_items is not None:
            annotations = []
            for item in input_items:
                record = {}
                pts = []
                for p in item.points:
                    pts.append(p.x)
                    pts.append(p.y)
                record['box'] = pts
                record['text'] = item.getCategory()
                annotations.append(record)

            # merging boxes if needed
            if param.merge_box:
                annotations = stitch_boxes_into_lines(annotations, max_x_dist=param.max_x_dist,
                                                      min_y_overlap_ratio=param.min_y_overlap_ratio)

            ann_info = self.kie_dataset._parse_anno_info(annotations)
            ann_info['ori_bboxes'] = ann_info.get('ori_bboxes',
                                                  ann_info['bboxes'])
            ann_info['gt_bboxes'] = ann_info.get('gt_bboxes',
                                                 ann_info['bboxes'])

            out = model_inference(self.model,
                                  srcImg,
                                  ann=ann_info,
                                  batch_mode=False,
                                  return_data=False)
            visual_img = np.zeros_like(srcImg)
            visual_img.fill(255)
            self.visualize_kie(out, numeric_output, annotations, visual_img)
            visual_output.setImage(visual_img)
        # Call endTaskRun to finalize process
        self.endTaskRun()

    def visualize_kie(self, results, numeric_output, annotations, visual_img):
        labels = np.argmax(results['nodes'].cpu().numpy(), axis=1).astype(dtype=float)
        values = np.max(results['nodes'].cpu().numpy(), axis=1)
        texts = [anno['text'] for anno in annotations]
        boxes = [anno['box'] for anno in annotations]

        # output for Ikomia Studio
        numeric_output.addValueList(list(labels), "Class", texts)

        # output in a textfile for other applications
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        output_file = os.path.join(output_dir, datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss"))
        if not (os.path.isdir(output_dir)):
            os.mkdir(output_dir)
        with open(output_file, 'w') as f:
            f.write("")
        with open(output_file, 'a') as f:
            for label, conf, text, box in zip(labels, values, texts, boxes):
                if int(label) != len(self.classes) - 1:
                    color = [255 * (1 - conf), 0, 255 * conf]

                    draw_text(visual_img, re.sub('[^A-Za-z0-9_]+', '', self.classes[int(label)]), box, color)
                    f.write(text + " " + str(int(label)) + " " + str(conf) + "\n")


def draw_text(img_display, text, box, color):
    x_b, y_b, w_b, h_b = polygon2bbox(box)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w_t, h_t), _ = cv2.getTextSize(text, fontFace=font, fontScale=1, thickness=1)
    fontscale = w_b / w_t
    org = (x_b, y_b + int((h_b + h_t * fontscale) / 2))
    cv2.putText(img_display, text, org, font, fontScale=fontscale, color=color, thickness=1)


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
