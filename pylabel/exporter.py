"""PyLabel currently supports exporting annotations in COCO, YOLO, and VOC PASCAL formats."""

import json
from typing import List
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import yaml
import shutil
from pylabel.shared import _ReindexCatIds
from tqdm import tqdm
import csv

from pathlib import PurePath, Path


class Export:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def ExportToVoc(
        self,
        output_path=None,
        segmented_=False,
        path_=False,
        database_=False,
        folder_=False,
        occluded_=False,
    ):
        """Writes annotation files to disk in VOC XML format and returns path to files.

        By default, tags with empty values will not be included in the XML output.
        You can optionally choose to include them if they are required for your solution.

        Args:
            output_path (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object.
            segmented_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            path_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            database_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            folder_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            occluded_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.

        Returns:
            A list with 1 or more paths (strings) to annotations files.

        Example:
            >>> dataset.export.ExportToVoc()
            ['data/voc_annotations/000000000322.xml', ...]
        """
        ds = self.dataset

        if output_path == None:
            output_path = ds.path_to_annotations
        else:
            output_path = output_path

        os.makedirs(output_path, exist_ok=True)

        output_file_paths = []

        def voc_xml_file_creation(
            data,
            file_name,
            output_file_path,
            segmented=True,
            path=True,
            database=True,
            folder=True,
            occluded=True,
        ):
            index = 0
            df_smaller = data[data["img_filename"] == file_name].reset_index()

            if len(df_smaller) == 1:
                # print('test')
                annotation_text_start = "<annotation>"

                flder_lkp = str(df_smaller.loc[index]["img_folder"])
                if folder == True and flder_lkp != "":
                    folder_text = "<folder>" + flder_lkp + "</folder>"
                else:
                    folder_text = ""

                filename_text = (
                    "<filename>"
                    + str(df_smaller.loc[index]["img_filename"])
                    + "</filename>"
                )

                pth_lkp = str(df_smaller.loc[index]["img_path"])
                if path == True and pth_lkp != "":
                    path_text = "<path>" + pth_lkp + "</path>"
                else:
                    path_text = ""

                sources_text = ""

                size_text_start = "<size>"
                width_text = (
                    "<width>" + str(df_smaller.loc[index]["img_width"]) + "</width>"
                )
                height_text = (
                    "<height>" + str(df_smaller.loc[index]["img_height"]) + "</height>"
                )
                depth_text = (
                    "<depth>" + str(df_smaller.loc[index]["img_depth"]) + "</depth>"
                )
                size_text_end = "</size>"

                seg_lkp = str(df_smaller.loc[index]["ann_segmented"])
                if segmented == True and seg_lkp != "":
                    segmented_text = (
                        "<segmented>"
                        + str(df_smaller.loc[index]["ann_segmented"])
                        + "</segmented>"
                    )
                else:
                    segmented_text = ""

                # If the image has no annotations, skip this part of the output
                no_annotations = (
                    pd.isnull(df_smaller.loc[index]["cat_id"])
                    or df_smaller.loc[index]["cat_id"] == ""
                )

                if not no_annotations:
                    object_text_start = "<object>"

                    name_text = (
                        "<name>" + str(df_smaller.loc[index]["cat_name"]) + "</name>"
                    )
                    pose_text = (
                        "<pose>" + str(df_smaller.loc[index]["ann_pose"]) + "</pose>"
                    )
                    truncated_text = (
                        "<truncated>"
                        + str(df_smaller.loc[index]["ann_truncated"])
                        + "</truncated>"
                    )
                    difficult_text = (
                        "<difficult>"
                        + str(df_smaller.loc[index]["ann_difficult"])
                        + "</difficult>"
                    )

                    occluded_text = ""

                    bound_box_text_start = "<bndbox>"

                    xmin_text = (
                        "<xmin>"
                        + str(int(df_smaller.loc[index]["ann_bbox_xmin"]))
                        + "</xmin>"
                    )
                    xmax_text = (
                        "<xmax>"
                        + str(int(df_smaller.loc[index]["ann_bbox_xmax"]))
                        + "</xmax>"
                    )
                    ymin_text = (
                        "<ymin>"
                        + str(int(df_smaller.loc[index]["ann_bbox_ymin"]))
                        + "</ymin>"
                    )
                    ymax_text = (
                        "<ymax>"
                        + str(int(df_smaller.loc[index]["ann_bbox_ymax"]))
                        + "</ymax>"
                    )

                    bound_box_text_end = "</bndbox>"
                    object_text_end = "</object>"
                else:
                    object_text_start = ""
                    name_text = ""
                    pose_text = ""
                    truncated_text = ""
                    difficult_text = ""
                    occluded_text = ""
                    bound_box_text_start = ""
                    xmin_text = ""
                    xmax_text = ""
                    ymin_text = ""
                    ymax_text = ""
                    bound_box_text_end = ""
                    object_text_end = ""

                # Continue this part even if there are no annotations for this image
                annotation_text_end = "</annotation>"

                xmlstring = (
                    annotation_text_start
                    + folder_text
                    + filename_text
                    + path_text
                    + sources_text
                    + size_text_start
                    + width_text
                    + height_text
                    + depth_text
                    + size_text_end
                    + segmented_text
                    + object_text_start
                    + name_text
                    + pose_text
                    + truncated_text
                    + difficult_text
                    + occluded_text
                    + bound_box_text_start
                    + xmin_text
                    + xmax_text
                    + ymin_text
                    + ymax_text
                    + bound_box_text_end
                    + object_text_end
                    + annotation_text_end
                )
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()

                with open(output_file_path, "w") as f:
                    f.write(pretty_xml_as_string)

                return output_file_path

            else:
                # When there are more than one annotations for the image

                # print('test')
                annotation_text_start = "<annotation>"

                flder_lkp = str(df_smaller.loc[index]["img_folder"])
                if folder == True and flder_lkp != "":
                    folder_text = "<folder>" + flder_lkp + "</folder>"
                else:
                    folder_text = ""

                filename_text = (
                    "<filename>"
                    + str(df_smaller.loc[index]["img_filename"])
                    + "</filename>"
                )

                pth_lkp = str(df_smaller.loc[index]["img_path"])
                if path == True and pth_lkp != "":
                    path_text = "<path>" + pth_lkp + "</path>"
                else:
                    path_text = ""

                # db_lkp = str(df_smaller.loc[index]['Databases'])
                # if database == True and db_lkp != '':
                #    sources_text = '<source>'+'<database>'+ db_lkp +'</database>'+'</source>'
                # else:
                sources_text = ""

                size_text_start = "<size>"
                width_text = (
                    "<width>" + str(df_smaller.loc[index]["img_width"]) + "</width>"
                )
                height_text = (
                    "<height>" + str(df_smaller.loc[index]["img_height"]) + "</height>"
                )
                depth_text = (
                    "<depth>" + str(df_smaller.loc[index]["img_depth"]) + "</depth>"
                )
                size_text_end = "</size>"

                seg_lkp = str(df_smaller.loc[index]["ann_segmented"])
                if segmented == True and seg_lkp != "":
                    segmented_text = (
                        "<segmented>"
                        + str(df_smaller.loc[index]["ann_segmented"])
                        + "</segmented>"
                    )
                else:
                    segmented_text = ""

                xmlstring = (
                    annotation_text_start
                    + folder_text
                    + filename_text
                    + path_text
                    + sources_text
                    + size_text_start
                    + width_text
                    + height_text
                    + depth_text
                    + size_text_end
                    + segmented_text
                )

                for obj in range(len(df_smaller)):
                    object_text_start = "<object>"

                    name_text = (
                        "<name>" + str(df_smaller.loc[obj]["cat_name"]) + "</name>"
                    )
                    pose_text = (
                        "<pose>" + str(df_smaller.loc[obj]["ann_pose"]) + "</pose>"
                    )
                    truncated_text = (
                        "<truncated>"
                        + str(df_smaller.loc[obj]["ann_truncated"])
                        + "</truncated>"
                    )
                    difficult_text = (
                        "<difficult>"
                        + str(df_smaller.loc[obj]["ann_difficult"])
                        + "</difficult>"
                    )

                    # occ_lkp = str(df_smaller.loc[index]['Object Occluded'])
                    # if occluded==True and occ_lkp != '':
                    #    occluded_text = '<occluded>'+occ_lkp+'</occluded>'
                    # else:
                    occluded_text = ""

                    bound_box_text_start = "<bndbox>"

                    xmin_text = (
                        "<xmin>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_xmin"]))
                        + "</xmin>"
                    )
                    xmax_text = (
                        "<xmax>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_xmax"]))
                        + "</xmax>"
                    )
                    ymin_text = (
                        "<ymin>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_ymin"]))
                        + "</ymin>"
                    )
                    ymax_text = (
                        "<ymax>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_ymax"]))
                        + "</ymax>"
                    )

                    bound_box_text_end = "</bndbox>"
                    object_text_end = "</object>"
                    annotation_text_end = "</annotation>"
                    index = index + 1

                    xmlstring = (
                        xmlstring
                        + object_text_start
                        + name_text
                        + pose_text
                        + truncated_text
                        + difficult_text
                        + occluded_text
                        + bound_box_text_start
                        + xmin_text
                        + xmax_text
                        + ymin_text
                        + ymax_text
                        + bound_box_text_end
                        + object_text_end
                    )

                xmlstring = xmlstring + annotation_text_end
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()

                with open(output_file_path, "w") as f:
                    f.write(pretty_xml_as_string)

                return output_file_path

        # Loop through all images in the dataframe and call voc_xml_file_creation for each one
        pbar = tqdm(
            desc="Exporting VOC files...",
            total=len(list(set(self.dataset.df.img_filename))),
        )
        for file_title in list(set(self.dataset.df.img_filename)):
            file_name = Path(file_title)
            file_name = str(file_name.with_suffix(".xml"))
            file_path = str(Path(output_path, file_name))
            voc_file_path = voc_xml_file_creation(
                ds.df,
                file_title,
                segmented=segmented_,
                path=path_,
                database=database_,
                folder=folder_,
                occluded=occluded_,
                output_file_path=file_path,
            )
            output_file_paths.append(voc_file_path)
            pbar.update()

        return output_file_paths

    @staticmethod
    def _df_to_csv(
        df: pd.DataFrame,
        file_path: str,
        sep: str = " ",
        float_format: str = "%0.4f",
        columns: List[str] = None
    ):
        """
        with the pandas to_csv method the output for a list (e.g. the keypoints) must include a quote character or an
        escape character. To avoid both of those, we need a custom function
        """
        if columns is None:
            columns = df.columns

        def _format_float(fl: float, fl_format: str):
            if np.isnan(fl):
                return ""
            else:
                return fl_format % fl

        with open(file_path, "w") as f:
            for row in df[columns].itertuples():
                row=row[1:]
                formatted_row = []
                for x in row:                
                    if isinstance(x, float):
                        formatted_row.append(_format_float(x, float_format))
                    elif isinstance(x, list):
                        formatted_row.extend(
                            [_format_float(y, float_format) if isinstance(y, float) else str(y) for y in x]
                        )
                    else:
                        formatted_row.append(str(x))
                f.write(sep.join(formatted_row) + '\n')

    def ExportToYoloV5(
        self,
        output_path="training/labels",
        yaml_file="dataset.yaml",
        copy_images=False,
        use_splits=False,
        cat_id_index=None,
        segmentation=False,
        keypoints=False,
    ):
        """Writes annotation files to disk in YOLOv5 format and returns the paths to files.

        Args:

            output_path (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object. If you are exporting images to train a model, the recommended path
                to use is 'training/labels'.
            yaml_file (str):
                If a file name (string) is provided, a YOLOv5 YAML file will be created with entries for the files
                and classes in this dataset. It will be created in the parent of the output_path directory.
                The recommended name for the YAML file is 'dataset.yaml'.
            copy_images (boolean):
                If True, then the annotated images will be copied to a directory next to the labels directory into
                a directory named 'images'. This will prepare your labels and images to be used as inputs to
                train a YOLOv5 model.
            use_splits (boolean):
                If True, then the images and annotations will be moved into directories based on the values in the split column.
                For example, if a row has the value split = "train" then the annotations for that row will be moved to directory
                /train. If a YAML file is specificied then the YAML file will use the splits to specify the folders user for the
                train, val, and test datasets.
            cat_id_index (int):
                Reindex the cat_id values so that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Yolo requires the set of annotations to start at 0 when training a model.
            segmentation (boolean):
                If true, then segmentation annotations will be exported instead of bounding box annotations.
                If there are no segmentation annotations, then no annotations will be empty.
            keypoints (boolean):
                If true, then keypoint annotations will be exported as well as bounding box annotations.
                It is not possible to export both segmentation and keypoint annotations at the same time in YOLO format.
                Each bounding box within a dataset should have the same number of keypoints defined e.g. 17 for COCO.
                Keypoints are a triplet of (x, y, visibility), see e.g. https://cocodataset.org/#format-data
                If some images have no keypoint annotations, then the bounding boxes will be followed by a series of
                delimiting spaces.
                If some bounding boxes within an image have no keypoint annotations, those keypoints will be a series of
                zeroes, denoting x=0, y=0, visibility=0.

        Returns:
            A list with 1 or more paths (strings) to annotations files. If a YAML file is created
            then the first item in the list will be the path to the YAML file.

        Examples:
            >>> dataset.export.ExportToYoloV5(output_path='training/labels',
            >>>     yaml_file='dataset.yaml', cat_id_index=0)
            ['training/dataset.yaml', 'training/labels/frame_0002.txt', ...]

        """
        ds = self.dataset

        assert not (segmentation and keypoints), "Only one of segmentation and keypoints can be exported in YOLO format"

        # Inspired by https://github.com/aws-samples/groundtruth-object-detection/blob/master/create_annot.py
        yolo_dataset = ds.df.copy(deep=True)
        # Convert nan values in the split column from nan to '' because those are easier to work with with when building paths
        yolo_dataset.split = yolo_dataset.split.fillna("")

        # Create all of the paths that will be used to manage the files in this dataset
        path_dict = {}

        # The output path is the main path that will be used to create the other relative paths
        path = PurePath(output_path)
        path_dict["label_path"] = output_path
        # The /images directory should be next to the /labels directory
        path_dict["image_path"] = str(PurePath(path.parent, "images"))
        # The root directory is in parent of the /labels and /images directories
        path_dict["root_path"] = str(PurePath(path.parent))
        # The YAML file should be in root directory
        path_dict["yaml_path"] = str(PurePath(path_dict["root_path"], yaml_file))
        # The root directory will usually be next to the yolov5 directory.
        # Specify the relative path
        path_dict["root_path_from_yolo_dir"] = str(PurePath("../"))
        # If these default values to not match the users environment then they can manually edit the YAML file

        if copy_images:
            # Create the folder that the images will be copied to
            Path(path_dict["image_path"]).mkdir(parents=True, exist_ok=True)

        # Drop rows that are not annotated
        # Note, having zero annotates can still be considered annotated
        # in cases when are no objects in the image thats should be indentified
        yolo_dataset = yolo_dataset.loc[yolo_dataset["annotated"] == 1]

        # yolo_dataset["cat_id"] = (
        #     yolo_dataset["cat_id"].astype("float").astype(pd.Int32Dtype())
        # )

        yolo_dataset.cat_id = yolo_dataset.cat_id.replace(r"^\s*$", np.nan, regex=True)

        pd.to_numeric(yolo_dataset["cat_id"])

        if cat_id_index != None:
            assert isinstance(cat_id_index, int), "cat_id_index must be an int."
            _ReindexCatIds(yolo_dataset, cat_id_index)

        # Convert empty bbox coordinates to nan to avoid math errors
        # If an image has no annotations then an empty label file will be created
        yolo_dataset.ann_bbox_xmin = yolo_dataset.ann_bbox_xmin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_ymin = yolo_dataset.ann_bbox_ymin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_width = yolo_dataset.ann_bbox_width.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_height = yolo_dataset.ann_bbox_height.replace(
            r"^\s*$", np.nan, regex=True
        )

        # If segmentation = False then export bounding boxes
        if segmentation == False:
            yolo_dataset["center_x_scaled"] = (
                yolo_dataset["ann_bbox_xmin"] + (yolo_dataset["ann_bbox_width"] * 0.5)
            ) / yolo_dataset["img_width"]
            yolo_dataset["center_y_scaled"] = (
                yolo_dataset["ann_bbox_ymin"] + (yolo_dataset["ann_bbox_height"] * 0.5)
            ) / yolo_dataset["img_height"]
            yolo_dataset["width_scaled"] = (
                yolo_dataset["ann_bbox_width"] / yolo_dataset["img_width"]
            )
            yolo_dataset["height_scaled"] = (
                yolo_dataset["ann_bbox_height"] / yolo_dataset["img_height"]
            )

            if keypoints:
                keypoints_yolo = [[] for _ in range(len(yolo_dataset.index))]
                for img_ix, row in yolo_dataset.iterrows():
                    img_width = row["img_width"]
                    img_height = row["img_height"]
                    keypoints_coco = row["ann_keypoints"]
                    if keypoints_coco:
                        for bbox_ix, kp in enumerate(keypoints_coco):
                            if bbox_ix % 3 == 0:
                                # x coordinate
                                keypoints_yolo[img_ix].append(kp / img_width)
                            elif bbox_ix % 3 == 1:
                                # y coordinate
                                keypoints_yolo[img_ix].append(kp / img_height)
                            else:
                                # visibility
                                keypoints_yolo[img_ix].append(kp)
                yolo_dataset["keypoints_scaled"] = keypoints_yolo

        # Create folders to store annotations
        if output_path == None:
            dest_folder = PurePath(
                ds.path_to_annotations, yolo_dataset.iloc[0].img_folder
            )
        else:
            dest_folder = output_path

        os.makedirs(dest_folder, exist_ok=True)

        unique_images = yolo_dataset["img_filename"].unique()
        output_file_paths = []
        pbar = tqdm(desc="Exporting YOLO files...", total=len(unique_images))
        for img_filename in unique_images:
            df_single_img_annots = yolo_dataset.loc[
                yolo_dataset.img_filename == img_filename
            ]

            basename, _ = os.path.splitext(img_filename)
            annot_txt_file = basename + ".txt"
            # Use the value of the split collumn to create a directory
            # The values should be train, val, test or ''
            if use_splits:
                split_dir = df_single_img_annots.iloc[0].split
            else:
                split_dir = ""
            destination = str(PurePath(dest_folder, split_dir, annot_txt_file))
            Path(
                dest_folder,
                split_dir,
            ).mkdir(parents=True, exist_ok=True)

            # If segmentation = false then output bounding boxes
            if segmentation == False:
                columns = [
                    "cat_id",
                    "center_x_scaled",
                    "center_y_scaled",
                    "width_scaled",
                    "height_scaled",
                ]
                if keypoints:
                    columns.append("keypoints_scaled")

                self._df_to_csv(
                    df=df_single_img_annots,
                    file_path=destination,
                    sep=" ",
                    float_format="%.4f",
                    columns=columns
                )

            # If segmentation = true then output the segmentation mask
            else:
                # Create one file for image
                with open(destination, "w") as file:
                    # Create one row per row in the data frame
                    for i in range(0, df_single_img_annots.shape[0]):
                        row = str(df_single_img_annots.iloc[i].cat_id)
                        segmentation_array = df_single_img_annots.iloc[
                            i
                        ].ann_segmentation[0]

                        # Iterate through every value of the segmentation array
                        # To normalize the coordinates from 0-1
                        for index, l in enumerate(segmentation_array):
                            # The first number in the array is the x value so divide by the width
                            if index % 2 == 0:
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_width
                                    )
                                )
                            else:
                                # The first number in the array is the x value so divide by the height
                                row += " " + (
                                    str(
                                        segmentation_array[index]
                                        / df_single_img_annots.iloc[i].img_height
                                    )
                                )

                        file.write(row + "\n")

            output_file_paths.append(destination)

            if copy_images:
                source_image_path = str(
                    Path(
                        ds.path_to_annotations,
                        df_single_img_annots.iloc[0].img_folder,
                        df_single_img_annots.iloc[0].img_filename,
                    )
                )

                current_file = Path(source_image_path)
                assert (
                    current_file.is_file
                ), f"File does not exist: {source_image_path}. Check img_folder column values."
                Path(path_dict["image_path"], split_dir).mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy(
                    str(source_image_path),
                    str(PurePath(path_dict["image_path"], split_dir, img_filename)),
                )
            pbar.update()

        # Create YAML file
        if yaml_file:
            # Make a set with all of the different values of the split column
            splits = set(yolo_dataset.split)
            # Build a dict with all of the values that will go into the YAML file
            dict_file = {}
            dict_file["path"] = path_dict["root_path_from_yolo_dir"]

            # If train is one of the splits, append train to path
            if use_splits and "train" in splits:
                dict_file["train"] = str(PurePath(path_dict["image_path"], "train"))
            else:
                dict_file["train"] = path_dict["image_path"]

            # If val is one of the splits, append val to path
            if use_splits and "val" in splits:
                dict_file["val"] = str(PurePath(path_dict["image_path"], "val"))
            else:
                # If there is no val split, use the train split as the val split
                dict_file["val"] = dict_file["train"]

            # If test is one of the splits, make a test param and add test to the path
            if use_splits and "test" in splits:
                dict_file["test"] = str(PurePath(path_dict["image_path"], "test"))

            dict_file["nc"] = ds.analyze.num_classes
            dict_file["names"] = ds.analyze.classes

            # Save the yamlfile
            with open(path_dict["yaml_path"], "w") as file:
                documents = yaml.dump(dict_file, file,encoding="utf-8",allow_unicode=True)
                output_file_paths = [path_dict["yaml_path"]] + output_file_paths

        return output_file_paths

    def ExportToCoco(self, output_path=None, cat_id_index=None):
        """
        Writes COCO annotation files to disk (in JSON format) and returns the path to files.

        Args:
            output_path (str):
                This is where the annotation files will be written. If not-specified then the path will be derived from the path_to_annotations and
                name properties of the dataset object.
            cat_id_index (int):
                Reindex the cat_id values so that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Some models like Yolo require starting from 0 and others like Detectron require starting from 1.

        Returns:
            A list with 1 or more paths (strings) to annotations files.

        Example:
            >>> dataset.exporter.ExportToCoco()
            ['data/labels/dataset.json']

        """
        # Prepare dataframe
        df = self.dataset.df.copy(deep=True)
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df["ann_iscrowd"] = df["ann_iscrowd"].fillna(0)
        
        if cat_id_index is not None:
            assert isinstance(cat_id_index, int), "cat_id_index must be int"
            _ReindexCatIds(df, cat_id_index)
        
        # Process unique images
        # Use img_filename as the unique identifier if img_id is not available
        images = (df.groupby('img_filename', as_index=False)
                 .agg({
                     'img_id': 'first',
                     'img_folder': 'first',
                     'img_path': 'first',
                     'img_width': 'first',
                     'img_height': 'first',
                     'img_depth': 'first'
                 })
                 .apply(_create_image_dict, axis=1)
                 .tolist())
        
        # Process annotations
        annotations = []
        categories_set = set()
        categories = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Exporting to COCO"):
            if pd.isna(row["cat_id"]):
                continue
                
            annotations.append(_create_annotation_dict(row, idx))
            
            # Add unique categories
            cat_id = int(row["cat_id"])
            if cat_id not in categories_set:
                categories.append(_create_category_dict(row))
                categories_set.add(cat_id)
        
        # Create COCO format JSON
        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }
        
        # Save to file
        if output_path is None:
            output_path = Path(self.dataset.path_to_annotations, f"{self.dataset.name}.json")
        
        # Replace NaN values with None before JSON serialization
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(x) for x in obj]
            elif pd.isna(obj):
                return None
            return obj
            
        coco_dict_clean = clean_nan(coco_dict)
        with open(output_path, "w") as f:
            json.dump(coco_dict_clean, f, indent=4)
        
        return [str(output_path)]

def _process_keypoints(row):
    """Handle keypoint processing for annotations"""
    if "ann_keypoints" not in row or np.isnan(row["ann_keypoints"]).all():
        return {}
    
    keypoints = row["ann_keypoints"]
    if not isinstance(keypoints, (list, np.ndarray)):
        raise TypeError('Keypoints must be list or numpy array')
    
    n_keypoints = len(keypoints) // 3 if isinstance(keypoints, list) else keypoints.size // 3
    return {"num_keypoints": n_keypoints, "keypoints": keypoints}

def _create_image_dict(row):
    """Create image dictionary from dataframe row"""
    return {
        "id": row["img_id"],
        "folder": row["img_folder"], 
        "file_name": row["img_filename"],
        "path": row["img_path"],
        "width": row["img_width"],
        "height": row["img_height"],
        "depth": row["img_depth"]
    }

def _create_annotation_dict(row, index):
    """Create annotation dictionary from dataframe row"""
    annotation = {
        "image_id": row["img_id"],
        "id": index,
        "segmented": row["ann_segmented"],
        "bbox": [
            row["ann_bbox_xmin"],
            row["ann_bbox_ymin"],
            row["ann_bbox_width"],
            row["ann_bbox_height"]
        ],
        "area": row["ann_area"],
        "segmentation": row["ann_segmentation"],
        "iscrowd": row["ann_iscrowd"],
        "pose": row["ann_pose"],
        "truncated": row["ann_truncated"],
        "category_id": int(row["cat_id"]),
        "difficult": row["ann_difficult"]
    }
    
    # Add keypoints if present
    keypoints_data = _process_keypoints(row)
    if keypoints_data:
        annotation.update(keypoints_data)
    
    return annotation

def _create_category_dict(row):
    """Create category dictionary from dataframe row"""
    return {
        "id": int(row["cat_id"]),
        "name": row["cat_name"],
        "supercategory": row["cat_supercategory"]
    }
