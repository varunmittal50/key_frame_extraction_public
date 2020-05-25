from __future__ import print_function

import os
from glob import glob
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_float
import hdbscan
# from multiprocessing import Pool, Process, cpu_count
import matplotlib.pyplot as plt
import seaborn as sns

import time

from  extracting_candidate_frames import Configs as config


class ImageSelector(object):
    """Class for selection of best top N images from input list of images, Currently following selection method are implemented:
    brightness filtering, contrast/entropy filtering, clustering of frames and variance of laplacian for non blurred images 
    selection
    :param object: base class inheritance
    :type object: class:`Object`
    """

    # def __init__(self): #, pool_obj):
        # Setting for Multiprocessing Pool Object
        #self.pool_obj = pool_obj

        # Setting for optimum Brightness values
        # self.min_brightness_value = config.min_brightness_value
        # self.max_brightness_value = config.max_brightness_value

        # # Setting for optimum Contrast/Entropy values
        # self.min_entropy_value = config.min_entropy_value
        # self.max_entropy_value = config.max_entropy_value

    def __get_brighness_score__(self, image):
        """Internal function to compute the brightness of input image , returns brightness score between 0 to 100.0 , 
        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image   
        :return: result of Brighness measurment 
        :rtype: float value between 0.0 to 100.0    
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        sum = np.sum(v, dtype=np.float32)
        num_of_pixels = v.shape[0] * v.shape[1]
        brightness_score = (sum * 100.0) / (num_of_pixels * 255.0)
        return brightness_score

    def __get_entropy_score__(self, image):
        """Internal function to compute the entropy/contrast of input image , returns entropy score between 0 to 10 , 
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image
        :return: result of Entropy measurment
        :rtype: float value between 0.0 to 10.0
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entr_img = entropy(gray, disk(5))
        all_sum = np.sum(entr_img)
        num_of_pixels = entr_img.shape[0] * entr_img.shape[1]
        entropy_score = (all_sum) / (num_of_pixels)

        return entropy_score

    def __variance_of_laplacian__(self, image):
        """Internal function to compute the laplacian of the image and then return the focus
        measure, which is simply the variance of the laplacian,
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param image: input image
        :type image: Opencv Numpy Image   
        :return: result of cv2.Laplacian
        :rtype: opencv image of type CV_64F    
        """

        return cv2.Laplacian(image, cv2.CV_64F).var()

    def __filter_optimum_brightness_and_contrast_images__(self, input_img_files):
        """ Internal function for selection of given input images with following parameters :optimum brightness and contrast range ,
        returns array of image files which are in optimum brigtness and contrast/entropy range.
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param files: list of input image files 
        :type files: python list of images
        :return: Returns list of filtered images  
        :rtype: python list of images 
        """

        n_files = len(input_img_files)
        # -------- calculating the brightness and entropy score by multiprocessing ------
        brightness_score = np.asarray(list(map(self.__get_brighness_score__, input_img_files)))
            # self.pool_obj.
        #     map(self.__get_brighness_score__, input_img_files)
        # )

        entropy_score = np.asarray(list(map(self.__get_entropy_score__, input_img_files)))
            # self.pool_obj.
        #     map(self.__get_entropy_score__, input_img_files)
        # )
        # print(list(entropy_score))
        # -------- Check if brightness and contrast scores are in the min and max defined range ------
        brightness_ok = np.where(
            np.logical_and(
                brightness_score > self.min_brightness_value,
                brightness_score < self.max_brightness_value,
            ),
            True,
            False,
        )
        contrast_ok = np.where(
            np.logical_and(
                entropy_score > self.min_entropy_value,
                entropy_score < self.max_entropy_value,
            ),
            True,
            False,
        )

        # Returning only thos images which are have good brightness and contrast

        return [
            input_img_files[i]
            for i in range(n_files)
            if brightness_ok[i] and contrast_ok[i]
        ]


    def __prepare_cluster_sets__hdbscan(self, files):
        """ Internal function for clustering input image files, returns array of indexs of each input file
        (which determines which cluster a given file belongs)
 
        :param object: base class inheritance
        :type object: class:`Object`
        :param files: list of input image files 
        :type files: python list of opencv numpy images
        :return: Returns array containing index for each file for cluster belongingness 
        :rtype: np.array   
        """

        # all_hists = []
        all_dst = []
        # Calculating the histograms for each image and adding them into **all_hists** list or all_dst** list
        for img_file in files:
            # img1 = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            # # (thresh, img) = cv2.threshold(img1, 150, 255, cv2.THRESH_BINARY)
            # hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
            # hist = hist.reshape((256))
            # all_hists.append(hist)
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (256, 256), img)
            imf = np.float32(img) / 255.0  # float conversion/scale
            dst = cv2.dct(imf)  # the dct
            dst = dst[:16, :16]
            dst = dst.reshape((256))
            all_dst.append(dst)


        # HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
        #         gen_min_span_tree=True, leaf_size=40, memory=Memory(cachedir=None),
        #         metric='euclidean', min_cluster_size=5, min_samples=None, p=None)
        # {'braycurtis': hdbscan.dist_metrics.BrayCurtisDistance,
        # 'canberra': hdbscan.dist_metrics.CanberraDistance,
        # 'chebyshev': hdbscan.dist_metrics.ChebyshevDistance,
        # 'cityblock': hdbscan.dist_metrics.ManhattanDistance,
        # 'dice': hdbscan.dist_metrics.DiceDistance,
        # 'euclidean': hdbscan.dist_metrics.EuclideanDistance,
        # 'hamming': hdbscan.dist_metrics.HammingDistance,
        # 'haversine': hdbscan.dist_metrics.HaversineDistance,
        # 'infinity': hdbscan.dist_metrics.ChebyshevDistance,
        # 'jaccard': hdbscan.dist_metrics.JaccardDistance,
        # 'kulsinski': hdbscan.dist_metrics.KulsinskiDistance,
        # 'l1': hdbscan.dist_metrics.ManhattanDistance,
        # 'l2': hdbscan.dist_metrics.EuclideanDistance,
        # 'mahalanobis': hdbscan.dist_metrics.MahalanobisDistance,
        # 'manhattan': hdbscan.dist_metrics.ManhattanDistance,
        # 'matching': hdbscan.dist_metrics.MatchingDistance,
        # 'minkowski': hdbscan.dist_metrics.MinkowskiDistance,
        # 'p': hdbscan.dist_metrics.MinkowskiDistance,
        # 'pyfunc': hdbscan.dist_metrics.PyFuncDistance,
        # 'rogerstanimoto': hdbscan.dist_metrics.RogersTanimotoDistance,
        # 'russellrao': hdbscan.dist_metrics.RussellRaoDistance,
        # 'seuclidean': hdbscan.dist_metrics.SEuclideanDistance,
        # 'sokalmichener': hdbscan.dist_metrics.SokalMichenerDistance,
        # 'sokalsneath': hdbscan.dist_metrics.SokalSneathDistance,
        # 'wminkowski': hdbscan.dist_metrics.WMinkowskiDistance}
        # Hdbascan = hdbscan.HDBSCAN(min_cluster_size=2,metric='manhattan').fit(all_hists)
        Hdbascan = hdbscan.HDBSCAN(min_cluster_size=2,metric='manhattan').fit(all_dst)
        labels = np.add(Hdbascan.labels_,1)
        nb_clusters = len(np.unique(Hdbascan.labels_))
        # x=self.__plots_for_clustering(Hdbascan,all_dst)
        # del x

        files_clusters_index_array = []
        files_clusters_index_array_of_only_one_image = []
        for i in np.arange(nb_clusters):
            # print(i)
            if i==0:
                index_array = np.where(labels == i)
                files_clusters_index_array_of_only_one_image.append(index_array)
            else:
                index_array = np.where(labels == i)
                files_clusters_index_array.append(index_array)

        files_clusters_index_array = np.array(files_clusters_index_array)
        return files_clusters_index_array,files_clusters_index_array_of_only_one_image

    def __plots_for_clustering(self,Hdbascan,all_dst):
        # cluster_spanning_tree = Hdbascan.minimum_spanning_tree_.plot(edge_cmap='viridis',
        #                                                             edge_alpha=0.6,
        #                                                             node_size=80,
        #                                                             # edge_linewidth=2)
        single_linkage_tree_dst = Hdbascan.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # cluster_spanning_tree.figure.savefig('cluster_spanning_tree_dst.jpeg')
        single_linkage_tree_dst.figure.savefig('cluster_hierarchy_plot .jpeg')


    def __get_laplacian_scores(self, files, n_images):
        """Function to iteratre over each image in the cluster and calculates the laplacian/blurryness 
           score and adds the score to a list
        :param files: list of input filenames 
        :type files: python list of string
        :param n_images: number of images in the given cluster
        :type n_images: int
        :return: Returns list of laplacian scores for each image in the given cluster
        :rtype: python list 
        """

        variance_laplacians = []
        # Iterate over all images in image list
        for image_i in n_images:
            img_file = files[n_images[image_i]]
            img = cv2.cvtColor(img_file, cv2.COLOR_BGR2GRAY)

            # Calculating the blurryness of image
            variance_laplacian = self.__variance_of_laplacian__(img)
            variance_laplacians.append(variance_laplacian)

        return variance_laplacians

    def __get_best_images_index_from_each_cluster__(
        self, files, files_clusters_index_array
    ):
        """ Internal function returns index of one best image from each cluster
        :param object: base class inheritance
        :type object: class:`Object`
        :param files: list of input filenames 
        :type files: python list of string
        :param files_clusters_index_array: Input is array containing index for each file for cluster belongingness 
        :type: np.array   
        :return: Returns list of filtered files which are best candidate from each cluster
        :rtype: python list 
        """

        filtered_items = []

        # Iterating over every image in each cluster to find the best images from every cluster
        clusters = np.arange(len(files_clusters_index_array))
        for cluster_i in clusters:
            curr_row = files_clusters_index_array[cluster_i][0] 
            # kp_lengths = []
            n_images = np.arange(len(curr_row))
            variance_laplacians = self.__get_laplacian_scores(files, n_images)

            # Selecting image with low burr(high laplacian) score
            try:
                selected_frame_of_current_cluster = curr_row[np.argmax(variance_laplacians)]
                filtered_items.append(selected_frame_of_current_cluster)
            except:
                break 

        return filtered_items

    def __getstate__(self):
        """Function to get the state of initialized class object and remove the pool object from it
        """
        self_dict = self.__dict__.copy()
        # del self_dict["pool_obj"]
        return self_dict

    def __setstate__(self, state):
        """Function to update the state of initialized class object woth the pool object
        """
        self.__dict__.update(state)

    def select_best_frames(self, input_key_frames,output_folder):
    # def select_best_frames(self, input_key_frames, number_of_frames):
        """[summary] Public function for Image selector class: takes list of key-frames images and number of required
        frames as input, returns list of filtered keyframes
        :param object: base class inheritance
        :type object: class:`Object`
        :param input_key_frames: list of input keyframes in list of opencv image format 
        :type input_key_frames: python list opencv images
        :param number_of_frames: Required number of images 
        :type: int   
        :return: Returns list of filtered image files 
        :rtype: python list of images
        """

        filtered_images_list = []

        # Selecting only those images which have good brishtness and contrast
        # input_key_frames = self.__filter_optimum_brightness_and_contrast_images__(
        #     input_key_frames
        # )
        
        # Selecting the best images from each cluster by first preparing the clusters on basis of histograms 
        # and then selecting the best images from every cluster
        # if len(input_key_frames) >= self.nb_clusters:
        if len(input_key_frames) >= 1:
            files_clusters_index_array,files_clusters_index_array_of_only_one_image = self.__prepare_cluster_sets__hdbscan(input_key_frames)
            selected_images_index = self.__get_best_images_index_from_each_cluster__(
                input_key_frames, files_clusters_index_array
            )
            files_clusters_index_array_of_only_one_image = [item for t in files_clusters_index_array_of_only_one_image for item in t]
            files_clusters_index_array_of_only_one_image = files_clusters_index_array_of_only_one_image[0].tolist()
            selected_images_index.extend(files_clusters_index_array_of_only_one_image)
            for index in selected_images_index:
                img = input_key_frames[index]
                filtered_images_list.append(img)
            # saving images of same clusters 
            i=0
            for images in files_clusters_index_array:
                # try:
                path = output_folder+'/'+str(i)
                try:
                    if not os.path.isdir(output_folder):
                        os.mkdir(output_folder)
                        print(output_folder)
                except OSError:
                    print ("Creation of the directory %s failed" % output_folder)
                try:
                    os.makedirs(path)
                except:
                    pass
                for image in images[0]:
                    cv2.imwrite(os.path.join(path, str(image)+'.jpeg'),input_key_frames[image])
                i=i+1
        else:
            # if the imput candidate frames are less than a single cluster.
            for img in input_key_frames:
                filtered_images_list.append(img)

        # saving clusters of single image cluster
        for images in files_clusters_index_array_of_only_one_image:
            print(files_clusters_index_array_of_only_one_image)
            path = output_folder+'/'+str(i)
            try:
                if not os.path.isdir(output_folder):
                    os.mkdir(output_folder)
                    print(output_folder)
            except OSError:
                print ("Creation of the directory %s failed" % output_folder)
            try:
                os.makedirs(path)
            except:
                pass
            cv2.imwrite(os.path.join(path, str(image)+'.jpeg'),input_key_frames[image])
            i=i+1

          
        return filtered_images_list