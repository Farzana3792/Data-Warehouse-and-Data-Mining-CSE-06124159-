import shap

import pandas as pd
import numpy as np
import statistics as st

import matplotlib.pyplot as plt
import seaborn as sn

import xgboost
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans




class disSHAP:

    # explainer_type = TreeExplainer (T) or DeepExplainer (D) or LinearExplainer (L)

    def __init__(self, model, x_train_data, x_test_data, y_train_data, y_test_data, feature_names, explainer_type, number_of_top_features=0, cluster_num=5):
        self.model = model
        self.x_train_data = x_train_data
        self.y_train_data = y_train_data
        max_allowed_x_test_data_size = 20000
        self.x_test_data = x_test_data[:min(max_allowed_x_test_data_size, len(x_test_data))]
        self.y_test_data = y_test_data[:min(max_allowed_x_test_data_size, len(y_test_data))]

        if len(set(self.y_test_data)) > 2 or len(set(self.y_train_data)) > 2:
            raise Exception('Please provide binary classification data!')
        
        if max(self.y_test_data) != 1 or  max(self.y_train_data) != 1:
            raise Exception('Please re-label your attack/positive class as 1.')
        
        if min(self.y_test_data) != 0 or min(self.y_train_data) != 0:
            raise Exception('Please re-label your benign/negative class as 0.')

        self.model_predicted_label = self.model.predict(self.x_test_data)

        self.feature_names = feature_names
        self.cluster_num = cluster_num
        self.number_of_top_features = max(number_of_top_features, math.floor(2 * math.sqrt(len(feature_names))))

        self.tp_instances = np.where((self.y_test_data == 1) & (self.model_predicted_label == 1))[0]
        self.tn_instances = np.where((self.y_test_data == 0) & (self.model_predicted_label == 0))[0]
        self.fp_instances = np.where((self.y_test_data == 0) & (self.model_predicted_label == 1))[0]
        self.fn_instances = np.where((self.y_test_data == 1) & (self.model_predicted_label == 0))[0]

        self.explainer_type = explainer_type
        self.model_explainer = None
        self.model_test_shap_values = np.array([])
        self.generating_shap_values()

        # for automated clustering
        # self.tp_shap_value_avg_centroids = self.Automated_K_Means_Clustering_SHAP_Values(self.model_test_shap_values[self.tp_instances])
        # self.tn_shap_value_avg_centroids = self.Automated_K_Means_Clustering_SHAP_Values(self.model_test_shap_values[self.tn_instances])
        # self.fp_shap_value_avg_centroids = self.Automated_K_Means_Clustering_SHAP_Values(self.model_test_shap_values[self.fp_instances])
        # self.fn_shap_value_avg_centroids = self.Automated_K_Means_Clustering_SHAP_Values(self.model_test_shap_values[self.fn_instances])


        # for manual clustering
        self.tp_shap_value_avg_centroids = self.Clustering_SHAP_Values(self.model_test_shap_values[self.tp_instances])
        self.tn_shap_value_avg_centroids = self.Clustering_SHAP_Values(self.model_test_shap_values[self.tn_instances])
        self.fp_shap_value_avg_centroids = self.Clustering_SHAP_Values(self.model_test_shap_values[self.fp_instances])
        self.fn_shap_value_avg_centroids = self.Clustering_SHAP_Values(self.model_test_shap_values[self.fn_instances])

        # print(self.tp_shap_value_avg_centroids.shape)
        # print(self.tn_shap_value_avg_centroids.shape)
        # print(self.fp_shap_value_avg_centroids.shape)
        # print(self.fn_shap_value_avg_centroids.shape)
        # print(len(self.tp_shap_value_avg_centroids))
        # print(self.tp_instances)
        # print(self.model_test_shap_values[self.tp_instances])
        # exit(0)



        self.top_tp_features_index = self.top_n_features(np.mean(self.model_test_shap_values[self.tp_instances], axis=0))
        self.top_tn_features_index = self.top_n_features(np.mean(self.model_test_shap_values[self.tn_instances], axis=0))
        self.top_fp_features_index = self.top_n_features(np.mean(self.model_test_shap_values[self.fp_instances], axis=0))
        self.top_fn_features_index = self.top_n_features(np.mean(self.model_test_shap_values[self.fn_instances], axis=0))




    def generating_shap_values(self):
        background_data = self.x_train_data[np.random.choice(self.x_train_data.shape[0], 1000, replace=False)]
        if self.explainer_type == 'T':
            self.model_explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == 'D':
            self.model_explainer = shap.DeepExplainer(self.model, background_data)
        elif self.explainer_type == 'L':
            self.model_explainer = shap.LinearExplainer(self.model, background_data)
        else:
            raise Exception("please provide supported explainer object type\nTreeExplainer (T) or DeepExplainer (D) or LinearExplainer (L)")
        
        
        batch_size = 500
        shap_values_list = []  # Use a list to collect batches

        for i in range(0, len(self.x_test_data), batch_size):
            print('Generating SHAP values for batch ====================>', 
                (i // batch_size) + 1, '/', (len(self.x_test_data) + batch_size - 1) // batch_size)
            batch_data = self.x_test_data[i:i+batch_size]
            batch_shap_values = self.model_explainer(batch_data)
            shap_values_list.append(batch_shap_values.values)
        
        self.model_test_shap_values = np.vstack(shap_values_list)
    



    # def top_n_features(self, shap_value_group):
    #     print('passed value shape', self.feature_names, np.array(shap_value_group))
    #     exit(0)
    #     shap_df = pd.DataFrame(shap_value_group, columns=self.feature_names)
    #     vals = np.abs(shap_df.values).mean(0)
    #     shap_importance = pd.DataFrame(list(zip(self.feature_names, vals)), columns = ['col_name', 'feature_importance_vals'])
    #     shap_importance.sort_values(by = ['feature_importance_vals'], ascending = False, inplace = True)
    #     return shap_importance.head(self.number_of_top_features).index


    def top_n_features(self, array):
        return np.argsort(array)[-self.number_of_top_features:][::-1]




    def same_sign_shap_value(self, shap_1, shap_2):
        ans = 0
        for i in range(len(shap_1)):
            if shap_1[i]<0 and shap_2[i]<0:
                ans += 1
            if shap_1[i]>0 and shap_2[i]>0:
                ans += 1
            if shap_1[i]==0 and shap_2[i]==0:
                ans += 1
        return ans




    def shap_value_mismatched(self, instance_1, instance_2):
        # print(instance_1, instance_2)
        # exit(0)
        pos_mismatch, neg_mismatch, _0_mismatch = 0, 0, 0
        for i in range(len(instance_1)):
            if instance_1[i]<0:
                if instance_2[i]>0:
                    neg_mismatch += 1
            elif instance_1[i]>0:
                if instance_2[i]<0:
                    pos_mismatch += 1
            else:
                if instance_2[i] != 0:
                    _0_mismatch += 1
        
        return pos_mismatch + neg_mismatch + _0_mismatch



    def old_disSHAP_Comparison(self, x_instances_shap):
        
        dist = []
        for i in range(self.cluster_num):
            dist.append([abs(np.sum(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index]-x_instances_shap[self.top_tp_features_index])), 
                abs(np.sum(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index]-x_instances_shap[self.top_tn_features_index])), 
                abs(np.sum(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index]-x_instances_shap[self.top_fp_features_index])), 
                abs(np.sum(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index]-x_instances_shap[self.top_fn_features_index]))])



        mismatches = []
        for i in range(self.cluster_num):
            mismatches.append([self.shap_value_mismatched(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index], x_instances_shap[self.top_tp_features_index]), 
                    self.shap_value_mismatched(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index], x_instances_shap[self.top_tn_features_index]), 
                    self.shap_value_mismatched(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index], x_instances_shap[self.top_fp_features_index]), 
                    self.shap_value_mismatched(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index], x_instances_shap[self.top_fn_features_index])])


        min_mismatches = [mismatches[i].index(min(mismatches[i])) for i in range(self.cluster_num)]
        min_dist = [dist[i].index(min(dist[i])) for i in range(self.cluster_num)]
          
        return [st.mode(sorted(min_mismatches)) , st.mode(sorted(min_dist))]
    



    def mitigate_misclassification_top_n(self, x_instances_shap):

        # self.top_tp_features_index = top_20_features[0]
        # top_20_tn_features_index = top_20_features[1]
        # top_20_fp_features_index = top_20_features[2]
        # top_20_fn_features_index = top_20_features[3]

        # self.tp = avg_shap_value_cluster_groups[0]
        # selected_avg_shap_value_tn_clustep = avg_shap_value_cluster_groups[1]
        # selected_avg_shap_value_fp_cluster = avg_shap_value_cluster_groups[2]
        # selected_avg_shap_value_fn_cluster = avg_shap_value_cluster_groups[3]


        # print('check check len',len(self.tp))
        # print('check check len',len(selected_avg_shap_value_tn_clustep))
        # print('check check len',len(selected_avg_shap_value_fp_cluster))
        # print('check check len',len(selected_avg_shap_value_fn_cluster))

        tp_grp_dist, tn_grp_dist, fp_grp_dist, fn_grp_dist = [],[],[],[]

        ## loop for the four groups and calculating the distance with the x_insitance
        for i in range(len(self.tp_shap_value_avg_centroids)):
            tp_grp_dist.append(abs(np.sum(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index]-x_instances_shap[self.top_tp_features_index])))

        for i in range(len(self.tn_shap_value_avg_centroids)):
            tp_grp_dist.append(abs(np.sum(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index]-x_instances_shap[self.top_tn_features_index])))

        for i in range(len(self.fp_shap_value_avg_centroids)):
            tp_grp_dist.append(abs(np.sum(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index]-x_instances_shap[self.top_fp_features_index])))

        for i in range(len(self.fn_shap_value_avg_centroids)):
            tp_grp_dist.append(abs(np.sum(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index]-x_instances_shap[self.top_fn_features_index])))



        dist = [tp_grp_dist, tn_grp_dist, fp_grp_dist, fn_grp_dist]



        tp_grp_mis, tn_grp_mis, fp_grp_mis, fn_grp_mis = [],[],[],[]

        ## loop for the four groups and calculating the distance with the x_insitance
        for i in range(len(self.tp_shap_value_avg_centroids)):
            tp_grp_mis.append(self.shap_value_mismatched(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index], x_instances_shap[self.top_tp_features_index]))

        for i in range(len(self.tn_shap_value_avg_centroids)):
            tn_grp_mis.append(self.shap_value_mismatched(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index], x_instances_shap[self.top_tn_features_index]))

        for i in range(len(self.fp_shap_value_avg_centroids)):
            fp_grp_mis.append(self.shap_value_mismatched(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index], x_instances_shap[self.top_fp_features_index]))

        for i in range(len(self.fn_shap_value_avg_centroids)):
            fn_grp_mis.append(self.shap_value_mismatched(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index], x_instances_shap[self.top_fn_features_index]))




        mismatches = [tp_grp_mis, tn_grp_mis, fp_grp_mis, fn_grp_mis]


        min_mismatches = [mismatches[i].index(min(mismatches[i])) for i in range(len(mismatches))]
        min_dist = [dist[i].index(min(dist[i])) for i in range(len(dist))]

        # print(dist,mismatches)
        
        # print(min_dist,min_mismatches)
        
        
        # print(lowest_dist_class, lowest_mismatch_class)  
        return [min_mismatches, min_dist]




    def disSHAP_Comparison(self, x_instances_shap_value):
        # print('check tp cluster len',len(self.tp_shap_value_avg_centroids), self.tp_shap_value_avg_centroids)
        # print('check tn cluster len',len(self.tn_shap_value_avg_centroids), self.tn_shap_value_avg_centroids)
        # print('check fp cluster len',len(self.fp_shap_value_avg_centroids), self.fp_shap_value_avg_centroids)
        # print('check fn cluster len',len(self.fn_shap_value_avg_centroids), self.fn_shap_value_avg_centroids)

        tp_grp_dist, tn_grp_dist, fp_grp_dist, fn_grp_dist = [], [], [], []
        tp_grp_mis, tn_grp_mis, fp_grp_mis, fn_grp_mis = [], [], [], []
        ## loop for the four groups and calculating the distance with the x_insitance
        for i in range(len(self.tp_shap_value_avg_centroids)):
            individual_tp_cluster_top_features = self.top_n_features(self.tp_shap_value_avg_centroids[i])
            tp_grp_dist.append(abs(np.sum(self.tp_shap_value_avg_centroids[i][individual_tp_cluster_top_features]-x_instances_shap_value[individual_tp_cluster_top_features])))
            tp_grp_mis.append(self.shap_value_mismatched(self.tp_shap_value_avg_centroids[i][individual_tp_cluster_top_features], x_instances_shap_value[individual_tp_cluster_top_features]))
            
            # tp_grp_dist.append(abs(np.sum(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index]-x_instances_shap_value[self.top_tp_features_index])))
            # tp_grp_mis.append(self.self.shap_value_mismatched(self.tp_shap_value_avg_centroids[i][self.top_tp_features_index], x_instances_shap_value[self.top_tp_features_index]))
            
        for i in range(len(self.tn_shap_value_avg_centroids)):
            individual_tn_cluster_top_features = self.top_n_features(self.tn_shap_value_avg_centroids[i])
            tn_grp_dist.append(abs(np.sum(self.tn_shap_value_avg_centroids[i][individual_tn_cluster_top_features]-x_instances_shap_value[individual_tn_cluster_top_features])))
            tn_grp_mis.append(self.shap_value_mismatched(self.tn_shap_value_avg_centroids[i][individual_tn_cluster_top_features], x_instances_shap_value[individual_tn_cluster_top_features]))

            # tn_grp_dist.append(abs(np.sum(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index]-x_instances_shap_value[self.top_tn_features_index])))
            # tn_grp_mis.append(self.self.shap_value_mismatched(self.tn_shap_value_avg_centroids[i][self.top_tn_features_index], x_instances_shap_value[self.top_tn_features_index]))
            
        for i in range(len(self.fp_shap_value_avg_centroids)):
            individual_fp_cluster_top_features = self.top_n_features(self.fp_shap_value_avg_centroids[i])
            fp_grp_dist.append(abs(np.sum(self.fp_shap_value_avg_centroids[i][individual_fp_cluster_top_features]-x_instances_shap_value[individual_fp_cluster_top_features])))
            fp_grp_mis.append(self.shap_value_mismatched(self.fp_shap_value_avg_centroids[i][individual_fp_cluster_top_features], x_instances_shap_value[individual_fp_cluster_top_features]))

            # fp_grp_dist.append(abs(np.sum(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index]-x_instances_shap_value[self.top_fp_features_index])))
            # fp_grp_mis.append(self.self.shap_value_mismatched(self.fp_shap_value_avg_centroids[i][self.top_fp_features_index], x_instances_shap_value[self.top_fp_features_index]))
            
        for i in range(len(self.fn_shap_value_avg_centroids)):
            individual_fn_cluster_top_features = self.top_n_features(self.fn_shap_value_avg_centroids[i])
            fn_grp_dist.append(abs(np.sum(self.fn_shap_value_avg_centroids[i][individual_fn_cluster_top_features]-x_instances_shap_value[individual_fn_cluster_top_features])))
            fn_grp_mis.append(self.shap_value_mismatched(self.fn_shap_value_avg_centroids[i][individual_fn_cluster_top_features], x_instances_shap_value[individual_fn_cluster_top_features]))

            # fn_grp_dist.append(abs(np.sum(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index]-x_instances_shap_value[self.top_fn_features_index])))
            # fn_grp_mis.append(self.self.shap_value_mismatched(self.fn_shap_value_avg_centroids[i][self.top_fn_features_index], x_instances_shap_value[self.top_fn_features_index]))
        
    

        dist = [min(tp_grp_dist), min(tn_grp_dist), min(fp_grp_dist), min(fn_grp_dist)]
        mismatches = [min(tp_grp_mis), min(tn_grp_mis), min(fp_grp_mis), min(fn_grp_mis)]

        return [mismatches.index(min(mismatches)), dist.index(min(dist))]


        # dist = [tp_grp_dist, tn_grp_dist, fp_grp_dist, fn_grp_dist]
        # mismatches = [tp_grp_mis, tn_grp_mis, fp_grp_mis, fn_grp_mis]

        # min_mismatches = [mismatches[i].index(min(mismatches[i])) for i in range(len(mismatches))]
        # min_dist = [dist[i].index(min(dist[i])) for i in range(len(dist))]

        # print(dist,mismatches)
        
        # print(min_dist,min_mismatches)
        
        
        # # print(lowest_dist_class, lowest_mismatch_class)  
        # return [min_mismatches, min_dist]
    




    def disSHAP_Verdict(self, X_single_incomming_instance):
        X_single_incomming_instance = np.array([X_single_incomming_instance])
        predicted_label = self.model.predict(X_single_incomming_instance)
        predicted_prob = self.model.predict_proba(X_single_incomming_instance)
        instance_shap_value = self.model_explainer(X_single_incomming_instance).values[0]
        # print(instance_shap_value)
        # exit(0)
        comparison_value = self.disSHAP_Comparison(instance_shap_value)
        # print(comparison_value)
        # exit(0)
        shap_compar_match, shap_compar_dist = comparison_value[0], comparison_value[1]
        

        if predicted_prob[0][predicted_label[0]]>=0.9:
            return predicted_label[0]
        
        # new logic

        # 0+3=3 or 1+2=3 
        # if shap_compar_dist != shap_compar_match and shap_compar_dist + shap_compar_match != 3:
        #     print("N/A, Model ---> pos | disSHAP ---> Confused")
        #     # print(shap_compar_match, shap_compar_dist)
        #     return predicted_label[0]
        
        # if (shap_compar_match == 0 or shap_compar_match == 3) or (shap_compar_dist == 0 or shap_compar_dist == 3):
        #     if predicted_label == 1:
        #         print("Agree, Model ---> pos | disSHAP ---> TP")
        #     else:
        #         print("Disagree, Model ---> neg | disSHAP ---> FP")
        #     return 1
            
        # else:
        #     if predicted_label == 0:
        #         print("Agree, Model ---> neg | disSHAP ---> TN")
        #     else:
        #         print("Disagree, Model ---> pos | disSHAP ---> FN")
        #     return 0



        if predicted_label == 1:
            if (shap_compar_match == 2 and shap_compar_dist == 2):
                print("Disgree, Model ---> pos | disSHAP ---> FP")
                return 0
            if (shap_compar_match == 0 or shap_compar_dist == 0):
                print("Agree, Model ---> pos | disSHAP ---> TP")
                return 1
            elif (shap_compar_match != 2 and shap_compar_dist != 2):
                print("Agree, Model ---> pos | disSHAP ---> TP")
                return 1
            else:
                print("N/A, Model ---> pos | disSHAP ---> Confused")
                return 1
            
        else:
            if (shap_compar_match == 3 and shap_compar_dist == 3):
                print("Disgree, Model ---> neg | disSHAP ---> FN")
                return 1
            if (shap_compar_match == 1 or shap_compar_dist == 1):
                print("Agree, Model ---> neg | disSHAP ---> TN")
                return 0
            elif (shap_compar_match != 3 and shap_compar_dist != 3):
                print("Agree, Model ---> neg | disSHAP ---> TN")
                return 0
            else:
                print("N/A, Model ---> pos | disSHAP ---> Confused")
                return 1




    def old_disSHAP_Verdict(self, X_single_incomming_instance):
        X_single_incomming_instance = np.array([X_single_incomming_instance])
        predicted_label = self.model.predict(X_single_incomming_instance)
        predicted_prob = self.model.predict_proba(X_single_incomming_instance)
        instance_shap_value = self.model_explainer(X_single_incomming_instance)
        # print(instance_shap_value)
        # exit(0)
        comparison_value = self.mitigate_misclassification_top_n(instance_shap_value)
        # print(comparison_value)
        shap_compar_match, shap_compar_dist = comparison_value[0], comparison_value[1]
        

        if predicted_prob[0][predicted_label[0]]>=0.9:
            return predicted_label[0]
        


        #0+3=3 or 1+2=3 
        # if shap_compar_dist != shap_compar_match and shap_compar_dist + shap_compar_match != 3:
        #     print("N/A, Model ---> pos | disSHAP ---> Confused")
        #     # print(shap_compar_match, shap_compar_dist)
        #     return predicted_label[0]
        
        # if predicted_label == 1:
        #     if (shap_compar_match != 2 or shap_compar_dist != 2):
        #         print("Agree, Model ---> pos | disSHAP ---> TP")
        #         return 1
        #     else:
        #         print("Disagree, Model ---> pos | disSHAP ---> FP")
        #         return 0
            
        # else:
        #     if (shap_compar_match != 3 or shap_compar_dist != 3):
        #         print("Agree, Model ---> neg | disSHAP ---> TN")
        #         return 0
        #     else:
        #         print("Disagree, Model ---> neg | disSHAP ---> FN")
        #         return 1



        # old logic

        if predicted_label == 1:
            if ((0 in shap_compar_dist or 3 in shap_compar_dist) or (0 in shap_compar_match or 3 in shap_compar_match)):
                print("Agree, Model ---> pos | disSHAP ---> TP")
                return 1
            else:
                print("Disagree, Model ---> pos | disSHAP ---> FP")
                return 0
            
        else:
            if ((1 in shap_compar_dist or 2 in shap_compar_dist) or (1 in shap_compar_match or 2 in shap_compar_match)):
                print("Agree, Model ---> neg | disSHAP ---> TN")
                return 0
            else:
                print("Disagree, Model ---> neg | disSHAP ---> FN")
                return 1

        

    

    # Clustering the SHAP values so that we can test with different avg portion


    def Automated_K_Means_Clustering_SHAP_Values(self, group_shap_value, max_k=5):

        if len(group_shap_value) < max_k + 1:
            return group_shap_value


        # Find the optimal number of clusters using Silhouette Score
        silhouette_scores = []
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = kmeans.fit_predict(group_shap_value)
            silhouette_scores.append(silhouette_score(group_shap_value, labels))

        # Optimal k is the one with the highest silhouette score
        optimal_k = np.argmax(silhouette_scores) + 2

        # Perform clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, n_init="auto", random_state=42)
        labels = kmeans.fit_predict(group_shap_value)
        # centroids  = kmeans.cluster_centers_

        # return centroids[labels]

        # Organize the clusters
        result = {}
        for i in range(optimal_k):
            result[i] = []
        for i, label in enumerate(labels):
            result[label].append(group_shap_value[i])
        
        for i in result:
            result[i] = np.mean(result[i], axis=0)
        
        return np.array(list(result.values()))
    


    def Clustering_SHAP_Values(self, shap_value):
        kmeans = KMeans(n_init="auto", n_clusters=self.cluster_num)
        kmeans.fit(shap_value)
        label = kmeans.labels_

        result = {}
        for i in range(self.cluster_num):
            result[i] = []
        for i in range(len(label)):
            result[label[i]].append(shap_value[i])
        
        for i in result:
            result[i] = np.mean(result[i], axis = 0)
        
        return result




# #-----------------implementation----------------

# data = pd.read_csv('Dataset/dataset_2.csv')

# #for dataset 2
# data['status'] = [1 if x=='phishing' else 0 for x in data['status']]
# feature_names = list(data.columns[1:-1])
# X = np.array(data[data.columns[1:-1]])
# Y = np.array(data[data.columns[-1]])


# #for dataset 1
# # feature_names = data.columns[:-1]
# # X = np.array(data[data.columns[:-1]])
# # Y = np.array(data[data.columns[-1]])

# X_train, Y_train = [], []
# X_test, Y_test = [], []
# X_val, Y_val = [], []

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=42)



# model = xgboost.XGBClassifier(max_depth=5, objective='binary:logistic', n_estimators=100)
# model.fit(X_train, Y_train)

# Y_pred_test = model.predict(X_test)
# test_report = classification_report(Y_test, Y_pred_test)
# print("\n\nModel's test report\n", test_report)


# Y_pred_val = model.predict(X_val)
# old_acc, old_pre, old_recall, old_f1 = accuracy_score(Y_val, Y_pred_val), precision_score(Y_val, Y_pred_val), recall_score(Y_val, Y_pred_val), f1_score(Y_val, Y_pred_val) 



# example_obj = disSHAP(model, np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test), feature_names, 'T')

# Y_pred_val_shap = [example_obj.disSHAP_Verdict(i) for i in X_val]
# accuracy = accuracy_score(Y_val, Y_pred_val_shap)
# precision = precision_score(Y_val, Y_pred_val_shap)
# recall = recall_score(Y_val, Y_pred_val_shap)
# f1 = f1_score(Y_val, Y_pred_val_shap)
# print("\n\nOld Accuracy:", (old_acc)*100, "Old Precision:", (old_pre)*100, "Old Recall:", (old_recall)*100, "Old F1-Score:", (old_f1)*100)
# print("New Accuracy:", (accuracy)*100, "New Precision:", (precision)*100, "New Recall:", (recall)*100, "New F1-Score:", (f1)*100)
# print("Accuracy dif:", (accuracy-old_acc)*100, "Precision dif:", (precision-old_pre)*100, "Recall dif:", (recall-old_recall)*100, "F1-Score dif:", (f1-old_f1)*100,'\n\n\n')