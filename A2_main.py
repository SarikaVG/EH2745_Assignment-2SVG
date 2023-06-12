# -*- coding: utf-8 -*-
"""
#Assignment 2 combines Python programming, combining the machine learning techniques and the 
power system modeling techniques.

@author: Sarika Vaiyapuri Gunassekaran
"""

import pandapower as pp
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.control import ConstControl
from collections import Counter
import os
import numpy as np
import pandas as pd
import tempfile
import matplotlib.pyplot as plt

# Power Grid Model
def powergrid_net():

    net = pp.create_empty_network()
    
    Bus_Voltage = 110 # in kv
    
    #Bus Data
    Bus1 = pp.create_bus(net, Bus_Voltage, name='Bus 1-CLARK-R1')
    Bus2 = pp.create_bus(net, Bus_Voltage, name='Bus 2-AMHERST-R1')
    Bus3 = pp.create_bus(net, Bus_Voltage, name='Bus 3-WINLOCK-R1')
    Bus4 = pp.create_bus(net, Bus_Voltage, name='Bus 4-BOWMAN-R2')
    Bus5 = pp.create_bus(net, Bus_Voltage, name='Bus 5-TROY-R2')
    Bus6 = pp.create_bus(net, Bus_Voltage, name='Bus 6-MAPLE-R2')
    Bus7 = pp.create_bus(net, Bus_Voltage, name='Bus 7-GRAND-R3')
    Bus8 = pp.create_bus(net, Bus_Voltage, name='Bus 8-WAUTAGA-R3')
    Bus9 = pp.create_bus(net, Bus_Voltage, name='Bus 9-CROSS-R3')
    
    
    # Generator data
    
    Gen1_MW = 0
    Gen2_MW = 163
    Gen3_MW = 85
    
    # Q generated is 0 for all generators
    
    Gen1 = pp.create_gen(net, Bus1,Gen1_MW, slack=True, name='Generator 1')
    Gen2 = pp.create_sgen(net, Bus2,Gen2_MW, q_mvar=0, name='Generator 2')
    Gen3 = pp.create_sgen(net, Bus3,Gen3_MW, q_mvar=0, name='Generator 3')


    # In order to create the loads we need to define the P and Q consumed by each load
    Load5_MW = 90
    Load5_MVAR = 30
    Load7_MW = 100
    Load7_MVAR = 35
    Load9_MW = 125
    Load9_MVAR = 50
    
    Load5 = pp.create_load(net, Bus5, Load5_MW , Load5_MVAR, name='Load 5')
    Load7 = pp.create_load(net, Bus7, Load7_MW , Load7_MVAR, name='Load 7')
    Load9 = pp.create_load(net, Bus9, Load9_MW , Load9_MVAR, name='Load 9')
    
    
    Line_length = 10 # in km
    Line1 = pp.create_line(net, Bus1, Bus4, Line_length, '149-AL1/24-ST1A 110.0', name='Line 1 to 4')
    Line2 = pp.create_line(net, Bus2, Bus8, Line_length, '149-AL1/24-ST1A 110.0', name='Line 2 to 8')
    Line3 = pp.create_line(net, Bus3, Bus6, Line_length, '149-AL1/24-ST1A 110.0', name='Line 3 to 6')
    Line4 = pp.create_line(net, Bus4, Bus5, Line_length, '149-AL1/24-ST1A 110.0', name='Line 4 to 5')
    Line5 = pp.create_line(net, Bus4, Bus9, Line_length, '149-AL1/24-ST1A 110.0', name='Line 4 to 9')
    Line6 = pp.create_line(net, Bus5, Bus6, Line_length, '149-AL1/24-ST1A 110.0', name='Line 5 to 6')
    Line7 = pp.create_line(net, Bus6, Bus7, Line_length, '149-AL1/24-ST1A 110.0', name='Line 6 to 7')
    Line8 = pp.create_line(net, Bus7, Bus8, Line_length, '149-AL1/24-ST1A 110.0', name='Line 7 to 8')
    Line9 = pp.create_line(net, Bus8, Bus9, Line_length, '149-AL1/24-ST1A 110.0', name='Line 8 to 9')
    
    
    return net

# Timeseries Power Flow Implementation, creation of operation states for varying P and Q
def create_data_source(net, n_timesteps=30,state=''):
  profiles = pd.DataFrame()
  if state == 'High Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 1.05 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 1.05 * net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) * net.load.q_mvar[i])
  elif state == 'Low Load':
        for i in range(len(net.load)):
            profiles['load{}_P'.format(str(i))] = 0.90 * net.load.p_mw[i] + (0.05 * np.random.random(n_timesteps) * net.load.p_mw[i])
            profiles['load{}_Q'.format(str(i))] = 0.90 *  net.load.q_mvar[i] + (0.05 * np.random.random(n_timesteps) *  net.load.q_mvar[i])
  ds = DFData(profiles)
  return profiles, ds
# Controller to vary P and Q of the bus and load
def create_controllers(net, ds):
    for i in range(len(net.load)):
        ConstControl(net, element='load', variable='p_mw', element_index=[i],
                     data_source=ds, profile_name=['load{}_P'.format(str(i))])
        ConstControl(net, element='load', variable='q_mvar', element_index=[i],
                     data_source=ds, profile_name=['load{}_Q'.format(str(i))])

    return net
# Output writer to save the results of the time series analysis
def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls", log_variables=list())
    # these variables are saved to the harddisk after / during the time series loop
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')
    return ow

net = powergrid_net()
n_time_steps=60

def high_load(net, n_time_steps):
    _net = net
    pf, ds = create_data_source(_net,n_timesteps=n_time_steps, state='High Load')
    _net = create_controllers(net, ds)
    time_steps = range(0, n_time_steps)
    print(time_steps)
    ow = create_output_writer(_net, time_steps, './High Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    
def low_load(net, n_time_steps):
    _net = net
    pf, ds = create_data_source(_net,n_timesteps=n_time_steps, state='Low Load')
    _net = create_controllers(net, ds)
    time_steps = range(0, n_time_steps)
    print(time_steps)
    ow = create_output_writer(_net, time_steps, './Low Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)


def high_load_gen3_discon(net, n_time_steps):
    _net = net
    index_sgen = pp.get_element_index(_net, 'sgen', 'Generator 3')    
    _net.sgen.in_service[index_sgen] = False
    pf, ds = create_data_source(_net, n_timesteps=n_time_steps,state='High Load')
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, './G3 High Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[index_sgen] = True

def low_load_gen3_discon(net, n_time_steps):
    _net = net
    index_sgen = pp.get_element_index(_net, 'sgen', 'Generator 3')
    _net.sgen.in_service[index_sgen] = False
    pf, ds = create_data_source(_net, n_timesteps=n_time_steps,state='Low Load')
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, './G3 Low Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.sgen.in_service[index_sgen] = True

def high_load_line8_discon(net, n_time_steps):
    _net = net
    index_line = pp.get_element_index(_net, 'line', 'Line 5 to 6')
    _net.line.in_service[index_line] = False
    pf, ds = create_data_source(_net, n_timesteps=n_time_steps,state='High Load')
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, './Line8 High Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line] = True

def low_load_line8_discon(net, n_time_steps):
    _net = net
    index_line = pp.get_element_index(_net, 'line', 'Line 5 to 6')
    _net.line.in_service[index_line] = False
    pf, ds = create_data_source(_net, n_timesteps=n_time_steps,state='Low Load')
    _net = create_controllers(_net, ds)
    time_steps = range(0, n_time_steps)
    ow = create_output_writer(_net, time_steps, './Line8 Low Load')
    run_timeseries(_net, time_steps, calculate_voltage_angles=True)
    _net.line.in_service[index_line] = True

# To read the files for each status
high_load(net, n_time_steps)
low_load(net, n_time_steps)
high_load_gen3_discon(net, n_time_steps)
low_load_gen3_discon(net, n_time_steps)
high_load_line8_discon(net, n_time_steps)
low_load_line8_discon(net, n_time_steps)

# Voltage in pu and angles are read from the excel files and merged
#HIGH LOAD
high_load_vpu_file = os.path.join('./High Load', "res_bus", "vm_pu.xls")
vpu_high_load = pd.read_excel(high_load_vpu_file, index_col=0)

high_load_angle_file = os.path.join('./High Load', "res_bus", "va_degree.xls")
angle_high_load = pd.read_excel(high_load_angle_file, index_col=0)

high_load_df = pd.concat([vpu_high_load, angle_high_load], axis=1, ignore_index=True)
high_load_df['check_os'] = 'high load'

# LOW LOAD
low_load_vpu_file = os.path.join('./Low Load', "res_bus", "vm_pu.xls")
vpu_low_load = pd.read_excel(low_load_vpu_file, index_col=0)

low_load_angle_file = os.path.join('./Low Load', "res_bus", "va_degree.xls")
angle_low_load = pd.read_excel(low_load_angle_file, index_col=0)

low_load_df = pd.concat([vpu_low_load, angle_low_load], axis=1, ignore_index=True)
low_load_df['check_os'] = 'low load'

# GEN3 DISCONECTED HIGH
vpu_gen3_disc_file_high = os.path.join('./G3 High Load', "res_bus", "vm_pu.xls")
vpu_gen3_disc_high = pd.read_excel(vpu_gen3_disc_file_high, index_col=0)

angle_gen3_disc_file_high = os.path.join('./G3 High Load', "res_bus", "va_degree.xls")
angle_gen3_disc_high = pd.read_excel(angle_gen3_disc_file_high, index_col=0)

gen_disc_high_df = pd.concat([vpu_gen3_disc_high, angle_gen3_disc_high], axis=1, ignore_index=True)
gen_disc_high_df['check_os'] = 'Generator disconnected high'

# GEN3 DISCONECTED LOW
vpu_gen3_disc_file_low = os.path.join('./G3 Low Load', "res_bus", "vm_pu.xls")
vpu_gen3_disc_low = pd.read_excel(vpu_gen3_disc_file_low, index_col=0)

angle_gen3_disc_file_low = os.path.join('./G3 Low Load', "res_bus", "va_degree.xls")
angle_gen3_disc_low = pd.read_excel(angle_gen3_disc_file_low, index_col=0)

gen_disc_low_df = pd.concat([vpu_gen3_disc_low, angle_gen3_disc_low], axis=1, ignore_index=True)
gen_disc_low_df['check_os'] = 'Generator disconnected low'

# LINE 5-6 DISCONNECTED HIGH LOAD
vpu_line_disc_file_high = os.path.join('./Line8 High Load', "res_bus", "vm_pu.xls")
vpu_line_disc_high = pd.read_excel(vpu_line_disc_file_high, index_col=0)


angle_line_disc_file_high = os.path.join('./Line8 High Load', "res_bus", "va_degree.xls")
angle_line_disc_high = pd.read_excel(angle_line_disc_file_high, index_col=0)

line_disc_high_df = pd.concat([vpu_line_disc_high, angle_line_disc_high], axis=1, ignore_index=True)
line_disc_high_df['check_os'] = 'Line disconnected high'

# LINE 5-6 DISCONNECTED LOW LOAD
vpu_line_disc_file_low = os.path.join('./Line8 Low Load', "res_bus", "vm_pu.xls")
vpu_line_disc_low = pd.read_excel(vpu_line_disc_file_low, index_col=0)


angle_line_disc_file_low = os.path.join('./Line8 Low Load', "res_bus", "va_degree.xls")
angle_line_disc_low = pd.read_excel(angle_line_disc_file_low, index_col=0)

line_disc_low_df = pd.concat([vpu_line_disc_low, angle_line_disc_low], axis=1, ignore_index=True)
line_disc_low_df['check_os'] = 'Line disconnected low'


dataset = pd.concat([high_load_df, low_load_df, gen_disc_high_df, gen_disc_low_df, line_disc_high_df, line_disc_low_df], 
                  axis=0, ignore_index=True)
print(np.shape(dataset))

# Plotting voltage and angle for all buses
def plot_simulation_result():
    fig, ax = plt.subplots(nrows=6, figsize=(6, 12))
    color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive']
    voltage_df = [vpu_high_load, vpu_low_load,
                      vpu_gen3_disc_high, vpu_gen3_disc_low,
                      vpu_line_disc_high, vpu_line_disc_low]
    angle_df = [angle_high_load, angle_low_load,
                    angle_gen3_disc_high, angle_gen3_disc_low,
                    angle_line_disc_high, angle_line_disc_low]
    title_list = ['Base configuration, high load',
                      'Base configuration, low load',
                      'Gen 3 disconnected, high load',
                      'Gen 3 disconnected, low load',
                      'Line 5-6 disconnected, high load', 
                      'Line 5-6 disconnected, low load']
    for j in range(0, 6):
          for i in range(0, 9):
              ax[j].scatter(voltage_df[j][i], angle_df[j][i], c=color[i], s=5, label='Bus {}'.format(i + 1))
              box = ax[j].get_position()
              ax[j].set_position([-0.075, box.y0, box.width, box.height])
              ax[j].set_title(title_list[j])
              ax[j].set_xlabel('Voltage')
              ax[j].set_ylabel('Angle')
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',ncol=1, fancybox=True, shadow=True)
    plt.show()
    fig_file_name = 'plot.png'
    fig.savefig(fig_file_name)  
plot = plot_simulation_result()
dataset_for_normalization = dataset.drop(['check_os'], axis=1)
data_normalized = dataset.copy()
print(dataset)

# Converting to normalized dataset using min-max normalization method

for i in range(1, 9):
    data_normalized[i] = np.divide(dataset_for_normalization[i] - dataset_for_normalization[i].min(),
                                      dataset_for_normalization[i].max() - dataset_for_normalization[i].min())

for i in range(10, 18):
    data_normalized[i] = np.divide(dataset_for_normalization[i] - dataset_for_normalization[i].min(),
                                      dataset_for_normalization[i].max() - dataset_for_normalization[i].min())

dataset_norm_labeled = data_normalized.copy()
dataset_norm_labeled['check_os'] = dataset['check_os'].copy()


# Split dataset into training data and test data
train_coefficient = 0.8
n_training = int(train_coefficient * n_time_steps)
dataset_train = pd.concat([dataset_norm_labeled[:n_training],
                           dataset_norm_labeled[n_time_steps:n_training + n_time_steps],
                           dataset_norm_labeled[2 * n_time_steps:n_training + 2 * n_time_steps],
                           dataset_norm_labeled[3 * n_time_steps:n_training + 3 * n_time_steps],
                           dataset_norm_labeled[4 * n_time_steps:n_training + 4 * n_time_steps],
                           dataset_norm_labeled[5 * n_time_steps:n_training + 5 * n_time_steps]],
                          axis=0, ignore_index=True)

dataset_test= pd.concat([dataset_norm_labeled[n_training:n_time_steps],
                         dataset_norm_labeled[n_training + n_time_steps:2 * n_time_steps],
                         dataset_norm_labeled[n_training + 2 * n_time_steps:3 * n_time_steps],
                         dataset_norm_labeled[n_training + 3 * n_time_steps:4 * n_time_steps],
                         dataset_norm_labeled[n_training + 4 * n_time_steps:5 * n_time_steps],
                         dataset_norm_labeled[n_training + 5 * n_time_steps:6 * n_time_steps]],
                        axis=0, ignore_index=True)

dataset.to_excel("dataset.xlsx")
dataset_norm_labeled = dataset_norm_labeled.sample(frac=1).reset_index(drop=True)
dataset_norm_labeled.to_excel("dataset_norm_labeled.xlsx")

dataset_train = dataset_train.sample(frac=1).reset_index(drop=True)
dataset_test= dataset_test.sample(frac=1).reset_index(drop=True)

Dataset = dataset.to_numpy()
Dataset_norm = data_normalized.to_numpy()

# k-means clustering algorithm
# Create the KMeans class, which will execute the clustering process.
#The number of maximum iterations for the elbow approach and locating local optimum as an input  Zero output
       
class kMeans:
   def __init__(self, max_iters=100):
        self.max_iters = max_iters
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

# Prdection is done by first initializing centroids and using elbow method find the minimum values
   def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

 # Elbow method:
        j_diff_threshold = 45 #Manually tuned
        diff_k_clusters = []
        for k in range(1,8):
 # Finding Local optima through iteration:
            clusters = []
            for i in range(self.max_iters):
                random_sample_idxs = np.random.choice(self.n_samples, k, replace=False)
                self.centroids = [self.X[idx] for idx in random_sample_idxs]
                self.clusters = [[] for _ in range(k)]
                for _ in range(self.max_iters):
                    self.clusters = self._create_clusters(self.centroids, k)
                    centroids_old = self.centroids
                    self.centroids = self._get_centroids(self.clusters, k)
                    if self._is_converged(centroids_old, self.centroids, k):
                        break
 # Classify samples as the index of their clusters     
                result = self._get_cluster_labels(self.clusters)
                j = self._j_metric(self.clusters, self.centroids)
                new_clusters = Clusters(i, k, self.centroids, self.clusters, result, j)
                clusters.append(new_clusters)

            min_j_idx = Clusters.lowest_j_idx
        
            diff_k_clusters.append(clusters[min_j_idx])
# Calculate j metrics 
            if k > 1:
                j_diff = diff_k_clusters[k-2].j - diff_k_clusters[k-1].j
                if j_diff <= j_diff_threshold:
                    #plot bus1
                    centroids_bus1 = np.array(diff_k_clusters[k-1].centroids)[:, [0,9]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus1, 0, 9, "Bus 1")
                    #plot bus2
                    centroids_bus2 = np.array(diff_k_clusters[k-1].centroids)[:, [1,10]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus2, 1, 10, "Bus 2")
                    #plot bus3
                    centroids_bus3 = np.array(diff_k_clusters[k-1].centroids)[:, [2,11]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus3, 2, 11, "Bus 3")
                    #plot bus4
                    centroids_bus4 = np.array(diff_k_clusters[k-1].centroids)[:, [3,12]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus4, 3, 12, "Bus 4")
                    #plot bus5
                    centroids_bus5 = np.array(diff_k_clusters[k-1].centroids)[:, [4,13]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus5, 4, 13, "Bus 5")
                    #plot bus6
                    centroids_bus6 = np.array(diff_k_clusters[k-1].centroids)[:, [5,14]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus6, 5, 14, "Bus 6")
                    #plot bus7
                    centroids_bus7 = np.array(diff_k_clusters[k-1].centroids)[:, [6,15]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus7, 6, 15, "Bus 7")                    
                    #plot bus8
                    centroids_bus8 = np.array(diff_k_clusters[k-1].centroids)[:, [7,16]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus8, 7, 16, "Bus 8")  
                    #plot bus9
                    centroids_bus9 = np.array(diff_k_clusters[k-1].centroids)[:, [8,17]]
                    self.plot(diff_k_clusters[k-1].sample_idx, centroids_bus9, 8, 17, "Bus 9")  

                    return diff_k_clusters[k-1].label, k

        def _create_clusters(self, centroids, k):
            clusters = [[] for _ in range(k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

        def _closest_centroid(self, sample, centroids):
            distances = [euclidean_distance(sample, point) for point in centroids]
            closest_index = np.argmin(distances)
            return closest_index

        def _get_centroids(self, clusters, k):           
            centroids = np.zeros((k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

        def _is_converged(self, centroids_old, centroids, k):
            distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(k)
        ]
        return sum(distances) == 0

        def _get_cluster_labels(self, clusters):
            labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

        def _j_metric(self, clusters, centroids):
            j = 0
        for cluster_idx, k in enumerate(clusters):
            centroid = centroids[cluster_idx]
            for idx in k:
                sample = self.X[idx]
                j += euclidean_distance(sample, centroid)
        return j
    
        def plot(self, clusters, centroids, col1, col2, title):
              
            fig, ax = plt.subplots(figsize=(15, 10))

        for i, index in enumerate(clusters):
            point = np.array(self.X[index])[:,[col1, col2]].T
            ax.scatter(*point)

        for point in centroids:
            ax.scatter(*point, marker="x", linewidth=2)

        plt.xlabel("Voltage Magnitude (normalized, in %)")
        plt.ylabel("Voltage Angle (normalized, in %)")
        plt.title(title)
        plt.show()
        
# Knn nearest neighbors algorithm
# euclidean distance method to find the nearest value
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

class knn:
    def __init__(self, K):
        self.K = K

    def fit(self, X, Y):
        self.X_train = X
        self.y_train = Y

    def predict_knn (self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

# To compute distances beween x and all sampleds in the training set
    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_idx = np.argsort(distances)[:self.K]
        k_neighbor_labels = [self.y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)
        return most_common[0][0]
            
# create the training set
X_training_set = dataset_train.drop(['check_os'], axis=1).to_numpy()
y_training_set = dataset_train['check_os'].to_numpy()

# create the testing set
X_testing_set = dataset_test.drop(['check_os'], axis=1).to_numpy()
y_testing_set = dataset_test['check_os'].to_numpy()

print(X_training_set)
print(y_training_set)
print(X_testing_set)
print(y_testing_set)

clf = knn(K=6)

clf.fit(X_training_set, y_training_set)
prediction = clf.predict_knn(X_testing_set)
print("prediction", prediction)
print("test data", y_testing_set)
print("Accuracy ", accuracy(y_testing_set, prediction))



