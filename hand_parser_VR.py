# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 11:54:05 2020

@author: TomNissens
"""

import pandas as pd
import numpy as np
from scipy import signal, interpolate
import os
#import sys
#import math
from time import time
#import operator
import matplotlib.pyplot as plt
import seaborn as sns

class Parser:
    def __init__(self):
        write_csv = 0
        self.plot_figs = 1
        
        self.count_resample_error = 0
        self.participants = {}
        self.resample_nr = 101
        # Walk through all files in the current directory, but analyze them only if they are .asc files
        self.session = {}
        nroffiles = 0
        for file in os.listdir("./data"):
            (root, ext) = os.path.splitext(file)
            if ext == ".csv" and root[:9] == "data_hand":
                
                #first load data_log for pp
                log = pd.read_csv("./data/data_log_" + root[10:] + ext)
                #only get experimtal trials
                self.data_log = log[log['session'] == 'experiment']
                #reset index
                self.data_log.index = range(len(self.data_log.index))
                
                self.pp_code = root[10:14]
                self.participants[self.pp_code] = {'pp_nr': root[10:12], 'pp_id': root[12:14]}
                print(self.pp_code)
                
                nroffiles = nroffiles + 1
                self.session[self.pp_code] = self.parse(file)
        
        if self.plot_figs:
            self.plotAvgTrajOverall(endpoint_x = 1)
        
        if self.count_resample_error > 0:
            raise RuntimeError('total nr of trials failed to resample (outside of current exclussion criteria): ' + str(self.count_resample_error))
        
        # write (in)dependent variables to file
        if write_csv:
            header_ready = 0
            header_str = ''
            header = []
            with open('parsed_data_{}.csv'.format(str(int(time()))), 'w') as f:
                for pp, pp_data in self.session.items():
                    for trialid, trial in pp_data.items():
                        one_trial_str = ''
                        for attrname, attrval in trial.items():
                            if header_ready == 0:
                                header_str += (str(attrname) + ', ')
                                header.append(attrname)
                            else:
                                if not attrname in header:
                                    ValueError('trial keys not same')
                            one_trial_str += (str(attrval) + ', ')
                        if header_ready == 0:
                            header_str = header_str[:-2]
                            header_str += '\n'
                            f.write(header_str)
                            header_ready = 1
                            del header_str
                        one_trial_str = one_trial_str[:-2]
                        one_trial_str += '\n'
                        f.write(one_trial_str)
         
    def parse(self, file):
        trial = {}
        parsing = 0
        trialid = -1
        
        # Process the file line-by-line
        for line in open(os.path.join("data", file)):
            # Split the line into separate words and convert these into floats and integers if possible
            str_word = line.split()
            word = []
            for term in str_word:
                try:
                    if "." in term:
                        word.append(float(term))
                    else:
                        word.append(int(term))
                except:
                    word.append(term)
                    
            #get pp nr and id
            if word[2] == 'file':
#                self.pp_id = word[4]
#                self.pp_nr = word[3]
                pass
            #detect trial start
            if word[1] == 'start_trial':
                trialid = int(word[2])
                trial[trialid] = {'TRIALNR' : trialid, 'FILE' : file, 'TRIAL_START' : int(word[0])}
                # Initialize the trial
                self.initialize(trial[trialid])
            elif word[0] == 'VAR':
                trial[trialid][word[1]] = word[2]
            elif word[0] == 'COORD':
                trial[trialid]['coord_'+word[1]] = [word[2], word[3], word[4]]
            elif word[2] == 'start_task':
                parsing = 1
                trial[trialid]['hand_samples'] = np.array([[0.0,0.0,0.0,0.0]])
                trial[trialid]['start_task_t'] = word[1]
            elif word[2] == 'on_shape':
                trial[trialid]['on_shape_t'] = word[1]
            elif word[2] == 'end_task':
                parsing = 0
                trial[trialid]['hand_samples'] = trial[trialid]['hand_samples'][1:,:]
                velocities = []
                #calculate velocities
                for idx in range(1,len(trial[trialid]['hand_samples'])):
                    velocities.append(self.velocity_calc(trial[trialid]['hand_samples'][idx-1],trial[trialid]['hand_samples'][idx]))
                trial[trialid]['velocities'] = velocities
                
                #interpolate missing samples
                trial[trialid]['hand_samples_interp'] = self.interpolate_t_3D(trial[trialid]['hand_samples'])
                    
                
                #filter
#                w = 12 / (90/2)
                b, a = signal.butter(5,12,'lowpass', fs = 90)
                output = np.empty_like(trial[trialid]['hand_samples_interp'])
                output[:,0] = trial[trialid]['hand_samples_interp'][:,0]
                for axis in range(1,4):
                    output[:,axis] = signal.filtfilt(b,a,trial[trialid]['hand_samples_interp'][:,axis])
                trial[trialid]['hand_samples_interp_filt'] = output
                #calculate filtered velocities
                velocities = []
                for idx in range(1,len(trial[trialid]['hand_samples_interp_filt'])):
                    velocities.append(self.velocity_calc(output[idx-1],output[idx]))
                    #get velocity at shape touch detection
                    if not trial[trialid]['on_shape_t'] == -1 and trial[trialid]['on_shape_velo'] == -1 and trial[trialid]['hand_samples_interp'][idx,0] > trial[trialid]['on_shape_t']:
                        trial[trialid]['on_shape_velo'] = velocities[-1]
                        trial[trialid]['on_shape_samplenr'] = len(velocities)
                    if velocities[-1] > 0.2 and trial[trialid]['move_start_samplenr'] == -1:
                        trial[trialid]['move_start_samplenr'] = len(velocities)
                        trial[trialid]['reach_latency'] = trial[trialid]['hand_samples_interp_filt'][len(velocities),0] - trial[trialid]['start_task_t']
                    elif velocities[-1] < 0.5 and not trial[trialid]['on_shape_samplenr'] == -1 and trial[trialid]['move_end_samplenr'] == -1:
                        trial[trialid]['move_end_samplenr'] = len(velocities)
                        
                trial[trialid]['velocities_filtered'] = velocities
                # #get min velocity after touch; note the samplenr is based on the non interpolated sample data
                # trial[trialid]['velocity_min'] = min(velocities[trial[trialid]['on_shape_samplenr']:])
                
                if not self.data_log.finger_on_shape[trialid] == -1 and trial[trialid]['exp_type'] == 'experiment':
                    #remove reach with start-stop-start based on velocity
                    trial[trialid]['reach_interuption'] = self.detectReachInterupt(np.copy(trial[trialid]['velocities_filtered'][trial[trialid]['move_start_samplenr']:trial[trialid]['on_shape_samplenr']]))
                    
                    if trial[trialid]['reach_interuption'] == 0:
                        #cut trajectory from samples
                        trial[trialid]['traj'] = np.copy(trial[trialid]['hand_samples_interp_filt'][trial[trialid]['move_start_samplenr']:trial[trialid]['on_shape_samplenr']])
                        #resample traj
                        trial[trialid]['traj_resample'], trial[trialid]['resample_success'] = self.resamplenorm(np.copy(trial[trialid]['traj']), self.resample_nr)
                        if trial[trialid]['resample_success'] == 1:
                            trial[trialid]['reach_end_coord'] = trial[trialid]['traj'][-1,1:4]
                                                
                if trial[trialid]['exp_type'] == 'practice':
                    del trial[trialid]
                
            elif parsing == 1 and not word[0] == 'MSG':
                trial[trialid]['hand_samples'] = np.append(trial[trialid]['hand_samples'],[[word[0], word[1], word[2], word[3]]],0)

        self.pp_process(trial)
        if self.plot_figs:
            self.pp_figs(trial)

        
        trial = self.cleanup_trial_forLogging(trial)
        
        return trial
        
    def pp_process(self,trial):
        self.participants[self.pp_code]['overall_traj'] = np.zeros([self.resample_nr,3])
        for tar_pos in range(4):
            '''
            get average trajectory for different conditions
            No Distractor condition: tar_pos 0,1,2,3
            '''
            self.participants[self.pp_code]['baseline_traj'+str(tar_pos)] = np.zeros([self.resample_nr,3])
            trial_count = 0
            for t_nr in self.data_log.index[(self.data_log.tar_pos == tar_pos) & (self.data_log.dis_condition == 'same') & (self.data_log.finger_on_target == 1)]:
                if trial[t_nr]['resample_success'] == 1:
                    self.participants[self.pp_code]['baseline_traj'+str(tar_pos)] += trial[t_nr]['traj_resample']
                    trial_count += 1
            self.participants[self.pp_code]['baseline_traj'+str(tar_pos)] /= trial_count     
            '''
            With Distractor condition: tar_pos 0,1,2,3 * dis_pos_rel -3,-2,-1,1,2,3
                possible combinations: tar_pos 0 * dis_pos 1,2,3
                                    tar_pos 1 * dis_pos -1,1,2
                                    tar_pos 2 * dis_pos -2,-1,1
                                    tar_pos 3 * dis_pos -3,-2,-1
            '''
            if tar_pos == 0:
                dis_range = [1,2,3]
            elif tar_pos == 1:
                dis_range = [-1,1,2]
            elif tar_pos == 2:
                dis_range = [-2,-1,1]
            elif tar_pos == 3:
                dis_range = [-3,-2,-1]
            for dis_pos in dis_range:
                self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)] = np.zeros([self.resample_nr,3])
                trial_count = 0
                for t_nr in self.data_log.index[(self.data_log.tar_pos == tar_pos) & (self.data_log.dis_pos_rel == dis_pos) & (self.data_log.dis_condition == 'different') & (self.data_log.finger_on_target == 1)]:
                    if trial[t_nr]['resample_success'] == 1:
                        self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)] += trial[t_nr]['traj_resample']
                        trial_count += 1
                        
                        #plot pp1, tarpos 0, dispos 2, individual trial trajs
#                        if self.pp_code == '02is' and tar_pos == 1 and dis_pos == 2:
#                                    fig = plt.figure()
#                                    x = trial[t_nr]['traj_resample'][:,0]
#                                    y = trial[t_nr]['traj_resample'][:,1]
#                                    z = trial[t_nr]['traj_resample'][:,2]
#                            
#                                    ax = plt.axes(projection='3d')
#                                    #line
#                                    ax.plot3D(x,z,y, '--', color = [0.2,0.2,0.0])
#                                    #scatter
#                                    ax.scatter3D(x,z,y, s=30, c = z, cmap='Wistia', marker = 'o', depthshade=False, edgecolors='none')
#                                    ax.set_title('Trial nr ' + str(t_nr))
#                                    ax.view_init(30, -90)
#                                    plt.show()
                        
                self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)] /= trial_count
                if dis_pos < 0:
                    self.participants[self.pp_code]['overall_traj'] -= (self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)] - self.participants[self.pp_code]['baseline_traj'+str(tar_pos)])
                else:
                    self.participants[self.pp_code]['overall_traj'] += (self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)] - self.participants[self.pp_code]['baseline_traj'+str(tar_pos)])
        self.participants[self.pp_code]['overall_traj'] /= 12
        
    def pp_figs(self,trial):
        '''
        visualize trajectories, velocity, latency distribution etc.
        '''
#        from mpl_toolkits import mplot3d
#        weird_trials = [369,470]
#        # plot one trial velocities at different preprocess levels
#        if self.pp_code == '02is':
#            for trialidx in weird_trials:
#                fig, ax = plt.subplots(2,1)
#                x = range(len(trial[trialidx]['velocities']))
#                ax[0].plot(x,trial[trialidx]['velocities'])
#                x = range(len(trial[trialidx]['velocities_filtered']))
#                ax[1].plot(x,trial[trialidx]['velocities_filtered'], '*-')
#                ax[1].axvline(trial[trialidx]['on_shape_samplenr']-1,ls='--',color='r')
#                ax[1].axvline(trial[trialidx]['move_start_samplenr']-1,ls='--',color='k')
#                ax[1].axvline(trial[trialidx]['move_end_samplenr']-1,ls='--',color='k')
#                plt.show()

            #plot random trajectory
            
#            x = trial[369]['hand_samples_interp_filt'][:,1]
#            y = trial[369]['hand_samples_interp_filt'][:,2]
#            z = trial[369]['hand_samples_interp_filt'][:,3]
#            fig = plt.figure()
#            ax = fig.add_subplot(1, 2, 1, projection='3d') #plt.axes(projection='3d')
#            #line
#            ax.plot3D(x,z,y, '--', color = [0.5,0.5,0.25])
#            #scatter
#            ax.scatter3D(x,z,y, s=50, c = z, cmap='Wistia', marker = 'o', depthshade=False, edgecolors='none')
#            ax.view_init(30, -90)
#            
#            #plot random reach trajectory
#            x = trial[369]['traj_resample'][:,0]
#            y = trial[369]['traj_resample'][:,1]
#            z = trial[369]['traj_resample'][:,2]
#    
#            ax = fig.add_subplot(1, 2, 2, projection='3d') #plt.axes(projection='3d')
#            #line
#            ax.plot3D(x,z,y, '--', color = [0.2,0.2,0.0])
#            #scatter
#            ax.scatter3D(x,z,y, s=30, c = z, cmap='Wistia', marker = 'o', depthshade=False, edgecolors='none')
#            ax.view_init(30, -90)
#            plt.show()
        
        #plot histogram of velocity at shape touch
#        on_shape_velo = []
#        velomin = []
#        for i in range(len(trial)):
#            if self.data_log['finger_on_target'][i]:
#                on_shape_velo.append(trial[i]['on_shape_velo'])
#                velomin.append(trial[i]['velocity_min'])
#        fig, ax = plt.subplots(2,1)
#        n, bins, patches = ax[0].hist(on_shape_velo,100)
#        ax[0].grid(True)
#        n, bins, patches = ax[1].hist(velomin,100)
#        ax[1].grid(True)
#        plt.show()

        #plot histogram of reach_end_coord
        # custom bins give warning -> ignore
        # Warning= RuntimeWarning: invalid value encountered in true_divide return n/db/n.sum(), bin_edges
        with np.errstate(divide='ignore',invalid='ignore'):
            fixed_bin = np.arange(-0.25,0.25,0.005)
            # fixed_bin = 50
            fig, axes = plt.subplots(4,1)
            for tar_pos in range(4):
                reach_ends_base = []
                reach_ends_dist_left = []
                reach_ends_dist_right = []
                for tr in self.data_log.index[(self.data_log.tar_pos == tar_pos) & (self.data_log.finger_on_target == 1)]:
                    if trial[tr]['resample_success'] == 1:
                        if trial[tr]['dis_condition'] == 'same':
                            reach_ends_base.append(trial[tr]['reach_end_coord'][0])
                        elif trial[tr]['dis_condition'] == 'different':
                            if trial[tr]['dis_pos_rel'] > 0:
                                reach_ends_dist_right.append(trial[tr]['reach_end_coord'][0])
                            if trial[tr]['dis_pos_rel'] < 0:
                                reach_ends_dist_left.append(trial[tr]['reach_end_coord'][0])
                # axes[tar_pos].hist(reach_ends_base,bins = fixed_bin, alpha = 0.5, color = 'b', density=True, label='base')
                # axes[tar_pos].hist(reach_ends_dist_left,bins = fixed_bin, alpha = 0.5, color = 'r', density=True, label='left')
                # axes[tar_pos].hist(reach_ends_dist_right,bins = fixed_bin, alpha = 0.5, color = 'g', density=True, label='right')
                
                sns.distplot(reach_ends_base,bins = fixed_bin, color = 'b', label='base', ax = axes[tar_pos])
                sns.distplot(reach_ends_dist_left,bins = fixed_bin, color = 'r', label='left', ax = axes[tar_pos])
                sns.distplot(reach_ends_dist_right,bins = fixed_bin, color = 'g', label='right', ax = axes[tar_pos])
                axes[tar_pos].legend(loc = 9)
                axes[tar_pos].set_title('target location ' + str(tar_pos))
                axes[tar_pos].set_xlim([-0.25,0.25])
                axes[tar_pos].set_ylabel('density')
                axes[tar_pos].set_xlabel('reach end x-coordinate')
            plt.tight_layout()
            plt.show()
        
        
        #plot histogram of reach_latency
        # custom bins give warning -> ignore
        with np.errstate(divide='ignore',invalid='ignore'):
            fixed_bin = np.arange(150,700,20)
            # fixed_bin = 30
            fig, axes = plt.subplots()
            reach_lat_base = []
            reach_lat_dist = []
            for tar_pos in range(4):
                for tr in range(len(trial)):
                    if trial[tr]['resample_success'] == 1 and trial[tr]['tar_pos'] == tar_pos:
                        if trial[tr]['dis_condition'] == 'same':
                            reach_lat_base.append(trial[tr]['reach_latency'])
                        elif trial[tr]['dis_condition'] == 'different':
                            reach_lat_dist.append(trial[tr]['reach_latency'])
    
            reach_lat_base = np.array(reach_lat_base)*1000
            reach_lat_dist = np.array(reach_lat_dist)*1000
            mean_lat_base = np.mean(reach_lat_base)
            mean_lat_dist = np.mean(reach_lat_dist)
            sns.distplot(reach_lat_base,bins = fixed_bin, color = 'b', label='base')
            sns.distplot(reach_lat_dist,bins = fixed_bin, color = 'r', label='distr')
            axes.legend(loc = 1)
            axes.scatter([mean_lat_base],[0.0155], color = 'b', alpha = 0.6)
            axes.scatter([mean_lat_dist],[0.0155], color = 'r', alpha = 0.6)
            axes.set_title('latency distribution')
            # axes.set_xlim([-0.25,0.25])
            axes.set_ylabel('density')
            axes.set_xlabel('reach latency (ms)')
            plt.tight_layout()
            plt.show()
        
    
        #plot average trajectory per participant
        self.plotAvgTrajConditionsPP(mode='together')
        # self.plotAvgTrajOverallPP()
        pass
    
    def plotAvgTrajOverallPP(self):
        x = range(self.resample_nr)
        y = self.participants[self.pp_code]['overall_traj'][:,0]
        plt.figure()
        plt.plot(x,y)
        plt.show()
        
    def plotAvgTrajOverall(self, endpoint_x = 1):
        trajs = np.zeros([self.resample_nr,len(self.participants)])
        x = range(self.resample_nr)
        fig, ax = plt.subplots()
        if endpoint_x == 0:
            ax.plot([0,self.resample_nr],[0,0], color = [0.5,0.5,0.5])
        for nr, pp in enumerate(self.participants):
            trajs[:,nr] = self.participants[pp]['overall_traj'][:,0]*1000
            if endpoint_x == 0:
                #rotate so that endpoint_x = 0
                line_start_end = np.linspace(trajs[0,nr],trajs[-1,nr],self.resample_nr)
                trajs[:,nr] -= line_start_end
            ax.plot(x,trajs[:,nr], '--', color = [0.7,0.7,0.7], linewidth=1)
        y = np.average(trajs, axis = 1)
        y_sem = np.std(trajs, axis = 1)/np.sqrt(len(self.participants))
        ax.plot(x,y, color = 'r')
        ax.fill_between(x,y - y_sem, y + y_sem, alpha = 0.2, color = 'r')
        ax.set_xlim([0, self.resample_nr-1])
        ax.set_ylabel('attraction score')
        ax.set_xlabel('normalized distance (%)')
        plt.show()
        
    
    def plotAvgTrajConditionsPP(self, version = 'baseline', mode = 'separate'):
        
        color_list = [[1,0.25,0.25],[0.9,0.25,0.25],[0.8,0.25,0.25],[0.25,0.25,0.8],[0.25,0.25,0.9],[0.25,0.25,1]]
        line_stylist = ['-.',':','-']
        
        fig = plt.figure()
        if mode == 'together':
            ax = plt.axes(projection='3d')
        for tar_pos in range(4):
            if tar_pos == 0:
                dis_range = [1,2,3]
            elif tar_pos == 1:
                dis_range = [-1,1,2]
            elif tar_pos == 2:
                dis_range = [-2,-1,1]
            elif tar_pos == 3:
                dis_range = [-3,-2,-1]
            #plot baseline
            x = np.copy(self.participants[self.pp_code]['baseline_traj'+str(tar_pos)][:,0])
            y = np.copy(self.participants[self.pp_code]['baseline_traj'+str(tar_pos)][:,1])
            z = np.copy(self.participants[self.pp_code]['baseline_traj'+str(tar_pos)][:,2])

            sub_x = 0
            sub_y = 0
            sub_z = 0
            if version == 'x':
                sub_x = np.copy(x)
                x -= sub_x
            elif version == 'y':
                sub_y = np.copy(y)
                y -= sub_y
            elif version == 'z':
                sub_z = np.copy(z)
                z -= sub_z
                
            if mode == 'separate':
                ax = fig.add_subplot(2, 2, tar_pos+1, projection='3d')
                
            #line
            if mode == 'separate':
                label_text = 'Baseline ' + str(tar_pos)
            elif mode == 'together':
                label_text = '_nolegend_'
                    
            ax.plot3D(x,z,y, '-', color = [0.25,0.25,0.25], label= label_text)
    
            #plot distractor condition in same plot
            for dis_pos in dis_range:
                if dis_pos < 0:
                    colored = color_list[dis_pos+3]
                    label_text = 'Distractor Left ' + str(abs(dis_pos))
                else:
                    colored = color_list[dis_pos+2]
                    label_text = 'Distractor Right ' + str(abs(dis_pos))
                line_style = line_stylist[abs(dis_pos)-1]

                if mode == 'together' and (tar_pos == 1 or tar_pos == 2):
                    label_text = '_nolegend_'

                x = self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)][:,0] - sub_x
                y = self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)][:,1] - sub_y
                z = self.participants[self.pp_code]['distractor_traj'+str(tar_pos)+str(dis_pos)][:,2] - sub_z
                ax.plot3D(x,z,y, line_style, color = colored, label= label_text, linewidth=3)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_zlabel('y')
            ax.legend()
            ax.view_init(30, -90)
        plt.show()
    
    def initialize(self, trial):
        trial['on_shape_samplenr'] = -1
        trial['on_shape_t'] = -1
        trial['on_shape_velo'] = -1
        # trial['velocity_min'] = -1
        trial['move_start_samplenr'] = -1
        trial['move_end_samplenr'] = -1
        trial['traj'] = -1
        trial['traj_resample'] = -1
        trial['resample_success'] = -1
        trial['reach_interuption'] = -1
        trial['reach_end_coord'] = -1
        trial['reach_latency'] = -1
        
    def cleanup_trial_forLogging(self, trial):
        
        to_del_keys = []
        for key, value in trial[0].items():
            if isinstance(value,(list,dict,tuple, np.ndarray)):
                to_del_keys.append(key)
        for tr_id in range(len(trial)):
            for key_del in to_del_keys:
                del trial[tr_id][key_del]
            if not len(trial[tr_id]) == len(trial[0]):
                raise ValueError('trial keys not same length')
        return trial
        
    def resamplenorm(self,traj,nr):
        '''
        traj: to resample trajectory
        nr: nr of samples to resample to
        resample based on amplitude
        '''
        
        def nearestPoint_onLine_fromPoint(p1, p2, p3):
            '''
            p1, p2: points for line segment
            p3: point outside of line
            
            returns p4: point on line segment that is closest to p3
                        i.e. line p3,p4 perpendicular to line p1,p2
            '''
            x1, y1, z1 = p1
            x2, y2, z2 = p2
            x3, y3, z3 = p3
            dx, dy, dz = x2-x1, y2-y1, z2-z1
            det = dx*dx + dy*dy + dz*dz
            a = (dy*(y3-y1)+dx*(x3-x1)+dz*(z3-z1))/det
            return x1+a*dx, y1+a*dy, z1+a*dz
        
        # set starting point to 0,0,0
        traj -= traj[0,:]
        
        start_point = traj[0,1:4]
        end_point = traj[-1,1:4]
        #calculate distance traveled along amplitude
        distance_ampli = [0]
        for sample in traj[1:-1,1:4]:
            # intersection with l1 of line through each samplepoint perpendicular to l1
            sample_ampli = nearestPoint_onLine_fromPoint(traj[0,1:4], traj[-1,1:4], sample)
            distance_ampli.append(np.linalg.norm(start_point - sample_ampli))
        #add endpoint = amplitude
        distance_ampli.append(np.linalg.norm(start_point - end_point))
        #resample axes
        distance_ampli_resample = np.linspace(distance_ampli[0],distance_ampli[-1],nr)
        new_traj = np.empty((nr,3))
        distance_ampli = np.array(distance_ampli)
        
        #remove samples with negative amplitude
        distance_ampli = distance_ampli[distance_ampli >= 0]
        traj = traj[distance_ampli >= 0, :]
        #remove samples after max amplitude (limit to 5)
        maxampli_id = np.argmax(distance_ampli)
        if 0 < ((len(distance_ampli)-1) - maxampli_id) <= 5:
            distance_ampli = distance_ampli[0:maxampli_id+1]
            traj = traj[0:maxampli_id+1, :]
        elif ((len(distance_ampli)-1) - maxampli_id) > 5:
            print('Exclussion: Max amplitude too far from movement endpoint')
            return -1, -1

        distance_ampli =np.sort(distance_ampli)
        
        for axis in range(1,4):
            # sort samples according to distance_ampli sort
            # [x for (y,x) in sorted(zip(sort_like_array,to_sort_array), key=lambda pair: pair[0])] 
            traj[:,axis] = [x for (y,x) in sorted(zip(distance_ampli,traj[:,axis]), key=lambda pair: pair[0])] 
            try: 
                inter_fct = interpolate.splrep(distance_ampli,traj[:,axis], k = 1)
                new_traj[:,axis-1] = np.array(interpolate.splev(distance_ampli_resample,inter_fct))

            except: 
                print('resample fail')
                self.count_resample_error += 1
                return -1,-1
                            
        #normalize
        new_traj /= distance_ampli_resample[-1]

        if np.isnan(new_traj).any():
                    print('nan detected')
                    print(new_traj)
                    raise TypeError('No nans allowed')
                    
        return new_traj, 1
        
    def detectReachInterupt(self, velocity):
        #restriction one: based on local minimum
        ids_min = signal.argrelextrema(velocity, np.less)[0]
        if len(ids_min) > 0:
            ids_max = signal.argrelextrema(velocity, np.greater)[0]
            ids_all = np.sort(np.append(ids_max,ids_min))
            for ids in ids_min:
                
                if min(ids_all) < ids:
                    first_upper_velo = velocity[ids_all[np.where(ids_all == ids)[0]-1]]
                else:
                    first_upper_velo = velocity[0]
                if max(ids_all) > ids:
                    second_upper_velo = velocity[ids_all[np.where(ids_all == ids)[0]+1]]
                else:
                    second_upper_velo = velocity[-1]
                lower_velo = velocity[ids]

                if lower_velo <= first_upper_velo/3.0 and lower_velo <= second_upper_velo/3.0:
                    print('Exclussion: Reach interupt detected R1')
                    return 1
            #restriction 2: based on lower velocity treshold after first velocity peak
            if (velocity[ids_max[0]:ids_max[-1]] < 0.2).any():
                print('Exclussion: Reach interupt detected R2')
                return 1
        
        return 0
        
        
    def interpolate_t_3D(self, samples):
        #manually interpolate missing t values if delta_t > 1000/90
        time_s = samples[:,0]
        time_full = np.array(round(time_s[0],7))
        for idx in range(1,len(samples)):
            delta_t = time_s[idx] - time_s[idx-1]
            #devide delta_t by sample rate to check how many samples are missing +1
            nr_missing = int(round(delta_t / (1.0/90)))
            if nr_missing >= 2:
                extra_t = np.arange(time_s[idx-1],time_s[idx],delta_t/nr_missing)
                for idy in range(1,nr_missing):
                    time_full = np.append(time_full,round(extra_t[idy],7))
            time_full = np.append(time_full,round(time_s[idx],7))
            
        if not len(time_s) == len(time_full):
            new_samples = np.empty((len(time_full),4))
            new_samples[:,0] = time_full
            #interpolate 3D coordinates
            #for each 3D coordinate calculate interpolation fct
            for axis in range(1,4):
                inter_fct = interpolate.splrep(time_s,samples[:,axis])
                new_samples[:,axis] = interpolate.splev(time_full,inter_fct)
            return new_samples
          
        else: return samples
        
    def velocity_calc(self, point1, point2):
        t_delta = point2[0] - point1[0]
        distance = np.linalg.norm(point2[1:] - point1[1:])
        return distance/t_delta
                
if __name__ == "__main__":
    parsed = Parser()
    print("'t Is gedaan!")
