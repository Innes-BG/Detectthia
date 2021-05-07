
import cv2
import numpy as np
from numpy import array, transpose, mean, stack, where

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import math 
import os
import time
from math import atan2, pi, sin, cos

def sort_points_by_angle(landmarks):
    #Sort points by angle
    angles=[]
    for p in landmarks:
        angle= atan2(p[0],p[1])*360/(2*pi)
        if angle<0:
            angle=angle+360
        angles.append(angle)
        
    angle_inds = np.array(angles).argsort()    
    landmarks = np.array(landmarks)[angle_inds]
    return landmarks

def get_landmarks_from_image(image, step=30, pca=True):
    print('Start')
    tic = time.perf_counter()
    img = cv2.imread(image,0)        
    
    #Fast way to go thrugh each pixels and get coordinates of pixels above 0
    limit=0
    x, y = (img > limit).nonzero()
    
    ##percentage of lesion area in the image.
    # per=len(x)/(len(img)*len(img[0]))
    # print(per)
    ##Show shape
    # plt.plot(x, y, 'ro')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    # plt.close()
    
    
    #Principal components
    pts1=np.transpose([x,y])
    
    if pca:
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(pts1)
        pts2 = pca.transform(pts1)
        
    else:#Not pca, only center
        x=transpose(pts1)[0]  
        y=transpose(pts1)[1]  
        x_c=x-mean(x)
        y_c=y-mean(y)
        pts2 = transpose([x_c, y_c])
        
    # Show shape after PCA
    x2=np.transpose(pts2)[0]  
    y2=np.transpose(pts2)[1]  
    plt.plot(x2, y2, 'ro')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    
    '''Landmarks are chosen every step of angle, 
    1st define the line for the angle, starting in 0,0
    2nd get points in that line with a pixel margin ->pos_points
    3rd find furthest pixel, it is the lesion limit
    '''    

    angle=0
    landmarks=[]
    while angle in range(0,180):

        angle_pi=angle*pi/180
        #line constants:
        a=sin(angle_pi)
        b=cos(angle_pi)  

        #Get points with distance to angle line below 0.5 pixels and is distance to the origin
        res = array([[p, (p[0]**2+p[1]**2)**0.5] for p in pts2 if abs(a*p[0]+b*p[1])/(a**2+b**2)<0.5], dtype="object")
        res_t = transpose(res)
        p_sel= stack(res_t[0], axis=0)
        d2zero= res_t[1] #distance to centre
        #Get point with max distance to zero
        max_ind = where(d2zero == np.max(d2zero))
        landmark1 = p_sel[max_ind][0]

        #Furthest point from landmark1 in angle line
        d2other = array([[((landmark1[0]-p[0])**2+(landmark1[1]-p[1])**2)**0.5] for p in p_sel], dtype="object")
        max_ind2 = where(d2other == np.max(d2other))
        landmark2 = p_sel[max_ind2[0]][0]  
        
        landmarks.append(landmark1)
        landmarks.append(landmark2)
       
            
        angle=angle+step
        
    #Sort points by angle
    landmarks = sort_points_by_angle(landmarks)
    
    xl=np.transpose(landmarks)[0]  
    yl=np.transpose(landmarks)[1]  
    plt.plot(xl, yl, 'bo')
    n=[1,2,3,4,5,6]
    for i, txt in enumerate(n):
        plt.annotate(txt, (xl[i], yl[i]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()   
    plt.close()
    toc = time.perf_counter()
    print('Time for 1 image and ' + str(step_value) + 'º step angle:' + str(round(toc - tic,0)) + ' seconds')        
            
    return landmarks



#If mirro two sets of landmarks are created for each image, original and mirrored
def get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value, pca=True, mirror=True):
    tic_ini = time.perf_counter()


    subfolders=(next(os.walk(directory))[1])
    i=0
    filenames=[]
    names=[]
    landmarks_list=[]
    labels=[]
    names_cor=[]
    for sf in subfolders:
        files = os.listdir(directory + '/' + sf)
        label=sf
        for file in files:
            if file.endswith(filetype):
                filenames.append(file)
                names.append(sf+str(i))
                names_cor.append(sf+str(i))#Only for correspondence file, not affected by mirroring
                labels.append(sf)
                landmarks=get_landmarks_from_image(directory + '/' + sf+ '/' + file, step_value, pca)
                landmarks_list.append(landmarks)
                if mirror:
                    names.append(sf+str(i)+'_m')
                    labels.append(sf)
                    landmarks_mirror=np.transpose([np.transpose(landmarks)[0],-np.transpose(landmarks)[1]])
                    landmarks_mirror = sort_points_by_angle(landmarks_mirror)     
                    landmarks_list.append(landmarks_mirror)

                i+=1
                
    '''Write morphologika file'''
    res = open(results_file, 'w')
    res.write('[individuals]'+'\n')
    res.write(str(len(names))+'\n')
    res.write('[landmarks]'+'\n')
    res.write(str(len(landmarks_list[0]))+'\n')
    res.write('[dimensions]'+'\n')
    res.write(str(len(landmarks_list[0][0]))+'\n')
    res.write('[names]'+'\n')
    for name in names:
        res.write(name + '\n')
    res.write('[labels]'+'\n')
    res.write('Sample'+'\n')
    res.write('[labelvalues]'+'\n')
    for label in labels:
        res.write(label + '\n')
    res.write('[rawpoints]'+'\n')
    i=0
    while i<len(names):
        res.write("\'" + names[i]+'\n')     
        lm = landmarks_list[i]
        for l in lm:
            res.write(str(l[0])+' ')
            res.write(str(l[1])+'\n')    
        res.write('\n')  
        i+=1
    res.close()
    
    '''Write matching names and filenames file'''
    cor = open(correspondences_file, 'w')
    i=0
    while i<len(filenames):
        cor.write(filenames[i] + ',' + names_cor[i]+'\n')
        i+=1

    cor.close()
        
            
    toc_fin = time.perf_counter()
    print('Time for ' +str(len(names)) + ' images and ' + str(step_value) + 'º step angle:' + str(round(toc_fin - tic_ini,0)) + ' seconds')        
        
        
'''Directory is a folder with one subfolder per label, the names of the subfolders are used as label'''  
directory='D:/USAL/Detectthia/Morfometrias/Pruebas_morfo/Images3'    
filetype='png'
correspondences_file= directory +'/correspondence.txt'

# results_file= directory +'/Landmarks_10g_mirror.txt'
# step_value=10 #Angle step between landmarks        
# get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)

# results_file= directory +'/Landmarks_12g_mirror.txt'
# step_value=12 #Angle step between landmarks        
# get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)

# results_file= directory +'/Landmarks_15g_mirror.txt'
# step_value=15 #Angle step between landmarks        
# get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)

# results_file= directory +'/Landmarks_20g_mirror.txt'
# step_value=20 #Angle step between landmarks        
# get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)

# results_file= directory +'/Landmarks_30g_mirror.txt'
# step_value=30 #Angle step between landmarks        
# get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)

results_file= directory +'/Landmarks_60g_mirror.txt'
step_value=60 #Angle step between landmarks        
get_landmarks_batch(directory, filetype, results_file, correspondences_file, step_value)








