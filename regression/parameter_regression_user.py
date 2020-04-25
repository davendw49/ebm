# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import progressbar
from scipy import stats

# filename='../RealDatasetValidation/data/BX-Book-Ratings.csv'
# filename='../RealDatasetValidation/data/AudioScrobbler.csv'
# filename='../RealDatasetValidation/data/ratings_Books.csv'
# filename='../RealDatasetValidation/data/ratings_CDs_and_Vinyl.csv'
# filename='../RealDatasetValidation/data/ratings_Electronics.csv'
filename='../RealDatasetValidation/data/ratings_Movies_and_TV.csv'


def get_valid_data():
    '''    first ,kick out invalid data -- rating time less than thresh   '''
    #allData=pd.read_csv('BX-Book-Ratings.csv')
    #print(allData.columns )
    allData=dict()
    pos = 0
    with open(filename,'r') as f:
            # filter the first line
        f.readline()  
        lines=f.readlines()
            # traverse and filter 
        for idx,line in enumerate(lines):
            dataInString=line.strip().split(',')  # list type
            for index,data in enumerate(dataInString):  # userName ISBN score
                dataInString[index]=data.replace('\"','')
            if not allData.get(dataInString[0]) :   # create the user's list if he does not exist
                allData[dataInString[0]]=list() 
            allData[dataInString[0] ].append(float( dataInString[2]) )
    #        miu.append( float(dataInString[2]) )
    return allData
    '''calculate miu1 and miu2 using allData but predict only with valid users'''

def get_valid_user(allData):
    # %% cell1
    ''' pick out valid user '''
    #num=0
    miu=[]    # to store all data
    rateFreq=dict() # key is rate and value is its frequency  rateFreq[userName][score]=frequency
    for key in allData.keys():      # userName
        if len(allData[key])> thresh:
    #        num+=1
                # create the dict for each valid user
            if not rateFreq.get( key ):
                rateFreq[ key ]= dict()     
                # create the dict entry for each score
            for score in allData[key]:
                if not rateFreq[key].get(score):
                    rateFreq[ key ][ score ]=0
                rateFreq[ key ][ score ]+=1
                miu.append(score)
    return rateFreq, miu

def cal_para_multiple(rateFreq, miu):
    # %% cell2
    '''calculate common para'''
    miu1=np.mean(miu);  theta1=np.std(miu)
    # four direction for miu2 and theta2 
    # deltamiu=[0.5,-0.5]; theta=[2,1,-1,-2]
    cp = [(0,1), (0,-1), (1,-1), (-1,1)]

    write=open('3resultWithPartData.txt','a');


    N=8*len(rateFreq)
    p = progressbar.ProgressBar(maxval=N)
    p.start() 
    progress=0

    userWeight=[None]*8   # list of 4 ele
    for i in range(2):
        for j in range(4):
            deltamiu = cp[j][0]
            theta = cp[j][1]

            miu2=miu1+deltamiu
            theta2=theta1+theta

            userWeight[i*4+j]=dict()
                # rateFreq[userName][score]=frequency
                # userScore[score]=frequency
            '''test'''
            print("************************************",file=write)
            print('i={0},j={1}'.format(i,j),file=write )
            sumMse_all = 0.0
            for user,userScore in rateFreq.items(): 
                    # cal MSE and choose the best weight
                bestWeight=0
                minMse=float('inf');
                '''对每个user 遍历一遍weight 把每个数据点都相加'''
                for weight in np.arange(0, 1.1 ,0.1): # arange exclude endPoint while linspace includes defaultly  
                    sumMse=0
                        # userScore[score]=frequency
                    '''对于每个user，他对每个score的mse要累计起来'''
                    '''这里需要check下data'''
                    for score,freq in userScore.items():    
                        sumMse+=(( weight*stats.norm(miu1, theta1).cdf(score)+ (1-weight)*stats.norm(miu2,theta2).cdf(score)- freq )/freq )**2
                    
                    '''test'''
                    print('user:{0},weight:{1},mse:{2}'.format(user,weight,sumMse),file=write )
                   
                    if sumMse < minMse:
                        minMse=sumMse
                        bestWeight=weight
                sumMse_all += minMse
                '''test'''
                print('user:{0},minMse:{1}'.format(user,minMse),file=write )
                userWeight[i*4+j][user]=bestWeight;   # change here   
                
                progress+=1
                p.update(progress)
            print(sumMse_all/len(rateFreq.keys()))

    write.close()
    print(userWeight)
    p.finish()

def cal_para_single(rateFreq, miu):
    # %% cell2
    '''calculate common para'''
    miu1=np.mean(miu);  theta1=np.std(miu)
    # four direction for miu2 and theta2 
    # deltamiu=[0.5,-0.5]; theta=[2,1,-1,-2]
    cp = [(0,1), (0,-1), (1,-1), (-1,1)]

    write=open('3resultWithPartData_AudioScrobbler.txt','a');


    N=8*len(rateFreq)
    p = progressbar.ProgressBar(maxval=N)
    p.start() 
    progress=0

    userWeight=[None]*8   # list of 4 ele
    for i in range(2):
        for j in range(1):
            
            userWeight[i*4+j]=dict()
                # rateFreq[userName][score]=frequency
                # userScore[score]=frequency
            '''test'''
            print("************************************",file=write)
            print('i={0},j={1}'.format(i,j),file=write )
            sumMse_all = 0.0
            for user,userScore in rateFreq.items(): 
                    # cal MSE and choose the best weight
                bestWeight=0
                minMse=float('inf');
                '''对每个user 遍历一遍weight 把每个数据点都相加'''
                for weight in np.arange(0, 1.1 ,0.1): # arange exclude endPoint while linspace includes defaultly  
                    sumMse=0
                        # userScore[score]=frequency
                    '''对于每个user，他对每个score的mse要累计起来'''
                    '''这里需要check下data'''
                    for score,freq in userScore.items():    
                        sumMse+=((weight*stats.norm(miu1, theta1).cdf(score)- freq )/freq )**2
                    
                    '''test'''
                    print('user:{0},weight:{1},mse:{2}'.format(user,weight,sumMse),file=write )
                   
                    if sumMse < minMse:
                        minMse=sumMse
                        bestWeight=weight
                sumMse_all += minMse
                '''test'''
                print('user:{0},minMse:{1}'.format(user,minMse),file=write )
                userWeight[i*4+j][user]=bestWeight;   # change here   
                
                progress+=1
                p.update(progress)
            print(sumMse_all/len(rateFreq.keys()))

    write.close()
    print(userWeight)
    p.finish()

if __name__=="__main__":
    
    for thresh in [150, 180, 200, 220]:
        print(thresh)
        a_dict, a_miu = get_valid_user(get_valid_data())
        print(thresh, "  ", len(a_dict.keys()))
    # thresh = 150
    # a_dict, a_miu = get_valid_user(get_valid_data())
    # cal_para_single(a_dict, a_miu)
    # cal_para_multiple(a_dict, a_miu)