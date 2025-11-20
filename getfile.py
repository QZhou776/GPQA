import copy
import os
import numpy as np
import open3d as o3d
import pickle
import csv
from sklearn import utils

def open3dTensorFile2PCD(tpcd):
    pcd=o3d.geometry.PointCloud()
    points=tpcd.point.positions.numpy()
    colors=tpcd.point.colors.numpy().astype('float')/255.0
    pcd.points=o3d.utility.Vector3dVector(points)
    pcd.colors=o3d.utility.Vector3dVector(colors)
    return pcd

def randomPatchSplit(pcd,patchSize=512,pertubation=32):
    '''
    get a set of randomly cropped pint clouds with size=patchsize from the input point cloud.
    At most 4 point clouds are croped from the input
    :param pcd: input point cloud sequence
    :param patchSize:  the output point cloud bounding box size
    :param pertubation: the random offset applied to boundingbox when cropping along 2^N grid. disabled when set to 0
    :return: cropped point clouds, with a descend order of point number;
             number of cropped point clouds;
             bounding box list of point clouds;
             num of points inside each cropped point cloud
    '''
    np.random.rand(42)

    curPCDList, upperPCD, idxList = get64blockfromPCD(pcd, patchSize)
    pcdnumlist,pointnumList=[],[]
    for i in idxList:
        pcdnumlist.append(len(i))
    pcdnumlist=np.array(pcdnumlist)
    pcdnumarg=np.argsort(pcdnumlist)[::-1]
    pcdnumlist=pcdnumlist
    num=len(idxList)
    numofpick=np.where(num>4,4,num)
    pickIdx=pcdnumarg[:numofpick]#np.random.choice(num, numofpick, replace=False)
    pcd512List, boundingbox = [],[]
    for idx in list(pickIdx):
        min_bound,max_bound=curPCDList[idx].get_min_bound(),curPCDList[idx].get_max_bound()
        np.random.rand(0)
        if pertubation==0:#no randomness
            randomOffset=0
        else:
            randomOffset = np.random.randint(pertubation, size=3) - (pertubation//2)
        min_bound = min_bound // patchSize * patchSize
        min_bound=min_bound+randomOffset
        max_bound=min_bound+patchSize-(1e-2)
        min_bound=np.where(min_bound<0,0,min_bound)
        bbox=o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
        finalpcd=pcd.crop(bbox)
        if finalpcd.has_points()==False:
            continue
        pcd512List.append(finalpcd)
        boundingbox.append([min_bound,max_bound])
        pointnumList.append(pcdnumlist[idx])
    return pcd512List, len(pcd512List),boundingbox,pointnumList

###read training set
def prepare_Training_Patch(pcdfolderpath,scorepath,patchpath):
    '''
    split into large voxels
    :param pcdfolderpath:
    :param scorepath:
    :param patchpath:
    :return:
    '''
    scoredata,newdata=[],[]
    with open(scorepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            scoredata.append(row[:])
    for i in range(len(scoredata)):
        if i==0:
            continue
        pcdname=scoredata[i][1]
        curpcd = o3d.t.io.read_point_cloud(pcdfolderpath+pcdname+'.ply')
        curpcd = open3dTensorFile2PCD(curpcd)
        patchpcd,num,boundingbox,pointnumList=randomPatchSplit(curpcd,patchSize=512,pertubation=512//16)
        for j in range(num):
            patchname=pcdname+'_'+str(j).zfill(2)
            tmpdata=copy.deepcopy(scoredata[i])
            tmpdata[1]=patchname
            tmpdata.append(boundingbox[j][0][0])
            tmpdata.append(boundingbox[j][0][1])
            tmpdata.append(boundingbox[j][0][2])
            tmpdata.append(boundingbox[j][1][0])
            tmpdata.append(boundingbox[j][1][1])
            tmpdata.append(boundingbox[j][1][2])
            tmpdata.append(j)
            tmpdata.append(pointnumList[j])
            newdata.append(tmpdata)
            o3d.io.write_point_cloud(patchpath+patchname+'.ply', patchpcd[j])
    with open('D:/PCVQA/my_csv2.csv', 'w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC,
                            delimiter=',')
        writer.writerows(newdata)





def testingPCDNormalization(pcd):
    minbound=pcd.get_min_bound()
    pcd=pcd.translate(-1*(pcd.get_min_bound()))
    return pcd,minbound



def get64blockfromPCD(pcd,blocksize=64):
    tmpPCDlist=[]
    see=pcd.get_min_bound()
    seemax=pcd.get_max_bound()
    downtuple = pcd.voxel_down_sample_and_trace(blocksize, pcd.get_min_bound()//blocksize*blocksize, pcd.get_max_bound())
    for i in range(len(downtuple[2])):
        chosenidx = np.asarray(downtuple[2][i])
        block64points = pcd.select_by_index(chosenidx)
        tmpPCDlist.append(block64points)
    return tmpPCDlist,downtuple[0],downtuple[2]

def gettestingPCDList(num,blocksize=64):
    pcdlist = []
    translatelist=[]
    path="D:/PCVQA/training/ppc/"
    strlist=["p03_vpcc_r05.ply"]
    pcdnumlist=num
    for i in pcdnumlist:
        filepath=path+strlist[i]
        tmppcd=o3d.io.read_point_cloud(filepath)
        tmppcd,minbound=testingPCDNormalization(tmppcd)
        pcdlist.extend(get64blockfromPCD(tmppcd,blocksize)[0])
        translatelist.append(minbound)
    return len(pcdlist), pcdlist, translatelist

def getPCDTree(pcd,blocksize=[8,64]):
    upperPCDTree=[]
    for i in range(len(blocksize)):
        see1=pcd.get_min_bound()
        see2=pcd.get_max_bound()
        check=np.asarray(pcd.points)
        seemax=np.max(check[:,0])
        curPCDList,upperPCD,idxList=get64blockfromPCD(pcd, blocksize[i])
        pcd=upperPCD
        upperPCDTree.append([curPCDList,[upperPCD],idxList])
    return upperPCDTree

def randomselectTraining(scorepath,numofPCD):
    scoredata=[]
    with open(scorepath, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            scoredata.append(row[:])
    selectedData=[]
    for row in scoredata:
        if row[-2]==str(0):
            selectedData.append(row)

    selectedData=utils.shuffle(selectedData,random_state=42)
    selectedData=np.array(selectedData)
    return selectedData[:numofPCD,:]





