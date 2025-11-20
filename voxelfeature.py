import copy

import numpy as np
import open3d as o3d
import scipy

import common_utils
import pointSampling
from saab import Saab
from skimage.measure import block_reduce
from evaluate import MSE
from skimage.util import view_as_windows
import matplotlib.pyplot as plt

def Shrink(X, win):
    ch=X.shape[0]
    X = view_as_windows(X, (1,win,win,win),(1,win,win,win))
    curshape=X.shape
    X = X.reshape(-1, X.shape[-3], X.shape[-2],X.shape[-1])
    #X = X.reshape(-1,X.shape[-1])
    return X,curshape

def Shrink2D(X, win):#input num*width*height*channel
    ch=X.shape[0]
    X = view_as_windows(X, (1,win,win,1),(1,win,win,1))
    curshape=X.shape
    X = X.reshape(-1, X.shape[-3], X.shape[-2],X.shape[-1])
    #X = X.reshape(-1,X.shape[-1])
    return X,curshape
def extDistFeat(pcdlist,radius,blocksize):
    expDist=np.zeros((len(pcdlist),blocksize,blocksize,blocksize))
    for i in range(len(pcdlist)):
        pcd=pcdlist[i]
        pointVec=np.asarray(pcd.points)
        allvoxel=np.unravel_index(np.arange(blocksize**3),(blocksize,blocksize,blocksize))
        allvoxel=np.asarray(allvoxel).transpose()
        # cubedist=scipy.spatial.distance.cdist(allvoxel, pointVec)
        # see=np.min(cubedist,axis=1)
        tree = scipy.spatial.KDTree(pointVec)
        mindist, minid = tree.query(allvoxel)
        distcube=mindist.reshape(blocksize,blocksize,blocksize)
        expDist[i]=np.exp(-0.5*distcube)
        check=0
    return expDist

def getSaabFeature(occulist,istraining=True,saabmodel=[]):
    shape0=occulist.shape
    occulist=occulist.reshape(occulist.shape[0],-1)
    if istraining==True:
        saab = Saab(num_kernels=-1, needBias=True, bias=0)
        saab.fit(occulist)
    else:
        saab=saabmodel
    Xt = saab.transform(occulist)
    if istraining==True:
        _=saab
    else:
        _=[]
    return Xt,_


def getmaxPooling(cubes,size=2):
    tmp = block_reduce(cubes[0], block_size=(size, size,size), func=np.max)
    resultcube=np.zeros((cubes.shape[0],)+tmp.shape)
    for i in range(cubes.shape[0]):
        resultcube[i] = block_reduce(cubes[i], block_size=(size,size,size), func=np.max)
    return resultcube

def RFTselector(feature,score,featureidx):
    featureScoreList=np.zeros(feature.shape[-1])
    for i in range(feature.shape[-1]):
        curfeat=feature[:,i]
        minfeat,maxfeat=np.min(curfeat),np.max(curfeat)
        nstep=(maxfeat-minfeat)/64
        RTscore=np.zeros(64)
        for j in range(64):
            val=minfeat+j*((maxfeat-minfeat)/64)
            idxSL,idxSR=curfeat<=val,curfeat>val
            meanL,meanR=np.mean(score[idxSL]),np.mean(score[idxSR])
            mseL=MSE(score[idxSL],meanL*np.ones_like(score[idxSL]))
            mseR=MSE(score[idxSR],meanR*np.ones_like(score[idxSR]))
            mseL,mseR=np.sqrt(mseL),np.sqrt(mseR)
            NL,NR=score[idxSL].shape[0],score[idxSR].shape[0]
            RTscore[j]=(NL*mseL+NR*mseR)/(NL+NR)
        Ropt=np.min(RTscore)
        featureScoreList[i]=Ropt
    featureScoreList=np.sort(featureScoreList)
    np.savetxt('testout'+str(featureidx)+'.txt', featureScoreList, delimiter=',')
    return featureScoreList

def extractPlaneFeat(pcd,numofPatch,score=[],mapSize=15):
    '''
    :param pcd: input pcd
    :param mapSize: project to a map of mapSize*mapSize
    :param numofPatch: how many patches extracted from the pcd
    :param score: score of each point
    :return: projected map

    '''
    # visualcolor = 0.5 * np.ones((score.shape[0], 3))
    visualcolor = score
    pcdwitcolor=copy.deepcopy(pcd)
    pcd.colors=o3d.utility.Vector3dVector(visualcolor)

    keypointlist=np.argsort(score[:,0])[::-1]
    points=np.asarray(pcd.points)
    pickedNum,i=0,0
    pcdminbound,pcdmaxbound=pcd.get_min_bound(),pcd.get_max_bound()
    while pickedNum<numofPatch:
        center=points[keypointlist[i]]
        if np.min(abs(pcdminbound-center))<(mapSize//8) or np.min(abs(pcdmaxbound-center))<(mapSize//8):
            #avoid center too close to boundary
            continue
        min_bound=center-(mapSize//2)-1e-3
        max_bound=min_bound+mapSize+2e-3
        bbox=o3d.geometry.AxisAlignedBoundingBox(min_bound,max_bound)
        croppcd=pcd.crop(bbox)
        croppedcolorpcd=pcdwitcolor.crop(bbox)
        ####get projmap#############
        visualcolor=np.asarray(croppedcolorpcd.colors)
        projimg=pointSampling.getpcdCloudDir(croppedcolorpcd,visualcolor,mapSize)

        ############################
        i=i+1
        pickedNum=pickedNum+1

def extractLocalFreq(pcd,numofPatch,radius, score=[],mapSize=15):
    visualcolor = score
    pcdwitcolor = copy.deepcopy(pcd)
    pcd.colors = o3d.utility.Vector3dVector(visualcolor)

    keypointlist = np.argsort(score[:, 0])[::-1]
    points = np.asarray(pcd.points)
    pickedNum, i = 0, 0
    pcdminbound, pcdmaxbound = pcd.get_min_bound(), pcd.get_max_bound()
    outimg,selectedKeypoints=[],[np.array([-1024,-1024,-1024])]
    while pickedNum < numofPatch:
        center = points[keypointlist[i]]
        i = i + 1
        if np.min(abs(pcdminbound - center)) < (mapSize // 8) or np.min(abs(pcdmaxbound - center)) < (mapSize // 8):
            # avoid center too close to boundary
            continue
        if np.min(common_utils.get3Ddistance(center,selectedKeypoints))<(mapSize//2):
            continue
        min_bound = center - (mapSize // 2) - 1e-3
        max_bound = min_bound + mapSize-1 + 2e-3
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        croppcd = pcd.crop(bbox)
        croppedcolorpcd = pcdwitcolor.crop(bbox)
        # ###########################
        # coor=np.asarray(croppcd.points)
        # N=coor.shape[0]
        # tree = scipy.spatial.KDTree(coor)
        # mindist, minid = tree.query(coor, k=15)
        # Laplacian=np.zeros((N,N))
        # for k in range(N):
        #     distance = mindist[k, 1:]
        #     tmp = np.exp(-1 * np.square(distance / (0.5 * radius)))
        #     weight = tmp
        #     Laplacian[k,minid[k,1:]]=weight
        #     Laplacian[k,k]=-1*sum(weight)
        # W,V=np.linalg.eig(Laplacian)
        # visualcolor=np.asarray(croppedcolorpcd.colors)
        # visualcolorinfreq=np.matmul(visualcolor[:,0].transpose(),V)
        # plt.subplot(2,1,1)
        # plt.plot(abs(visualcolorinfreq))
        plt.subplot(2,1,2)
        visualcolor = np.asarray(croppedcolorpcd.colors)
        projimg,validpixnum = pointSampling.getpcdCloudDir(croppedcolorpcd, visualcolor, mapSize,centerPCD=center,cropsize=mapSize-2)
        outimg.append(projimg)
        plt.imshow(projimg)
        plt.show()
        ############################
        selectedKeypoints.append(center)
        pickedNum = pickedNum + 1
    return np.array(outimg)

def getMeanMaxStd(feature):#input is N by 8*8*8
    std=np.std(feature,axis=-1).reshape(-1,1)
    mean=np.mean(feature,axis=-1).reshape(-1,1)
    max=np.max(feature,axis=-1).reshape(-1,1)
    return np.concatenate((max,mean,std),axis=-1)

def getMeanStd(feature):#input is N by 8*8*8
    std=np.std(feature,axis=-1).reshape(-1,1)
    mean=np.mean(feature,axis=-1).reshape(-1,1)
    return np.concatenate((mean,std),axis=-1)