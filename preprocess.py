import copy
import os
import numpy as np
import open3d as o3d
from scipy import ndimage
from sklearn import cluster
from sklearn import preprocessing
import pickle



def relocatetoOrigin(pcd,blocksize):
    offset=pcd.get_min_bound()
    pcd=pcd.translate(-1*(offset//blocksize*blocksize))
    return pcd


def getOccufromPCDwithOffset(pcd,mapSize):
    occu = np.zeros((mapSize, mapSize, mapSize))
    realpcdcoor=np.asarray(pcd.points)
    pcdoffset64=realpcdcoor//mapSize*mapSize
    pcdcoor=(realpcdcoor - pcdoffset64).astype("int")
    occu[pcdcoor[:,0],pcdcoor[:,1],pcdcoor[:,2]]=1
    return occu

def getFeatureVoxfromPCDwithOffset(pcd,mapSize,featureVec):
    occu = np.zeros((mapSize, mapSize, mapSize))
    realpcdcoor=np.asarray(pcd.points)
    pcdoffset64=realpcdcoor//mapSize*mapSize
    pcdcoor=(realpcdcoor - pcdoffset64).astype("int")
    occu[pcdcoor[:,0],pcdcoor[:,1],pcdcoor[:,2]]=featureVec
    return occu
# def getMeanGradient(pcdList,blocksize=64):
#     for i in len(pcdList):
#         occu=getOccufromPCDwithOffset(pcdList[i],blocksize)
#         gradx=ndimage.gaussian_gradient_magnitude(occu, sigma=[5,0,0])
#         grady=ndimage.gaussian_gradient_magnitude(occu, sigma=[0,5,0])
#         gradz=ndimage.gaussian_gradient_magnitude(occu, sigma=[0,0,5])

###alignment
def alignCubeList(cubelist,reflectmat,permuMat):
    blocksize=cubelist.shape[1]
    for i in range(cubelist.shape[0]):
        for j in range(3):
            if reflectmat[i,j,j]<0:
                cubelist[i]=np.flip(cubelist[i],axis=j)
        curpermute=permuMat[i,0:3,0:3]
        case=np.sum(np.diag(curpermute))
        posi=np.argwhere((curpermute-np.diag(curpermute))>0)
        if case==1:
            axis0,axis1=posi[0,0],posi[0,1]
            cubelist[i]=np.swapaxes(cubelist[i],axis1=axis0,axis2=axis1)
        elif case==0:
            axis0,axis1=posi[0,0],posi[0,1]
            cubelist[i]=np.swapaxes(cubelist[i],axis1=axis0,axis2=axis1)
            axis0, axis1 = posi[1, 0], posi[1, 1]
            cubelist[i] = np.swapaxes(cubelist[i], axis1=axis0, axis2=axis1)
    return cubelist

def alignpcdlist(pcdList,blocksize=64):

    occu2,occu,occu3=np.zeros((blocksize,blocksize,blocksize)),np.zeros((blocksize,blocksize,blocksize)),np.zeros((blocksize,blocksize,blocksize))
    reflectMatList,permuteMatList=[],[]
    for i in range(len(pcdList)):
        curpcd=pcdList[i]
        curpcd=relocatetoOrigin(curpcd,blocksize)
        occu = occu+getOccufromPCDwithOffset(curpcd, blocksize)
        mean,cov=curpcd.compute_mean_and_covariance()
        ###reflect mat
        signOfMean=np.where(mean-(blocksize/2)>=0,-1,1)
        reflectMat=np.eye(4)
        reflectMat[0,0],reflectMat[0,3]=signOfMean[0], blocksize-1 if signOfMean[0]<0 else 0
        reflectMat[1,1],reflectMat[1,3]=signOfMean[1], blocksize-1 if signOfMean[1]<0 else 0
        reflectMat[2,2],reflectMat[2,3]=signOfMean[2], blocksize-1 if signOfMean[2]<0 else 0
        alignedPCD = curpcd.transform(reflectMat)
        occu3 = occu3 + getOccufromPCDwithOffset(alignedPCD, blocksize)

        mean,cov=alignedPCD.compute_mean_and_covariance()
        evalue,evector=np.linalg.eig(cov)
        idx = evalue.argsort()[::-1]
        evalue,evector = evalue[idx],evector[:, idx]

        permuteMat=np.zeros((4,4))
        permuteMat[3,3]=1
        tmp=np.eye(3)
        idx0=np.argmax(abs(np.matmul(tmp,evector[:,0])))
        permuteMat[0,0:3],tmp[idx0,:]=tmp[idx0,:],0
        idx1=np.argmax(abs(np.matmul(tmp,evector[:,1])))
        permuteMat[1,0:3],tmp[idx1,:]=tmp[idx1,:],0
        idx2=np.argmax(abs(np.matmul(tmp,evector[:,2])))
        permuteMat[2,0:3],tmp[idx2,:]=tmp[idx2,:],0
        alignedPCD=curpcd.transform(permuteMat)
        mean, cov = alignedPCD.compute_mean_and_covariance()
        evalue2, evector2 = np.linalg.eig(cov)
        idx = evalue2.argsort()[::-1]
        evalue2,evector2 = evalue2[idx],evector2[:, idx]

        tmpoccu=getOccufromPCDwithOffset(alignedPCD,blocksize)
        occu2=occu2+getOccufromPCDwithOffset(alignedPCD,blocksize)
        pcdList[i]=alignedPCD
        # finalMat.append(np.matmul(reflectMat,permuteMat))
        reflectMatList.append(reflectMat)
        permuteMatList.append(permuteMat)
    return pcdList,[np.array(reflectMatList),np.array(permuteMatList)]

def clusterPCD(pcdList,n_clusters=8,blocksize=64):
    featureVecList=[]
    clusteredPCD=[None]*n_clusters
    clusteredPCD2 = [None] * n_clusters
    occu=np.zeros((len(pcdList),blocksize,blocksize,blocksize))
    for i in range(len(pcdList)):
        mean, cov = pcdList[i].compute_mean_and_covariance()
        evalue, evector = np.linalg.eigh(cov)
        idx = evalue.argsort()[::-1]
        evalue,evector = evalue[idx],evector[:, idx]
        featureVecList.append(np.concatenate((mean,evalue)))
        occu[i]=getOccufromPCDwithOffset(pcdList[i],blocksize)
    featureVecList=np.array(featureVecList)
    # scaler =preprocessing.StandardScaler(with_mean=True,with_std=True)
    # scaler.fit(featureVecList)
    # featureVecList=scaler.transform(featureVecList)
    KM = cluster.KMeans(n_clusters=n_clusters, n_init=11)
    KM.fit(featureVecList)

    idx=KM.predict(featureVecList)
    for i in range(n_clusters):
        clusteredPCD[i]=(occu[idx==i])
        clusteredPCD2[i]=[]
    for ii in range(idx.shape[0]):
        clusteredPCD2[idx[ii]].append(pcdList[ii])
    check=0
    return clusteredPCD2

def getVoxelHop(feature,treeIdx,upperPCD,neighborhood,n_layer):
    blockSize=neighborhood[n_layer]//neighborhood[n_layer-1]
    featCube=np.zeros((len(treeIdx),blockSize,blockSize,blockSize))
    pcdwithfeature=[]
    for i in range(len(treeIdx)):
        ###scale pcd
        curpcd=copy.deepcopy(upperPCD[i])
        curpcd.scale(1/neighborhood[n_layer-1], center=(0, 0, 0))
        chosenidx = np.asarray(treeIdx[i])
        chosenFeature=feature[chosenidx]
        featCube[i]=getFeatureVoxfromPCDwithOffset(curpcd,blockSize,chosenFeature)

        pcdwithfeature.append(curpcd)
    return featCube

def appendPCDTree(pcdtreeList,addedPCDtree):
    if pcdtreeList == []:
        pcdtreeList = addedPCDtree
    else:
        for i in range(len(pcdtreeList)):
            pcdtreeList[i][0].extend(addedPCDtree[i][0])
            pcdtreeList[i][1].extend(addedPCDtree[i][1])
            pcdtreeList[i][2].extend(addedPCDtree[i][2])
    return pcdtreeList