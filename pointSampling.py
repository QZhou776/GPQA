import copy

import numpy as np
import open3d as o3d
import scipy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import common_utils

def getpcdCloudDir(pcd,color,mapSize,centerPCD,cropsize):
    mean, cov = pcd.compute_mean_and_covariance()
    evalue, evector = np.linalg.eig(cov)
    idx = evalue.argsort()[::-1]
    evalue, evector = evalue[idx], evector[:, idx]
    # pcd=pcd.translate(-1*mean)


##template
    # template=np.sqrt(1/2)*np.array([
    #     [1,1,0],[1,-1,0],[1,0,1],[1,0,-1],[0,1,1],[0,1,-1]
    # ])
    # template2=np.concatenate((template,np.eye(3)),axis=0)
    # normidx=np.argmax(abs(np.matmul(template2, evector[:, 2])))
    if True:#normidx>5:
        ###align to 3axes
        tmp = np.eye(3)
        idx0 = np.argmax(abs(np.matmul(tmp, evector[:, 0])))
        evector[:, 0], tmp[idx0, :] = tmp[idx0, :].transpose(), 0
        idx1 = np.argmax(abs(np.matmul(tmp, evector[:, 1])))
        evector[:, 1], tmp[idx1, :] = tmp[idx1, :].transpose(), 0
        idx2 = np.argmax(abs(np.matmul(tmp, evector[:, 2])))
        evector[:, 2], tmp[idx2, :] = tmp[idx2, :].transpose(), 0
    # else:
    #     tmp=np.zeros((3,3))
    #     evector[:, 2]=template[normidx,:]
    #     if np.sum(np.abs(evector[:, 2]))!=1:
    #         evector[:, 2]=np.sqrt(0.5)*evector[:, 2]
    #     evector[:, 0]=np.cross(evector[:, 2],np.array([1,0,0]))
    #     evector[:, 0] /= np.linalg.norm(evector[:, 0])
    #     if np.sum(np.abs(evector[:, 0]))!=1:
    #         evector[:, 0]=np.sqrt(0.5)*evector[:, 0]
    #     evector[:, 1] = np.cross(evector[:, 2], evector[:, 0])
    #     evector[:, 1] /= np.linalg.norm(evector[:, 1])
    #     if np.sum(np.abs(evector[:, 1]))!=1:
    #         evector[:, 1]=np.sqrt(0.5)*evector[:, 1]
    #################
    points=np.asarray(pcd.points)
    projectedPCD=np.matmul(points,evector[:,0:2])
    centerPCD=np.matmul(centerPCD,evector[:,0:2])
    ##################
    projmap=np.zeros((mapSize,mapSize,3))
    normalized2Dpcd=projectedPCD-centerPCD
    normalized2Dpcd=((normalized2Dpcd//1)+(mapSize//2)).astype("int")
    # ravelIdx=np.ravel_multi_index(normalized2Dpcd.transpose(), (mapSize, mapSize))
    # uniqueInfo = np.unique(ravelIdx, return_counts=True)
    # projmap[normalized2Dpcd[:,0],normalized2Dpcd[:,1],0]=color[:,0]
    # projmap[normalized2Dpcd[:,0],normalized2Dpcd[:,1],1]=color[:,1]
    # projmap[normalized2Dpcd[:,0],normalized2Dpcd[:,1],2]=color[:,2]
    # check0=copy.deepcopy(projmap[:,:,0])
    # projmap=common_utils.interpImg(projmap,mapSize)
    # check=projmap[:,:,0]
##################
    grid_x, grid_y = np.mgrid[0:17:17j, 0:17:17j]
    projimg=np.zeros((mapSize,mapSize,3))
    projimg[:,:,0] = griddata(normalized2Dpcd, color[:,0], (grid_x, grid_y), method='nearest')
    projimg[:,:,1] = griddata(normalized2Dpcd, color[:,1], (grid_x, grid_y), method='nearest')
    projimg[:,:,2] = griddata(normalized2Dpcd, color[:,2], (grid_x, grid_y), method='nearest')
    offset=(mapSize-cropsize)//2
    projimg=projimg[offset:-offset,offset:-offset,:]
    validpix=color.shape[0]
    # projmap=np.where(np.isnan(projmap),0,projmap)
    # projmap = projmap[offset:-offset, offset:-offset, :]
##################
    # minx,maxx,miny,maxy=np.min(projectedPCD[:,0]),np.max(projectedPCD[:,0]),np.min(projectedPCD[:,1]),np.max(projectedPCD[:,1])
    # width=max(abs(maxx),abs(minx),abs(maxy),abs(miny))*2
    # minx,maxx,miny,maxy=-(width/2)-1e-2,(width/2)+1e-2,-(width/2)-1e-2,(width/2)+1e-2
    # grid_x, grid_y = np.mgrid[minx:maxx:64j, miny:maxy:64j]
    # projimg = griddata(projectedPCD, color, (grid_x, grid_y), method='nearest')
    return projimg,validpix

# def get_flattened_pcds2(source,A,B,C,D,x0,y0,z0):
#     x1 = np.asarray(source.points)[:,0]
#     y1 = np.asarray(source.points)[:,1]
#     z1 = np.asarray(source.points)[:,2]
#     x0 = x0 * np.ones(x1.size)
#     y0 = y0 * np.ones(y1.size)
#     z0 = z0 * np.ones(z1.size)
#     r = np.power(np.square(x1-x0)+np.square(y1-y0)+np.square(z1-z0),0.5)
#     a = (x1-x0)/r
#     b = (y1-y0)/r
#     c = (z1-z0)/r
#     t = -1 * (A * np.asarray(source.points)[:,0] + B * np.asarray(source.points)[:,1] + C * np.asarray(source.points)[:,2] + D)
#     t = t / (a*A+b*B+c*C)
#     np.asarray(source.points)[:,0] = x1 + a * t
#     np.asarray(source.points)[:,1] = y1 + b * t
#     np.asarray(source.points)[:,2] = z1 + c * t
#     return source

def getPointScore(pcd,NumNeighbor,isCoor=True):
    coor=np.asarray(pcd.points)
    N=coor.shape[0]
    tree = scipy.spatial.KDTree(coor)
    linsparray=np.round(np.linspace(0,N-1,100)).astype("int")
    mindist, minid = tree.query(coor[linsparray],k=NumNeighbor)
    radius=np.max(mindist[:,-1])

    mindist,minid=tree.query(coor,k=NumNeighbor)

    batch_size=1000
    batch_num=np.ceil(N/batch_size).astype("int")
    # zz_batch=np.zeros((batch_size,coor.shape[-1]))
    zz=np.zeros((N,coor.shape[-1]))
    Z,score=[],[]
    if isCoor==True:
        Z.append(coor)
    else:
        Z.append(np.asarray(pcd.colors))
    for iterL in range(1,5):##iteration
        z=Z[iterL-1]
        for i_batch in range(0,batch_num):
            batch_index=np.arange(i_batch*batch_size,min((i_batch+1)*batch_size,N))
            zz_batch = np.zeros((batch_index.shape[0],coor.shape[-1]))
            for j in range(batch_index.shape[0]):
                i=batch_index[j]
                distance=mindist[i,1:]
                tmp=np.exp(-1*np.square(distance/(0.5*radius)))

                if np.sum(tmp)>1e-6:
                    weight=tmp/np.sum(tmp)
                    zz_batch[j,:]=np.dot(weight,z[minid[i,1:],:])
                    del weight, tmp, distance
                else:
                    zz_batch[j,:]=z[i,:]
                    del tmp, distance
            zz[batch_index,:]=zz_batch
            # zz_batch=np.zeros_like(zz_batch)
        Z.append(zz)
        zz = np.zeros_like(zz)
        score.append(np.sum(np.square(Z[0]-Z[iterL]),axis=1))
        print(np.mean(score[-1]))
        print((np.std(score[-1])))
    ###binarize###
    score[-1] = np.where(score[-1] > 0.15, 1, score[-1])
    ####assign color to it
    # plt.plot(np.sort(score[-1]))
    # plt.show()
    colornew=np.zeros((N,3))

    minval,maxval=np.min(score[-1]),np.max(score[-1])
    colornew[:,0]=(score[-1]-minval)/(maxval-minval)
    colornew[:,0]=np.power(colornew[:,0],0.5)
    colornew[:,1],colornew[:,2]=0.5,0.5
    # pcd.colors=o3d.utility.Vector3dVector(colornew)
    o3d.io.write_point_cloud('newcheckdistortedGEO.ply', pcd)
    return score,colornew,radius

