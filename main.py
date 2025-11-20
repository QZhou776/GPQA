# This is a sample Python script.
import getfile
import pointSampling
import preprocess
import voxelfeature
import open3d as o3d
import numpy as np
import GreenBPCQA
import pickle
import common_utils
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from scipy import stats
import logging
import time
import os

# Press Shift+F10 to execute it or replace it with your code.
def set_logger():
    t=time.time()
    strlist=str(t)
    log_file = os.path.join(strlist+'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def trainSaabbyBatch(mymodel,pcdinfolist,pcdpath):
    neighborhood = [3, 192]
    ###add pooling
    # totalexpDistCube=np.zeros((0,neighborhood[0]//2,neighborhood[0]//2,neighborhood[0]//2))
    totalexpDistCube=np.zeros((0,neighborhood[0],neighborhood[0],neighborhood[0]))
    pcdtree,score=[],[]
    for i in range(pcdinfolist.shape[0]):
        if i%10==0:
            print(i)
        pcd512=o3d.io.read_point_cloud(pcdpath+pcdinfolist[i][1]+'.ply')
        pcd512, minbound = getfile.testingPCDNormalization(pcd512)
        pcd512=common_utils.pcdcolorConversion([pcd512],mode=1)[0];#YUV to YUV
        tmppcd=getfile.randomPatchSplit(pcd512,patchSize=192,pertubation=0)[0][0]
        addedpcdtree = getfile.getPCDTree(tmppcd, neighborhood)
        alignedpcd = preprocess.alignpcdlist(addedpcdtree[0][0], blocksize=neighborhood[0])[0]
        clusteredPCD = [alignedpcd]
        expDistCube = voxelfeature.extDistFeat(clusteredPCD[0], 0, blocksize=neighborhood[0])
        # ####pooling over expDistCube
        # expDistCube=voxelfeature.getmaxPooling(expDistCube, size=2)
        ############################
        totalexpDistCube=np.concatenate((totalexpDistCube,expDistCube),axis=0)
        pcdtree=preprocess.appendPCDTree(pcdtree,addedpcdtree)
        score.append(float(pcdinfolist[i][2]))
    score=np.array(score)
#####layer 0#########
    featLayer0 = mymodel.SaabFeatureLayer(totalexpDistCube, istraining=True, modelIdx=0)
    sortedfeat=np.sort(featLayer0[:,0])
    counts, bins = np.histogram(sortedfeat,bins=64)
    plt.plot(sortedfeat)
    plt.show()
    print('done')
    del totalexpDistCube
    featureACLayer0 = []
    for i in range(0, featLayer0.shape[-1]):
        curACcube = preprocess.getVoxelHop(featLayer0[:, i], pcdtree[1][2], pcdtree[1][0], neighborhood, n_layer=1)
        if i == 0:
            featCubeDCLayer0 = curACcube
        else:
            curACcube = preprocess.getVoxelHop(featLayer0[:, i], pcdtree[1][2], pcdtree[1][0], neighborhood, n_layer=1)
            curACcube = np.abs(curACcube)
            featureACLayer0.append(voxelfeature.getmaxPooling(curACcube, size=4))
    totalfeat_layer_0_Num = featLayer0.shape[-1]
#####output layer 0: alignment and aggregation#########
    transMat=preprocess.alignpcdlist(pcdtree[1][0],blocksize=neighborhood[1])[1]
    alignedfeatDC=preprocess.alignCubeList(featCubeDCLayer0,transMat[0],transMat[1])
    for i in range(0,totalfeat_layer_0_Num-1):
        featureACLayer0[i]=preprocess.alignCubeList(featureACLayer0[i],transMat[0],transMat[1])
    featureACLayer0=np.array(featureACLayer0)
    tmpshape=featureACLayer0.shape
    featureACLayer0 = mymodel.SaabFeatureLayer(featureACLayer0.reshape((-1,)+featureACLayer0.shape[2:]), istraining=True, modelIdx=2)
    # reshape
    featureACLayer0=featureACLayer0.reshape(tmpshape)
    featureACLayer0=np.swapaxes(featureACLayer0,0,1)
    featureACLayer0=featureACLayer0.reshape(featureACLayer0.shape[0],-1)
    # concate mean max std
    addfeature=voxelfeature.getMeanMaxStd(featureACLayer0)
    featureACLayer0=np.concatenate((featureACLayer0,addfeature),axis=-1)
    voxelfeature.RFTselector(featureACLayer0,score,featureidx=0)
    del featureACLayer0
#####layer 1##########
    ####block split, Saab kernel 4x4x4
    layer_1_DCfeat, subshape = voxelfeature.Shrink(alignedfeatDC, win=4)
    featLayer1 = mymodel.SaabFeatureLayer(layer_1_DCfeat, istraining=True, modelIdx=1)
    featLayer1 = featLayer1.reshape(subshape[0:4] + (-1,))
####output layer 1: pooling alignment and aggregation#########
    # transMat = preprocess.alignpcdlist(pcdtree[1][2], blocksize=neighborhood[1])[1]
    # layer_1_DCfeat = preprocess.alignCubeList(layer_1_DCfeat, transMat[0], transMat[1])
    featureACLayer1 = []
    for i in range(0, featLayer1.shape[-1]):
        if i == 0:
            featCubeDCLayer1 = featLayer1[:, :, :, :, i]
        else:
            featLayer1[:, :, :, :, i] = np.abs(featLayer1[:, :, :, :, i])
            featureACLayer1.append(voxelfeature.getmaxPooling(featLayer1[:, :, :, :, i], size=2))
    featureACLayer1 = np.array(featureACLayer1)
    ###currently no alignment
    tmpshape2 = featureACLayer1.shape
    featureAC2Layer1 = mymodel.SaabFeatureLayer(featureACLayer1.reshape((-1,) + featureACLayer1.shape[2:]),
                                                istraining=True, modelIdx=3)
    # reshape
    featureAC2Layer1 = featureAC2Layer1.reshape(tmpshape2)
    featureAC2Layer1 = np.swapaxes(featureAC2Layer1, 0, 1)
    featureAC2Layer1 = featureAC2Layer1.reshape(featureAC2Layer1.shape[0], -1)
    #################
    addfeature = voxelfeature.getMeanMaxStd(featureAC2Layer1)
    featureAC2Layer1 = np.concatenate((featureAC2Layer1, addfeature), axis=-1)
    voxelfeature.RFTselector(featureAC2Layer1, score, featureidx=1)
    ####layer 1, DC last Saab
    tmpshape3 = featCubeDCLayer1.shape
    featureDC2Layer1 = mymodel.SaabFeatureLayer(featCubeDCLayer1, istraining=True, modelIdx=4)
    #################
    addfeature = voxelfeature.getMeanStd(featureDC2Layer1)
    featureDC2Layer1 = np.concatenate((featureDC2Layer1, addfeature), axis=-1)
    voxelfeature.RFTselector(featureDC2Layer1, score,featureidx=2)
    with open('GreenBPCQAv1', 'wb') as f:
        pickle.dump(mymodel, f)
    check2 = 0

def trainSaabbyBatchV2(mymodel,pcdinfolist,pcdpath):
    pcdtree,score,picpatternList=[],[],[]
    for i in range(pcdinfolist.shape[0]):
        print(pcdinfolist[i][1])
        if 'p72' in pcdinfolist[i][1] or 'p27' in pcdinfolist[i][1] or 'p56_gpcc' in pcdinfolist[i][1] or 'p55_gpcc' in pcdinfolist[i][1]:
            continue
        if i%10==0:
            print(i)
        pcd512=o3d.io.read_point_cloud(pcdpath+pcdinfolist[i][1]+'.ply')
        pcd512, minbound = getfile.testingPCDNormalization(pcd512)
        # pcd512=common_utils.pcdcolorConversion([pcd512],mode=1)[0];#YUV to YUV
        tmppcd=getfile.randomPatchSplit(pcd512,patchSize=192,pertubation=0)[0][0]
        _, visualcolor, radius = pointSampling.getPointScore(tmppcd, 50, isCoor=False)
        projimgList=voxelfeature.extractLocalFreq(tmppcd, numofPatch=24, radius=radius, score=visualcolor, mapSize=17)
        picpatternList.extend(projimgList)
        scorelist=np.ones(projimgList.shape[0])*float(pcdinfolist[i][2])
        score.extend(scorelist)
    picpatternList=np.array(picpatternList)
#####layer 0#########
    #cut images into 3x3
    windowImg,shape=voxelfeature.Shrink2D(picpatternList,win=3)
    windowImg=windowImg.reshape(windowImg.shape[0],-1)
    featLayer0 = mymodel.SaabFeatureLayer(windowImg, istraining=True, modelIdx=0)
    sortedfeat = np.sort(featLayer0[:, 0])
    counts, bins = np.histogram(sortedfeat, bins=64)
    plt.plot(sortedfeat)
    plt.show()
    print('done')
    featLayer0=featLayer0.reshape(shape[0:4]+(-1,))
    featLayer0 = np.swapaxes(featLayer0, 2, 3)#swap color channel to 2nd axis
    featLayer0 = np.swapaxes(featLayer0, 1, 2)
    del windowImg
    featureACLayer0 = []
    for i in range(0, featLayer0.shape[-1]):
        if i == 0:
            featCubeDCLayer0 = featLayer0[:,:,:,:,0]
        else:
            featureACLayer0.append(featLayer0[:,:,:,:,i])#probably need pooling here
    totalfeat_layer_0_Num = featLayer0.shape[-1]
    #####layer 0 AC feature aggregation#########
    featureACLayer0 = np.array(featureACLayer0)
    tmpshape = featureACLayer0.shape
    featureACLayer0=featureACLayer0.reshape(-1,featureACLayer0.shape[-2]*featureACLayer0.shape[-1])
    featureACLayer0 = mymodel.SaabFeatureLayer(featureACLayer0, istraining=True, modelIdx=2)

    # reshape
    featureACLayer0=featureACLayer0.reshape(tmpshape)
    featureACLayer0=np.swapaxes(featureACLayer0,0,1)
    featureACLayer0=featureACLayer0.reshape(featureACLayer0.shape[0],-1)
    # concate mean max std
    addfeature=voxelfeature.getMeanMaxStd(featureACLayer0)
    featureACLayer0=np.concatenate((featureACLayer0,addfeature),axis=-1)
    voxelfeature.RFTselector(featureACLayer0,np.array(score),featureidx=0)
#####layer 1##########
    DCLayer0shape=featCubeDCLayer0.shape
    featCubeDCLayer0=featCubeDCLayer0.reshape(-1,featCubeDCLayer0.shape[-2]*featCubeDCLayer0.shape[-1])
    featLayer1 = mymodel.SaabFeatureLayer(featCubeDCLayer0, istraining=True, modelIdx=1)
    featLayer1 = featLayer1.reshape(DCLayer0shape)

    # featLayer1 = np.swapaxes(featLayer1, 0, 1)
    featLayer1 = featLayer1.reshape(featLayer1.shape[0], -1)
    # concate mean max std
    addfeature = voxelfeature.getMeanMaxStd(featLayer1)
    featLayer1 = np.concatenate((featLayer1, addfeature), axis=-1)
    voxelfeature.RFTselector(featLayer1, np.array(score), featureidx=1)
    check=0
    ######concat feature#############
    finalfeat=np.concatenate((featLayer1,featureACLayer0),axis=1)

    ##########split dataset##########
    # using the train test split function
    X_train, X_valid, y_train, y_valid = train_test_split(finalfeat, np.array(score),
                                       random_state=42,
                                       test_size=0.25,
                                       shuffle=True)
    # X_train, X_valid, y_train, y_valid=finalfeat[:-960,:],finalfeat[-960:,:],np.array(score)[:-960],np.array(score)[-960:]
    X_train, X_valid, y_train, y_valid=finalfeat[:-3840,:],finalfeat[-3840:,:],np.array(score)[:-3840],np.array(score)[-3840:]
    ##########xgboost############
    t = time.time()
    eval_set = [(X_train, y_train), (X_valid, y_valid)]
    reg = xgb.XGBRegressor(objective='reg:squarederror',
                           max_depth=5,
                           n_estimators=1500,
                           subsample=0.6,
                           eta=0.08,
                           colsample_bytree=0.4,
                           min_child_weight=4)

    reg.fit(X_train, y_train, eval_set=eval_set, eval_metric=['rmse'],
            early_stopping_rounds=100, verbose=False)
    logging.info('Regressor trained in {} secs...'.format(time.time()-t))

    bst = reg.get_booster()
    bst.save_model('xgboost.json')

    logging.info('Validating...')
    pred_valid_mos = reg.predict(X_valid)

    SRCC = stats.spearmanr(pred_valid_mos, y_valid)
    logging.info("SRCC: {}".format(SRCC[0]))

    corr, _ = pearsonr(pred_valid_mos, y_valid)
    logging.info("PLCC: {}".format(corr))

#########################
    np.savetxt('Finalscore.txt', pred_valid_mos, delimiter=',')
    np.savetxt('realscore.txt', y_valid, delimiter=',')
    plt.scatter(y_valid,pred_valid_mos)
    plt.show()


if __name__ == '__main__':
    # pred_valid_mos = np.loadtxt('Finalscore.txt')
    # y_valid = np.loadtxt('realscore.txt')
    # uniquescore,idx=np.unique(y_valid,return_inverse=True)
    # meanpred=np.zeros_like(uniquescore)
    # for i in range(len(uniquescore)):
    #     meanpred[i]=np.mean(pred_valid_mos[y_valid==uniquescore[i]])
    # plt.scatter(uniquescore, meanpred)
    # plt.show()
    # SRCC = stats.spearmanr(meanpred, uniquescore)
    # corr, _ = pearsonr(meanpred, uniquescore)
    # print(SRCC)
    # print(corr)
#
# ##debug
    getfile.prepare_Training_Patch("D:/PCVQA/training/ppc/", "D:/PCVQA/trainset_mos_std_ci.csv",
                                   "D:/PCVQA/trainingPatch2/")
#     curpcd=o3d.t.io.read_point_cloud('p72_gpcc-octree-raht_r03.ply')
#     realpcd=getfile.open3dTensorFile2PCD(curpcd)
#     o3d.io.write_point_cloud('newcheckdistortedGEzz.ply', realpcd)

# # curpcd=o3d.io.read_point_cloud('p03_gpcc-octree-predlift_r02_cut3.ply')
    curpcd=o3d.io.read_point_cloud('D:/PCVQA/trainingPatch/p53_vpcc_r00_00.ply')
    # curpcd=o3d.io.read_point_cloud('p03Cloud_cut2.ply')
    # curpcd=o3d.io.read_point_cloud('p03Cloud_cutplane.ply')
    score,visualcolor,radius=pointSampling.getPointScore(curpcd,50,isCoor=False)
    # voxelfeature.extractPlaneFeat(curpcd, numofPatch=100, score=visualcolor, mapSize=31)
    voxelfeature.extractLocalFreq(curpcd, numofPatch=100,radius=radius, score=visualcolor, mapSize=17)

    set_logger()
    mymodel=GreenBPCQA.GreenPCBVQA()
    pcdinfolist=getfile.randomselectTraining("D:/PCVQA/my_csv.csv",800)
    # trainSaabbyBatch(mymodel, pcdinfolist,"D:/PCVQA/trainingPatch/")
    trainSaabbyBatchV2(mymodel, pcdinfolist,"D:/PCVQA/trainingPatch/")











#     ###################################
#     tmppcd = o3d.io.read_point_cloud("D:/PCVQA/training/ppc/p03_vpcc_r05.ply")
#     tmppcd, minbound = getfile.testingPCDNormalization(tmppcd)
#     # getfile.prepare_Training_Patch("D:/PCVQA/training/ppc/","D:/PCVQA/trainset_mos_std_ci.csv","D:/PCVQA/trainingPatch/")
#
#     neighborhood=[16,256,256]#[8,64,512]
#     pcdtree=getfile.getPCDTree(tmppcd,neighborhood)
#
# #####layer 0#########
#     alignedpcd=preprocess.alignpcdlist(pcdtree[0][0],blocksize=neighborhood[0])[0]
#     # clusteredPCD=preprocess.clusterPCD(alignedpcd,n_clusters=4,blocksize=8)
#     clusteredPCD=[alignedpcd]
#     expDistCube=voxelfeature.extDistFeat(clusteredPCD[0],0,blocksize=neighborhood[0])
#     featLayer0=mymodel.SaabFeatureLayer(expDistCube,istraining=True,modelIdx=0)
#     del expDistCube
#     # featCubeDCLayer0=preprocess.getVoxelHop(featLayer0[:,0],pcdtree[1][2],pcdtree[1][0],neighborhood,n_layer=1)
#     featureACLayer0=[]
#     for i in range(0,featLayer0.shape[-1]):
#         curACcube=preprocess.getVoxelHop(featLayer0[:,i],pcdtree[1][2],pcdtree[1][0],neighborhood,n_layer=1)
#         if i==0:
#             featCubeDCLayer0=curACcube
#         else:
#             curACcube = preprocess.getVoxelHop(featLayer0[:, i], pcdtree[1][2], pcdtree[1][0], neighborhood, n_layer=1)
#             curACcube = np.abs(curACcube)
#             featureACLayer0.append(voxelfeature.getmaxPooling(curACcube,size=4))
#     totalfeat_layer_0_Num=featLayer0.shape[-1]
#     del featLayer0
# #####output layer 0: alignment and aggregation#########
#     transMat=preprocess.alignpcdlist(pcdtree[1][0],blocksize=neighborhood[1])[1]
#     alignedfeatDC=preprocess.alignCubeList(featCubeDCLayer0,transMat[0],transMat[1])
#     for i in range(0,totalfeat_layer_0_Num-1):
#         featureACLayer0[i]=preprocess.alignCubeList(featureACLayer0[i],transMat[0],transMat[1])
#     featureACLayer0=np.array(featureACLayer0)
#     # featACLayer1Shape=featureACLayer0.shape
#     featureACLayer0 = mymodel.SaabFeatureLayer(featureACLayer0.reshape((-1,)+featureACLayer0.shape[2:]), istraining=True, modelIdx=2)
#     # featureAC2Layer1=featureAC2Layer1.reshape(featACLayer1Shape)
#     # concate mean max std
#     addfeature=voxelfeature.getMeanMaxStd(featureACLayer0)
#     featureACLayer0=np.concatenate((featureACLayer0,addfeature),axis=-1)
#     ###release memory#########3
#
# #####layer 1##########
#     ####block split, Saab kernel 4x4x4
#     layer_1_DCfeat,subshape=voxelfeature.Shrink(alignedfeatDC,win=4)
#     featLayer1 = mymodel.SaabFeatureLayer(layer_1_DCfeat, istraining=True, modelIdx=1)
#     featLayer1=featLayer1.reshape(subshape[0:4]+(-1,))
#     ####output layer 1: pooling alignment and aggregation#########
#     # transMat = preprocess.alignpcdlist(pcdtree[1][2], blocksize=neighborhood[1])[1]
#     # layer_1_DCfeat = preprocess.alignCubeList(layer_1_DCfeat, transMat[0], transMat[1])
#     featureACLayer1=[]
#     for i in range(0, featLayer1.shape[-1]):
#         if i==0:
#             featCubeDCLayer1=featLayer1[:,:,:,:,i]
#         else:
#             featLayer1[:,:,:,:,i] = np.abs(featLayer1[:,:,:,:,i])
#             featureACLayer1.append(voxelfeature.getmaxPooling(featLayer1[:,:,:,:,i], size=2))
#     featureACLayer1=np.array(featureACLayer1)
#     ###currently no alignment
#     featureAC2Layer1 = mymodel.SaabFeatureLayer(featureACLayer1.reshape((-1,)+featureACLayer1.shape[2:]), istraining=True, modelIdx=3)
#     addfeature=voxelfeature.getMeanMaxStd(featureAC2Layer1)
#     featureAC2Layer1=np.concatenate((featureAC2Layer1,addfeature),axis=-1)
#     ####layer 1, DC last Saab
#     featureDC2Layer1 = mymodel.SaabFeatureLayer(featCubeDCLayer1, istraining=True, modelIdx=4)
#     addfeature=voxelfeature.getMeanStd(featureDC2Layer1)
#     featureDC2Layer1 = np.concatenate((featureDC2Layer1, addfeature), axis=-1)
#     with open('GreenBPCQAv1', 'wb') as f:
#         pickle.dump(mymodel, f)
#     check2=0
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
