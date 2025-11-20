import numpy as np
import skimage
import open3d as o3d

def pcdcolorConversion(pcdlist,mode=1):
    '''
    mode 1: RGB2YUV
    :param pcd:
    :return: pcd with new color space
    '''
    if mode==1:
        srcSpace='RGB'
        dstSpace='YUV'
        convertmat=np.array([[0.299, 0.587, 0.114],[ -0.1687, - 0.3313, 0.5],[0.5, - 0.4187, - 0.0813]]).transpose()
    for i in range(len(pcdlist)):
        pcd=pcdlist[i]
        colors=np.asarray(pcd.colors)
        colors=255*colors
        # newcolors=skimage.color.convert_colorspace(colors, srcSpace, dstSpace)
        # newcolors=skimage.color.rgb2yuv(colors)
        newcolors=np.matmul(colors,convertmat)
        newcolors[:,1],newcolors[:,2]=newcolors[:,1]+128,newcolors[:,2]+128
        newcolors=newcolors/255.0
        pcd.colors=o3d.utility.Vector3dVector(newcolors)
        pcdlist[i]=pcd
    return pcdlist

def interpImg(img,size):
    for ch in range(3):
        for i in range(0,size,2):
            for j in range(0,size,2):
                pattern=img[i:i+2,j:j+2,ch]
                existpointval=np.mean(pattern[pattern>0])
                pattern[pattern==0]=existpointval
                img[i:i + 2, j:j + 2, ch]=pattern
    for ch in range(3):
        for i in range(0,size,4):
            for j in range(0,size,4):
                pattern=img[i:i+4,j:j+4,ch]
                existpointval=np.mean(pattern[pattern>0])
                pattern[pattern==0]=existpointval
                img[i:i + 4, j:j + 4, ch]=pattern
    return img

def get3Ddistance(point,list):
    distance=np.zeros(len(list))
    for i in range(len(list)):
        distance[i]=np.linalg.norm(point-list[i])
    return distance
