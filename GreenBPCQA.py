import voxelfeature


class GreenPCBVQA():
    def __init__(self):
        self.layer_0_saab=[]
        self.layer_1_saab=[]
        self.layer_0_AC_saab=[]
        self.layer_1_AC_saab=[]
        self.layer_1_DC_saab=[]
        self.SaabList=[self.layer_0_saab,
                  self.layer_1_saab,
                  self.layer_0_AC_saab,
                  self.layer_1_AC_saab,
                  self.layer_1_DC_saab,]

    def SaabFeatureLayer(self,occulist,istraining=True,modelIdx=0):
        if istraining==True:
            Xt,model=voxelfeature.getSaabFeature(occulist, istraining=True)
            self.SaabList[modelIdx]=model
        else:
            Xt,_=voxelfeature.getSaabFeature(occulist, istraining=False,saabmodel=self.SaabList[modelIdx])
        return Xt