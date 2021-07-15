import torch
import torch.nn as nn
import bc
binare = False #True
fm_prued = 0
ops_pruned = 0
row1 = 0
row3 = 0
colum1 = 0
colum3 = 0
center = 0
row1_total = 0
row3_total = 0
colum1_total = 0
colum3_total = 0
total_kernel = 0
class CRN(torch.nn.Conv2d):

    def __init__(self, in_features, out_features, stride=1, bias=False, padding = 0, kernel_size=3):
        super(CRN, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size)
        self.in_features = in_features
        self.out_features = out_features
        self.k = kernel_size

        self.crnWeights = nn.parameter.Parameter(torch.FloatTensor(out_features, 1, 1)) # in_features, 5))#.to('cuda')
        nn.init.ones_(self.crnWeights)

    def forward(self,input):
        #self.crnWeights = clampNoGradient.apply(self.crnWeights)
        #nn.init.ones_(self.crnWeights)
        global binare
        global fm
        global ops
        global row1
        global row3
        global colum1
        global colum3
        global total_kernel
        global center
        if binare:
            self.crnWeights.data.copy_(torch.clamp(torch.abs(self.crnWeights),0,1).round())
            #fm_pruned +=  input.shape[2]
            #ops_prunes +=
        #row1+= torch.sum(torch.abs(self.crnWeights.data[:,:,0]))
        #row3+= torch.sum(torch.abs(self.crnWeights.data[:,:,1]))
        #colum1+= torch.sum(torch.abs(self.crnWeights.data[:,:,2]))
        #colum3+= torch.sum(torch.abs(self.crnWeights.data[:,:,3]))
        #center+= torch.sum(torch.abs(self.crnWeights.data[:,:,4]))
        bc.ones += torch.sum(torch.abs(self.crnWeights.data))
        bc.total += self.crnWeights.data.nelement()
        #total_kernel += self.out_features * self.in_features
        crn = torch.FloatTensor(self.out_features, self.in_features, self.k,self.k).cuda()
        crn[:,:,0,:self.k] = self.weight[:,:,0,:self.k] * self.crnWeights[:]#,:,0].resize(self.out_features, self.in_features,1)
        crn[:,:,self.k-1,:self.k] = self.weight[:,:,self.k-1,:self.k] * self.crnWeights[:]#,:,1].resize(self.out_features, self.in_features,1)
        crn[:,:,:self.k,0] = self.weight[:,:,:self.k,0] * self.crnWeights[:]#,:,2].resize(self.out_features, self.in_features,1)
        crn[:,:,:self.k,self.k-1] = self.weight[:,:,:self.k,self.k-1] * self.crnWeights[:]#,:,3].resize(self.out_features, self.in_features,1)
        crn[:,:,self.k//2,self.k//2] = self.weight[:,:,self.k//2,self.k//2]# * self.crnWeights[:,:,4]
        return nn.functional.conv2d(input, crn, bias = self.bias, stride = self.stride, padding = self.padding)
class clampNoGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.abs(x),0,1).round()

    @staticmethod
    def backward(ctx, g):
        return g


class CRN_Loss(torch.nn.Module):
    def __init__(self,model,reg_strength):
        super(CRN_Loss,self).__init__()
        self.XEL_loss = torch.nn.CrossEntropyLoss()
        self.Alpha_loss = torch.nn.L1Loss()
        self.model=model
        self.reg_strength=reg_strength

    def forward(self,x,y):
        XEL = self.XEL_loss(x,y)
        alpha_reg=1.0
        max=0.0
        wd=0
        alpha=torch.tensor([0.0]).cuda()
        target=torch.tensor([0.0]).cuda()
        for m in self.model.modules():
            if isinstance(m, CRN):# or isinstance(m, Quant_IP):
                alpha+=torch.sum(torch.abs(m.crnWeights))
                #alpha+=m.n_bit_a*alpha_reg
                max=max+2*8.0
            if isinstance(m,CRN) or isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.BatchNorm2d) :
                wd+=torch.sum(torch.pow(m.weight,2))
        alpha_loss=self.Alpha_loss(alpha/max,target)
        return XEL,alpha_loss,wd

