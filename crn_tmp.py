import torch
import torch.nn as nn
import bc
p = False
nb_pruned = 0
all_k = 0
ops = 0
all_ops = 0
class CRN(torch.nn.Conv2d):

    def __init__(self, in_features, out_features, stride=1, bias=False, padding = 0, kernel_size=3,p = True):
        super(CRN, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size)
        self.in_features = in_features
        self.out_features = out_features
        self.k = kernel_size
        #self.th = torch.tensor(0.1)
        self.crnWeights = nn.parameter.Parameter(torch.FloatTensor(out_features, in_features, 1, 1))#.to('cuda') # 5))#.to('cuda')
        nn.init.ones_(self.crnWeights)
        self.p = p
        self.tmp_to_count = torch.FloatTensor(self.out_features, self.in_features, 1, 1)
    def forward(self,input):
        global nb_pruned
        global all_k
        global ops
        global all_ops
        #self.crnWeights = clampNoGradient.apply(self.crnWeights)
        #nn.init.ones_(self.crnWeights)
        if(self.p):
            #self.crnWeights.data.copy_(torch.clamp(torch.abs(self.crnWeights),0,1).round())
            #self.crnWeights = clampNoGradient.apply(self.crnWeights)
            masked_weight = self.weight * clampNoGradient.apply(self.crnWeights)#self.crnWeights
            #print(masked_weight)
            masked_weight[:,:,1,1] = self.weight[:,:,1,1]
            self.tmp_to_count.data.copy_(clampNoGradient.apply(self.crnWeights))
            nb_pruned += torch.sum(self.tmp_to_count)#self.crnWeights)
            all_k += self.in_features * self.out_features

            y = nn.functional.conv2d(input, masked_weight, bias = self.bias, stride = self.stride, padding = self.padding)
            ops +=  torch.sum(self.tmp_to_count) * self.weight.shape[2]*self.weight.shape[3]*y.shape[2]*y.shape[3] + (self.in_features * self.out_features - torch.sum(self.tmp_to_count))*y.shape[2]*y.shape[3]
            all_ops+= self.in_features * self.out_features * self.weight.shape[2]*self.weight.shape[3]*y.shape[2]*y.shape[3]
            return y
        y = nn.functional.conv2d(input, self.weight, bias = self.bias, stride = self.stride, padding = self.padding)
        all_k += self.in_features * self.out_features
        nb_pruned+= self.in_features * self.out_features
        ops+= self.in_features * self.out_features * self.weight.shape[2]*self.weight.shape[3]*y.shape[2]*y.shape[3]
        all_ops+= self.in_features * self.out_features * self.weight.shape[2]*self.weight.shape[3]*y.shape[2]*y.shape[3]
        return y


class clampNoGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.clamp(torch.abs(x),0,1).round().cuda()

    @staticmethod
    def backward(ctx, g):
        return g


class CRN_Loss(torch.nn.Module):
    def __init__(self,model,reg_strength):
        super(CRN_Loss,self).__init__()
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