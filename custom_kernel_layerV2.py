import torch
import torch.nn
import parameters

class Custom_Kernel_Layer(torch.nn.Conv2d):	

    def __init__(self, in_features, out_features, stride=1, bias=False, padding = 0, kernel_size=3):
        super(Custom_Kernel_Layer, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        
        self.threshold = 0.1
        self.attentionWeightsV = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features, 1 ,2))
        torch.nn.init.ones_(self.attentionWeightsV)
        self.attentionWeightsH = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features, 2 ,1))
        torch.nn.init.ones_(self.attentionWeightsH)
        #self.custom_weight = torch.FloatTensor(out_features, in_features, kernel_size,kernel_size).cuda()

    def forward(self,input):
        attention = torch.FloatTensor(self.out_features, self.in_features, self.kernel_size ,self.kernel_size).cuda()

        if parameters.binary:
            
            self.attentionWeightsV = (torch.abs(self.attentionWeightsV) > self.threshold) * self.attentionWeightsV 
            print(torch.sum(torch.abs(self.attentionWeightsV) > self.threshold))
            self.attentionWeightsH = (torch.abs(selfattentionWeightsH) > self.threshold) * self.attentionWeightsH 
            print(torch.sum(torch.abs(self.attentionWeightsH) > self.threshold))
        
        attention[:,:,:,0] = attention[:,:,:,0] * self.attentionWeightsV[:,:,:,0]
        attention[:,:,:,self.kernel_size-1] = attention[:,:,:,self.kernel_size-1] * self.attentionWeightsV[:,:,:,1]

        attention[:,:,0,:] = attention[:,:,0,:] * self.attentionWeightsH[:,:,0,:]
        attention[:,:,self.kernel_size-1,:] = attention[:,:,self.kernel_size-1,:] * self.attentionWeightsH[:,:,1,:]
        attention[:,:,1,1] = self.weight[:,:,1,1]

        return torch.nn.functional.conv2d(input, attention, bias = self.bias, stride = self.stride, padding = self.padding)

class CKL_Loss(torch.nn.Module):
    def __init__(self,model,reg_strength):
        super(CKL,self).__init__()
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
            if isinstance(m, Custom_Kernel_Layer):# or isinstance(m, Quant_IP):
                alpha+=torch.sum(torch.abs(m.attentionWeightsH))
                alpha+=torch.sum(torch.abs(m.attentionWeightsV))
                #alpha+=m.n_bit_a*alpha_reg
                max=max+2*8.0
            if isinstance(m,Custom_Kernel_Layer) or isinstance(m,torch.nn.Linear) or isinstance(m,torch.nn.BatchNorm2d) :
                wd+=torch.sum(torch.pow(m.weight,2))
        alpha_loss=self.Alpha_loss(alpha/max,target)
        return XEL,alpha_loss,wd