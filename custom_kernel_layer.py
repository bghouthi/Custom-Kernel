import torch
import torch.nn
import parameters

class Custom_Kernel_Layer(torch.nn.Conv2d):	

    def __init__(self, in_features, out_features, stride=1, bias=False, padding = 0, kernel_size=3):
        super(Custom_Kernel_Layer, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_width = kernel_size
        
        self.attentionWeights = torch.nn.parameter.Parameter(torch.FloatTensor(out_features, in_features,kernel_size ,kernel_size))
        torch.nn.init.ones_(self.attentionWeights)
        #self.custom_weight = torch.FloatTensor(out_features, in_features, kernel_size,kernel_size).cuda()

    def forward(self,input):
        #attention = self.attentionWeights + 0
        #attention[:,:,:,1] = attention[:,:,:,0]
        #attention[:,:,:,2] = attention[:,:,:,0]

        if parameters.binary:
            
            attention = (torch.abs(attention) > 0.01) * attention
            print(torch.sum(torch.abs(attention) > 0.01))

            #attention = torch.abs(self.attentionWeights) > 0.1 * self.attentionWeights
            #print(sum(torch.abs(self.attentionWeights) > 0.1))
        #else:
        #    attention = self.attentionWeights
        
        custom_weight =  self.weight * self.attentionWeights
        #self.custom_weight[:,:,:,0] = attention[:,:,:,0] * self.weight[:,:,:,0] #+ attention[:,:,:,1] * self.weight[:,:,:,1] + attention[:,:,:,2] * self.weight[:,:,:,2] 
        #self.custom_weight[:,:,:,1] = attention[:,:,:,1] * self.weight[:,:,:,1]
        #self.custom_weight[:,:,:,2] = attention[:,:,:,2] * self.weight[:,:,:,2]
     
        return torch.nn.functional.conv2d(input, custom_weight, bias = self.bias, stride = self.stride, padding = self.padding)
        #return torch.nn.functional.conv2d(input, self.weight, bias = self.bias, stride = self.stride, padding = self.padding)