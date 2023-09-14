import torch
from torch import nn
import torch.nn.functional as F
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.restoformer_3d_block import TransformerBlock,LayerNorm


class UNetrasformer3D(nn.Module):

    ## Conv 3
    
    def __init__(self,  num_classes, input_channels=1,embed_dims=[8, 16, 32],num_heads = 1, norm_layer=LayerNorm, **kwargs):
        super().__init__()
        
        self.encoder1 = nn.Conv3d(input_channels,   embed_dims[0], 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv3d(embed_dims[0],  embed_dims[2], 3, stride=1, padding=1)  
        
        self.encoder3 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)
        
        self.ebn1 = nn.BatchNorm3d(embed_dims[0])
        self.ebn2 = nn.BatchNorm3d(embed_dims[2])
        
        self.ebn3 = nn.BatchNorm3d(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[2],LayerNorm_type='bias')
        self.norm4 = norm_layer(embed_dims[2],LayerNorm_type='bias')
        
        self.trasformerblock1 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)
        self.trasformerblock2 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)
        self.trasformerblock3 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)
        self.trasformerblock4 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)
        
        
        
        self.dnorm3 = norm_layer(embed_dims[2],LayerNorm_type='bias')
        self.dnorm4 = norm_layer(embed_dims[2],LayerNorm_type='bias')
    
        
        self.decoder1 = nn.Conv3d(embed_dims[2], embed_dims[2], (3,3,3), stride=1, padding=(1,1,1))  
        self.decoder2 = nn.Conv3d(embed_dims[2], embed_dims[2], (3,3,3), stride=1, padding=(1,1,1))  
        
        self.decoder4 = nn.Conv3d(embed_dims[2], embed_dims[0], (3,3,3), stride=1, padding=(1,1,1))
        self.decoder5 = nn.Conv3d(embed_dims[0], embed_dims[0], (3,3,3), stride=1, padding=(1,1,1))
        
       
        
        self.decoder3 = TransformerBlock(dim=embed_dims[2], num_heads=num_heads)

        self.dbn1 = nn.BatchNorm3d(embed_dims[2])
        self.dbn2 = nn.BatchNorm3d(embed_dims[2])
        self.dbn3 = nn.BatchNorm3d(embed_dims[2])
        self.dbn4 = nn.BatchNorm3d(embed_dims[0])
        
        
        self.final = nn.Conv3d(embed_dims[0], num_classes,kernel_size=1)
        #self.soft = nn.Softmax(dim =1)

    def forward(self, x):
        ### Encoder
        ### Conv Stage

        ### Stage 1
        
        
        out = self.ebn1(self.encoder1(x))
        out = F.relu(F.max_pool3d(out,2,2))
        t1 = out
        ### Stage 2
        out = self.ebn2(self.encoder2(out))
        out = F.relu(F.max_pool3d(out,2,2))
        t2 = out
        ### Stage 3       
        out = self.ebn3(self.encoder3(out))
        out = F.relu(F.max_pool3d(out,2,2))
        t3 = out
        
        #Transformer stage
        #Stage 4
        out = self.trasformerblock1(out)
        out = F.relu(F.max_pool3d(out,(1,2,2),(1,2,2)))
        out = self.norm3(out)
        t4=out
        
        ### Bottleneck
        out = self.trasformerblock2(out)
        out = F.relu(F.max_pool3d(out,(1,2,2),(1,2,2)))
        out = self.norm4(out)
         
        #stage4
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(1,2,2),mode ='trilinear'))
        out = torch.add(out,t4) 
        out = self.trasformerblock3(out) 
        ### Stage 3   
        out = self.dnorm3(out)    
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(1,2,2),mode ='trilinear'))
        out = torch.add(out,t3)
        out = self.trasformerblock4(out)
        ### Stage 2
        out = self.dnorm4(out)
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t2)
        
        
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2,2),mode ='trilinear'))
        out = torch.add(out,t1)
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2,2),mode ='trilinear'))
        
        
        
        out = self.final(out)
        #out = self.soft(out)
   
        return out