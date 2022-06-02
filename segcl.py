#import pdb
import sys
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange,repeat

def identity(t):
    return t

def rand_true(prob):
    return random.random() < prob

# loss fn

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

class PPM(nn.Module):
    def __init__(self, nchannels, gamma):
        super(PPM, self).__init__()
        self.transform_layer = nn.Conv2d(nchannels, nchannels, 1)
        self.gamma = gamma

    def forward(self, x):
        x_reshape = x.reshape((x.size(0),x.size(1),-1))
        
        xi = x_reshape[:, :, :, None] # Bx256x49x1
        xj = x_reshape[:, :, None, :] # Bx256x1x49
        
        size = xi.size(2)
        for i in range(64):
            if i ==0:
                similarity = F.relu(F.cosine_similarity(xi,xj[:,:,:,i*(size//64):(i+1)*(size//64)], dim = 1))** self.gamma
            else:
                similarity = torch.cat([similarity,F.relu(F.cosine_similarity(xi,xj[:,:,:,i*(size//64):(i+1)*(size//64)], dim = 1))** self.gamma],dim=2)
                           
        #print(similarity.size())
        similarity = similarity.reshape((similarity.size(0),x.size(2),x.size(3),x.size(2),x.size(3)))
        #print(similarity.size())
        transform_out = self.transform_layer(x) # Bx256x7x7
        out = torch.einsum('b x y h w, b c h w -> b c x y', similarity, transform_out) # Bx256x7x7
        return out
     


class SegCL(nn.Module):
    """
    Ref https://github.com/lucidrains/pixel-level-contrastive-learning
    """
    def __init__(self, base_encoder, use_pixpro=True, dim=256, T = 0.3, m = 0.99, alpha = 1, C=1600, K=100,ln=0.02):
        '''
        dis_thre: threhold of coordiantes distance
        dim: last feat channel (default 256)
        T: softmax temperature (default 0.2)
        '''
        super(SegCL, self).__init__()
        self.use_pixpro = use_pixpro
        self.K = K
        self.C = C
        self.T = T
        self.m = m
        self.base_m = m
        self.alpha = alpha
        self.ln = ln
        
        # flip augmentation
        self.rand_flip_fn = lambda t: torch.flip(t, dim = (-1, ))
        
        self.encoder_q = base_encoder()
        self.encoder_k = base_encoder()
        #self.encoder_t = base_encoder()

        # replace fc with 1x1 conv to keep 7x7 resolution
        self.encoder_q.classifier.out1 = nn.Sequential(
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, dim, 1))
        
        self.encoder_k.classifier.out1 = nn.Sequential(
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, dim, 1))
        
        self.encoder_q.classifier.out2 = nn.Sequential(
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, dim, 1))
        
        self.encoder_k.classifier.out2 = nn.Sequential(
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1), 
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, dim, 1))
        
        self.propix = PPM(nchannels=256, gamma=2)
        
        # create the queue
        self.register_buffer("queue1", torch.zeros(C, dim, K))
        self.register_buffer("mask1", torch.zeros((C, K), dtype=torch.long))
        self.register_buffer("queue2", torch.zeros(C, dim, K))
        self.register_buffer("mask2", torch.zeros((C, K), dtype=torch.long))   
        #self.register_buffer("instance_queue1", torch.zeros(C, dim, K))
        #self.register_buffer("instance_queue2", torch.zeros(C, dim, K))           
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("class_ptr", torch.zeros(1, dtype=torch.long))

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
            
    def _dequeue_and_enqueue(self, keys1, keys2, masks1, masks2, fake_keys1, fake_keys2, fake_masks1, fake_masks2, low_fake_masks1, low_fake_masks2):
        # gather keys before updating queue
        keys1 = concat_all_gather(keys1)
        keys2 = concat_all_gather(keys2)
        #ins_keys1 = concat_all_gather(ins_keys1)
        #ins_keys2 = concat_all_gather(ins_keys2)         
        masks1 = concat_all_gather(masks1)
        masks2 = concat_all_gather(masks2)

        fake_keys1 = concat_all_gather(fake_keys1)
        fake_keys2 = concat_all_gather(fake_keys2)
        #fake_ins_keys1 = concat_all_gather(fake_ins_keys1)
        #fake_ins_keys2 = concat_all_gather(fake_ins_keys2)         
        fake_masks1 = concat_all_gather(fake_masks1)
        fake_masks2 = concat_all_gather(fake_masks2)
        low_fake_masks1 = concat_all_gather(low_fake_masks1)
        low_fake_masks2 = concat_all_gather(low_fake_masks2)        
        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        class_ptr = int(self.class_ptr)
        # assert self.K % batch_size == 0  # for simplicity
        keys1 = rearrange(keys1, 'K C dim -> C dim K')
        keys2 = rearrange(keys2, 'K C dim -> C dim K')
        #ins_keys1 = rearrange(ins_keys1, 'K C dim -> C dim K')
        #ins_keys2 = rearrange(ins_keys2, 'K C dim -> C dim K')         
        masks1 = rearrange(masks1, 'K C -> C K')
        masks2 = rearrange(masks2, 'K C -> C K')

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, :, ptr] = keys1.sum(dim=-1)
        self.queue2[:, :, ptr] = keys2.sum(dim=-1)
        #self.instance_queue1[:, :, ptr] = ins_keys1.sum(dim=-1)
        #self.instance_queue2[:, :, ptr] = ins_keys2.sum(dim=-1)          
        self.mask1[:, ptr] = masks1.sum(dim=-1)
        self.mask2[:, ptr] = masks2.sum(dim=-1)
        
        #combine similar prototype
#         cal1_mean = 0
#         cal2_mean = 0
#         cal1 = {}
#         cal2 = {}
#         raw_cal1=[]
#         raw_cal2=[]
#         for i in range(self.C):
#             if self.mask1[i].sum() != 0:
#                 temp_crop1_feat_k = (self.queue1[i]).sum(-1) / self.mask1[i].sum(-1)  # 256xk->256
#                 temp_crop1_feat_k = temp_crop1_feat_k.unsqueeze(0)  # 256->1x256
#                 if cal1_mean == 0:
#                     queue_crop1_feat_k = temp_crop1_feat_k
#                 else:
#                     queue_crop1_feat_k = torch.cat([queue_crop1_feat_k, temp_crop1_feat_k], dim=0)  # cx256
#                 raw_cal1.append(i)
#                 cal1[i]=cal1_mean
#                 cal1_mean += 1
#             if self.mask2[i].sum() != 0:
#                 temp_crop2_feat_k = (self.queue2[i]).sum(-1) / self.mask2[i].sum(-1)  # 256xk->256
#                 temp_crop2_feat_k = temp_crop2_feat_k.unsqueeze(0)  # 256->1x256
#                 if cal2_mean == 0:
#                     queue_crop2_feat_k = temp_crop2_feat_k
#                 else:
#                     queue_crop2_feat_k = torch.cat([queue_crop2_feat_k, temp_crop2_feat_k], dim=0)  # cx256
#                 raw_cal2.append(i)
#                 cal2[i]=cal2_mean
#                 cal2_mean += 1
#         raw_cal1 = torch.LongTensor(raw_cal1).cuda()
#         raw_cal2 = torch.LongTensor(raw_cal2).cuda()
                

#         queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
#         queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256

#         trans_queue_crop1_feat_k = rearrange(queue_crop1_feat_k,'c dim -> dim c')
#         trans_queue_crop2_feat_k = rearrange(queue_crop2_feat_k,'c dim -> dim c')
#         sim_queue1 = (queue_crop1_feat_k @ trans_queue_crop1_feat_k) #cxc
#         sim_queue2 = (queue_crop2_feat_k @ trans_queue_crop2_feat_k) #cxc

#         q_m1, q_class1 = sim_queue1.topk(2, dim=1, sorted=True) #cx2
#         q_m2, q_class2 = sim_queue2.topk(2, dim=1, sorted=True) #cx2

#        for i in range(self.C):    
#            if self.mask1[i].sum()!=0:
#                if q_m1[cal1[i],-1]>0.85: 
#                    temp_class1 = raw_cal1[q_class1[cal1[i],-1]] 
#                    if temp_class1 < 21:
#                        continue
#                    self.queue1[i] += self.queue1[temp_class1]
#                    self.mask1[i] += self.mask1[temp_class1]
#                    self.queue1[temp_class1] = 0
#                    self.mask1[temp_class1] = 0
#
#            if self.mask2[i].sum()!=0:
#                if q_m2[cal2[i],-1]>0.85: 
#                    temp_class2 = raw_cal2[q_class2[cal2[i],-1]] 
#                    if temp_class2 < 21:
#                        continue
#                    self.queue2[i] += self.queue2[temp_class2]
#                    self.mask2[i] += self.mask2[temp_class2]
#                    self.queue2[temp_class2] = 0
#                    self.mask2[temp_class2] = 0

        fake_keys1 = rearrange(fake_keys1, 'K C dim -> C dim K')
        fake_keys2 = rearrange(fake_keys2, 'K C dim -> C dim K')
        #fake_ins_keys1 = rearrange(fake_ins_keys1, 'K C dim -> C dim K')
        #fake_ins_keys2 = rearrange(fake_ins_keys2, 'K C dim -> C dim K')             
        fake_masks1 = rearrange(fake_masks1, 'K C -> C K')
        fake_masks2 = rearrange(fake_masks2, 'K C -> C K')
        low_fake_masks1 = rearrange(low_fake_masks1, 'K C -> C K')
        low_fake_masks2 = rearrange(low_fake_masks2, 'K C -> C K')

        fake_keys1 = fake_keys1.sum(dim=-1)
        fake_keys2 = fake_keys2.sum(dim=-1)
        #fake_ins_keys1 = fake_ins_keys1.sum(dim=-1)
        #fake_ins_keys2 = fake_ins_keys2.sum(dim=-1)        
        fake_masks_cnt1 = fake_masks1.sum(dim=-1)
        fake_masks_cnt2 = fake_masks2.sum(dim=-1)
        low_fake_masks_cnt1 = low_fake_masks1.sum(dim=-1)
        low_fake_masks_cnt2 = low_fake_masks2.sum(dim=-1)
        conf1 = (low_fake_masks_cnt1/(fake_masks_cnt1 + 1e-10))>0.5
        conf2 = (low_fake_masks_cnt2/(fake_masks_cnt2 + 1e-10))>0.5
        fake_masks1 = (fake_masks1!=0).sum(dim=-1)
        fake_masks2 = (fake_masks2!=0).sum(dim=-1)       
        ct1,loc1 = (fake_masks_cnt1*conf1.float()).topk(10,largest=True)
        ct2,loc2 = (fake_masks_cnt2*conf2.float()).topk(10,largest=True)
        #re_ct1,re_loc1 = (low_fake_masks_cnt1).topk(10,largest=True)

        cnt = self.C - class_ptr
        if cnt>=10:
            if class_ptr<255 and 255-class_ptr+1<=10:
                cnt2 = 255-class_ptr
                cnt3 = 10-cnt2
                self.queue1[class_ptr:class_ptr+cnt2, :, ptr] = fake_keys1[loc1[:cnt2]]
                self.queue2[class_ptr:class_ptr+cnt2, :, ptr] = fake_keys2[loc2[:cnt2]]
                #self.instance_queue1[class_ptr:class_ptr+cnt2, :, ptr] = fake_ins_keys1[loc1[:cnt2]]
                #self.instance_queue2[class_ptr:class_ptr+cnt2, :, ptr] = fake_ins_keys2[loc2[:cnt2]]
                self.mask1[class_ptr:class_ptr+cnt2, ptr] = fake_masks1[loc1[:cnt2]]
                self.mask2[class_ptr:class_ptr+cnt2, ptr] = fake_masks2[loc2[:cnt2]]
                self.queue1[256:256+cnt3, :, ptr] = fake_keys1[loc1[cnt2:]]
                self.queue2[256:256+cnt3, :, ptr] = fake_keys2[loc2[cnt2:]]
                #self.instance_queue1[256:256+cnt3, :, ptr] = fake_ins_keys1[loc1[cnt2:]]
                #self.instance_queue2[256:256+cnt3, :, ptr] = fake_ins_keys2[loc2[cnt2:]]
                self.mask1[256:256+cnt3, ptr] = fake_masks1[loc1[cnt2:]]
                self.mask2[256:256+cnt3, ptr] = fake_masks2[loc2[cnt2:]] 
                class_ptr = 256 + cnt3                   

            else:
                self.queue1[class_ptr:class_ptr+10, :, ptr] = fake_keys1[loc1]
                self.queue2[class_ptr:class_ptr+10, :, ptr] = fake_keys2[loc2]
                #self.instance_queue1[class_ptr:class_ptr+10, :, ptr] = fake_ins_keys1[loc1]
                #self.instance_queue2[class_ptr:class_ptr+10, :, ptr] = fake_ins_keys2[loc2]                
                self.mask1[class_ptr:class_ptr+10, ptr] = fake_masks1[loc1]
                self.mask2[class_ptr:class_ptr+10, ptr] = fake_masks2[loc2]
                class_ptr = class_ptr + 10
        elif cnt>0:
            self.queue1[class_ptr:class_ptr+cnt, :, ptr] = fake_keys1[loc1[:cnt]]
            self.queue2[class_ptr:class_ptr+cnt, :, ptr] = fake_keys2[loc2[:cnt]]
            #self.instance_queue1[class_ptr:class_ptr+cnt, :, ptr] = fake_ins_keys1[loc1[:cnt]]
            #self.instance_queue2[class_ptr:class_ptr+cnt, :, ptr] = fake_ins_keys2[loc2[:cnt]]            
            self.mask1[class_ptr:class_ptr+cnt, ptr] = fake_masks1[loc1[:cnt]]
            self.mask2[class_ptr:class_ptr+cnt, ptr] = fake_masks2[loc2[:cnt]]    
            class_ptr = class_ptr + cnt         
        
        else:
            
            ct1,loc1 = (fake_masks1*conf1.float()).topk(10,largest=True)
            ct2,loc2 = (fake_masks2*conf2.float()).topk(10,largest=True)            
            queue_masks1 = self.mask1.sum(dim=-1)
            queue_masks2 = self.mask2.sum(dim=-1)
            qct1, qloc1 = queue_masks1.topk(11,largest=False)
            qct2, qloc2 = queue_masks2.topk(11,largest=False)
            qct1 = qct1[qloc1!=255]
            qct2 = qct2[qloc2!=255]
            qloc1 = qloc1[qloc1!=255]
            qloc2 = qloc2[qloc2!=255]
            #print(qct1,qloc1)
            #print(ct1,loc1)
            for i in range(10):
                if qct1[i]<ct1[i]:
                   self.queue1[qloc1[i], :, :]=0
                   #self.instance_queue1[qloc1[i], :, :]=0
                   self.mask1[qloc1[i], :]=0
                   self.queue1[qloc1[i], :, ptr]=fake_keys1[loc1[i]]
                   #self.instance_queue1[qloc1[i], :, ptr]=fake_ins_keys1[loc1[i]]
                   self.mask1[qloc1[i], ptr]=fake_masks1[loc1[i]]
                if qct2[i]<ct2[i]:
                   self.queue2[qloc2[i], :, :]=0
                   #self.instance_queue2[qloc2[i], :, :]=0
                   self.mask2[qloc2[i], :]=0
                   self.queue2[qloc2[i], :, ptr]=fake_keys2[loc2[i]]
                   #self.instance_queue2[qloc2[i], :, ptr]=fake_ins_keys2[loc2[i]]
                   self.mask2[qloc2[i], ptr]=fake_masks2[loc2[i]]
        
        # print(ptr)       
        ptr = (ptr + 1) % self.K  # move pointer

        self.queue_ptr[0] = ptr
        self.class_ptr[0] = class_ptr
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def generate_pseudo_label(self,crop1,crop2,crop1_seg, crop2_seg, crop1_mask, crop2_mask):
        shape = crop1_seg.size()
        with torch.no_grad():
            self._momentum_update_key_encoder()
            crop1_feat_k, crop1_instance_k = self.encoder_k(crop1)
            crop2_feat_k, crop2_instance_k = self.encoder_k(crop2)

            #crop1_feat_seg = rearrange(crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            #crop2_feat_seg = rearrange(crop2_seg, 'b c h w -> b (h w) c') # Bx49x1

            crop1_feat_k, crop2_feat_k = \
                    list(map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), \
                    (crop1_feat_k, crop2_feat_k)))

            cal1_mean = 0
            cal2_mean = 0
            cal1 = []
            cal2 = []
            for i in range(self.C):
                if self.mask1[i].sum() != 0:
                    temp_crop1_feat_k = (self.queue1[i]).sum(-1) / self.mask1[i].sum(-1)  # 256xk->256
                    temp_crop1_feat_k = temp_crop1_feat_k.unsqueeze(0)  # 256->1x256
                    if cal1_mean == 0:
                        queue_crop1_feat_k = temp_crop1_feat_k
                    else:
                        queue_crop1_feat_k = torch.cat([queue_crop1_feat_k, temp_crop1_feat_k], dim=0)  # cx256
                    cal1.append(i)
                    cal1_mean += 1
                if self.mask2[i].sum() != 0:
                    temp_crop2_feat_k = (self.queue2[i]).sum(-1) / self.mask2[i].sum(-1)  # 256xk->256
                    temp_crop2_feat_k = temp_crop2_feat_k.unsqueeze(0)  # 256->1x256
                    if cal2_mean == 0:
                        queue_crop2_feat_k = temp_crop2_feat_k
                    else:
                        queue_crop2_feat_k = torch.cat([queue_crop2_feat_k, temp_crop2_feat_k], dim=0)  # cx256
                    cal2.append(i)
                    cal2_mean += 1
                
            cal1 = torch.LongTensor(cal1).cuda()
            cal2 = torch.LongTensor(cal2).cuda()

            temp_crop1_feat_t = F.normalize(crop1_feat_k, dim=1)  # bx256x49
            temp_crop2_feat_t = F.normalize(crop2_feat_k, dim=1)
            queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
            queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256

            sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_t) 
            sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_t) 
            sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
            sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc
            t_m1, t_class1 = sim_crop1t_2k.topk(2, dim=2, sorted=True)  # bx49x3
            t_m2, t_class2 = sim_crop2t_1k.topk(2, dim=2, sorted=True)  # bx49x3


            over_thre1 = (t_m1[:,:,0] > 0.8).long()
            over_thre2 = (t_m2[:,:,0] > 0.8).long()
            low_thre1 = (t_m1[:,:,0] < 0.7).long()
            low_thre2 = (t_m2[:,:,0] < 0.7).long() 
            #print(over_thre1.sum())
            #print(t_m1.max(),t_m1.min())

            crop1_feat_seg_semi = ((cal2[t_class1[:,:,0]] + 1)*over_thre1).squeeze().long()
            crop2_feat_seg_semi = ((cal1[t_class2[:,:,0]] + 1)*over_thre2).squeeze().long()
            
            sorted_t_class1, _ = torch.sort(t_class1) 
            sorted_t_class2, _ = torch.sort(t_class2) 
            
            #hash to save time
            hash_num = torch.LongTensor([[[1,4]]]).cuda()
            #hash = hash.unsqueeze(0).unsqueeze(0)

            fake_crop1_feat_seg = (((cal2[sorted_t_class1]*hash_num).sum(dim=-1) + 1)).squeeze().long() 
            fake_crop2_feat_seg = (((cal1[sorted_t_class2]*hash_num).sum(dim=-1) + 1)).squeeze().long()
                
            fake_crop1_feat_seg = fake_crop1_feat_seg * (1 - crop1_mask)
            fake_crop2_feat_seg = fake_crop2_feat_seg * (1 - crop2_mask)

            low_fake_crop1_feat_seg = fake_crop1_feat_seg * low_thre1
            low_fake_crop2_feat_seg = fake_crop2_feat_seg * low_thre2
            #crop1_feat_seg = (crop1_feat_seg+1).squeeze().long() * crop1_mask + crop1_feat_seg_semi * (1 - crop1_mask)
            #crop2_feat_seg = (crop2_feat_seg+1).squeeze().long() * crop2_mask + crop2_feat_seg_semi * (1 - crop2_mask) 

            crop1_feat_seg_semi = rearrange(crop1_feat_seg_semi, 'b (h w) -> b h w',h=crop1_seg.size(2)//4,w=crop1_seg.size(3)//4).unsqueeze(1) #bxcxhxw
            crop2_feat_seg_semi = rearrange(crop2_feat_seg_semi, 'b (h w) -> b h w',h=crop2_seg.size(2)//4,w=crop2_seg.size(3)//4).unsqueeze(1)
            
            fake_crop1_feat_seg = rearrange(fake_crop1_feat_seg, 'b (h w) -> b h w',h=crop1_seg.size(2)//4,w=crop1_seg.size(3)//4).unsqueeze(1)#bxcxhxw
            fake_crop2_feat_seg = rearrange(fake_crop2_feat_seg, 'b (h w) -> b h w',h=crop2_seg.size(2)//4,w=crop2_seg.size(3)//4).unsqueeze(1)
            low_fake_crop1_feat_seg = rearrange(low_fake_crop1_feat_seg, 'b (h w) -> b h w',h=crop1_seg.size(2)//4,w=crop1_seg.size(3)//4).unsqueeze(1)#bxcxhxw
            low_fake_crop2_feat_seg = rearrange(low_fake_crop2_feat_seg, 'b (h w) -> b h w',h=crop2_seg.size(2)//4,w=crop2_seg.size(3)//4).unsqueeze(1)

            crop1_feat_seg_semi = F.interpolate(crop1_feat_seg_semi.float(), size=(shape[2],shape[3]), mode='nearest').long()
            crop2_feat_seg_semi = F.interpolate(crop2_feat_seg_semi.float(), size=(shape[2],shape[3]), mode='nearest').long()
            #print(crop1_seg.size(),crop1_mask.size(),crop1_feat_seg_semi.size())
            crop1_feat_seg = (crop1_seg+1).long() * crop1_mask.unsqueeze(1).unsqueeze(1) + crop1_feat_seg_semi * (1 - crop1_mask).unsqueeze(1).unsqueeze(1)
            crop2_feat_seg = (crop2_seg+1).long() * crop2_mask.unsqueeze(1).unsqueeze(1) + crop2_feat_seg_semi * (1 - crop2_mask).unsqueeze(1).unsqueeze(1)
            
            return crop1_feat_seg, crop2_feat_seg, fake_crop1_feat_seg, fake_crop2_feat_seg, low_fake_crop1_feat_seg, low_fake_crop2_feat_seg


    def forward(self, crop1, crop2, crop1_seg, crop2_seg, crop1_mask, crop2_mask, fake_crop1_seg, fake_crop2_seg, low_fake_crop1_seg, low_fake_crop2_seg, mix_crop1, mix_crop2, mix_crop1_seg, mix_crop2_seg, add_crop1,add_crop2,add_mix_crop1,add_mix_crop2, c1_bbx1=0, c1_bbx2=0, c1_bby1=0, c1_bby2=0, c2_bbx1=0, c2_bbx2=0, c2_bby1=0, c2_bby2=0, lam1=0, lam2=0, is_cutmix=True, full=True):
        '''
        Args:
            crop1&crop2: Bx3x224x224
            crop1_seg&crop2_seg: Bx1x7x7
        '''
        if is_cutmix:
            shape = crop1_seg.size()            
            # spatial augmentations #
            # define rand horizontal flip augmentaitons
            is_flip_crop1 = rand_true(0.5)
            is_flip_crop2 = rand_true(0.5)
            is_flip_mix_crop1 = rand_true(0.5)
            is_flip_mix_crop2 = rand_true(0.5)        
            flip_fn = lambda t: torch.flip(t, dims=(-1,)) # horizentoal flip

            flip_crop1_fn = flip_fn if is_flip_crop1 else identity
            flip_crop2_fn = flip_fn if is_flip_crop2 else identity
            flip_mix_crop1_fn = flip_fn if is_flip_mix_crop1 else identity
            flip_mix_crop2_fn = flip_fn if is_flip_mix_crop2 else identity
        
            # flip croped image
            crop1 = flip_crop1_fn(crop1)
            crop2 = flip_crop2_fn(crop2)
            crop1_seg = flip_crop1_fn(crop1_seg)
            crop2_seg = flip_crop2_fn(crop2_seg)
            add_crop1 = flip_crop1_fn(add_crop1)
            add_crop2 = flip_crop2_fn(add_crop2)
            fake_crop1_seg = flip_crop1_fn(fake_crop1_seg)
            fake_crop2_seg = flip_crop2_fn(fake_crop2_seg)
        
            mix_crop1 = flip_mix_crop1_fn(mix_crop1)
            mix_crop2 = flip_mix_crop2_fn(mix_crop2)
            
            mix_crop1_seg = flip_mix_crop1_fn(mix_crop1_seg)
            mix_crop2_seg = flip_mix_crop2_fn(mix_crop2_seg)  

            add_mix_crop1 = flip_mix_crop1_fn(add_mix_crop1)
            add_mix_crop2 = flip_mix_crop2_fn(add_mix_crop2)  
            ##########################################################


        
            # get forward feature #
            crop1_feat_q, crop1_instance_q = self.encoder_q(crop1) # Bx256x7x7
            crop2_feat_q, crop2_instance_q = self.encoder_q(crop2) # Bx256x7x7
            mix_crop1_feat_q, mix_crop1_instance_q = self.encoder_q(mix_crop1) # Bx256x7x7
            mix_crop2_feat_q, mix_crop2_instance_q = self.encoder_q(mix_crop2) # Bx256x7x7        
            with torch.no_grad():
                self._momentum_update_key_encoder()
                crop1_feat_k, crop1_instance_k = self.encoder_k(crop1)
                crop2_feat_k, crop2_instance_k = self.encoder_k(crop2)
                mix_crop1_feat_k, mix_crop1_instance_k = self.encoder_k(mix_crop1)
                mix_crop2_feat_k, mix_crop2_instance_k = self.encoder_k(mix_crop2)                 
                crop1_feat_t, crop1_instance_t = self.encoder_k(add_crop1)
                crop2_feat_t, crop2_instance_t = self.encoder_k(add_crop2)   
                mix_crop1_feat_t, mix_crop1_instance_t = self.encoder_k(add_mix_crop1)
                mix_crop2_feat_t, mix_crop2_instance_t = self.encoder_k(add_mix_crop2)                            
            nfeat_pix = (crop1_feat_q.size(-1))**2
            ##########################################################


            # get postive maks #
            # reshape coord format 
            # sort order: [[r1,c1],[r1,c2],
            #              [r2,c1],[r2,c2]]
            crop1_feat_seg = rearrange(crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            crop2_feat_seg = rearrange(crop2_seg, 'b c h w -> b (h w) c') # Bx49x1
            fake_crop1_feat_seg = rearrange(fake_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            fake_crop2_feat_seg = rearrange(fake_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1
            low_fake_crop1_feat_seg = rearrange(low_fake_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            low_fake_crop2_feat_seg = rearrange(low_fake_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1            
            
            mix_crop1_feat_seg = rearrange(mix_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            mix_crop2_feat_seg = rearrange(mix_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1
            
            ##########################################################
        
            # compute final pix loss #
            if not self.use_pixpro:
                # pix-contrast loss #
                # crop1_feat_q: Bx256x49
            
                mix_crop1_feat_q, mix_crop2_feat_q, mix_crop1_feat_k, mix_crop2_feat_k, crop1_feat_q, crop2_feat_q, crop1_feat_k, crop2_feat_k = \
                    list(map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), \
                    (mix_crop1_feat_q, mix_crop2_feat_q, mix_crop1_feat_k, mix_crop2_feat_k, crop1_feat_q, crop2_feat_q, crop1_feat_k, crop2_feat_k)))

                mix_crop1_feat_t, mix_crop2_feat_t, crop1_feat_t, crop2_feat_t = \
                    list(map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), \
                    (mix_crop1_feat_t, mix_crop2_feat_t, crop1_feat_t, crop2_feat_t)))


                #print(crop2_feat_k.size())
                dim = crop1_feat_q.size(1)
                batch_size=crop1_feat_q.size(0) 
                
                # update
                mean_crop1_feat_k = torch.zeros(batch_size, self.C, dim).cuda()
                mask_crop1 = torch.zeros((batch_size, self.C), dtype=torch.long).cuda()
                mean_crop2_feat_k = torch.zeros(batch_size, self.C, dim).cuda()
                mask_crop2 = torch.zeros((batch_size, self.C), dtype=torch.long).cuda()
                #mean_crop1_instance_k = torch.zeros(batch_size,self.C,dim).cuda()
                #mean_crop2_instance_k = torch.zeros(batch_size,self.C,dim).cuda()   

                fake_mean_crop1_feat_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                #fake_mean_crop1_instance_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                fake_mask_crop1 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                fake_mean_crop2_feat_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                #fake_mean_crop2_instance_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                fake_mask_crop2 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                low_fake_mask_crop1 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                low_fake_mask_crop2 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()


                for i in range(batch_size):
                    crop1_label = torch.unique(crop1_feat_seg[i])
                    crop2_label = torch.unique(crop2_feat_seg[i])
                    fake_crop1_label = torch.unique(fake_crop1_feat_seg[i])
                    fake_crop2_label = torch.unique(fake_crop2_feat_seg[i])
                    for j in crop1_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (crop1_feat_seg[i].squeeze() == j)  # 49
                        mask_crop1[i][j.item() - 1] = 1
                        mean_crop1_feat_k[i][j.item() - 1] = (crop1_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #mean_crop1_instance_k[i][j.item()-1] = (crop1_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()

                    for j in crop2_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (crop2_feat_seg[i].squeeze() == j)  # 49
                        mask_crop2[i][j.item() - 1] = 1
                        mean_crop2_feat_k[i][j.item() - 1] = (crop2_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #mean_crop2_instance_k[i][j.item()-1] = (crop2_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()    

                    for j in fake_crop1_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (fake_crop1_feat_seg[i].squeeze() == j)  # 49
                        low_temp_mask = (low_fake_crop1_feat_seg[i].squeeze() == j)  # 49
                        fake_mask_crop1[i][j.item() - 1] = temp_mask.float().sum()
                        low_fake_mask_crop1[i][j.item() - 1] = low_temp_mask.float().sum()
                        fake_mean_crop1_feat_k[i][j.item() - 1] = (crop1_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #fake_mean_crop1_instance_k[i][j.item()-1] = (crop1_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()    

                    for j in fake_crop2_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (fake_crop2_feat_seg[i].squeeze() == j)  # 49
                        low_temp_mask = (low_fake_crop2_feat_seg[i].squeeze() == j)  # 49
                        fake_mask_crop2[i][j.item() - 1] = temp_mask.float().sum()
                        low_fake_mask_crop2[i][j.item() - 1] = low_temp_mask.float().sum()
                        fake_mean_crop2_feat_k[i][j.item() - 1] = (crop2_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #fake_mean_crop2_instance_k[i][j.item()-1] = (crop2_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()            
                
                self._dequeue_and_enqueue(mean_crop1_feat_k, mean_crop2_feat_k, mask_crop1, mask_crop2, fake_mean_crop1_feat_k, fake_mean_crop2_feat_k,  fake_mask_crop1, fake_mask_crop2, low_fake_mask_crop1, low_fake_mask_crop2)

                cal1_mean = 0
                cal2_mean = 0
                cal1 = {}
                cal2 = {}
                raw_cal1=[]
                raw_cal2=[]
                for i in range(self.C):
                    if self.mask1[i].sum() != 0:
                        temp_crop1_feat_k = (self.queue1[i]).sum(-1) / self.mask1[i].sum(-1)  # 256xk->256
                        temp_crop1_feat_k = temp_crop1_feat_k.unsqueeze(0)  # 256->1x256
                        #temp_crop1_instance_k = (self.instance_queue1[i]*self.mask1[i]).sum(-1)/self.mask1[i].sum(-1) #256xk->256
                        #temp_crop1_instance_k = temp_crop1_instance_k.unsqueeze(0)#256->1x256                            
                              
                        if cal1_mean == 0:
                            queue_crop1_feat_k = temp_crop1_feat_k
                            #queue_crop1_instance_k = temp_crop1_instance_k
                        else:
                            queue_crop1_feat_k = torch.cat([queue_crop1_feat_k, temp_crop1_feat_k], dim=0)  # cx256
                            #queue_crop1_instance_k = torch.cat([queue_crop1_instance_k, temp_crop1_instance_k],dim=0) #cx256
                        raw_cal1.append(i)
                        cal1[i]=cal1_mean
                        cal1_mean += 1
                    if self.mask2[i].sum() != 0:
                        temp_crop2_feat_k = (self.queue2[i]).sum(-1) / self.mask2[i].sum(-1)  # 256xk->256
                        temp_crop2_feat_k = temp_crop2_feat_k.unsqueeze(0)  # 256->1x256
                        #temp_crop2_instance_k = (self.instance_queue2[i]*self.mask2[i]).sum(-1)/self.mask2[i].sum(-1) #256xk->256
                        #temp_crop2_instance_k = temp_crop2_instance_k.unsqueeze(0)#256->1x256
                        if cal2_mean == 0:
                            queue_crop2_feat_k = temp_crop2_feat_k
                            #queue_crop2_instance_k = temp_crop2_instance_k
                        else:
                            queue_crop2_feat_k = torch.cat([queue_crop2_feat_k, temp_crop2_feat_k], dim=0)  # cx256
                            #queue_crop2_instance_k = torch.cat([queue_crop2_instance_k, temp_crop2_instance_k],dim=0) #cx256
                        raw_cal2.append(i)
                        cal2[i]=cal2_mean
                        cal2_mean += 1
                raw_cal1 = torch.LongTensor(raw_cal1).cuda()
                raw_cal2 = torch.LongTensor(raw_cal2).cuda()                
                #print(cal1_mean,cal2_mean) 

                temp_crop1_feat_f = F.normalize(crop1_feat_k, dim=1)  # bx256x49
                temp_crop2_feat_f = F.normalize(crop2_feat_k, dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256

                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_f) 
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_f) 
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc
                t_m1, t_class1 = sim_crop1t_2k.topk(1, dim=2, sorted=True)  # bx49
                t_m2, t_class2 = sim_crop2t_1k.topk(1, dim=2, sorted=True)

                over_thre1 = (t_m1 > 0.8).long()
                over_thre2 = (t_m2 > 0.8).long()

                crop1_feat_seg_semi = ((raw_cal2[t_class1] + 1)*over_thre1).squeeze().long().unsqueeze(2)
                crop2_feat_seg_semi = ((raw_cal1[t_class2] + 1)*over_thre2).squeeze().long().unsqueeze(2)

                crop1_feat_seg = crop1_feat_seg * crop1_mask.unsqueeze(1) + crop1_feat_seg_semi * (1 - crop1_mask).unsqueeze(1)
                crop2_feat_seg = crop2_feat_seg * crop2_mask.unsqueeze(1) + crop2_feat_seg_semi * (1 - crop2_mask).unsqueeze(1)

                #update mix label
                crop1_seg = rearrange(crop1_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1
                crop2_seg = rearrange(crop2_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1
                mix_crop1_seg = rearrange(mix_crop1_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1
                mix_crop2_seg = rearrange(mix_crop2_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1    
                crop1_seg = flip_crop1_fn(crop1_seg)
                crop2_seg = flip_crop2_fn(crop2_seg) 
                mix_crop1_seg = flip_mix_crop1_fn(mix_crop1_seg)
                mix_crop2_seg = flip_mix_crop2_fn(mix_crop2_seg)

                temp_crop1_seg =  F.interpolate(crop1_seg.float(), size=(shape[2]*4,shape[3]*4), mode='nearest').long()
                temp_crop2_seg =  F.interpolate(crop2_seg.float(), size=(shape[2]*4,shape[3]*4), mode='nearest').long()
                mix_crop1_seg =  F.interpolate(mix_crop1_seg.float(), size=(shape[2]*4,shape[3]*4), mode='nearest').long()
                mix_crop2_seg =  F.interpolate(mix_crop2_seg.float(), size=(shape[2]*4,shape[3]*4), mode='nearest').long()
                temp_crop1_seg[:, :, c1_bbx1:c1_bbx2, c1_bby1:c1_bby2] = mix_crop1_seg[:, :, c1_bbx1:c1_bbx2, c1_bby1:c1_bby2]
                temp_crop2_seg[:, :, c2_bbx1:c2_bbx2, c2_bby1:c2_bby2] = mix_crop2_seg[:, :, c2_bbx1:c2_bbx2, c2_bby1:c2_bby2]
                mix_crop1_seg = F.interpolate(temp_crop1_seg.float(), size=(shape[2],shape[3]), mode='nearest').long()
                mix_crop2_seg = F.interpolate(temp_crop2_seg.float(), size=(shape[2],shape[3]), mode='nearest').long()

                mix_crop1_seg = flip_mix_crop1_fn(mix_crop1_seg)
                mix_crop2_seg = flip_mix_crop2_fn(mix_crop2_seg)
                mix_crop1_feat_seg = rearrange(mix_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
                mix_crop2_feat_seg = rearrange(mix_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1



                loss_pix_1q_2k = 0
                loss_pix_2q_1k = 0    


                temp_crop1_feat_q = F.normalize(crop1_feat_q,dim=1)#bx256x49
                temp_crop2_feat_q = F.normalize(crop2_feat_q,dim=1)

                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49

                sim_crop1q_2k = sim_crop1q_2k.squeeze().transpose(2,1) #bx49xc
                sim_crop2q_1k = sim_crop2q_1k.squeeze().transpose(2,1) #bx49xc 

                exp_sim_crop1q_2k = sim_crop1q_2k.exp()
                exp_sim_crop2q_1k = sim_crop2q_1k.exp()
            
                sum_exp_sim_crop1q_2k = exp_sim_crop1q_2k.sum(dim=-1)
                sum_exp_sim_crop2q_1k = exp_sim_crop2q_1k.sum(dim=-1)

                for i in range(batch_size):
                    
                    temp_loss_pix_1q_2k = 0
                    temp_loss_pix_2q_1k = 0
                    crop1_label,count1_label = torch.unique(crop1_feat_seg[i],return_counts=True)
                    crop2_label,count2_label = torch.unique(crop2_feat_seg[i],return_counts=True)
                    _, loc1_label = count1_label.topk(min(count1_label.size(0),10))
                    _, loc2_label = count2_label.topk(min(count2_label.size(0),10))
                    crop1_label = crop1_label[loc1_label]
                    crop2_label = crop2_label[loc2_label]
                    cnt1 = crop1_label.size()[0]
                    cnt2 = crop2_label.size()[0]
                    #print(cnt1,cnt2)
                    for j in crop1_label:
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt1 -=1
                            continue                            
                        mask_crop1 = (crop1_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal2:
                            t_exp_sim_crop1q_2k = exp_sim_crop1q_2k[i,:,cal2[j.item()-1]]
                            #print(t_exp_sim_crop1q_2k.size(),cal2[j.item()-1],j.item()-1)
                            log_sim_crop1q_2k = (-torch.log(t_exp_sim_crop1q_2k/sum_exp_sim_crop1q_2k[i])) 
                            #print(t_exp_sim_crop1q_2k.min(),t_exp_sim_crop1q_2k.max(),sum_exp_sim_crop1q_2k[i],mask_crop1.size())
                            temp_loss_pix_1q_2k += (log_sim_crop1q_2k.masked_select(mask_crop1)).mean()
                            #print(temp_loss_pix_1q_2k)
                        else:
                            cnt1 -= 1 
                    
                    for j in crop2_label: 
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt2 -=1
                            continue                            
                        mask_crop2 = (crop2_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal1:
                            t_exp_sim_crop2q_1k = exp_sim_crop2q_1k[i,:,cal1[j.item()-1]]
                            log_sim_crop2q_1k = (-torch.log(t_exp_sim_crop2q_1k/sum_exp_sim_crop2q_1k[i]))
                            temp_loss_pix_2q_1k +=(log_sim_crop2q_1k.masked_select(mask_crop2)).mean()
                        else:
                            cnt2 -=1
                    if cnt1!=0:
                        temp_loss_pix_1q_2k = temp_loss_pix_1q_2k/cnt1
                    if cnt2!=0:
                        temp_loss_pix_2q_1k = temp_loss_pix_2q_1k/cnt2
                    
                    loss_pix_1q_2k+=temp_loss_pix_1q_2k
                    loss_pix_2q_1k+=temp_loss_pix_2q_1k
                    
                loss_pix_1q_2k = loss_pix_1q_2k/batch_size
                loss_pix_2q_1k = loss_pix_2q_1k/batch_size
                loss_pix1 = (loss_pix_1q_2k + loss_pix_2q_1k) / 2 

                #temp_crop1_feat_q = F.normalize(crop1_feat_q, dim=1)  # bx256x49
                #temp_crop2_feat_q = F.normalize(crop2_feat_q, dim=1)
                #temp_crop1_feat_k = F.normalize(crop1_feat_k, dim=1)  # bx256x49
                #temp_crop2_feat_k = F.normalize(crop2_feat_k, dim=1)
                #queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                #queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                #sim_crop2k_2k = (queue_crop2_feat_k @ temp_crop2_feat_k) / 0.04  # bxcx49
                #sim_crop1k_1k = (queue_crop1_feat_k @ temp_crop1_feat_k) / 0.04  # bxcx49
                #sim_crop2k_2k = sim_crop2k_2k.squeeze().transpose(1, 2)  # bx49xc
                #sim_crop1k_1k = sim_crop1k_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                #sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                #sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                #sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                #sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                #sim1 = (temp_crop2_feat_k.transpose(1, 2) @ temp_crop1_feat_q) / self.T  # bx49x49
                #sim2 = (temp_crop1_feat_k.transpose(1, 2) @ temp_crop2_feat_q) / self.T  # bx49x49
                #sim1 = sim1.transpose(1, 2)  # bx49x49
                #sim2 = sim2.transpose(1, 2)  # bx49x49

                #_,loc1 = sim1.topk(1)
                #loc1_repeat = loc1.repeat(1,1,sim_crop2k_2k.size(2))                
                #_,loc2 = sim2.topk(1)
                #loc2_repeat = loc2.repeat(1,1,sim_crop1k_1k.size(2))

                #sim_crop2k_2k_gather = torch.gather(sim_crop2k_2k,1,loc1_repeat)
                #sim_crop1k_1k_gather = torch.gather(sim_crop1k_1k,1,loc2_repeat)

                #soft_sim_crop2k_2k = F.softmax(sim_crop2k_2k_gather,dim=-1).detach()
                #soft_sim_crop1k_1k = F.softmax(sim_crop1k_1k_gather,dim=-1).detach()
                #loss_con_1q_2k = torch.sum(-soft_sim_crop2k_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                #loss_con_2q_1k = torch.sum(-soft_sim_crop1k_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                #loss_con1 = (loss_con_1q_2k + loss_con_2q_1k) / 2    
                
                temp_crop1_feat_q = F.normalize(crop1_feat_q, dim=1)  # bx256x49
                temp_crop2_feat_q = F.normalize(crop2_feat_q, dim=1)
                temp_crop1_feat_t = F.normalize(crop1_feat_t, dim=1)  # bx256x49
                temp_crop2_feat_t = F.normalize(crop2_feat_t, dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_t) / 0.04  # bxcx49
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_t) / 0.04  # bxcx49
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                soft_sim_crop1t_2k = F.softmax(sim_crop1t_2k,dim=-1).detach()
                soft_sim_crop2t_1k = F.softmax(sim_crop2t_1k,dim=-1).detach()
                loss_con_1q_2k = torch.sum(-soft_sim_crop1t_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                loss_con_2q_1k = torch.sum(-soft_sim_crop2t_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                loss_con1 = (loss_con_1q_2k + loss_con_2q_1k) / 2                 
                
                loss_pix_1q_2k = 0
                loss_pix_2q_1k = 0  

                temp_mix_crop1_feat_q = F.normalize(mix_crop1_feat_q,dim=1)#bx256x49
                temp_mix_crop2_feat_q = F.normalize(mix_crop2_feat_q,dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k,dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k,dim=1)#cx256
                    
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_mix_crop1_feat_q) /self.T #bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_mix_crop2_feat_q) /self.T #bxcx49
                    
                sim_crop1q_2k = sim_crop1q_2k.squeeze().transpose(2,1) #bx49xc
                sim_crop2q_1k = sim_crop2q_1k.squeeze().transpose(2,1) #bx49xc
                    
                exp_sim_crop1q_2k = sim_crop1q_2k.exp()
                exp_sim_crop2q_1k = sim_crop2q_1k.exp()
                    
                sum_exp_sim_crop1q_2k = exp_sim_crop1q_2k.sum(dim=-1)
                sum_exp_sim_crop2q_1k = exp_sim_crop2q_1k.sum(dim=-1)   
                
                for i in range(batch_size):
                    temp_loss_pix_1q_2k = 0
                    temp_loss_pix_2q_1k = 0                  
                    mix_crop1_label,mix_count1_label = torch.unique(mix_crop1_feat_seg[i],return_counts=True)
                    mix_crop2_label,mix_count2_label = torch.unique(mix_crop2_feat_seg[i],return_counts=True)
                    _, mix_loc1_label = mix_count1_label.topk(min(mix_count1_label.size(0),10))
                    _, mix_loc2_label = mix_count2_label.topk(min(mix_count2_label.size(0),10))
                    mix_crop1_label = mix_crop1_label[mix_loc1_label]
                    mix_crop2_label = mix_crop2_label[mix_loc2_label]
                    cnt1 = mix_crop1_label.size()[0]
                    cnt2 = mix_crop2_label.size()[0]             
                    #print(cnt1,cnt2)
                    for j in mix_crop1_label:
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt1 -=1
                            continue
                        mask_crop1 = (mix_crop1_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal2:
                            t_exp_sim_crop1q_2k = exp_sim_crop1q_2k[i,:,cal2[j.item()-1]]
                            log_sim_crop1q_2k = (-torch.log(t_exp_sim_crop1q_2k/sum_exp_sim_crop1q_2k[i]))
                            temp_loss_pix_1q_2k +=(log_sim_crop1q_2k.masked_select(mask_crop1)).mean()
                        else:
                            cnt1 -= 1 
                    
                    for j in mix_crop2_label: 
                        if j.item()==256:
                            continue  
                        if j.item()==0:
                            cnt2 -=1
                            continue 
                        mask_crop2 = (mix_crop2_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal1: 
                            t_exp_sim_crop2q_1k = exp_sim_crop2q_1k[i,:,cal1[j.item()-1]]
                            log_sim_crop2q_1k = (-torch.log(t_exp_sim_crop2q_1k/sum_exp_sim_crop2q_1k[i]))
                            temp_loss_pix_2q_1k +=(log_sim_crop2q_1k.masked_select(mask_crop2)).mean()
                        else:
                            cnt2 -=1
                    if cnt1!=0:
                        temp_loss_pix_1q_2k = temp_loss_pix_1q_2k/cnt1
                    if cnt2!=0:
                        temp_loss_pix_2q_1k = temp_loss_pix_2q_1k/cnt2
                    
                    loss_pix_1q_2k+=temp_loss_pix_1q_2k
                    loss_pix_2q_1k+=temp_loss_pix_2q_1k
                    
                loss_pix_1q_2k = loss_pix_1q_2k/batch_size
                loss_pix_2q_1k = loss_pix_2q_1k/batch_size
                #print(loss_pix_1q_2k,loss_pix_2q_1k)
                loss_pix2 = (loss_pix_1q_2k + loss_pix_2q_1k) / 2
                
                #loss_pix = loss_pix1
                loss_pix = (loss_pix1 + loss_pix2)/2  

                #temp_crop1_feat_q = F.normalize(mix_crop1_feat_q, dim=1)  # bx256x49
                #temp_crop2_feat_q = F.normalize(mix_crop2_feat_q, dim=1)
                #temp_crop1_feat_k = F.normalize(mix_crop1_feat_k, dim=1)  # bx256x49
                #temp_crop2_feat_k = F.normalize(mix_crop2_feat_k, dim=1)
                #queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                #queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                #sim_crop2k_2k = (queue_crop2_feat_k @ temp_crop2_feat_k) / 0.04  # bxcx49
                #sim_crop1k_1k = (queue_crop1_feat_k @ temp_crop1_feat_k) / 0.04  # bxcx49
                #sim_crop2k_2k = sim_crop2k_2k.squeeze().transpose(1, 2)  # bx49xc
                #sim_crop1k_1k = sim_crop1k_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                #sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                #sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                #sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                #sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                #sim1 = (temp_crop2_feat_k.transpose(1, 2) @ temp_crop1_feat_q) / self.T  # bx49x49
                #sim2 = (temp_crop1_feat_k.transpose(1, 2) @ temp_crop2_feat_q) / self.T  # bx49x49
                #sim1 = sim1.transpose(1, 2)  # bx49x49
                #sim2 = sim2.transpose(1, 2)  # bx49x49

                #_,loc1 = sim1.topk(1)
                #loc1_repeat = loc1.repeat(1,1,sim_crop2k_2k.size(2))                
                #_,loc2 = sim2.topk(1)
                #loc2_repeat = loc2.repeat(1,1,sim_crop1k_1k.size(2))

                #sim_crop2k_2k_gather = torch.gather(sim_crop2k_2k,1,loc1_repeat)
                #sim_crop1k_1k_gather = torch.gather(sim_crop1k_1k,1,loc2_repeat)

                #soft_sim_crop2k_2k = F.softmax(sim_crop2k_2k_gather,dim=-1).detach()
                #soft_sim_crop1k_1k = F.softmax(sim_crop1k_1k_gather,dim=-1).detach()
                #loss_con_1q_2k = torch.sum(-soft_sim_crop2k_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                #loss_con_2q_1k = torch.sum(-soft_sim_crop1k_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                #loss_con2 = (loss_con_1q_2k + loss_con_2q_1k) / 2 
                
                temp_crop1_feat_q = F.normalize(mix_crop1_feat_q, dim=1)  # bx256x49
                temp_crop2_feat_q = F.normalize(mix_crop2_feat_q, dim=1)
                temp_crop1_feat_t = F.normalize(mix_crop1_feat_t, dim=1)  # bx256x49
                temp_crop2_feat_t = F.normalize(mix_crop2_feat_t, dim=1)

                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_t) / 0.04  # bxcx49
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_t) / 0.04  # bxcx49
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                soft_sim_crop1t_2k = F.softmax(sim_crop1t_2k,dim=-1).detach()
                soft_sim_crop2t_1k = F.softmax(sim_crop2t_1k,dim=-1).detach()
                loss_con_1q_2k = torch.sum(-soft_sim_crop1t_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                loss_con_2q_1k = torch.sum(-soft_sim_crop2t_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)                               
                loss_con2 = (loss_con_1q_2k + loss_con_2q_1k) / 2
                
                loss_con = (loss_con1 + loss_con2)/2 #loss_con1               
                return loss_pix, 0.2*loss_con

            else:
                #not used

                return 0, 0
            ##########################################################
        else:
            # spatial augmentations #
            # define rand horizontal flip augmentaitons
            shape = crop1_seg.size()            
            is_flip_crop1 = rand_true(0.5)
            is_flip_crop2 = rand_true(0.5)
            is_flip_mix_crop1 = rand_true(0.5)
            is_flip_mix_crop2 = rand_true(0.5)
            flip_fn = lambda t: torch.flip(t, dims=(-1,)) # horizentoal flip

            flip_crop1_fn = flip_fn if is_flip_crop1 else identity
            flip_crop2_fn = flip_fn if is_flip_crop2 else identity
            flip_mix_crop1_fn = flip_fn if is_flip_mix_crop1 else identity
            flip_mix_crop2_fn = flip_fn if is_flip_mix_crop2 else identity
        
            # flip mix croped image first to avoid bugs
            mix_crop1 = flip_mix_crop1_fn(mix_crop1)
            mix_crop2 = flip_mix_crop2_fn(mix_crop2)
        
            mix_crop1_sega = flip_mix_crop1_fn(crop1_seg)
            mix_crop2_sega = flip_mix_crop2_fn(crop2_seg)
            mix_crop1_segb = flip_mix_crop1_fn(mix_crop1_seg)
            mix_crop2_segb = flip_mix_crop2_fn(mix_crop2_seg)

            add_mix_crop1 = flip_mix_crop1_fn(add_mix_crop1)
            add_mix_crop2 = flip_mix_crop2_fn(add_mix_crop2)              
            # flip croped image
            crop1 = flip_crop1_fn(crop1)
            crop2 = flip_crop2_fn(crop2)
            crop1_seg = flip_crop1_fn(crop1_seg)
            crop2_seg = flip_crop2_fn(crop2_seg)
            fake_crop1_seg = flip_crop1_fn(fake_crop1_seg)
            fake_crop2_seg = flip_crop2_fn(fake_crop2_seg)

            add_crop1 = flip_crop1_fn(add_crop1)
            add_crop2 = flip_crop2_fn(add_crop2)          
            ##########################################################
       
            # get forward feature #
            crop1_feat_q, crop1_instance_q = self.encoder_q(crop1) # Bx256x7x7
            crop2_feat_q, crop2_instance_q = self.encoder_q(crop2) # Bx256x7x7
            mix_crop1_feat_q, mix_crop1_instance_q = self.encoder_q(mix_crop1) # Bx256x7x7
            mix_crop2_feat_q, mix_crop2_instance_q = self.encoder_q(mix_crop2) # Bx256x7x7
            #crop1_feat_q = F.interpolate(crop1_feat_q,size=(crop1_feat_q.size(2)//2,crop1_feat_q.size(3)//2), mode='bilinear', align_corners=False)
            #crop2_feat_q = F.interpolate(crop2_feat_q,size=(crop2_feat_q.size(2)//2,crop2_feat_q.size(3)//2), mode='bilinear', align_corners=False)      
            with torch.no_grad():
                self._momentum_update_key_encoder()
                #if full:
                crop1_feat_k, crop1_instance_k = self.encoder_k(crop1)
                crop2_feat_k, crop2_instance_k = self.encoder_k(crop2) 
                mix_crop1_feat_k, mix_crop1_instance_k = self.encoder_k(mix_crop1)
                mix_crop2_feat_k, mix_crop2_instance_k = self.encoder_k(mix_crop2)                  
                crop1_feat_t, crop1_instance_t = self.encoder_k(add_crop1)
                crop2_feat_t, crop2_instance_t = self.encoder_k(add_crop2)   
                mix_crop1_feat_t, mix_crop1_instance_t = self.encoder_k(add_mix_crop1)
                mix_crop2_feat_t, mix_crop2_instance_t = self.encoder_k(add_mix_crop2)   

            nfeat_pix = (crop1_feat_q.size(-1))**2
            ##########################################################



            # get postive masks #
            # reshape coord format 
            # sort order: [[r1,c1],[r1,c2],
            #              [r2,c1],[r2,c2]]
            crop1_feat_seg = rearrange(crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            crop2_feat_seg = rearrange(crop2_seg, 'b c h w -> b (h w) c') # Bx49x1
            fake_crop1_feat_seg = rearrange(fake_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            fake_crop2_feat_seg = rearrange(fake_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1 
            low_fake_crop1_feat_seg = rearrange(low_fake_crop1_seg, 'b c h w -> b (h w) c') # Bx49x1
            low_fake_crop2_feat_seg = rearrange(low_fake_crop2_seg, 'b c h w -> b (h w) c') # Bx49x1            
            
            mix_crop1_feat_sega = rearrange(mix_crop1_sega, 'b c h w -> b (h w) c') # Bx49x1
            mix_crop2_feat_sega = rearrange(mix_crop2_sega, 'b c h w -> b (h w) c') # Bx49x1
            mix_crop1_feat_segb = rearrange(mix_crop1_segb, 'b c h w -> b (h w) c') # Bx49x1
            mix_crop2_feat_segb = rearrange(mix_crop2_segb, 'b c h w -> b (h w) c') # Bx49x1
            
            ##########################################################
        
            #positive_num = pos_mask_crop1_2.sum()
            #print(positive_num)

            # compute final pix loss #
            if not self.use_pixpro:
                # pix-contrast loss #
                # crop1_feat_q: Bx256x49
            
                mix_crop1_feat_q, mix_crop2_feat_q, mix_crop1_feat_k, mix_crop2_feat_k, crop1_feat_q, crop2_feat_q, crop1_feat_k, crop2_feat_k = \
                    list(map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), \
                    (mix_crop1_feat_q, mix_crop2_feat_q, mix_crop1_feat_k, mix_crop2_feat_k, crop1_feat_q, crop2_feat_q, crop1_feat_k, crop2_feat_k)))

                mix_crop1_feat_t, mix_crop2_feat_t, crop1_feat_t, crop2_feat_t = \
                    list(map(lambda t: rearrange(t, 'b c h w -> b c (h w)'), \
                    (mix_crop1_feat_t, mix_crop2_feat_t, crop1_feat_t, crop2_feat_t)))
            
                #print(crop2_feat_k.size())
                dim = crop1_feat_q.size(1)
                batch_size=crop1_feat_q.size(0)  
                
                # update
                mean_crop1_feat_k = torch.zeros(batch_size, self.C, dim).cuda()
                mask_crop1 = torch.zeros((batch_size, self.C), dtype=torch.long).cuda()
                mean_crop2_feat_k = torch.zeros(batch_size, self.C, dim).cuda()
                mask_crop2 = torch.zeros((batch_size, self.C), dtype=torch.long).cuda()
                #mean_crop1_instance_k = torch.zeros(batch_size,self.C,dim).cuda()
                #mean_crop2_instance_k = torch.zeros(batch_size,self.C,dim).cuda()   

                fake_mean_crop1_feat_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                #fake_mean_crop1_instance_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                fake_mask_crop1 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                fake_mean_crop2_feat_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                #fake_mean_crop2_instance_k = torch.zeros(batch_size, 5*self.C, dim).cuda()
                fake_mask_crop2 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                low_fake_mask_crop1 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()
                low_fake_mask_crop2 = torch.zeros((batch_size, 5*self.C), dtype=torch.long).cuda()


                for i in range(batch_size):
                    crop1_label = torch.unique(crop1_feat_seg[i])
                    crop2_label = torch.unique(crop2_feat_seg[i])
                    fake_crop1_label = torch.unique(fake_crop1_feat_seg[i])
                    fake_crop2_label = torch.unique(fake_crop2_feat_seg[i])
                    for j in crop1_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (crop1_feat_seg[i].squeeze() == j)  # 49
                        mask_crop1[i][j.item() - 1] = 1
                        mean_crop1_feat_k[i][j.item() - 1] = (crop1_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #mean_crop1_instance_k[i][j.item()-1] = (crop1_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()

                    for j in crop2_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (crop2_feat_seg[i].squeeze() == j)  # 49
                        mask_crop2[i][j.item() - 1] = 1
                        mean_crop2_feat_k[i][j.item() - 1] = (crop2_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #mean_crop2_instance_k[i][j.item()-1] = (crop2_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()    

                    for j in fake_crop1_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (fake_crop1_feat_seg[i].squeeze() == j)  # 49
                        low_temp_mask = (low_fake_crop1_feat_seg[i].squeeze() == j)  # 49
                        fake_mask_crop1[i][j.item() - 1] = temp_mask.float().sum()
                        low_fake_mask_crop1[i][j.item() - 1] = low_temp_mask.float().sum()
                        fake_mean_crop1_feat_k[i][j.item() - 1] = (crop1_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #fake_mean_crop1_instance_k[i][j.item()-1] = (crop1_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()    

                    for j in fake_crop2_label:
                        if j.item() == 256 or j.item() == 0:
                            continue
                        temp_mask = (fake_crop2_feat_seg[i].squeeze() == j)  # 49
                        low_temp_mask = (low_fake_crop2_feat_seg[i].squeeze() == j)  # 49
                        fake_mask_crop2[i][j.item() - 1] = temp_mask.float().sum()
                        low_fake_mask_crop2[i][j.item() - 1] = low_temp_mask.float().sum()
                        fake_mean_crop2_feat_k[i][j.item() - 1] = (crop2_feat_k[i] * temp_mask.float()).sum(
                            -1) / temp_mask.float().sum()
                        #fake_mean_crop2_instance_k[i][j.item()-1] = (crop2_instance_k[i]*temp_mask.float()).sum(-1)/temp_mask.float().sum()            
                
                self._dequeue_and_enqueue(mean_crop1_feat_k, mean_crop2_feat_k, mask_crop1, mask_crop2, fake_mean_crop1_feat_k, fake_mean_crop2_feat_k, fake_mask_crop1, fake_mask_crop2, low_fake_mask_crop1, low_fake_mask_crop2)
                                
                cal1_mean = 0
                cal2_mean = 0
                cal1 = {}
                cal2 = {}
                raw_cal1=[]
                raw_cal2=[]
                for i in range(self.C):
                    if self.mask1[i].sum() != 0:
                        temp_crop1_feat_k = (self.queue1[i]).sum(-1) / self.mask1[i].sum(-1)  # 256xk->256
                        temp_crop1_feat_k = temp_crop1_feat_k.unsqueeze(0)  # 256->1x256
                        #temp_crop1_instance_k = (self.instance_queue1[i]*self.mask1[i]).sum(-1)/self.mask1[i].sum(-1) #256xk->256
                        #temp_crop1_instance_k = temp_crop1_instance_k.unsqueeze(0)#256->1x256                            
                              
                        if cal1_mean == 0:
                            queue_crop1_feat_k = temp_crop1_feat_k
                            #queue_crop1_instance_k = temp_crop1_instance_k
                        else:
                            queue_crop1_feat_k = torch.cat([queue_crop1_feat_k, temp_crop1_feat_k], dim=0)  # cx256
                            #queue_crop1_instance_k = torch.cat([queue_crop1_instance_k, temp_crop1_instance_k],dim=0) #cx256
                        raw_cal1.append(i)
                        cal1[i]=cal1_mean
                        cal1_mean += 1
                    if self.mask2[i].sum() != 0:
                        temp_crop2_feat_k = (self.queue2[i]).sum(-1) / self.mask2[i].sum(-1)  # 256xk->256
                        temp_crop2_feat_k = temp_crop2_feat_k.unsqueeze(0)  # 256->1x256
                        #temp_crop2_instance_k = (self.instance_queue2[i]*self.mask2[i]).sum(-1)/self.mask2[i].sum(-1) #256xk->256
                        #temp_crop2_instance_k = temp_crop2_instance_k.unsqueeze(0)#256->1x256
                        
                        if cal2_mean == 0:
                            queue_crop2_feat_k = temp_crop2_feat_k
                            #queue_crop2_instance_k = temp_crop2_instance_k
                        else:
                            queue_crop2_feat_k = torch.cat([queue_crop2_feat_k, temp_crop2_feat_k], dim=0)  # cx256
                            #queue_crop2_instance_k = torch.cat([queue_crop2_instance_k, temp_crop2_instance_k],dim=0) #cx256
                        raw_cal2.append(i)
                        cal2[i]=cal2_mean
                        cal2_mean += 1
                raw_cal1 = torch.LongTensor(raw_cal1).cuda()
                raw_cal2 = torch.LongTensor(raw_cal2).cuda()                
                #print(cal1_mean,cal2_mean) 

                temp_crop1_feat_f = F.normalize(crop1_feat_k, dim=1)  # bx256x49
                temp_crop2_feat_f = F.normalize(crop2_feat_k, dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256

                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_f) 
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_f) 
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc
                t_m1, t_class1 = sim_crop1t_2k.topk(1, dim=2, sorted=True)  # bx49
                t_m2, t_class2 = sim_crop2t_1k.topk(1, dim=2, sorted=True)

                over_thre1 = (t_m1 > 0.8).long()
                over_thre2 = (t_m2 > 0.8).long()

                crop1_feat_seg_semi = ((raw_cal2[t_class1] + 1)*over_thre1).squeeze().long().unsqueeze(2)
                crop2_feat_seg_semi = ((raw_cal1[t_class2] + 1)*over_thre2).squeeze().long().unsqueeze(2)

                crop1_feat_seg = crop1_feat_seg * crop1_mask.unsqueeze(1) + crop1_feat_seg_semi * (1 - crop1_mask).unsqueeze(1)
                crop2_feat_seg = crop2_feat_seg * crop2_mask.unsqueeze(1) + crop2_feat_seg_semi * (1 - crop2_mask).unsqueeze(1)

                #update mix label
                crop1_seg = rearrange(crop1_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1
                crop2_seg = rearrange(crop2_feat_seg, 'b (h w) c -> b c h w', h=shape[2] , w=shape[3]) # Bx49x1
                #mix_crop1_seg = rearrange(mix_crop1_feat_seg, 'b (h w) c -> b c h w', h=mix_crop1_seg.size(2) , w=mix_crop1_seg.size(3)) # Bx49x1
                #mix_crop2_seg = rearrange(mix_crop2_feat_seg, 'b (h w) c -> b c h w', h=mix_crop2_seg.size(2) , w=mix_crop2_seg.size(3)) # Bx49x1    
                crop1_seg = flip_crop1_fn(crop1_seg)
                crop2_seg = flip_crop2_fn(crop2_seg) 
                #mix_crop1_seg = flip_mix_crop1_fn(mix_crop1_seg)
                #mix_crop2_seg = flip_mix_crop2_fn(mix_crop2_seg)

                mix_crop1_sega = crop1_seg
                mix_crop2_sega = crop2_seg


                mix_crop1_sega = flip_mix_crop1_fn(mix_crop1_sega)
                mix_crop2_sega = flip_mix_crop2_fn(mix_crop2_sega)
                mix_crop1_feat_sega = rearrange(mix_crop1_sega, 'b c h w -> b (h w) c') # Bx49x1
                mix_crop2_feat_sega = rearrange(mix_crop2_sega, 'b c h w -> b (h w) c') # Bx49x1

                loss_pix_1q_2k = 0
                loss_pix_2q_1k = 0    
                #t_dim_1 = queue_crop2_feat_k.size(0)
                #t_dim_2 = queue_crop1_feat_k.size(0)
                #print(int(t_dim_1*self.ln),int(t_dim_2*self.ln))

                temp_crop1_feat_q = F.normalize(crop1_feat_q,dim=1)#256x49
                temp_crop2_feat_q = F.normalize(crop2_feat_q,dim=1)
                    
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) /self.T #cx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) /self.T #cx49
                    
                sim_crop1q_2k = sim_crop1q_2k.squeeze().transpose(2,1) #49xc
                sim_crop2q_1k = sim_crop2q_1k.squeeze().transpose(2,1) #49xc
                    
                exp_sim_crop1q_2k = sim_crop1q_2k.exp()
                exp_sim_crop2q_1k = sim_crop2q_1k.exp()

                sum_exp_sim_crop1q_2k = exp_sim_crop1q_2k.sum(dim=-1)
                sum_exp_sim_crop2q_1k = exp_sim_crop2q_1k.sum(dim=-1)

                for i in range(batch_size):                    
                    temp_loss_pix_1q_2k = 0
                    temp_loss_pix_2q_1k = 0
                    crop1_label,count1_label = torch.unique(crop1_feat_seg[i],return_counts=True)
                    crop2_label,count2_label = torch.unique(crop2_feat_seg[i],return_counts=True)
                    _, loc1_label = count1_label.topk(min(count1_label.size(0),10))
                    _, loc2_label = count2_label.topk(min(count2_label.size(0),10))
                    crop1_label = crop1_label[loc1_label]
                    crop2_label = crop2_label[loc2_label]
                    cnt1 = crop1_label.size()[0]
                    cnt2 = crop2_label.size()[0]
                    for j in crop1_label:
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt1 -=1
                            continue                            
                        mask_crop1 = (crop1_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal2:
                            t_exp_sim_crop1q_2k = exp_sim_crop1q_2k[i,:,cal2[j.item()-1]]
                            log_sim_crop1q_2k = (-torch.log(t_exp_sim_crop1q_2k/sum_exp_sim_crop1q_2k[i]))
                            temp_loss_pix_1q_2k +=(log_sim_crop1q_2k.masked_select(mask_crop1)).mean()

                        else:
                            cnt1 -= 1 
                    
                    for j in crop2_label: 
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt2 -=1
                            continue                            
                        mask_crop2 = (crop2_feat_seg[i].squeeze()==j)#49
                        if j.item()-1 in cal1: 
                            t_exp_sim_crop2q_1k = exp_sim_crop2q_1k[i,:,cal1[j.item()-1]]
                            log_sim_crop2q_1k = (-torch.log(t_exp_sim_crop2q_1k/sum_exp_sim_crop2q_1k[i]))
                            temp_loss_pix_2q_1k +=(log_sim_crop2q_1k.masked_select(mask_crop2)).mean()
                        else:
                            cnt2 -=1
                    if cnt1!=0:
                        temp_loss_pix_1q_2k = temp_loss_pix_1q_2k/cnt1
                    if cnt2!=0:
                        temp_loss_pix_2q_1k = temp_loss_pix_2q_1k/cnt2
                    
                    loss_pix_1q_2k+=temp_loss_pix_1q_2k
                    loss_pix_2q_1k+=temp_loss_pix_2q_1k

                loss_pix_1q_2k = loss_pix_1q_2k/batch_size
                loss_pix_2q_1k = loss_pix_2q_1k/batch_size
                #print(loss_pix_1q_2k,loss_pix_2q_1k)
                loss_pix1 = (loss_pix_1q_2k + loss_pix_2q_1k) / 2 

                #temp_crop1_feat_q = F.normalize(crop1_feat_q, dim=1)  # bx256x49
                #temp_crop2_feat_q = F.normalize(crop2_feat_q, dim=1)
                #temp_crop1_feat_k = F.normalize(crop1_feat_k, dim=1)  # bx256x49
                #temp_crop2_feat_k = F.normalize(crop2_feat_k, dim=1)
                #queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                #queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                #sim_crop2k_2k = (queue_crop2_feat_k @ temp_crop2_feat_k) / 0.04  # bxcx49
                #sim_crop1k_1k = (queue_crop1_feat_k @ temp_crop1_feat_k) / 0.04  # bxcx49
                #sim_crop2k_2k = sim_crop2k_2k.squeeze().transpose(1, 2)  # bx49xc
                #sim_crop1k_1k = sim_crop1k_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                #sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                #sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                #sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                #sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                #sim1 = (temp_crop2_feat_k.transpose(1, 2) @ temp_crop1_feat_q) / self.T  # bx49x49
                #sim2 = (temp_crop1_feat_k.transpose(1, 2) @ temp_crop2_feat_q) / self.T  # bx49x49
                #sim1 = sim1.transpose(1, 2)  # bx49x49
                #sim2 = sim2.transpose(1, 2)  # bx49x49

                #_,loc1 = sim1.topk(1)
                #loc1_repeat = loc1.repeat(1,1,sim_crop2k_2k.size(2))                
                #_,loc2 = sim2.topk(1)
                #loc2_repeat = loc2.repeat(1,1,sim_crop1k_1k.size(2))

                #sim_crop2k_2k_gather = torch.gather(sim_crop2k_2k,1,loc1_repeat)
                #sim_crop1k_1k_gather = torch.gather(sim_crop1k_1k,1,loc2_repeat)

                #soft_sim_crop2k_2k = F.softmax(sim_crop2k_2k_gather,dim=-1).detach()
                #soft_sim_crop1k_1k = F.softmax(sim_crop1k_1k_gather,dim=-1).detach()
                #loss_con_1q_2k = torch.sum(-soft_sim_crop2k_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                #loss_con_2q_1k = torch.sum(-soft_sim_crop1k_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                #loss_con1 = (loss_con_1q_2k + loss_con_2q_1k) / 2  
                
                temp_crop1_feat_q = F.normalize(crop1_feat_q, dim=1)  # bx256x49
                temp_crop2_feat_q = F.normalize(crop2_feat_q, dim=1)
                temp_crop1_feat_t = F.normalize(crop1_feat_t, dim=1)  # bx256x49
                temp_crop2_feat_t = F.normalize(crop2_feat_t, dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_t) / 0.04  # bxcx49
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_t) / 0.04  # bxcx49
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                soft_sim_crop1t_2k = F.softmax(sim_crop1t_2k,dim=-1).detach()
                soft_sim_crop2t_1k = F.softmax(sim_crop2t_1k,dim=-1).detach()
                loss_con_1q_2k = torch.sum(-soft_sim_crop1t_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                loss_con_2q_1k = torch.sum(-soft_sim_crop2t_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                loss_con1 = (loss_con_1q_2k + loss_con_2q_1k) / 2 

                loss_pix_1q_2k = 0
                loss_pix_2q_1k = 0
                #t_dim_1 = queue_crop2_feat_k.size(0)
                #t_dim_2 = queue_crop1_feat_k.size(0)

                temp_mix_crop1_feat_q = F.normalize(mix_crop1_feat_q,dim=1)#256x49
                temp_mix_crop2_feat_q = F.normalize(mix_crop2_feat_q,dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k,dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k,dim=1)#cx256
                    
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_mix_crop1_feat_q)/self.T #cx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_mix_crop2_feat_q)/self.T #cx49
                    
                sim_crop1q_2k = sim_crop1q_2k.squeeze().transpose(2,1) #49xc
                sim_crop2q_1k = sim_crop2q_1k.squeeze().transpose(2,1) #49xc
                    
                exp_sim_crop1q_2k = sim_crop1q_2k.exp()
                exp_sim_crop2q_1k = sim_crop2q_1k.exp()
                                        
                sum_exp_sim_crop1q_2k = exp_sim_crop1q_2k.sum(dim=-1)
                sum_exp_sim_crop2q_1k = exp_sim_crop2q_1k.sum(dim=-1) 

                for i in range(batch_size):                   
                    temp_loss_pix_1q_2k = 0
                    temp_loss_pix_2q_1k = 0
                    temp_loss_pix_1q_2k_semi = 0
                    temp_loss_pix_2q_1k_semi = 0
                    mix_crop1_labela,mix_count1_labela = torch.unique(mix_crop1_feat_sega[i],return_counts=True)
                    mix_crop2_labela,mix_count2_labela = torch.unique(mix_crop2_feat_sega[i],return_counts=True)
                    _, mix_loc1_labela = mix_count1_labela.topk(min(mix_count1_labela.size(0),10))
                    _, mix_loc2_labela = mix_count2_labela.topk(min(mix_count2_labela.size(0),10))
                    mix_crop1_labela = mix_crop1_labela[mix_loc1_labela]
                    mix_crop2_labela = mix_crop2_labela[mix_loc2_labela]
                    cnt1 = mix_crop1_labela.size()[0]
                    cnt2 = mix_crop2_labela.size()[0]             
                    for j in mix_crop1_labela:
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt1 -=1
                            continue                                  
                        mask_crop1 = (mix_crop1_feat_sega[i].squeeze()==j)#49
                        if j.item()-1 in cal2:
                            t_exp_sim_crop1q_2k = exp_sim_crop1q_2k[i,:,cal2[j.item()-1]]
                            log_sim_crop1q_2k = (-torch.log(t_exp_sim_crop1q_2k/sum_exp_sim_crop1q_2k[i]))
                            temp_loss_pix_1q_2k +=lam1*(log_sim_crop1q_2k.masked_select(mask_crop1)).mean()
                        else:
                            cnt1 -= 1 
                        
                    for j in mix_crop2_labela: 
                        if j.item()==256:
                            continue
                        if j.item()==0:
                            cnt2 -=1
                            continue                                
                        mask_crop2 = (mix_crop2_feat_sega[i].squeeze()==j)#49
                        if j.item()-1 in cal1: 
                            t_exp_sim_crop2q_1k = exp_sim_crop2q_1k[i,:,cal1[j.item()-1]]
                            log_sim_crop2q_1k = (-torch.log(t_exp_sim_crop2q_1k/sum_exp_sim_crop2q_1k[i]))
                            temp_loss_pix_2q_1k +=lam2*(log_sim_crop2q_1k.masked_select(mask_crop2)).mean()
                        else:
                            cnt2 -=1
                    if cnt1!=0:
                        temp_loss_pix_1q_2k = temp_loss_pix_1q_2k/cnt1
                    if cnt2!=0:
                        temp_loss_pix_2q_1k = temp_loss_pix_2q_1k/cnt2
                        
                    loss_pix_1q_2k+=temp_loss_pix_1q_2k
                    loss_pix_2q_1k+=temp_loss_pix_2q_1k
 
                    temp_loss_pix_1q_2k = 0
                    temp_loss_pix_2q_1k = 0
                    mix_crop1_labelb,mix_count1_labelb = torch.unique(mix_crop1_feat_segb[i],return_counts=True)
                    mix_crop2_labelb,mix_count2_labelb = torch.unique(mix_crop2_feat_segb[i],return_counts=True)
                    _, mix_loc1_labelb = mix_count1_labelb.topk(min(mix_count1_labelb.size(0),10))
                    _, mix_loc2_labelb = mix_count2_labelb.topk(min(mix_count2_labelb.size(0),10))
                    mix_crop1_labelb = mix_crop1_labelb[mix_loc1_labelb]
                    mix_crop2_labelb = mix_crop2_labelb[mix_loc2_labelb]
                    cnt1 = mix_crop1_labelb.size()[0]
                    cnt2 = mix_crop2_labelb.size()[0]              
                
                    for j in mix_crop1_labelb:
                        if j.item()==256:
                            continue
                        mask_crop1 = (mix_crop1_feat_segb[i].squeeze()==j)#49
                        if j.item()-1 in cal2:
                            t_exp_sim_crop1q_2k = exp_sim_crop1q_2k[i,:,cal2[j.item()-1]]
                            log_sim_crop1q_2k = (-torch.log(t_exp_sim_crop1q_2k/sum_exp_sim_crop1q_2k[i]))
                            temp_loss_pix_1q_2k += (1-lam1) * (log_sim_crop1q_2k.masked_select(mask_crop1)).mean()
                        else:
                            #print(j.item())
                            cnt1 -= 1 
                    
                    for j in mix_crop2_labelb: 
                        if j.item()==256:
                            continue                  
                        mask_crop2 = (mix_crop2_feat_segb[i].squeeze()==j)#49

                        if j.item()-1 in cal1: 
                            t_exp_sim_crop2q_1k = exp_sim_crop2q_1k[i,:,cal1[j.item()-1]]
                            log_sim_crop2q_1k = (-torch.log(t_exp_sim_crop2q_1k/sum_exp_sim_crop2q_1k[i]))
                            temp_loss_pix_2q_1k += (1-lam2) * (log_sim_crop2q_1k.masked_select(mask_crop2)).mean()
                        else:
                            #print(j.item())
                            cnt2 -=1
                    if cnt1!=0:
                        temp_loss_pix_1q_2k = temp_loss_pix_1q_2k/cnt1
                    if cnt2!=0:
                        temp_loss_pix_2q_1k = temp_loss_pix_2q_1k/cnt2                    
                    
                    loss_pix_1q_2k+=temp_loss_pix_1q_2k
                    loss_pix_2q_1k+=temp_loss_pix_2q_1k
                        
                loss_pix_1q_2k = loss_pix_1q_2k/batch_size
                loss_pix_2q_1k = loss_pix_2q_1k/batch_size
                #print(loss_pix_1q_2k,loss_pix_2q_1k)
                loss_pix2 = (loss_pix_1q_2k + loss_pix_2q_1k) / 2 
                
                loss_pix = (loss_pix1 + loss_pix2)/2
                
                #loss_pix = loss_pix1
                
                #temp_crop1_feat_q = F.normalize(mix_crop1_feat_q, dim=1)  # bx256x49
                #temp_crop2_feat_q = F.normalize(mix_crop2_feat_q, dim=1)
                #temp_crop1_feat_k = F.normalize(mix_crop1_feat_k, dim=1)  # bx256x49
                #temp_crop2_feat_k = F.normalize(mix_crop2_feat_k, dim=1)
                #queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                #queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256
                
                #sim_crop2k_2k = (queue_crop2_feat_k @ temp_crop2_feat_k) / 0.04  # bxcx49
                #sim_crop1k_1k = (queue_crop1_feat_k @ temp_crop1_feat_k) / 0.04  # bxcx49
                #sim_crop2k_2k = sim_crop2k_2k.squeeze().transpose(1, 2)  # bx49xc
                #sim_crop1k_1k = sim_crop1k_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                #sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                #sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                #sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                #sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                #sim1 = (temp_crop2_feat_k.transpose(1, 2) @ temp_crop1_feat_q) / self.T  # bx49x49
                #sim2 = (temp_crop1_feat_k.transpose(1, 2) @ temp_crop2_feat_q) / self.T  # bx49x49
                #sim1 = sim1.transpose(1, 2)  # bx49x49
                #sim2 = sim2.transpose(1, 2)  # bx49x49

                #_,loc1 = sim1.topk(1)
                #loc1_repeat = loc1.repeat(1,1,sim_crop2k_2k.size(2))                
                #_,loc2 = sim2.topk(1)
                #loc2_repeat = loc2.repeat(1,1,sim_crop1k_1k.size(2))

                #sim_crop2k_2k_gather = torch.gather(sim_crop2k_2k,1,loc1_repeat)
                #sim_crop1k_1k_gather = torch.gather(sim_crop1k_1k,1,loc2_repeat)

                #soft_sim_crop2k_2k = F.softmax(sim_crop2k_2k_gather,dim=-1).detach()
                #soft_sim_crop1k_1k = F.softmax(sim_crop1k_1k_gather,dim=-1).detach()
                #loss_con_1q_2k = torch.sum(-soft_sim_crop2k_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                #loss_con_2q_1k = torch.sum(-soft_sim_crop1k_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)               
                #loss_con2 = (loss_con_1q_2k + loss_con_2q_1k) / 2 
                
                temp_crop1_feat_q = F.normalize(mix_crop1_feat_q, dim=1)  # bx256x49
                temp_crop2_feat_q = F.normalize(mix_crop2_feat_q, dim=1)
                temp_crop1_feat_t = F.normalize(mix_crop1_feat_t, dim=1)  # bx256x49
                temp_crop2_feat_t = F.normalize(mix_crop2_feat_t, dim=1)
                queue_crop1_feat_k = F.normalize(queue_crop1_feat_k, dim=1)
                queue_crop2_feat_k = F.normalize(queue_crop2_feat_k, dim=1)  # cx256

                sim_crop1t_2k = (queue_crop2_feat_k @ temp_crop1_feat_t) / 0.04  # bxcx49
                sim_crop2t_1k = (queue_crop1_feat_k @ temp_crop2_feat_t) / 0.04  # bxcx49
                sim_crop1t_2k = sim_crop1t_2k.squeeze().transpose(1, 2)  # bx49xc
                sim_crop2t_1k = sim_crop2t_1k.squeeze().transpose(1, 2)  # bx49xc  
            
                sim_crop1q_2k = (queue_crop2_feat_k @ temp_crop1_feat_q) / self.T  # bxcx49
                sim_crop2q_1k = (queue_crop1_feat_k @ temp_crop2_feat_q) / self.T  # bxcx49
                sim_crop1q_2k = sim_crop1q_2k.transpose(1, 2)  # bx49xc
                sim_crop2q_1k = sim_crop2q_1k.transpose(1, 2)  # bx49xc

                soft_sim_crop1t_2k = F.softmax(sim_crop1t_2k,dim=-1).detach()
                soft_sim_crop2t_1k = F.softmax(sim_crop2t_1k,dim=-1).detach()
                loss_con_1q_2k = torch.sum(-soft_sim_crop1t_2k * F.log_softmax(sim_crop1q_2k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1)
                loss_con_2q_1k = torch.sum(-soft_sim_crop2t_1k * F.log_softmax(sim_crop2q_1k,dim=-1), dim=-1).mean(dim=-1).mean(dim=-1) 
                loss_con2 = (loss_con_1q_2k + loss_con_2q_1k) / 2
                
                loss_con = (loss_con1 + loss_con2)/2

                return loss_pix, 0.2*loss_con


            else:
                # not used
                return 0, 0
            ##########################################################        

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


if __name__ == '__main__':
    sys.path.append('..')
    from arch.modeling import *
    model = SegCL(deeplabv3plus_resnet50, use_pixpro=False).cuda()
    img1 = img2 = torch.randn(4,3,224,224).cuda()
    coord1 = coord2 = torch.ones((4,1,56,56),dtype=torch.long).cuda()
    out = model(img1, img2, coord1, coord2)
    print(out)
