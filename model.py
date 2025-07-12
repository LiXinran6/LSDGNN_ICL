import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np, itertools, random, copy, math
from transformers import BertModel, BertConfig
from transformers import AutoTokenizer, AutoModelWithLMHead
from model_utils import *


class BertERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)
        # bert_encoder
        self.bert_config = BertConfig.from_json_file(args.bert_model_dir + 'config.json')

        self.bert = BertModel.from_pretrained(args.home_dir + args.bert_model_dir, config = self.bert_config)
        in_dim =  args.bert_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers- 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, content_ids, token_types,utterance_len,seq_len):

        # the embeddings for bert
        # if len(content_ids)>512:
        #     print('ll')

        #
        ## w token_type_ids
        # lastHidden = self.bert(content_ids, token_type_ids = token_types)[1] #(N , D)
        ## w/t token_type_ids
        lastHidden = self.bert(content_ids)[1] #(N , D)

        final_feature = self.dropout(lastHidden)

        # pooling

        outputs = self.out_mlp(final_feature) #(N, D)

        return outputs


class DAGERC(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_emb = nn.Embedding(2,args.hidden_dim)
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        else:
            gats = []
            for _ in range(args.gnn_layers):
                gats += [Gatdot(args.hidden_dim) if args.no_rel_attn else Gatdot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus = []
        for _ in range(args.gnn_layers):
            grus += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus = nn.ModuleList(grus)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

    def forward(self, features, adj,s_mask):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :return:
        '''
        num_utter = features.size()[1]
        if self.rel_attn:
            rel_ft = self.rel_emb(s_mask) # (B, N, N, D)

        H0 = F.relu(self.fc1(features)) # (B, N, D)
        H = [H0]
        for l in range(self.args.gnn_layers):
            H1 = self.grus[l](H[l][:,0,:]).unsqueeze(1) # (B, 1, D)
            for i in range(1, num_utter):
                if not self.rel_attn:
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                else:
                    _, M = self.gather[l](H[l][:, i, :], H1, H1, adj[:, i, :i], rel_ft[:, i, :i, :])
                H1 = torch.cat((H1 , self.grus[l](H[l][:,i,:], M).unsqueeze(1)), dim = 1)
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
            H0 = H1
        H.append(features)
        H = torch.cat(H, dim = 2) #(B, N, l*D)
        logits = self.out_mlp(H)
        return logits



class DAGERC_fushion(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats = []
            for _ in range(args.gnn_layers):
                # gats += [GAT_dialoggcn(args.hidden_dim)]
                gats += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)

        grus_c = []
        for _ in range(args.gnn_layers):
            grus_c += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c = nn.ModuleList(grus_c)

        grus_p = []
        for _ in range(args.gnn_layers):
            grus_p += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p = nn.ModuleList(grus_p)
        
        fcs = []
        for _ in range(args.gnn_layers):
            fcs += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs = nn.ModuleList(fcs)

        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)

        self.nodal_att_type = args.nodal_att_type
        
        in_dim = args.hidden_dim * (args.gnn_layers + 1) + args.emb_dim

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features, adj,s_mask,s_mask_onehot, lengths):
        '''
        :param features: (B, N, D)
        :param adj: (B, N, N)
        :param s_mask: (B, N, N)
        :param s_mask_onehot: (B, N, N, 2)
        :return:
        '''
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        # H0 = self.dropout(H0)
        H = [H0]
        for l in range(self.args.gnn_layers):
            C = self.grus_c[l](H[l][:,0,:]).unsqueeze(1) 
            M = torch.zeros_like(C).squeeze(1) 
            # P = M.unsqueeze(1) 
            P = self.grus_p[l](M, H[l][:,0,:]).unsqueeze(1)  
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i])
                    else:
                        _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask[:, i, :i])

                C = self.grus_c[l](H[l][:,i,:], M).unsqueeze(1)
                P = self.grus_p[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P
                H1 = torch.cat((H1 , H_temp), dim = 1)  
                # print('H1', H1.size())
                # print('----------------------------------------------------')
            H.append(H1)
        H.append(features)
        
        H = torch.cat(H, dim = 2) 

        H = self.attentive_node_features(H,lengths,self.nodal_att_type) 

        logits = self.out_mlp(H)

        return logits


#仅仅使用最后一层的short和long，concat；只用过去特征
#Only use the final layer's short and long features, concatenated; use only past features.
class DAGERC_new_1(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            #短距离
            gats_short = []
            gats_long = []
            for _ in range(args.gnn_layers):
                gats_short += [GAT_dialoggcn_v1(args.hidden_dim)]
            for _ in range(args.gnn_layers):
                gats_long += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather_short = nn.ModuleList(gats_short)
            self.gather_long = nn.ModuleList(gats_long)

        # 近距离 GRU
        grus_c_short = []
        for _ in range(args.gnn_layers):
            grus_c_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_short = nn.ModuleList(grus_c_short)

        # 远距离 GRU
        grus_c_long = []
        for _ in range(args.gnn_layers):
            grus_c_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_long = nn.ModuleList(grus_c_long)

        grus_p_short = []
        for _ in range(args.gnn_layers):
            grus_p_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_short = nn.ModuleList(grus_p_short)

        grus_p_long = []
        for _ in range(args.gnn_layers):
            grus_p_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_long = nn.ModuleList(grus_p_long)        
        
        #近距离全链接层
        fcs_short = []
        for _ in range(args.gnn_layers):
            fcs_short += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_short = nn.ModuleList(fcs_short)

        # 远距离全连接层
        fcs_long = []
        for _ in range(args.gnn_layers):
            fcs_long += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_long = nn.ModuleList(fcs_long)


        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        
        self.nodal_att_type = args.nodal_att_type
        
        in_dim = ((args.hidden_dim*2)+ args.emb_dim) 
   #     print(in_dim)
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

        self.affine1 = nn.Parameter(torch.empty(size=((args.hidden_dim)  , (args.hidden_dim)  )))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=((args.hidden_dim)  , (args.hidden_dim) )))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)
        self.beta = args.diffloss

    def forward(self, features, adj_1, adj_2 ,s_mask, s_mask_onehot, lengths):
        # 检查 H1 和 H2 是否完全相等
        are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(adj_1, adj_2))
      #  print("adj1 和 adj2 是否完全相等:", are_equal)
      #  print('adj1',adj_1)
      #  print('----------------------------------------------------')

     #  print('adj2',adj_2)
     #   print('----------------------------------------------------')
    
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        #print('H0', H0.size())
        # H0 = self.dropout(H0)
        H = [H0]
        H_combined_short_list = []
        #对短距离特征进行处理
        for l in range(self.args.gnn_layers):
            C = self.grus_c_short[l](H[l][:,0,:]).unsqueeze(1) #针对每一层的第一个节点，使用 GRU 单元更新节点特征并聚合信息。
            M = torch.zeros_like(C).squeeze(1) #初始化一个聚合信息张量 M（全零张量），并使用它与节点特征结合生成额外的特征 P。
            # P = M.unsqueeze(1) 
            P = self.grus_p_short[l](M, H[l][:,0,:]).unsqueeze(1)  #使用 M（全零张量）和第一个节点的特征 H[l][:, 0, :] 作为输入，得到额外特征 P，形状为 (B, D)
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P#将更新后的特征 C 与额外特征 P 相加，生成新的节点特征 H1，为后续层的计算做准备。
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    #将 H[l][:, i, :]（当前节点特征）,H1（之前节点的特征聚合结果）,adj[:, i, :i]（当前节点与之前节点的邻接矩阵）
                    #s_mask[:, i, :i]（当前节点的掩码）,得到聚合结果 M
                    _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i])
                    else:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:, i, :i])

                #使用 GRU 单元 self.grus_c[l] 来处理当前节点的特征 H[l][:, i, :] 和聚合后的特征 M，得到新的特征 C。
                # 这表明当前节点的特征更新与其邻居的聚合信息有关。
                C = self.grus_c_short[l](H[l][:,i,:], M).unsqueeze(1)
                #使用另一个 GRU 单元 self.grus_p[l] 来处理聚合特征 M 和当前节点的特征 H[l][:, i, :]，得到额外的特征 P。
                P = self.grus_p_short[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P#将更新后的特征 C 和额外特征 P 进行相加，生成新的节点特征 H_temp
                H1 = torch.cat((H1 , H_temp), dim = 1)  #将当前节点的特征 H_temp 拼接到 H1 中。
               # print('H1', H1.size())
                #print('----------------------------------------------------')
            H.append(H1)
            H_combined_short_list.append(H[l+1])
        '''
        下面对长距离特征进行处理    The following processes the long-distance features.
        '''
        H_long = [H0]  # 初始化 H_long
        H_combined_long_list = []  # 存储长距离处理的结果

        # 对长距离特征进行处理
        for l in range(self.args.gnn_layers):
            C_long = self.grus_c_long[l](H_long[l][:,0,:]).unsqueeze(1)  # 使用 GRU 更新长距离的第一个节点
            M_long = torch.zeros_like(C_long).squeeze(1)  # 初始化长距离的聚合信息张量 M_long
            P_long = self.grus_p_long[l](M_long, H_long[l][:,0,:]).unsqueeze(1)  # 生成额外的特征 P_long
            
            H1_long = C_long + P_long  # 生成新的长距离节点特征 H1_long
            for i in range(1, num_utter):
                # 依据不同的 attention 类型，进行特征聚合
                if self.args.attn_type == 'rgcn':
                    _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])
                else:
                    if not self.rel_attn:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i])
                    else:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])

                # 使用 GRU 更新当前节点的特征 C_long 和 M_long
                C_long = self.grus_c_long[l](H_long[l][:,i,:], M_long).unsqueeze(1)
                P_long = self.grus_p_long[l](M_long, H_long[l][:,i,:]).unsqueeze(1)   
                
                H_temp_long = C_long + P_long  # 将更新后的特征 C_long 和 P_long 相加生成新特征
                H1_long = torch.cat((H1_long, H_temp_long), dim=1)  # 将特征拼接到 H1_long 中
            H_long.append(H1_long)  # 更新 H_long 列表
            H_combined_long_list.append(H_long[l+1])
     
        '''
        两个通道特征都提取完毕！    Both short- and long-distance channel features have been extracted!
        '''    
      #  print('H_combined_short_list',H_combined_short_list)
       # print('H_combined_long_list',H_combined_long_list)
      #  are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(H_combined_short_list, H_combined_long_list))
      #  print("H_combined_short_list 和  H_combined_long_list 是否完全相等:", are_equal)
     #   for idx, tensor in enumerate(H_combined_short_list):
      #      print(f"H_combined_short_list[{idx}] shape: {tensor.shape}")
        H_final = []
      #  print("H2 shape:", H2.shape)
       # 计算差异正则化损失
        diff_loss = 0
        for l in range(self.args.gnn_layers):
        #    print('周期：', l)
            HShort_prime = H_combined_short_list[l]
            HLong_prime = H_combined_long_list[l]
         #   print("HShort_prime:", HShort_prime)
        #    print("HLong_prime:", HLong_prime)
        #    print("HShort_prime shape:", HShort_prime.shape)
        #    print("HLong_prime shape:", HLong_prime.shape)
            diff_loss = self.diff_loss(HShort_prime, HLong_prime) + diff_loss
           # print("diff_loss:", diff_loss)
          #  print(diff_loss.item())
            # 互交叉注意力机制
            A1 = F.softmax(torch.bmm(torch.matmul(HShort_prime, self.affine1), torch.transpose(HLong_prime, 1, 2)), dim=2)
            A2 = F.softmax(torch.bmm(torch.matmul(HLong_prime, self.affine2), torch.transpose(HShort_prime, 1, 2)), dim=2)

            HShort_prime_new = torch.bmm(A1, HLong_prime)  # 更新的短时特征
            HLong_prime_new = torch.bmm(A2, HShort_prime)    # 更新的长时特征

            HShort_prime_out = self.dropout(HShort_prime_new) if l < self.args.gnn_layers - 1 else HShort_prime_new
            HLong_prime_out = self.dropout(HLong_prime_new) if l <self.args.gnn_layers - 1 else HLong_prime_new

            H_final.append(HShort_prime_out)
            H_final.append(HLong_prime_out)
        H_final.append(features)

        H_final = torch.cat([H_final[-3],H_final[-2],H_final[-1]], dim = 2)
     #   print("H shape:", H.shape)
       # print("H:", H.shape)
     #   print("H_final shape after cat:", H_final.shape)
        H_final = self.attentive_node_features(H_final,lengths,self.nodal_att_type) 
     #   print("H_final shape after attentive_node_features:", H_final.shape)
        logits = self.out_mlp(H_final)
     #  print(diff_loss)
        return logits, self.beta * diff_loss


#仅仅使用最后一层的short和long，concat；使用了过去和未来双特征
#Only the final-layer short and long features are used and concatenated; both past and future features are utilized.
class DAGERC_new_2(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            #短距离
            gats_short = []
            gats_long = []
            for _ in range(args.gnn_layers):
                gats_short += [GAT_dialoggcn_v1(args.hidden_dim)]
            for _ in range(args.gnn_layers):
                gats_long += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather_short = nn.ModuleList(gats_short)
            self.gather_long = nn.ModuleList(gats_long)

        # 近距离 GRU
        grus_c_short = []
        for _ in range(args.gnn_layers):
            grus_c_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_short = nn.ModuleList(grus_c_short)

        # 远距离 GRU
        grus_c_long = []
        for _ in range(args.gnn_layers):
            grus_c_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_long = nn.ModuleList(grus_c_long)

        grus_p_short = []
        for _ in range(args.gnn_layers):
            grus_p_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_short = nn.ModuleList(grus_p_short)

        grus_p_long = []
        for _ in range(args.gnn_layers):
            grus_p_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_long = nn.ModuleList(grus_p_long)        
        
        #近距离全链接层
        fcs_short = []
        for _ in range(args.gnn_layers):
            fcs_short += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_short = nn.ModuleList(fcs_short)

        # 远距离全连接层
        fcs_long = []
        for _ in range(args.gnn_layers):
            fcs_long += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_long = nn.ModuleList(fcs_long)


        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        
        self.nodal_att_type = args.nodal_att_type
        
        in_dim = ((args.hidden_dim*2)*2 + args.emb_dim) 
   #     print(in_dim)
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

        self.affine1 = nn.Parameter(torch.empty(size=((args.hidden_dim*2)  , (args.hidden_dim*2)  )))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=((args.hidden_dim*2)  , (args.hidden_dim*2) )))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)
        self.beta = args.diffloss

    def forward(self, features, adj_1, adj_2 ,s_mask, s_mask_onehot, lengths):
        # 检查 H1 和 H2 是否完全相等
        are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(adj_1, adj_2))
      #  print("adj1 和 adj2 是否完全相等:", are_equal)
      #  print('adj1',adj_1)
      #  print('----------------------------------------------------')

     #  print('adj2',adj_2)
     #   print('----------------------------------------------------')
    
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        #print('H0', H0.size())
        # H0 = self.dropout(H0)
        H = [H0]
        H_combined_short_list = []
        #对短距离特征进行处理
        for l in range(self.args.gnn_layers):
            C = self.grus_c_short[l](H[l][:,0,:]).unsqueeze(1) #针对每一层的第一个节点，使用 GRU 单元更新节点特征并聚合信息。
            M = torch.zeros_like(C).squeeze(1) #初始化一个聚合信息张量 M（全零张量），并使用它与节点特征结合生成额外的特征 P。
            # P = M.unsqueeze(1) 
            P = self.grus_p_short[l](M, H[l][:,0,:]).unsqueeze(1)  #使用 M（全零张量）和第一个节点的特征 H[l][:, 0, :] 作为输入，得到额外特征 P，形状为 (B, D)
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P#将更新后的特征 C 与额外特征 P 相加，生成新的节点特征 H1，为后续层的计算做准备。
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    #将 H[l][:, i, :]（当前节点特征）,H1（之前节点的特征聚合结果）,adj[:, i, :i]（当前节点与之前节点的邻接矩阵）
                    #s_mask[:, i, :i]（当前节点的掩码）,得到聚合结果 M
                    _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i])
                    else:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:, i, :i])

                #使用 GRU 单元 self.grus_c[l] 来处理当前节点的特征 H[l][:, i, :] 和聚合后的特征 M，得到新的特征 C。
                # 这表明当前节点的特征更新与其邻居的聚合信息有关。
                C = self.grus_c_short[l](H[l][:,i,:], M).unsqueeze(1)
                #使用另一个 GRU 单元 self.grus_p[l] 来处理聚合特征 M 和当前节点的特征 H[l][:, i, :]，得到额外的特征 P。
                P = self.grus_p_short[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P#将更新后的特征 C 和额外特征 P 进行相加，生成新的节点特征 H_temp
                H1 = torch.cat((H1 , H_temp), dim = 1)  #将当前节点的特征 H_temp 拼接到 H1 中。
               # print('H1', H1.size())
                #print('----------------------------------------------------')
            H.append(H1)
        
            # 将输入特征反转
        # 反向特征提取
        features_reversed = torch.flip(features, dims=[1])  # 反转特征顺序
        adj_reversed = torch.flip(adj_1, dims=[1, 2])  # 反转邻接矩阵
        s_mask_reversed = torch.flip(s_mask, dims=[1, 2])  # 反转掩码
        
        H0_reversed = F.relu(self.fc1(features_reversed))
        H_reversed = [H0_reversed]

        for l in range(self.args.gnn_layers):
            C = self.grus_c_short[l](H_reversed[l][:, 0, :]).unsqueeze(1)
            M = torch.zeros_like(C).squeeze(1)
            P = self.grus_p_short[l](M, H_reversed[l][:, 0, :]).unsqueeze(1)
            H1_reversed = C + P

            for i in range(1, num_utter):
                if self.args.attn_type == 'rgcn':
                    _, M = self.gather_short[l](H_reversed[l][:, i, :], H1_reversed, H1_reversed, adj_reversed[:, i, :i], s_mask_reversed[:, i, :i])
                else:
                    if not self.rel_attn:
                        _, M = self.gather_short[l](H_reversed[l][:, i, :], H1_reversed, H1_reversed, adj_reversed[:, i, :i])
                    else:
                        _, M = self.gather_short[l](H_reversed[l][:, i, :], H1_reversed, H1_reversed, adj_reversed[:, i, :i], s_mask_reversed[:, i, :i])

                C = self.grus_c_short[l](H_reversed[l][:, i, :], M).unsqueeze(1)
                P = self.grus_p_short[l](M, H_reversed[l][:, i, :]).unsqueeze(1)
                H_temp_reversed = C + P
                H1_reversed = torch.cat((H1_reversed, H_temp_reversed), dim=1)
            H_reversed.append(H1_reversed)
            H_combined = torch.cat((H[l+1], H_reversed[l+1]), dim=2)  # 在第二维度拼接
            H_combined_short_list.append(H_combined)  # 将拼接后的结果添加到新列表中
        
        '''
        下面对长距离特征进行处理    The following processes the long-distance features.
        '''
        H_long = [H0]  # 初始化 H_long
        H_combined_long_list = []  # 存储长距离处理的结果

        # 对长距离特征进行处理
        for l in range(self.args.gnn_layers):
            C_long = self.grus_c_long[l](H_long[l][:,0,:]).unsqueeze(1)  # 使用 GRU 更新长距离的第一个节点
            M_long = torch.zeros_like(C_long).squeeze(1)  # 初始化长距离的聚合信息张量 M_long
            P_long = self.grus_p_long[l](M_long, H_long[l][:,0,:]).unsqueeze(1)  # 生成额外的特征 P_long
            
            H1_long = C_long + P_long  # 生成新的长距离节点特征 H1_long
            for i in range(1, num_utter):
                # 依据不同的 attention 类型，进行特征聚合
                if self.args.attn_type == 'rgcn':
                    _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])
                else:
                    if not self.rel_attn:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i])
                    else:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])

                # 使用 GRU 更新当前节点的特征 C_long 和 M_long
                C_long = self.grus_c_long[l](H_long[l][:,i,:], M_long).unsqueeze(1)
                P_long = self.grus_p_long[l](M_long, H_long[l][:,i,:]).unsqueeze(1)   
                
                H_temp_long = C_long + P_long  # 将更新后的特征 C_long 和 P_long 相加生成新特征
                H1_long = torch.cat((H1_long, H_temp_long), dim=1)  # 将特征拼接到 H1_long 中
            H_long.append(H1_long)  # 更新 H_long 列表
        
        # 反转特征顺序，进行逆向长距离特征提取
        features_reversed_long = torch.flip(features, dims=[1])  # 反转特征顺序
        adj_reversed_long = torch.flip(adj_2, dims=[1, 2])  # 反转长距离邻接矩阵
        s_mask_reversed_long = torch.flip(s_mask, dims=[1, 2])  # 反转掩码
        
        H0_reversed_long = F.relu(self.fc1(features_reversed_long))
        H_reversed_long = [H0_reversed_long]

        for l in range(self.args.gnn_layers):
            C_long = self.grus_c_long[l](H_reversed_long[l][:, 0, :]).unsqueeze(1)
            M_long = torch.zeros_like(C_long).squeeze(1)
            P_long = self.grus_p_long[l](M_long, H_reversed_long[l][:, 0, :]).unsqueeze(1)
            H1_reversed_long = C_long + P_long

            for i in range(1, num_utter):
                if self.args.attn_type == 'rgcn':
                    _, M_long = self.gather_long[l](H_reversed_long[l][:, i, :], H1_reversed_long, H1_reversed_long, adj_reversed_long[:, i, :i], s_mask_reversed_long[:, i, :i])
                else:
                    if not self.rel_attn:
                        _, M_long = self.gather_long[l](H_reversed_long[l][:, i, :], H1_reversed_long, H1_reversed_long, adj_reversed_long[:, i, :i])
                    else:
                        _, M_long = self.gather_long[l](H_reversed_long[l][:, i, :], H1_reversed_long, H1_reversed_long, adj_reversed_long[:, i, :i], s_mask_reversed_long[:, i, :i])

                C_long = self.grus_c_long[l](H_reversed_long[l][:, i, :], M_long).unsqueeze(1)
                P_long = self.grus_p_long[l](M_long, H_reversed_long[l][:, i, :]).unsqueeze(1)
                H_temp_reversed_long = C_long + P_long
                H1_reversed_long = torch.cat((H1_reversed_long, H_temp_reversed_long), dim=1)
            H_reversed_long.append(H1_reversed_long)

            # 将正向和逆向的长距离特征进行拼接
            H_combined_long = torch.cat((H_long[l+1], H_reversed_long[l+1]), dim=2)
            H_combined_long_list.append(H_combined_long)

        '''
        两个通道特征都提取完毕！    Both short- and long-distance channel features have been extracted!
        '''    
      #  print('H_combined_short_list',H_combined_short_list)
       # print('H_combined_long_list',H_combined_long_list)
      #  are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(H_combined_short_list, H_combined_long_list))
      #  print("H_combined_short_list 和  H_combined_long_list 是否完全相等:", are_equal)
     #   for idx, tensor in enumerate(H_combined_short_list):
      #      print(f"H_combined_short_list[{idx}] shape: {tensor.shape}")
        H_final = []
      #  print("H2 shape:", H2.shape)
       # 计算差异正则化损失
        diff_loss = 0
        for l in range(self.args.gnn_layers):
        #    print('周期：', l)
            HShort_prime = H_combined_short_list[l]
            HLong_prime = H_combined_long_list[l]
            print("HShort_prime:", HShort_prime)
            print("HLong_prime:", HLong_prime)
            print("HShort_prime shape:", HShort_prime.shape)
            print("HLong_prime shape:", HLong_prime.shape)
            diff_loss = self.diff_loss(HShort_prime, HLong_prime) + diff_loss
           # print("diff_loss:", diff_loss)
          #  print(diff_loss.item())
            # 互交叉注意力机制
            A1 = F.softmax(torch.bmm(torch.matmul(HShort_prime, self.affine1), torch.transpose(HLong_prime, 1, 2)), dim=2)
            A2 = F.softmax(torch.bmm(torch.matmul(HLong_prime, self.affine2), torch.transpose(HShort_prime, 1, 2)), dim=2)

            HShort_prime_new = torch.bmm(A1, HLong_prime)  # 更新的短时特征
            HLong_prime_new = torch.bmm(A2, HShort_prime)    # 更新的长时特征

            HShort_prime_out = self.dropout(HShort_prime_new) if l < self.args.gnn_layers - 1 else HShort_prime_new
            HLong_prime_out = self.dropout(HLong_prime_new) if l <self.args.gnn_layers - 1 else HLong_prime_new

            H_final.append(HShort_prime_out)
            H_final.append(HLong_prime_out)
        H_final.append(features)

        H_final = torch.cat([H_final[-3],H_final[-2],H_final[-1]], dim = 2)
     #   print("H shape:", H.shape)
       # print("H:", H.shape)
     #   print("H_final shape after cat:", H_final.shape)
        H_final = self.attentive_node_features(H_final,lengths,self.nodal_att_type) 
     #   print("H_final shape after attentive_node_features:", H_final.shape)
        logits = self.out_mlp(H_final)
     #  print(diff_loss)
        return logits, self.beta * diff_loss


#使用所有层的short和long，使用sum加每一层，不使用双特征融合技术
#All-layer short and long features are used, with a sum over each layer; dual-feature fusion is not applied.
class DAGERC_new_3(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            #短距离
            gats_short = []
            gats_long = []
            for _ in range(args.gnn_layers):
                gats_short += [GAT_dialoggcn_v1(args.hidden_dim)]
            for _ in range(args.gnn_layers):
                gats_long += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather_short = nn.ModuleList(gats_short)
            self.gather_long = nn.ModuleList(gats_long)

        # 近距离 GRU
        grus_c_short = []
        for _ in range(args.gnn_layers):
            grus_c_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_short = nn.ModuleList(grus_c_short)

        # 远距离 GRU
        grus_c_long = []
        for _ in range(args.gnn_layers):
            grus_c_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_long = nn.ModuleList(grus_c_long)

        grus_p_short = []
        for _ in range(args.gnn_layers):
            grus_p_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_short = nn.ModuleList(grus_p_short)

        grus_p_long = []
        for _ in range(args.gnn_layers):
            grus_p_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_long = nn.ModuleList(grus_p_long)        
        
        #近距离全链接层
        fcs_short = []
        for _ in range(args.gnn_layers):
            fcs_short += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_short = nn.ModuleList(fcs_short)

        # 远距离全连接层
        fcs_long = []
        for _ in range(args.gnn_layers):
            fcs_long += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_long = nn.ModuleList(fcs_long)


        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        
        self.nodal_att_type = args.nodal_att_type
        
        in_dim = (args.hidden_dim * (args.gnn_layers + 1)) + args.emb_dim
   #     print(in_dim)
        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

    def forward(self, features, adj_1, adj_2 ,s_mask, s_mask_onehot, lengths):
        # 检查 H1 和 H2 是否完全相等
        are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(adj_1, adj_2))
      #  print("adj1 和 adj2 是否完全相等:", are_equal)
      #  print('adj1',adj_1)
      #  print('----------------------------------------------------')

     #  print('adj2',adj_2)
     #   print('----------------------------------------------------')
    
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        #print('H0', H0.size())
        # H0 = self.dropout(H0)
        H = [H0]
        H_combined_short_list = []
        #对短距离特征进行处理
        for l in range(self.args.gnn_layers):
            C = self.grus_c_short[l](H[l][:,0,:]).unsqueeze(1) #针对每一层的第一个节点，使用 GRU 单元更新节点特征并聚合信息。
            M = torch.zeros_like(C).squeeze(1) #初始化一个聚合信息张量 M（全零张量），并使用它与节点特征结合生成额外的特征 P。
            # P = M.unsqueeze(1) 
            P = self.grus_p_short[l](M, H[l][:,0,:]).unsqueeze(1)  #使用 M（全零张量）和第一个节点的特征 H[l][:, 0, :] 作为输入，得到额外特征 P，形状为 (B, D)
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P#将更新后的特征 C 与额外特征 P 相加，生成新的节点特征 H1，为后续层的计算做准备。
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    #将 H[l][:, i, :]（当前节点特征）,H1（之前节点的特征聚合结果）,adj[:, i, :i]（当前节点与之前节点的邻接矩阵）
                    #s_mask[:, i, :i]（当前节点的掩码）,得到聚合结果 M
                    _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i])
                    else:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:, i, :i])

                #使用 GRU 单元 self.grus_c[l] 来处理当前节点的特征 H[l][:, i, :] 和聚合后的特征 M，得到新的特征 C。
                # 这表明当前节点的特征更新与其邻居的聚合信息有关。
                C = self.grus_c_short[l](H[l][:,i,:], M).unsqueeze(1)
                #使用另一个 GRU 单元 self.grus_p[l] 来处理聚合特征 M 和当前节点的特征 H[l][:, i, :]，得到额外的特征 P。
                P = self.grus_p_short[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P#将更新后的特征 C 和额外特征 P 进行相加，生成新的节点特征 H_temp
                H1 = torch.cat((H1 , H_temp), dim = 1)  #将当前节点的特征 H_temp 拼接到 H1 中。
               # print('H1', H1.size())
                #print('----------------------------------------------------')
            H.append(H1)
        '''
        下面对长距离特征进行处理
        '''
        H_long = [H0]  # 初始化 H_long
        H_combined_long_list = []  # 存储长距离处理的结果

        # 对长距离特征进行处理
        for l in range(self.args.gnn_layers):
            C_long = self.grus_c_long[l](H_long[l][:,0,:]).unsqueeze(1)  # 使用 GRU 更新长距离的第一个节点
            M_long = torch.zeros_like(C_long).squeeze(1)  # 初始化长距离的聚合信息张量 M_long
            P_long = self.grus_p_long[l](M_long, H_long[l][:,0,:]).unsqueeze(1)  # 生成额外的特征 P_long
            
            H1_long = C_long + P_long  # 生成新的长距离节点特征 H1_long
            for i in range(1, num_utter):
                # 依据不同的 attention 类型，进行特征聚合
                if self.args.attn_type == 'rgcn':
                    _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])
                else:
                    if not self.rel_attn:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i])
                    else:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])

                # 使用 GRU 更新当前节点的特征 C_long 和 M_long
                C_long = self.grus_c_long[l](H_long[l][:,i,:], M_long).unsqueeze(1)
                P_long = self.grus_p_long[l](M_long, H_long[l][:,i,:]).unsqueeze(1)   
                
                H_temp_long = C_long + P_long  # 将更新后的特征 C_long 和 P_long 相加生成新特征
                H1_long = torch.cat((H1_long, H_temp_long), dim=1)  # 将特征拼接到 H1_long 中
            H_long.append(H1_long)  # 更新 H_long 列表
       # for i, h in enumerate(H):
        #     print(f"H[{i}] shape: {h.shape}")

        H_combined = torch.cat(H, dim=2)
        H_long_combined = torch.cat(H_long, dim=2)
        sum_features = H_combined + H_long_combined
      #  print('sum_features Shape:', sum_features.shape)
       # print('features Shape:', features.shape)
        H_combined_final = torch.cat((sum_features, features), dim=2)

        H_final = self.attentive_node_features(H_combined_final,lengths,self.nodal_att_type)  
     #   print("H_final shape after attentive_node_features:", H_final.shape)
        logits = self.out_mlp(H_final)
     #  print(diff_loss)
        return logits

#使用过去的所有层的short和long，每一层都concat，使用特征融合技术。
#All past-layer short and long features are used; features from each layer are concatenated, and feature fusion techniques are applied.
class DAGERC_new_4(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        # gcn layer

        self.dropout = nn.Dropout(args.dropout)

        self.gnn_layers = args.gnn_layers

        if not args.no_rel_attn:
            self.rel_attn = True
        else:
            self.rel_attn = False

        if self.args.attn_type == 'linear':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatLinear(args.hidden_dim) if args.no_rel_attn else GatLinear_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'dotprod':
            gats = []
            for _ in range(args.gnn_layers):
                gats += [GatDot(args.hidden_dim) if args.no_rel_attn else GatDot_rel(args.hidden_dim)]
            self.gather = nn.ModuleList(gats)
        elif self.args.attn_type == 'rgcn':
            gats_short = []
            gats_long = []
            for _ in range(args.gnn_layers):
                gats_short += [GAT_dialoggcn_v1(args.hidden_dim)]
            for _ in range(args.gnn_layers):
                gats_long += [GAT_dialoggcn_v1(args.hidden_dim)]
            self.gather_short = nn.ModuleList(gats_short)
            self.gather_long = nn.ModuleList(gats_long)

        # 近距离 GRU
        # short distance GRU
        grus_c_short = []
        for _ in range(args.gnn_layers):
            grus_c_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_short = nn.ModuleList(grus_c_short)

        # 远距离 GRU
        #  long distance GRU
        grus_c_long = []
        for _ in range(args.gnn_layers):
            grus_c_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_c_long = nn.ModuleList(grus_c_long)

        grus_p_short = []
        for _ in range(args.gnn_layers):
            grus_p_short += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_short = nn.ModuleList(grus_p_short)

        grus_p_long = []
        for _ in range(args.gnn_layers):
            grus_p_long += [nn.GRUCell(args.hidden_dim, args.hidden_dim)]
        self.grus_p_long = nn.ModuleList(grus_p_long)        
        
        #近距离全链接层
        #Fully Connected Layer for Short-Range Features
        fcs_short = []
        for _ in range(args.gnn_layers):
            fcs_short += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_short = nn.ModuleList(fcs_short)

        # 远距离全连接层
        # Fully Connected Layer for Long-Range Features
        fcs_long = []
        for _ in range(args.gnn_layers):
            fcs_long += [nn.Linear(args.hidden_dim * 2, args.hidden_dim)]
        self.fcs_long = nn.ModuleList(fcs_long)


        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        
        self.nodal_att_type = args.nodal_att_type
        
        in_dim = (((args.hidden_dim*2))*(args.gnn_layers + 1) + args.emb_dim) 

        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [self.dropout]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.attentive_node_features = attentive_node_features(in_dim)

        self.affine1 = nn.Parameter(torch.empty(size=((args.hidden_dim)  , (args.hidden_dim)  )))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=((args.hidden_dim)  , (args.hidden_dim) )))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        self.diff_loss = DiffLoss(args)
        self.beta = args.diffloss

    def forward(self, features, adj_1, adj_2 ,s_mask,s_mask_onehot, lengths):
        # 检查 H1 和 H2 是否完全相等
        are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(adj_1, adj_2))
      #  print("adj1 和 adj2 是否完全相等:", are_equal)
      #  print('adj1',adj_1)
      #  print('----------------------------------------------------')

     #  print('adj2',adj_2)
     #   print('----------------------------------------------------')
    
        num_utter = features.size()[1]

        H0 = F.relu(self.fc1(features))
        #print('H0', H0.size())
        # H0 = self.dropout(H0)
        H = [H0]
        H_combined_short_list = []
        #对短距离特征进行处理   Process short-range features.
        for l in range(self.args.gnn_layers):
            C = self.grus_c_short[l](H[l][:,0,:]).unsqueeze(1) #针对每一层的第一个节点，使用 GRU 单元更新节点特征并聚合信息。For the first node of each layer, use a GRU unit to update the node features and aggregate information.
            M = torch.zeros_like(C).squeeze(1) #初始化一个聚合信息张量 M（全零张量），并使用它与节点特征结合生成额外的特征 P。Initialize an aggregation tensor M (a zero tensor), and use it together with the node features to generate additional features P.
            # P = M.unsqueeze(1) 
            P = self.grus_p_short[l](M, H[l][:,0,:]).unsqueeze(1)  
            #H1 = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
            #H1 = F.relu(C+P)
            H1 = C+P
            for i in range(1, num_utter):
                # print(i,num_utter)
                if self.args.attn_type == 'rgcn':
                    _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:,i,:i])
                    # _, M = self.gather[l](H[l][:,i,:], H1, H1, adj[:,i,:i], s_mask_onehot[:,i,:i,:])
                else:
                    if not self.rel_attn:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i])
                    else:
                        _, M = self.gather_short[l](H[l][:,i,:], H1, H1, adj_1[:,i,:i], s_mask[:, i, :i])


                C = self.grus_c_short[l](H[l][:,i,:], M).unsqueeze(1)

                P = self.grus_p_short[l](M, H[l][:,i,:]).unsqueeze(1)   
                # P = M.unsqueeze(1)
                #H_temp = F.relu(self.fcs[l](torch.cat((C,P) , dim = 2)))  
                #H_temp = F.relu(C+P)
                H_temp = C+P#将更新后的特征 C 和额外特征 P 进行相加，生成新的节点特征 H_temp
                H1 = torch.cat((H1 , H_temp), dim = 1)  #将当前节点的特征 H_temp 拼接到 H1 中。
               # print('H1', H1.size())
                #print('----------------------------------------------------')
            H.append(H1)
            H_combined_short_list.append(H[l+1])
        
        '''
        下面对长距离特征进行处理    The following processes the long-distance features.
        '''
        H_long = [H0]  # 初始化 H_long
        H_combined_long_list = []  # 存储长距离处理的结果

        # 对长距离特征进行处理
        for l in range(self.args.gnn_layers):
            C_long = self.grus_c_long[l](H_long[l][:,0,:]).unsqueeze(1)  # 使用 GRU 更新长距离的第一个节点
            M_long = torch.zeros_like(C_long).squeeze(1)  # 初始化长距离的聚合信息张量 M_long
            P_long = self.grus_p_long[l](M_long, H_long[l][:,0,:]).unsqueeze(1)  # 生成额外的特征 P_long
            
            H1_long = C_long + P_long  # 生成新的长距离节点特征 H1_long
            for i in range(1, num_utter):
                # 依据不同的 attention 类型，进行特征聚合
                if self.args.attn_type == 'rgcn':
                    _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])
                else:
                    if not self.rel_attn:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i])
                    else:
                        _, M_long = self.gather_long[l](H_long[l][:,i,:], H1_long, H1_long, adj_2[:,i,:i], s_mask[:,i,:i])

                # 使用 GRU 更新当前节点的特征 C_long 和 M_long
                C_long = self.grus_c_long[l](H_long[l][:,i,:], M_long).unsqueeze(1)
                P_long = self.grus_p_long[l](M_long, H_long[l][:,i,:]).unsqueeze(1)   
                
                H_temp_long = C_long + P_long  # 将更新后的特征 C_long 和 P_long 相加生成新特征
                H1_long = torch.cat((H1_long, H_temp_long), dim=1)  # 将特征拼接到 H1_long 中
            H_long.append(H1_long)  # 更新 H_long 列表    
            H_combined_long_list.append(H_long[l+1])
        '''
        两个通道特征都提取完毕！Both short- and long-distance channel features have been extracted!
        '''    
      #  print('H_combined_short_list',H_combined_short_list)
       # print('H_combined_long_list',H_combined_long_list)
      #  are_equal = all(torch.equal(h1, h2) for h1, h2 in zip(H_combined_short_list, H_combined_long_list))
      #  print("H_combined_short_list 和  H_combined_long_list 是否完全相等:", are_equal)
     #   for idx, tensor in enumerate(H_combined_short_list):
      #      print(f"H_combined_short_list[{idx}] shape: {tensor.shape}")
        H_final = []
        H_0_final = torch.cat([H0, H0], dim=2)
        H_final.append(H_0_final)
      #  print("H2 shape:", H2.shape)
       # 计算差异正则化损失
        diff_loss = 0
        for l in range(self.args.gnn_layers):
        #    print('周期：', l)
            HShort_prime = H_combined_short_list[l]
            HLong_prime = H_combined_long_list[l]
            #print("HShort_prime:", HShort_prime.shape)
           # print("HLong_prime:", HLong_prime.shape)
        #    print("HShort_prime shape:", HShort_prime.shape)
        #    print("HLong_prime shape:", HLong_prime.shape)
            diff_loss = self.diff_loss(HShort_prime, HLong_prime) + diff_loss
            #print("diff_loss:", diff_loss)
          #  print(diff_loss.item())
            # 互交叉注意力机制
            A1 = F.softmax(torch.bmm(torch.matmul(HShort_prime, self.affine1), torch.transpose(HLong_prime, 1, 2)), dim=2)
            A2 = F.softmax(torch.bmm(torch.matmul(HLong_prime, self.affine2), torch.transpose(HShort_prime, 1, 2)), dim=2)

            HShort_prime_new = torch.bmm(A1, HLong_prime)  # 更新的短时特征
            HLong_prime_new = torch.bmm(A2, HShort_prime)    # 更新的长时特征

            HShort_prime_out = self.dropout(HShort_prime_new) if l < self.args.gnn_layers - 1 else HShort_prime_new
            HLong_prime_out = self.dropout(HLong_prime_new) if l <self.args.gnn_layers - 1 else HLong_prime_new

            H_layer = torch.cat([HShort_prime_out, HLong_prime_out], dim=2)
            H_final.append(H_layer)
        H_final = torch.cat(H_final, dim=2)
        H_final = torch.cat([H_final, features], dim=2)


     #   print("H_final shape:", H_final.shape)
       # print("H:", H.shape)
     #   print("H_final shape after cat:", H_final.shape)
        H_final = self.attentive_node_features(H_final,lengths,self.nodal_att_type) 
     #   print("H_final shape after attentive_node_features:", H_final.shape)
        logits = self.out_mlp(H_final)
     #  print(diff_loss)
        return logits, self.beta * diff_loss