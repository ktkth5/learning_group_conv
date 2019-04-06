
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import time

class FLGC(nn.Module):
    """
    train()
    model.before_inference()
    validation()
    """

    def __init__(self, input_channel,output_channel,kernel_size,stride=1,padding=0,
                 dilation=1,group_num=1, bias=False):
        super(FLGC, self).__init__()

        self.conv = nn.Parameter(torch.randn(output_channel, input_channel, kernel_size, kernel_size))
        self.S = nn.Parameter(torch.randn(input_channel, group_num))
        self.T = nn.Parameter(torch.randn(output_channel, group_num))

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group_num = group_num

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.final_inference = False

        self.time_cp0 = 0
        self.time_cp1 = 0
        self.time_cp2 = 0
        self.time_cp3 = 0

    def forward(self, x):
        """
        :param x: shape(B, input_channel, H, W)
        :return: shape(B, output_channel, H, W)
        """
        # print("self.final_inference", self.final_inference)
        if not self.final_inference: # if self.training
            B, _, H, W = x.size()
            s_hat = torch.softmax(self.S, dim=1)
            t_hat = torch.softmax(self.T, dim=1)
            s = torch.argmax(s_hat, dim=1)
            t = torch.argmax(t_hat, dim=1)

            out = torch.Tensor()
            out_index = []
            for i in range(self.group_num):
                # Previous Method
                # xi = x * s_hat[:,i].view(1,-1,1,1).expand(B,self.input_channel,H,W)
                # ti = self.conv * t_hat[:,i].view(-1,1,1,1).expand(self.output_channel,self.input_channel,1,1)
                # if i==0:
                #     out = F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)
                # else:
                #     out += F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)

                # New Method
                x_index = list(np.where(s.cpu() == i)[0])
                f_index = list(np.where(t.cpu() == i)[0])
                out_index += f_index
                xi = x[:, x_index] * s_hat[x_index, i].view(1,-1,1,1).expand(B, len(x_index),H,W)
                ti = self.conv[f_index][:, x_index] * t_hat[f_index,i].view(-1,1,1,1).expand(len(f_index),len(x_index),1,1)
                # print("xi ti", xi.shape, ti.shape)
                if i==0:
                    out = F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)
                else:
                    out = torch.cat([out, F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)], 1)
                # print("out",out.shape)
            out = out[:, out_index]


            return out

        else: # if not self.training
            end = time.time()

            feature_index = [i for i in range(self.input_channel)]
            s_hat = torch.softmax(self.S, dim=1)
            t_hat = torch.softmax(self.T, dim=1)
            s = torch.argmax(s_hat, dim=1)
            t = torch.argmax(t_hat, dim=1)
            # s = torch.LongTensor([0, 1, 2, 2])
            # t = torch.LongTensor([0, 1, 1, 2, 2])
            out = None
            debug_num_filter = 0
            self.time_cp0 += time.time() - end
            end = time.time()
            for i in range(self.group_num):
                num_input = self.num_input_list[i] # num_input  = torch.sum(s == i).item()
                num_filter = self.num_filter_list[i] # num_filter = torch.sum(t == i).item()
                debug_num_filter += num_filter
                if num_input*num_filter==0:
                    continue
                # print(i,"f num input, num filter", num_input, num_filter)
                group_index = self.group_index_list[i] # group_index = np.where(s.cpu()==i)[0]
                index_new = self.index_new_list[i] # index_new = [feature_index.index(index) for index in group_index]
                # print(i,"f input", x[:,index_new,:,:].shape)
                # print(f"{i} x", x.shape)
                self.time_cp1 += time.time() - end
                end = time.time()
                if out is None:
                    out = self.conv_test[i](x[:,index_new,:,:])
                else:
                    out = torch.cat([out, self.conv_test[i](x[:,index_new,:,:])], 1)
                self.time_cp2 += time.time() - end
                end = time.time()

            out_new = out[:, self.output_index]
            # out_new = torch.zeros_like(out)
            # for i, index in enumerate(self.output_index):
            #     print("i index", i, index)
                # out_new[:,index,:,:] = out[:,i,:,:]
            self.time_cp3 += time.time() - end
            # print("TIME", self.time_cp0, self.time_cp1, self.time_cp2, self.time_cp3)
            print("S T", s_hat, t_hat)
            return out_new


    def before_inference(self):
        self.final_inference = True

        s_hat = torch.softmax(self.S, dim=1)
        t_hat = torch.softmax(self.T, dim=1)
        s = torch.argmax(s_hat, dim=1)
        t = torch.argmax(t_hat, dim=1)
        # s = torch.LongTensor([0,1,2,2])
        # t = torch.LongTensor([0,1,1,2,2])
        # print("S & T",s,t)
        self.conv_test = nn.ModuleList()
        self.output_index = []
        self.num_input_list = []
        self.num_filter_list = []
        self.group_index_list = []
        self.index_new_list = []
        feature_index = [i for i in range(self.input_channel)]
        for i in range(self.group_num):
            num_input = torch.sum(s==i).item()
            num_filter = torch.sum(t==i).item()
            self.num_input_list.append(num_input)
            self.num_filter_list.append(num_filter)
            self.group_index_list.append(np.where(s.cpu()==i)[0])
            self.index_new_list.append([feature_index.index(index) for index in self.group_index_list[-1]])
            # print(i,"num input, num filter", num_input, num_filter)
            if num_input*num_filter==0:
                self.conv_test.append(None)
            else:
                conv = nn.Conv2d(num_input,num_filter,kernel_size=self.kernel_size,
                                                stride=self.stride,padding=self.padding,bias=False).to(self.device)
                conv.weight.data = self.conv[list(np.where(t.cpu()==i)[0])][:,list(np.where(s.cpu()==i)[0])]
                self.conv_test.append(conv)
            self.output_index += list(np.where(t.cpu()==i)[0])
        return self.output_index


# def cycle_check(seq):
#     for module in seq.children():
#         if isinstance()

def apply_module(self):
    if isinstance(self, FLGC):
        self.before_inference()

def eval_set(self):
    self.apply(apply_module)
    # for module in self.children():
    #     # print("module", module, "\n")
    #     if isinstance(module, FLGC):
    #         print("module FLGC", module)
    #         module.before_inference()
    #     if isinstance(module, nn.Sequential):
    #         eval_set(module)
            # for module_seq in module.children():
            #     # print("module seq", module_seq)
            #     if isinstance(module_seq, FLGC):
            #         print("module_seq FLGC", module_seq)
            #         module_seq.before_inference()


def add_eval_set(net_main_module):
    net_main_module.eval_set = eval_set.__get__(net_main_module)

    return net_main_module