
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FLGC(nn.Module):
    """
    train()
    model.before_inference()
    validation()
    """

    def __init__(self, input_channel,output_channel,kernel_size,stride=1,padding=0,
                 dilation=1,group_num=1):
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

    def forward(self, x, feature_index=None, if_flgc_is_next=False):
        """
        :param x: shape(B, input_channel, H, W)
        :return: shape(B, output_channel, H, W)
        """

        if feature_index is None and self.training: # if self.training
            B, _, H, W = x.size()
            s_hat = torch.softmax(self.S, dim=1)
            t_hat = torch.softmax(self.T, dim=1)

            out = torch.Tensor()
            for i in range(self.group_num):
                xi = x * s_hat[:,i].view(1,-1,1,1).expand(B,self.input_channel,H,W)
                ti = self.conv * t_hat[:,i].view(-1,1,1,1).expand(self.output_channel,self.input_channel,1,1)
                if i==0:
                    out = F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)
                else:
                    out += F.conv2d(xi,ti,stride=self.stride,padding=self.padding,dilation=self.dilation)
            return out

        else: # if not self.training
            if feature_index is None:
                feature_index = [i for i in range(self.input_channel)]
            s_hat = torch.softmax(self.S, dim=1)
            t_hat = torch.softmax(self.T, dim=1)
            s = torch.argmax(s_hat, dim=1)
            t = torch.argmax(t_hat, dim=1)
            # s = torch.LongTensor([0, 1, 2, 2])
            # t = torch.LongTensor([0, 1, 1, 2, 2])
            out = None
            for i in range(self.group_num):
                num_input = sum(s==i).item()
                num_filter = sum(t == i).item()
                if num_input*num_filter==0:
                    continue
                # print(i,"f num input, num filter", num_input, num_filter)
                group_index = np.where(s==i)[0]
                index_new = [feature_index.index(index) for index in group_index]
                # print(i,"f input", x[:,index_new,:,:].shape)
                # print(f"{i} x", x.shape)
                if out is None:
                    out = self.conv_test[i](x[:,index_new,:,:])
                else:
                    out = torch.cat([out, self.conv_test[i](x[:,index_new,:,:])], 1)
            # print("out", out.shape)
            if if_flgc_is_next:
                return out
            else:
                out_new = torch.zeros_like(out)
                # print("out new out", out_new.shape, out.shape)
                for i, index in enumerate(self.output_index):
                    # print("i index", i, index)
                    out_new[:,index,:,:] = out[:,i,:,:]
                return out_new


    def before_inference(self):
        s_hat = torch.softmax(self.S, dim=1)
        t_hat = torch.softmax(self.T, dim=1)
        s = torch.argmax(s_hat, dim=1)
        t = torch.argmax(t_hat, dim=1)
        # s = torch.LongTensor([0,1,2,2])
        # t = torch.LongTensor([0,1,1,2,2])
        # print("S & T",s,t)
        self.conv_test = nn.ModuleList()
        self.output_index = []
        for i in range(self.group_num):
            num_input = sum(s==i).item()
            num_filter = sum(t==i).item()
            print(i,"num input, num filter", num_input, num_filter)
            if num_input*num_filter==0:
                self.conv_test.append(None)
            else:
                self.conv_test.append(nn.Conv2d(num_input,num_filter,kernel_size=self.kernel_size,
                                                stride=self.stride,padding=self.padding))
            self.output_index += list(np.where(t==i)[0])
        return self.output_index


def test_training():
    _input = torch.randn(1, 4, 5, 5)
    conv = FLGC(4, 5, 1, 1, 0, 1, 3)
    target = torch.rand(1,5,5,5)
    criterion = nn.MSELoss()
    conv.train()
    optimizer = torch.optim.SGD(conv.parameters(), lr=0.01)
    for i in range(10):
        input = _input.clone()
        out = conv(input)
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item(), conv.conv.mean().item(), conv.S.mean().item(), conv.T.mean().item())

    conv.before_inference()
    conv.eval()
    with torch.no_grad():
        input = _input.clone()
        out = conv(input)
        loss = criterion(out, target)
        print(loss.item())



if __name__=="__main__":
    x = torch.randn(3,2,1,1)
    x = torch.softmax(x, dim=1)
    t = torch.argmax(x, dim=1).view(-1)

    # print("t", t, t.shape)
    # print(sum(t==1).item())
    # print(list(np.where(t==1)[0]))


    test_training()