
import torch
import torch.nn as nn



def flgc_loss(self):
    if not hasattr(self, "st_loss"):
        self.st_loss = 0

    loss = 0
    for module in self.children():

        s_loss = torch.tensor([0]).float()
        if hasattr(module, "S"):
            s_hat = torch.softmax(module.S, dim=1)
            s_group_mean = s_hat.mean(dim=0)
            standard_value = torch.tensor([1/module.input_channel]).float()

            # print("s_group_mean", s_group_mean, standard_value)
            for i in range(module.group_num):
                # print("s_group_mean[i]", s_group_mean[i])
                # print("standard-s_group_mean", standard_value-s_group_mean[i])
                # print("s pow", torch.pow(standard_value-s_group_mean[i], 2))
                num_input = sum(torch.argmax(s_hat, dim=1)==i).item()
                if num_input == 0:
                    s_loss += torch.pow(standard_value+0.3-s_group_mean[i], 2)

        t_loss = torch.tensor([0]).float()
        if hasattr(module, "T"):
            t_hat = torch.softmax(module.T, dim=1)
            t_group_mean = t_hat.mean(dim=0)
            standard_value = torch.tensor([1 / module.output_channel]).float()
            for i in range(module.group_num):
                num_input = sum(torch.argmax(s_hat, dim=1) == i).item()
                if num_input == 0:
                # if t_group_mean[i] < standard_value:
                    t_loss += torch.pow(standard_value+0.3 - t_group_mean[i], 2)

        loss += (s_loss+t_loss)
    return loss



def add_flgc_loss(net_main_module):
    net_main_module.flgc_loss = flgc_loss.__get__(net_main_module)

    return net_main_module


if __name__=="__main__":
    import model
    for i in range(100):
        net = model.test_cifar()
        net = add_flgc_loss(net)
        loss = net.flgc_loss()
        print(loss)
