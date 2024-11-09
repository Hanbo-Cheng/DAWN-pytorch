import torch
from tqdm import tqdm
import time


def train_or_test(model, model_d, optimizer_g, optimizer_d, iterator, device, mode="train", epoch = 0):
    if mode == "train":
        model.train()
        model_d.train()
        grad_env = torch.enable_grad
    elif mode == "test":
        model.eval()
        model_d.eval()
        grad_env = torch.no_grad
    else:
        raise ValueError("This mode is not recognized.")

    # loss of the epoch
    dict_loss = {loss: 0 for loss in (model.losses)}
    dict_loss['Dloss'] = 0
    dict_loss['Gloss'] = 0

    with grad_env():
        # start_time = time.time()  # end
        # print(f'load time {end_time- start_time}')

        for i, batch in tqdm(enumerate(iterator), desc="Computing batch"):
            # Put everything in device

            # end_time = time.time()
            # print("load_cost: ", - start_time + end_time)
            # start_time = time.time() 

            batch = {key: val.to(device) for key, val in batch.items() if key!='videoname'}

            # end_time = time.time()
            # print("tocuda_cost: ", - start_time + end_time)
            # start_time = time.time()

            if mode == "train":
                # update the gradients to zero
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

            # forward pass
            batch = model(batch)
            mixed_loss, losses = model.compute_loss(batch, epoch)
            D_loss, G_loss = model_d.calculate_GAN_loss(batch)

            # end_time = time.time()
            # print("forward: ", - start_time + end_time)
            # start_time = time.time()
            
            for key in dict_loss.keys():
                if key != 'Gloss' and key != 'Dloss':
                    dict_loss[key] += losses[key]
            
            dict_loss['Dloss'] += D_loss.item()
            dict_loss['Gloss'] += G_loss.item()

            if mode == "train":
                # backward pass
                ((mixed_loss + (G_loss + D_loss) )).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                # update the weights
                optimizer_g.step()
                optimizer_d.step()
            
            # end_time = time.time()
            # print("back: ", - start_time + end_time)
            # start_time = time.time()

            # if i % 10 == 0:
            #     print(dict_loss)
    return dict_loss


def train(model, model_d, optimizer_g, optimizer_d, iterator, device, epoch):
    return train_or_test(model, model_d, optimizer_g, optimizer_d, iterator, device, mode="train", epoch = epoch)


def test(model, model_d, optimizer_g, optimizer_d, iterator, device):
    return train_or_test(model, model_d, optimizer_g, optimizer_d, iterator, device, mode="test")
