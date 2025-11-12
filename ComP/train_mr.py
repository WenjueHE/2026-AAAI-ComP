import time
import datetime
import random
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, recall_score
import sys
from utils import Logger, get_loaders, build_model, generate_mask, generate_inputs, gradient_weight
from loss import MaskedCELoss, MaskedMSELoss
import os
import warnings
sys.path.append('./')
warnings.filterwarnings("ignore")
import config
import torch.nn as nn
from loss import *

from sklearn.preprocessing import OneHotEncoder
from numpy.random import randint

## follow cpm-net's and GCNet's masking manner
def random_mask(view_num, input_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    """

    assert missing_rate is not None
    one_rate = 1 - missing_rate      

    if one_rate <= (1 / view_num): # 
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(input_len, 1))).toarray() # only select one view [avoid all zero input]
        return view_preserve # [samplenum, viewnum] => one value set=1, others=0

    if one_rate == 1:
        matrix = randint(1, 2, size=(input_len, view_num)) # [samplenum, viewnum] => all ones
        return matrix

    ## for one_rate between [1 / view_num, 1] => can have multi view input
    ## ensure at least one of them is avaliable 
    ## since some sample is overlapped, which increase difficulties
    if input_len < 32:
        alldata_len = 32
    else:
        alldata_len = input_len
    error = 1
    while error >= 0.005:

        ## gain initial view_preserve
        enc = OneHotEncoder(categories=[np.arange(view_num)])
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray() # [samplenum, viewnum=2] => one value set=1, others=0

        ## further generate one_num samples
        one_num = view_num * alldata_len * one_rate - alldata_len  # left one_num after previous step
        ratio = one_num / (view_num * alldata_len)                 # now processed ratio
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32) # based on ratio => matrix_iter
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int32)) # a: overlap number
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int32)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int32)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    
    matrix = matrix[:input_len, :]
    return matrix


def train_or_eval_model(args, model, reg_loss, cls_loss, dataloader, total_sqlen, optimizer=None, train=False, first_stage=True, mark='train', mask_rate = 0, 
                        coordinator=True, communication=True, prompt_former=True, gradient_modulator=True):
    weight = []
    preds, preds_a, preds_t, preds_v, masks, labels = [], [], [], [], [], []
    losses, losses1, losses2, losses3 = [], [], [], []
    preds_test_condition = []
    dataset = args.dataset
    cuda = torch.cuda.is_available() and not args.no_cuda
    view_num = 3

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    for data_idx, data in enumerate(dataloader):
        vidnames = []
        if train: optimizer.zero_grad()
        
        ## read dataloader and generate all missing conditions
        """
        audio_host, text_host, visual_host: [seqlen, batch, dim]
        audio_guest, text_guest, visual_guest: [seqlen, batch, dim]
        qmask: speakers, [batch, seqlen]
        umask: has utt, [batch, seqlen]
        label: [batch, seqlen]
        """
        seqlen_origion, batch = data[0].size(0), data[0].size(1)
        ## pad umask, qmask, label to total_sqlen
        assert total_sqlen >= seqlen_origion, f'total_sqlen < seqlen, {total_sqlen} < {seqlen_origion}'
        if total_sqlen > seqlen_origion:
            pad_len = total_sqlen - seqlen_origion
            for i in range(6):
                data[i] = F.pad(data[i], (0, 0, 0, 0, 0,  pad_len), 'constant', 0) # [seqlen_orition, batch, dim] --> [total_sqlen, batch, dim]
            for i in range(6, 9):
                data[i] = F.pad(data[i], (0, pad_len), 'constant', 0) # [batch, seqlen_orition] --> [batch, total_sqlen]
        
        audio_host, text_host, visual_host = data[0], data[1], data[2]
        audio_guest, text_guest, visual_guest = data[3], data[4], data[5]
        qmask, umask, label = data[6], data[7], data[8] # qmask is empty (for cmu datasets) and denotes speakers (for iemocap), umask is 1 for utterance and 0 for filled
        vidnames += data[-1]
        seqlen = audio_host.size(0)
        batch = audio_host.size(1)


        ## using cmp-net masking manner [at least one view exists]
        ## host mask
        # matrix = generate_mask(seqlen, batch, args.test_condition, first_stage) # [seqlen*batch, view_num]
        matrix = random_mask(view_num, seqlen*batch, mask_rate) # We can implement it in this way because 'shuffle = False' in dataloader
        #  all 1 if first_stage, else only 1 for observed modal
        audio_host_mask = np.reshape(matrix[:,0], (batch, seqlen, 1))
        text_host_mask = np.reshape(matrix[:,1], (batch, seqlen, 1))
        visual_host_mask = np.reshape(matrix[:,2], (batch, seqlen, 1))
        audio_host_mask = torch.LongTensor(audio_host_mask.transpose(1, 0, 2))
        text_host_mask = torch.LongTensor(text_host_mask.transpose(1, 0, 2))
        visual_host_mask = torch.LongTensor(visual_host_mask.transpose(1, 0, 2))
        # guest mask
        matrix = random_mask(view_num, seqlen*batch, mask_rate)
        audio_guest_mask = np.reshape(matrix[:,0], (batch, seqlen, 1))
        text_guest_mask = np.reshape(matrix[:,1], (batch, seqlen, 1))
        visual_guest_mask = np.reshape(matrix[:,2], (batch, seqlen, 1))
        audio_guest_mask = torch.LongTensor(audio_guest_mask.transpose(1, 0, 2))
        text_guest_mask = torch.LongTensor(text_guest_mask.transpose(1, 0, 2))
        visual_guest_mask = torch.LongTensor(visual_guest_mask.transpose(1, 0, 2))

        masked_audio_host = audio_host * audio_host_mask
        masked_audio_guest = audio_guest * audio_guest_mask
        masked_text_host = text_host * text_host_mask
        masked_text_guest = text_guest * text_guest_mask
        masked_visual_host = visual_host * visual_host_mask
        masked_visual_guest = visual_guest * visual_guest_mask

        n_classes = args.n_classes

        ## add cuda for tensor
        if cuda:
            masked_audio_host, audio_host_mask = masked_audio_host.to(device), audio_host_mask.to(device)
            masked_text_host, text_host_mask = masked_text_host.to(device), text_host_mask.to(device)
            masked_visual_host, visual_host_mask = masked_visual_host.to(device), visual_host_mask.to(device)
            masked_audio_guest, audio_guest_mask = masked_audio_guest.to(device), audio_guest_mask.to(device)
            masked_text_guest, text_guest_mask = masked_text_guest.to(device), text_guest_mask.to(device)
            masked_visual_guest, visual_guest_mask = masked_visual_guest.to(device), visual_guest_mask.to(device)

            qmask = qmask.to(device)
            umask = umask.to(device)
            label = label.to(device)


        ## generate mask_input_features: ? * [seqlen, batch, dim], input_features_mask: ? * [seq_len, batch, 3] speaker related
        masked_input_features = generate_inputs(masked_audio_host, masked_text_host, masked_visual_host, \
                                                masked_audio_guest, masked_text_guest, masked_visual_guest, qmask, total_sqlen)
        input_features_mask = generate_inputs(audio_host_mask, text_host_mask, visual_host_mask, \
                                                audio_guest_mask, text_guest_mask, visual_guest_mask, qmask, total_sqlen)
        mask_a, mask_t, mask_v = input_features_mask[0][:,:,0].transpose(0,1), input_features_mask[0][:,:,1].transpose(0,1), input_features_mask[0][:,:,2].transpose(0,1)
        '''
        # masked_input_features, input_features_mask: ?*[seqlen, batch, dim]
        # qmask: speakers, [batch, seqlen]
        # umask: has utt, [batch, seqlen]
        # label: [batch, seqlen]
        # log_prob: [seqlen, batch, num_classes]
        '''
        ## forward
        hidden, out, out_a, out_t, out_v, weight_save, proj_avt, proj_avt_rec, shared_atv, private_atv, out_atv_private = model(masked_input_features[0], label, input_features_mask[0], umask, first_stage)
        D_e = args.hidden
        shared_a, shared_t, shared_v = shared_atv[:,:,:D_e], shared_atv[:,:,D_e:2*D_e], shared_atv[:,:,2*D_e:]
        private_a, private_t, private_v = private_atv[:,:,:D_e], private_atv[:,:,D_e:2*D_e], private_atv[:,:,2*D_e:]
        if first_stage:
            out_a_private, out_t_private, out_v_private = out_atv_private[:,:,:n_classes], out_atv_private[:,:,n_classes:2*n_classes], out_atv_private[:,:,2*n_classes:]
        else:
            out_a_private, out_t_private, out_v_private = torch.rand(batch, seqlen, n_classes), torch.rand(batch, seqlen, n_classes), torch.rand(batch, seqlen, n_classes)

        ## save analysis result
        weight.append(weight_save)
        in_mask = torch.clone(input_features_mask[0].permute(1, 0, 2))
        in_mask[umask == 0] = 0
        weight.append(np.array(in_mask.cpu()))
        weight.append(label.detach().cpu().numpy())
        weight.append(vidnames)

        ## calculate loss
        lp_ = out.view(-1, out.size(2)) # [batch*seq_len, n_classes]
        lp_a, lp_t, lp_v = out_a.view(-1, out_a.size(2)), out_t.view(-1, out_t.size(2)), out_v.view(-1, out_v.size(2))
        lp_a_private, lp_t_private, lp_v_private = out_a_private.view(-1, out_a_private.size(2)), out_t_private.view(-1, out_t_private.size(2)), out_v_private.view(-1, out_v_private.size(2))
        labels_ = label.view(-1) # [batch*seq_len]
        proj_avt_rec_loss = nn.MSELoss()(proj_avt_rec, proj_avt)

        private_list = [private_a, private_t, private_v]
        shared_list = [shared_a, shared_t, shared_v]
        disentangled_loss = 0
        disentangled_loss_1 = 0
        disentangled_loss_2 = 0
        for i in range(len(private_list)):
            for j in range(i+1, len(private_list)):
                disentangled_loss += DisentangledLoss()(private_list[i], private_list[j],umask)
                disentangled_loss_1 += DisentangledLoss().foward_1(private_list[i], private_list[j], shared_list[i], shared_list[j], umask)
                disentangled_loss_2 += DisentangledLoss().foward_2(private_list[i], private_list[j], shared_list[i], shared_list[j], umask)
        
        if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            if first_stage:
                all_ones = torch.ones_like(lp_a)
                loss_a = cls_loss(lp_a, labels_, umask) + cls_loss(all_ones - lp_a_private, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask) + cls_loss(all_ones - lp_t_private, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask) + cls_loss(all_ones - lp_v_private, labels_, umask)
            else:
                loss_a = cls_loss(lp_a, labels_, umask)
                loss_t = cls_loss(lp_t, labels_, umask)
                loss_v = cls_loss(lp_v, labels_, umask)
                loss_all = cls_loss(lp_, labels_, umask) + loss_a + loss_t + loss_v + proj_avt_rec_loss
                # non_zeros = np.array([i for i, e in enumerate(labels_) if e != torch.zeros_like(labels_[0])])
                mask = umask.view(-1,1)
                non_zeros = np.array([i for i, e in enumerate(mask) if e != 0]) # remove 0, and remove mask
        if dataset in ['CMUMOSI', 'CMUMOSEI']:
            if first_stage:
                reverse_labels = torch.ones_like(labels_) - labels_
                all_ones = torch.ones_like(lp_a)

                noise = torch.randn_like(labels_)  # Generate Gaussian noise with the same shape as labels_
                noise = noise * 1.5  # Scale to range of approximately -3 to 3
                noise = noise.clamp(min=-3, max=3)  # Clamp the values to the range [-3, 3]

                loss_a = reg_loss(lp_a, labels_, umask) + reg_loss(lp_a_private, noise, umask)
                loss_t = reg_loss(lp_t, labels_, umask) + reg_loss(lp_t_private, noise, umask)
                loss_v = reg_loss(lp_v, labels_, umask) + reg_loss(lp_v_private, noise, umask)
            else:
                loss_a = reg_loss(lp_a, labels_, umask)
                loss_t = reg_loss(lp_t, labels_, umask)
                loss_v = reg_loss(lp_v, labels_, umask)
                loss_all = reg_loss(lp_, labels_, umask) + loss_a + loss_t + loss_v + proj_avt_rec_loss
                non_zeros = np.array([i for i, e in enumerate(labels_) if e != 0]) # remove 0, and remove mask


        ## save batch results
        preds_a.append(lp_a.data.cpu().numpy())
        preds_t.append(lp_t.data.cpu().numpy())
        preds_v.append(lp_v.data.cpu().numpy())
        preds.append(lp_.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        # print(f'---------------{mark} loss: {loss}-------------------')
        preds_test_condition.append(out.view(-1, out.size(2)).data.cpu().numpy())

        if train and first_stage:
            loss = loss_a + loss_t + loss_v + disentangled_loss_2 #4-2
            loss.backward()
            optimizer.step()
        if train and not first_stage:
            loss = loss_all + (loss_a + loss_t + loss_v) + disentangled_loss_2 #4-2
            loss.backward()

            if gradient_modulator:
                if dataset in ['CMUMOSI', 'CMUMOSEI']:
                    # adjust gradient
                    W_a, W_t, W_v = gradient_weight(labels_[non_zeros], lp_a[non_zeros], lp_t[non_zeros], lp_v[non_zeros])
                    modal_dict = ['a', 't', 'v']
                    W_atv = [W_a, W_t, W_v]
                    for i in range(len(modal_dict)):
                        W_i_expanded = torch.ones_like(labels_).reshape(-1,1).to(device)
                        W_i_expanded[non_zeros] = W_atv[i]
                        W_i_expanded = W_i_expanded.reshape(batch, -1) #[batch, seq_len]
                        mask = torch.zeros_like(labels_).reshape(-1,1).to(device)
                        mask[non_zeros] = 1
                        mask = mask.reshape(batch, -1) #[batch, seq_len]
                        W_i = torch.sum(W_i_expanded * mask, dim=0)/torch.sum(mask, dim=0)
                        W_i = torch.where(torch.isnan(W_i), torch.ones_like(W_i), W_i) 
                        for names, para in model.named_parameters():
                            if f'prop_former_{modal_dict[i]}.proj_n.fc1' in names:
                                if 'bias' in names:
                                    pass
                                else:
                                    n, _ = para.shape
                                    para.grad = para.grad * W_i.reshape(1, -1).repeat(n, 1) 
                
                elif dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
                    label = labels_.unsqueeze(1)
                    logit_a = torch.gather(lp_a, 1, label)
                    logit_t = torch.gather(lp_t, 1, label)
                    logit_v = torch.gather(lp_v, 1, label)
                    W_a, W_t, W_v = gradient_weight(label[non_zeros], logit_a[non_zeros], logit_t[non_zeros], logit_v[non_zeros])
                    modal_dict = ['a', 't', 'v']
                    W_atv = [W_a, W_t, W_v]
                    for i in range(len(modal_dict)):
                        W_i_expanded = torch.ones_like(labels_).reshape(-1,1).to(device)
                        W_i_expanded = W_i_expanded.float()
                        W_i_expanded[non_zeros] = W_atv[i]
                        W_i_expanded = W_i_expanded.reshape(batch, -1) #[batch, seq_len]
                        mask = torch.zeros_like(labels_).reshape(-1,1).to(device)
                        mask[non_zeros] = 1
                        mask = mask.reshape(batch, -1) #[batch, seq_len]
                        W_i = torch.sum(W_i_expanded * mask, dim=0)/torch.sum(mask, dim=0)
                        W_i = torch.where(torch.isnan(W_i), torch.ones_like(W_i), W_i) 
                        for names, para in model.named_parameters():
                            if f'prop_former_{modal_dict[i]}.proj_n.fc1' in names:
                                if 'bias' in names:
                                    pass
                                else:
                                    n, _ = para.shape
                                    para.grad = para.grad * W_i.reshape(1, -1).repeat(n, 1)


            optimizer.step()

    assert preds!=[], f'Error: no dataset in dataloader'
    preds  = np.concatenate(preds)
    preds_a = np.concatenate(preds_a)
    preds_t = np.concatenate(preds_t)
    preds_v = np.concatenate(preds_v)
    labels = np.concatenate(labels)
    masks  = np.concatenate(masks)

    # all
    if dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
        preds = np.argmax(preds, 1)
        preds_a = np.argmax(preds_a, 1)
        preds_t = np.argmax(preds_t, 1)
        preds_v = np.argmax(preds_v, 1)
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_accuracy = accuracy_score(labels, preds, sample_weight=masks)
        avg_fscore = f1_score(labels, preds, sample_weight=masks, average='weighted')
        mae = 0
        ua = recall_score(labels, preds, sample_weight=masks, average='macro')
        avg_acc_a = accuracy_score(labels, preds_a, sample_weight=masks)
        avg_acc_t = accuracy_score(labels, preds_t, sample_weight=masks)
        avg_acc_v = accuracy_score(labels, preds_v, sample_weight=masks)
        return mae, ua, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight

    elif dataset in ['CMUMOSI', 'CMUMOSEI']: #binary classification
        non_zeros = np.array([i for i, e in enumerate(labels) if e != 0]) # remove 0, and remove mask
        avg_loss = round(np.sum(losses)/np.sum(masks), 4)
        avg_accuracy = accuracy_score((labels[non_zeros] > 0), (preds[non_zeros] > 0))
        avg_fscore = f1_score((labels[non_zeros] > 0), (preds[non_zeros] > 0), average='weighted')
        mae = np.mean(np.absolute(labels[non_zeros] - preds[non_zeros].squeeze()))
        corr = np.corrcoef(labels[non_zeros], preds[non_zeros].squeeze())[0][1]
        avg_acc_a = accuracy_score((labels[non_zeros] > 0), (preds_a[non_zeros] > 0))
        avg_acc_t = accuracy_score((labels[non_zeros] > 0), (preds_t[non_zeros] > 0))
        avg_acc_v = accuracy_score((labels[non_zeros] > 0), (preds_v[non_zeros] > 0))
        return mae, corr, avg_accuracy, avg_fscore, [avg_acc_a, avg_acc_t, avg_acc_v], vidnames, avg_loss, weight




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ## Params for input
    parser.add_argument('--audio-feature', type=str, default=None, help='audio feature name')
    parser.add_argument('--text-feature', type=str, default=None, help='text feature name')
    parser.add_argument('--video-feature', type=str, default=None, help='video feature name')
    parser.add_argument('--dataset', type=str, default='IEMOCAPFour', help='dataset type')

    ## Params for model
    parser.add_argument('--time-attn', action='store_true', default=False, help='whether to use nodal attention in graph model: Equation 4,5,6 in Paper')
    parser.add_argument('--depth', type=int, default=4, help='')
    parser.add_argument('--num_heads', type=int, default=2, help='')
    parser.add_argument('--drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--attn_drop_rate', type=float, default=0.0, help='')
    parser.add_argument('--hidden', type=int, default=100, help='hidden size in model training')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes [defined by args.dataset]')
    parser.add_argument('--n_speakers', type=int, default=2, help='number of speakers [defined by args.dataset]')

    ## Params for training
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--gpu', type=int, default=2, help='index of gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--num-folder', type=int, default=5, help='folders for cross-validation [defined by args.dataset]')
    parser.add_argument('--seed', type=int, default=100, help='make split manner is same with same seed')
    parser.add_argument('--stage_epoch', type=float, default=100, help='number of epochs of the first stage')
    parser.add_argument('--lbd', type=float, default=0.5, help='lambda for prompt updating')
    parser.add_argument('--mask_rate', type=float, default=0.1, help='mask rate, ranges from 0 to 0.7')

    args = parser.parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    args.device = device
    save_folder_name = f'{args.dataset}'
    save_log = os.path.join(config.LOG_DIR, 'main_result', f'{save_folder_name}')
    if not os.path.exists(save_log): os.makedirs(save_log)
    time_dataset = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{args.dataset}"
    sys.stdout = Logger(filename=f"{save_log}/{time_dataset}_batchsize-{args.batch_size}_lr-{args.lr}_seed-{args.seed}_mask_rate-{args.mask_rate}.txt",
                        stream=sys.stdout)

    ## seed
    def seed_torch(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)  
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    seed_torch(args.seed)


    ## dataset
    if args.dataset in ['CMUMOSI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
        args.seq_len = 65 #fill all items with the same seqlen
    elif args.dataset in ['CMUMOSEI']:
        args.num_folder = 1
        args.n_classes = 1
        args.n_speakers = 1
        args.seq_len = 150
    elif args.dataset == 'IEMOCAPFour':
        args.num_folder = 5
        args.n_classes = 4
        args.n_speakers = 2
        args.seq_len = 150
    elif args.dataset == 'IEMOCAPSix':
        args.num_folder = 5
        args.n_classes = 6
        args.n_speakers = 2
        args.seq_len = 150
    cuda = torch.cuda.is_available() and not args.no_cuda

    filename = f"{args.dataset}-{args.mask_rate}.txt"
    print(args)
    with open(filename, 'a') as f:
        f.write(f"lbd:{args.lbd}\t seq_len:{args.seq_len}\n")

    ## reading data
    print (f'====== Reading Data =======')
    audio_feature, text_feature, video_feature = args.audio_feature, args.text_feature, args.video_feature
    audio_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], audio_feature)
    text_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], text_feature)
    video_root = os.path.join(config.PATH_TO_FEATURES[args.dataset], video_feature)
    print(audio_root)
    print(text_root)
    print(video_root)
    assert os.path.exists(audio_root) and os.path.exists(text_root) and os.path.exists(video_root), f'features not exist!'
    train_loaders, test_loaders, adim, tdim, vdim = get_loaders(audio_root=audio_root,
                                                                            text_root=text_root,
                                                                            video_root=video_root,
                                                                            num_folder=args.num_folder,
                                                                            batch_size=args.batch_size,
                                                                            dataset=args.dataset,
                                                                            num_workers=0)
    assert len(train_loaders) == args.num_folder, f'Error: folder number'

    
    print (f'====== Training and Testing =======')
    folder_mae = []
    folder_corr = []
    folder_acc = []
    folder_f1 = []
    folder_model = []
    for ii in range(args.num_folder):
        print (f'>>>>> Cross-validation: training on the {ii+1} folder >>>>>')
        train_loader = train_loaders[ii]
        test_loader = test_loaders[ii]
        start_time = time.time()

        print('-'*80)
        print (f'Step1: build model (each folder has its own model)')
        model = build_model(args, adim, tdim, vdim)
        reg_loss = MaskedMSELoss()
        cls_loss = MaskedCELoss()
        if cuda:
            model.to(device)
        optimizer = optim.Adam([{'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.l2}])
        print('-'*80)


        print (f'Step2: training (multiple epoches)')
        train_acc_as, train_acc_ts, train_acc_vs = [], [], []
        test_acc_as, test_acc_ts, test_acc_vs = [], [], []
        test_fscores, test_accs, test_maes, test_corrs = [], [], [], []
        models = []
        start_first_stage_time = time.time()

        print("------- Starting the first stage! -------")
        for epoch in range(args.epochs):
            first_stage = True if epoch < args.stage_epoch else False
            ## training and testing (!!! if IEMOCAP, the ua is equal to corr !!!)
            train_mae, train_corr, train_acc, train_fscore, train_acc_atv, train_names, train_loss, weight_train = train_or_eval_model(args, model, reg_loss, cls_loss, train_loader, \
                                                                            total_sqlen=args.seq_len, optimizer=optimizer, train=True, first_stage=first_stage, mark='train', mask_rate = args.mask_rate)
            test_mae, test_corr, test_acc, test_fscore, test_acc_atv, test_names, test_loss, weight_test = train_or_eval_model(args, model, reg_loss, cls_loss, test_loader, \
                                                                            total_sqlen=args.seq_len, optimizer=None, train=False, first_stage=first_stage, mark='test', mask_rate = args.mask_rate)


            ## save
            test_accs.append(test_acc)
            test_fscores.append(test_fscore)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)
            models.append(model)
            train_acc_as.append(train_acc_atv[0])
            train_acc_ts.append(train_acc_atv[1])
            train_acc_vs.append(train_acc_atv[2])
            test_acc_as.append(test_acc_atv[0])
            test_acc_ts.append(test_acc_atv[1]) 
            test_acc_vs.append(test_acc_atv[2])

            if first_stage:
                print(f'epoch:{epoch}; a_acc_train:{train_acc_atv[0]:.3f}; t_acc_train:{train_acc_atv[1]:.3f}; v_acc_train:{train_acc_atv[2]:.3f}')
                print(f'epoch:{epoch}; a_acc_test:{test_acc_atv[0]:.3f}; t_acc_test:{test_acc_atv[1]:.3f}; v_acc_test:{test_acc_atv[2]:.3f}')
            else:
                print(f'epoch:{epoch}; train_mae_{args.mask_rate}:{train_mae:.3f}; train_corr_{args.mask_rate}:{train_corr:.3f}; train_fscore_{args.mask_rate}:{train_fscore:2.2%}; train_acc_{args.mask_rate}:{train_acc:2.2%}; train_loss_{args.mask_rate}:{train_loss}')
                print(f'epoch:{epoch}; test_mae_{args.mask_rate}:{test_mae:.3f}; test_corr_{args.mask_rate}:{test_corr:.3f}; test_fscore_{args.mask_rate}:{test_fscore:2.2%}; test_acc_{args.mask_rate}:{test_acc:2.2%}; test_loss_{args.mask_rate}:{test_loss}')
            print('-'*10)
            ## update the parameter for the 2nd stage
            if epoch == args.stage_epoch-1:
                model = models[-1]

                model_idx_a = int(torch.argmax(torch.Tensor(train_acc_as)))
                print(f'best_epoch_a: {model_idx_a}')
                model_a = models[model_idx_a]
                transformer_a_para_dict = {k: v for k, v in model_a.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_a_para_dict)

                model_idx_t = int(torch.argmax(torch.Tensor(train_acc_ts)))
                print(f'best_epoch_t: {model_idx_t}')
                model_t = models[model_idx_t]
                transformer_t_para_dict = {k: v for k, v in model_t.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_t_para_dict)

                model_idx_v = int(torch.argmax(torch.Tensor(train_acc_vs)))
                print(f'best_epoch_v: {model_idx_v}')
                model_v = models[model_idx_v]
                transformer_v_para_dict = {k: v for k, v in model_v.state_dict().items() if 'Transformer' in k}
                model.state_dict().update(transformer_v_para_dict)

                end_first_stage_time = time.time()
                print("------- Starting the second stage! -------")

        end_second_stage_time = time.time()
        print("-"*80)
        print(f"Time of first stage: {end_first_stage_time - start_first_stage_time}s")
        print(f"Time of second stage: {end_second_stage_time - end_first_stage_time}s")
        print("-" * 80)

        print(f'Step3: saving and testing on the {ii+1} folder')
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            best_index_test = np.argmax(np.array(test_fscores))
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            best_index_test = np.argmax(np.array(test_accs))

        bestmae = test_maes[best_index_test]
        bestcorr = test_corrs[best_index_test]
        bestf1 = test_fscores[best_index_test]
        bestacc = test_accs[best_index_test]
        bestmodel = models[best_index_test]

        best_acc_a = test_acc_as[best_index_test]
        best_acc_t = test_acc_ts[best_index_test]
        best_acc_v = test_acc_vs[best_index_test]

        folder_mae.append(bestmae)
        folder_corr.append(bestcorr)
        folder_f1.append(bestf1)
        folder_acc.append(bestacc)
        folder_model.append(bestmodel)
        end_time = time.time()

        
        with open(filename,"a") as f:
            if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
                f.write(f"The best(acc) epoch of mask_rate ({args.mask_rate}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.\n")
                f.write(f"acc_a: {best_acc_a}, acc_t: {best_acc_t}, acc_v: {best_acc_v}\n")
                f.write(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>\n')
                print(f"The best(acc) epoch of mask_rate ({args.mask_rate}): {best_index_test} --test_mae {bestmae} --test_corr {bestcorr} --test_fscores {bestf1} --test_acc {bestacc}.")
                print(f"acc_a: {best_acc_a}, acc_t: {best_acc_t}, acc_v: {best_acc_v}")
                print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')
            if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
                f.write(f"The best(acc) epoch of mask_rate ({args.mask_rate}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
                f.write(f"acc_a: {best_acc_a}, acc_t: {best_acc_t}, acc_v: {best_acc_v}")
                f.write(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')
                print(f"The best(acc) epoch of mask_rate ({args.mask_rate}): {best_index_test} --test_acc {bestacc} --test_ua {bestcorr}.")
                print(f"acc_a: {best_acc_a}, acc_t: {best_acc_t}, acc_v: {best_acc_v}")
                print(f'>>>>> Finish: training on the {ii+1} folder, duration: {end_time - start_time} >>>>>')

    with open(filename,"a") as f:
        print('-'*80)
        f.write('-'*80)
        f.write(f"\n")
        if args.dataset in ['CMUMOSI', 'CMUMOSEI']:
            print(f"Folder avg: mask_rate ({args.mask_rate}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}")
            f.write(f"Folder avg: mask_rate ({args.mask_rate}) --test_mae {np.mean(folder_mae)} --test_corr {np.mean(folder_corr)} --test_fscores {np.mean(folder_f1)} --test_acc{np.mean(folder_acc)}\n\n")
        if args.dataset in ['IEMOCAPFour', 'IEMOCAPSix']:
            print(f"Folder avg: mask_rate ({args.mask_rate}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}")
            f.write(f"Folder avg: mask_rate ({args.mask_rate}) --test_acc{np.mean(folder_acc)} --test_ua {np.mean(folder_corr)}\n\n")

