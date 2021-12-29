# Training script
# Originally Forked from https://github.com/ExplorerFreda/VGNSL?files=1
# Modified by Noriyuki Kojima (nk654@cornell.edu)

import argparse
import logging
import os
import pickle
import shutil
import time

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from vocab import Vocabulary
from model import VGNSL
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data, test_trees
from test import f1_score
import data as data

def train(opt, train_loader, model, epoch, val_loader, vocab, writer):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # Always reset to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data, epoch=epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logger.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))
            writer.add_scalar('Loss/loss', float(str(model.logger.meters["Loss"]).replace("(", "").replace(")", "").split(" ")[0]), model.Eiters)
            writer.add_scalar('Loss/MatchLoss', float(str(model.logger.meters["MatchLoss"]).replace("(", "").replace(")", "").split(" ")[0]), model.Eiters)
            writer.add_scalar('Loss/RL-Loss', float(str(model.logger.meters["RL-Loss"]).replace("(", "").replace(")", "").split(" ")[0]), model.Eiters)
            writer.add_scalar('Score/Cumu-Reward-Score', float(str(model.logger.meters["Cumu-Reward"]).replace("(", "").replace(")", "").split(" ")[0]), model.Eiters)
            writer.add_scalar('Score/Entropy', float(str(model.logger.meters["Entropy"]).replace("(", "").replace(")", "").split(" ")[0]), model.Eiters)
        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            currscore = validate(opt, val_loader, model, vocab)
            writer.add_scalar('Score/Dev-Retrival-Score', currscore, model.Eiters)

def validate(opt, val_loader, model, vocab):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(
        model, val_loader, opt.log_step, logger.info, vocab)
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, measure='cosine')
    logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, measure='cosine')
    logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    return currscore


def save_checkpoint(state, is_best, curr_epoch, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    shutil.copyfile (prefix + filename, prefix + str(curr_epoch) + '.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # hyper parameters
    parser = argparse.ArgumentParser()
    # general arguments
    parser.add_argument('--data_path', default='../data/mscoco',
                        help='path to datasets')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='rank loss margin')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='number of training epochs')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='size of a training mini-batch')
    parser.add_argument('--word_dim', default=512, type=int,
                        help='dimensionality of the word embedding')
    parser.add_argument('--embed_size', default=512, type=int,
                        help='dimensionality of the joint embedding')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='gradient clipping threshold')
    parser.add_argument('--learning_rate', default=.0005, type=float,
                        help='initial learning rate')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='number of epochs to update the learning rate')
    parser.add_argument('--workers', default=0, type=int,
                        help='number of data loader workers')
    parser.add_argument('--log_step', default=10, type=int,
                        help='number of steps to print and record the log')
    parser.add_argument('--val_step', default=500, type=int,
                        help='number of steps to run validation')
    parser.add_argument('--logger_name', default='../output/',
                        help='path to save the model and log')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='dimensionality of the image embedding')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='optimizer, can be Adam, SGD, etc.')
    parser.add_argument('--init_embeddings', type=int, default=0)
    parser.add_argument('--init_embeddings_type', choices=['override', 'partial', 'partial-fixed'], default='override')
    parser.add_argument('--init_embeddings_key', choices=['glove', 'fasttext'], default='override')
    parser.add_argument('--init_embeddings_partial_dim', type=int, default=0)
    parser.add_argument('--no_imgnorm', action='store_true', help='Do not normalize the image embeddings.')
    parser.add_argument('--syntax_score_hidden', type=int, default=128)
    parser.add_argument('--vse_reward_alpha', type=float, default=1.0)
    parser.add_argument('--vse_loss_alpha', type=float, default=1.0)
    parser.add_argument('--syntax_dim', type=int, default=300)
    parser.add_argument('--lambda_hi', type=float, default=0,
                        help='penalization for head-initial inductive bias')

    # model (tagger) experiment arguments
    parser.add_argument('--bottleneck_dim', type=int, default=2)
    parser.add_argument('--tagger_fn', type=str, default='none',
                        help='none | linear | deeper | deeper_recursive | linear_recursive | bottleneck_dim')
    parser.add_argument('--score_fn', type=str, default='ws',
                        help='ws | max | mean | mean_hi ')
                        # help='linear| max | mean | mean_hi ')
    parser.add_argument('--combine_fn', type=str, default='mean',
                        help='mean | max')
    parser.add_argument('--seed', type=int, default=-1)

    # bounding box experiment options
    parser.add_argument('--loss_type', type=str, default='sh-loss',
                        help='sh-loss | mh-loss')

    # additional experiment options
    parser.add_argument('--load_checkpoint', action='store_true',  help='restart training from a checkpoint')
    parser.add_argument('--candidate', type=str, default="")
    parser.add_argument('--debug', action='store_true', help='debug flag')
    parser.add_argument('--data_shuffle', action='store_false')

    opt = parser.parse_args()

    # setup logger
    if os.path.exists(opt.logger_name):
        print(f'Warning: the folder {opt.logger_name} exists.')
    os.system('mkdir {:s}'.format(opt.logger_name))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(os.path.join(opt.logger_name, 'train.log'), 'w')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)
    logger.propagate = False

    # Setting up a manual_seed
    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(opt.seed)


    # set up writer
    assert("../outputs" in opt.logger_name)
    writer_logger_name = "../outputs/tensorboard/" + opt.logger_name.replace("../outputs/", "")
    os.makedirs(writer_logger_name, exist_ok=True)
    writer = SummaryWriter(log_dir=writer_logger_name)

    # load predefined vocabulary and pretrained word embeddings if applicable
    vocab = pickle.load(open(os.path.join(opt.data_path, 'vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)

    if opt.init_embeddings:
        opt.vocab_init_embeddings = os.path.join(
            opt.data_path, f'vocab.pkl.{opt.init_embeddings_key}_embeddings.npy'
        )

    # Load data loaders
    if opt.debug:
        train_loader, val_loader = data.get_debug_loaders(
            opt.data_path, vocab, opt.batch_size, opt.workers,
        )
    else:
        train_loader, val_loader = data.get_train_loaders(
            opt.data_path, vocab, opt.batch_size, opt.workers, is_shuffle=opt.data_shuffle
        )

    # construct the model
    model = VGNSL(opt)

    start_epoch = 0

    # starting training from a checkpoint
    if opt.load_checkpoint:
        checkpoint = torch.load(opt.candidate)
        model.load_state_dict(checkpoint['model'])
        model.Eiters = checkpoint['Eiters']
        start_epoch = checkpoint['epoch'] - 1
        logger_name = opt.logger_name
        opt = checkpoint['opt']
        opt.logger_name = logger_name
        opt.learning_rate = checkpoint['model'][2]['param_groups'][0]['lr']

    best_rsum = 0
    for ct in range(opt.num_epochs-start_epoch):
        epoch = ct + start_epoch
        print(epoch)
        adjust_learning_rate(opt, model.optimizer, epoch)
        # evaluate on validation set using VSE metrics
        rsum = validate(opt, val_loader, model, vocab)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, epoch, prefix=opt.logger_name + '/')

        # validate on evaluation objectives
        model_name = opt.logger_name + '/' + "{}.pth.tar".format(epoch)
        trees, ground_truth = test_trees(model_name, "dev")
        f1, _, _ =  f1_score(trees, ground_truth, gap_stats=False)
        writer.add_scalar('Eval/Dev-F1', f1, model.Eiters)
        writer.add_text("Eval/Dev-Prediction-1", trees[1], model.Eiters)
        writer.add_text("Eval/Dev-Prediction-2", trees[10], model.Eiters)
        writer.add_text("Eval/Dev-Prediction-3", trees[100], model.Eiters)
        writer.add_text("Eval/Dev-Prediction-4", trees[1000], model.Eiters)
        writer.add_text("Eval/Dev-Prediction-5", trees[2000], model.Eiters)
        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, vocab, writer)

    # closing writer
    try:
        writer.close()
    except KeyboardInterrupt:
        print("Interupted by a keyboard.")
        writer.close()
