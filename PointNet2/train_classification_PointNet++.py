import numpy as np
import math
import random
import os
import sys
import torch
import scipy.spatial.distance
from torch.utils.data import Dataset, DataLoader

from PointNet2.data_util.ModelnetDataloader import *
from PointNet2.classification.models.Pointnet2_ssg import *


def parse_args(args=None):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='9', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    # 是否使用法向量，默认为 False
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    # 是否将数据保存到本地，默认为 False
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    # 是否使用均匀采样，默认为 False
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    return parser.parse_args(args)



def test(model, loader, num_class=40):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    total_loss = 0

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, trans_feat = classifier(points)
        pred_choice = pred.data.max(1)[1]

        loss = criterion(pred, target.long(), trans_feat)
        total_loss += loss.item()

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    val_avg_loss = total_loss / len(loader)
    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)

    return val_avg_loss, instance_acc, class_acc


logger = logging.getLogger("Model")
def log_string(str):
    logger.info(str)
    print(str)

def main(args):
    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.cuda.set_device(int(args.gpu))
    print(f"Current device: {torch.cuda.get_device_name()}")
    print(f"Current device index: {torch.cuda.current_device()}")


    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)

    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=20, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=20)

    '''MODEL LOADING'''
    num_class = args.num_category
    # model = importlib.import_module(args.model)

    classifier = get_model(num_class, normal_channel=args.use_normals)
    criterion = get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    log_string('No existing model, starting training from scratch...')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    start_epoch = 0
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    train_loss, train_acc, val_loss, val_acc, val_class_acc, lr_history = [], [], [], [], [], []
    for epoch in range(start_epoch, args.epoch):
        log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        total_loss = 0

        scheduler.step()  # 学习率调度器（learning rate scheduler）的一种方法，用于更新优化器的学习率
        lr = scheduler.get_last_lr()[0]
        # print learning rate
        print("Epoch:", epoch + 1, "LR:", lr)
        lr_history.append(lr)
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),
                                               smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            # 因为random_point_dropout涉及需要np进行操作，需要将数据先转为np再转为tensor
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

            total_loss += loss.item()

        train_avg_loss = total_loss / len(trainDataLoader)
        log_string('Train Instance Loss: %f' % train_avg_loss)
        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        train_loss.append(train_avg_loss)
        #     print(f"Train Instance Loss: {train_loss[0]}")
        train_acc.append(train_instance_acc)
        #     print(f"train_acc:{train_acc}")

        with torch.no_grad():
            val_avg_loss, instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
            log_string(f'Test Instance Loss: %f' % val_avg_loss)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            val_loss.append(val_avg_loss)

            val_acc.append(instance_acc)
            #         print(f"val_acc{val_acc}")
            val_class_acc.append(class_acc)
            global_epoch += 1

    logger.info('End of training...')

    all_preds = []
    all_labels = []
    with torch.no_grad():
        #     val_avg_loss, instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
        for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            points = points.transpose(2, 1)
            pred, trans_feat = classifier(points)
            # softmax计算后的值选出最大的一个
            pred_choice = pred.data.max(1)[1]

            all_preds += list(pred_choice.cpu().numpy())
            all_labels += list(target.cpu().numpy())

    from sklearn.metrics import classification_report
    target_names = list(train_dataset.classes.keys())
    print(classification_report(all_labels, all_preds, target_names=target_names))


if __name__ == '__main__':
    sys.argv = ['--use_cpu', '--model', 'pointnet2_cls_ssg', '--epoch', '200', '--batch_size', '32', '--process_data']
    args = parse_args()
    main(args)