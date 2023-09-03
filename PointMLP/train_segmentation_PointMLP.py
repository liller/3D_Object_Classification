
import sys
import torch.backends.cudnn as cudnn
import argparse
import logging
from PointMLP.segmentation.model.PointMLP import *
from PointMLP.data_util.ShapeNetDataloader import *
from PointMLP.utils import *



def parse_args(args=None):

    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='PointMLP1')
    parser.add_argument('--exp_name', type=str, default='demo1', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    return parser.parse_args(args)

# 传入命令行参数
sys.argv = ['--batch_size', '--epoch','350','--gpu', '7']
args = parse_args()
print(args)

assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(int(args.gpu))


num_part = 50
# device = torch.device("cuda" if args.cuda else "cpu")



# io.cprint(str(model))
model = PointMLP(num_part).cuda()
# model = model.to(device)

model.apply(weight_init)



if args.use_sgd:
    print("Use SGD")
    opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
else:
    print("Use Adam")
    opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

if args.scheduler == 'cos':
    print("Use CosLR")
    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
else:
    print("Use StepLR")
    scheduler = StepLR(opt, step_size=args.step, gamma=0.5)


best_acc = 0
best_class_iou = 0
best_instance_iou = 0
num_part = 50
num_classes = 16



train_dataset = PartNormalDataset(root=data_path, npoints=args.num_points, split='trainval', normal_channel=False)
test_dataset = PartNormalDataset(root=data_path, npoints=args.num_points, split='test', normal_channel=False)

trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)


def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes):
    train_loss = 0.0
    total_loss = 0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = {}
    model.train()

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader),
                                                            smoothing=0.9):
        batch_size, num_point, _ = points.size()

        points, label, target, norm_plt = points.float().cuda(), label.long().squeeze(1).cuda(), target.long().cuda(), \
                                          norm_plt.float().cuda()
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)


        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # seg_pred: b,n,50
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target,
                                              num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)],
                                              dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.contiguous().data.max(1)[1]
        correct = pred_choice.eq(target.contiguous().data).sum()

        # sum
        shape_ious += batch_shapeious.item()
        count += batch_size
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))



    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5

    lr = opt.param_groups[0]['lr']
    printf('Learning rate: %f' % opt.param_groups[0]['lr'])

    train_avg_loss = train_loss * 1.0 / count

    metrics['accuracy'] = np.mean(accuracy)
    train_avg_acc = metrics['accuracy']

    metrics['shape_avg_iou'] = shape_ious * 1.0 / count
    train_avg_iou = metrics['shape_avg_iou']

    outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch + 1, train_avg_loss,
                                                                       train_avg_acc, train_avg_iou)
    printf(outstr)
    return train_avg_loss, train_avg_acc, train_avg_iou, lr


train_loss, train_acc, train_iou, val_loss, val_acc, val_iou, val_class_iou, lr_history = [], [], [], [], [], [], [], []

classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']

for epoch in range(args.epochs):

    train_avg_loss, train_avg_acc, train_avg_iou, lr = train_epoch(trainDataLoader, model, opt, scheduler, epoch,
                                                                   num_part, num_classes)
    lr_history.append(lr)
    train_loss.append(train_avg_loss)
    train_acc.append(train_avg_acc)
    train_iou.append(train_avg_iou)

    test_avg_loss, test_metrics, total_per_cat_iou = test_epoch(testDataLoader, model, epoch, num_part, num_classes)
    val_loss.append(test_avg_loss)
    val_acc.append(test_metrics['accuracy'])
    val_iou.append(test_metrics['shape_avg_iou'])


    if test_metrics['accuracy'] > best_acc:
        best_acc = test_metrics['accuracy']
        printf('Max Acc:%.5f' % best_acc)

    if test_metrics['shape_avg_iou'] > best_instance_iou:
        best_instance_iou = test_metrics['shape_avg_iou']
        printf('Max instance iou:%.5f' % best_instance_iou)

    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
    avg_class_iou = class_iou / 16
    val_class_iou.append(avg_class_iou)

    if avg_class_iou > best_class_iou:
        best_class_iou = avg_class_iou
        # print the iou of each class:
        for cat_idx in range(16):
            printf(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
        printf('Max class iou:%.5f' % best_class_iou)


printf('Final Max Acc:%.5f' % best_acc)
printf('Final Max instance iou:%.5f' % best_instance_iou)
printf('Final Max class iou:%.5f' % best_class_iou)
