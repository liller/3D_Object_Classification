
import sys
import torch.backends.cudnn as cudnn
import argparse
import logging
from PointMLP.classification.models.PointMLP import *
from PointMLP.data_util.ModelnetDataloader import *
from PointMLP.utils import *


def parse_args(args=None):
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=10, type=int, help='workers')
    return parser.parse_args(args)

# 传入命令行参数
sys.argv = ['--batch_size', '--epoch','300']
args = parse_args()

if args.seed is None:
    args.seed = np.random.randint(1, 10000)
#这是为了避免在使用多进程读取 HDF5 文件时出现问题。
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
assert torch.cuda.is_available(), "Please ensure codes are executed in cuda."
device = 'cuda'

screen_logger = logging.getLogger("Model")
screen_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
screen_logger.addHandler(file_handler)

if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
#     save_args(args)
    logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
    logger.set_names(["Epoch-Num", 'Learning-Rate',
                      'Train-Loss', 'Train-acc-B', 'Train-acc',
                      'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])

def printf(str):
    screen_logger.info(str)
    print(str)

printf('==> Preparing data..')
train_dataset = ModelNet40(partition='train', num_points=args.num_points)
test_dataset = ModelNet40(partition='test', num_points=args.num_points)

classes = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup',
           'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop',
           'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa',
           'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
train_loader = DataLoader(train_dataset, num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

# Model
printf(f"args: {args}")
printf('==> Building model..')
net = pointMLP()
criterion = cal_loss
net = net.to(device)
# criterion = criterion.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

best_test_acc = 0.  # best test accuracy
best_train_acc = 0.
best_test_acc_avg = 0.
best_train_acc_avg = 0.
best_test_loss = float("inf")
best_train_loss = float("inf")
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
optimizer_dict = None

from torch.optim.lr_scheduler import CosineAnnealingLR
optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
if optimizer_dict is not None:
    optimizer.load_state_dict(optimizer_dict)
scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)


train_loss, train_acc, train_class_acc, val_loss, val_acc, val_class_acc, lr_history = [], [], [], [], [], [], []
for epoch in range(start_epoch, args.epoch):
    printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
    train_out = train(net, train_loader, optimizer, criterion, device)  # {"loss", "acc", "acc_avg", "time"}
    test_out = validate(net, test_loader, criterion, device)

    # metrics
    train_loss.append(train_out["loss"])
    train_acc.append(train_out["acc"])
    train_class_acc.append(train_out["acc_avg"])

    val_loss.append(test_out["loss"])
    val_acc.append(test_out["acc"])
    val_class_acc.append(test_out["acc_avg"])
    lr_history.append(optimizer.param_groups[0]['lr'])

    scheduler.step()

    if test_out["acc"] > best_test_acc:
        best_test_acc = test_out["acc"]
        is_best = True
    else:
        is_best = False

    best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
    best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
    best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
    best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
    best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
    best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss


    logger.append([epoch, optimizer.param_groups[0]['lr'],
                   train_out["loss"], train_out["acc_avg"], train_out["acc"],
                   test_out["loss"], test_out["acc_avg"], test_out["acc"]])

logger.close()

printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
printf(f"++++++++" * 5)

all_preds = []
all_labels = []
with torch.no_grad():
    #     val_avg_loss, instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
    for batch_idx, (data, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits = net(data)
        preds = logits.max(dim=1)[1]

        all_preds += list(preds.cpu().numpy())
        all_labels += list(label.cpu().numpy())

from sklearn.metrics import classification_report
target_names = list(classes)
print(classification_report(all_labels, all_preds, target_names=target_names))



