
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from PointConv.classification.model.PointConv import *
from PointConv.classification.provider import *
from PointConv.data_util.ModelnetDataloader import *



def parse_args(args=None):
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointConv')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch',  default=400, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=16, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='pointconv', help='model name')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    return parser.parse_args(args)


# 传入命令行参数
sys.argv = ['--batchsize', '--gpu', '6', '--num_workers', '10']
args = parse_args()
# '''HYPER PARAMETER'''
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
torch.cuda.set_device(int(args.gpu))
print(f"Current device: {torch.cuda.get_device_name()}")
print(f"Current device index: {torch.cuda.current_device()}")


'''LOG'''
#     args = parse_args()
logger = logging.getLogger(args.model_name)
logger.info('---------------------------------------------------TRANING---------------------------------------------------')
logger.info('PARAMETER ...')
logger.info(args)


def log_string(str):
    logger.info(str)
    print(str)
'''DATA LOADING'''
log_string('Load dataset ...')
DATA_PATH = '/scratch/zczqlzh/Dataset/ModelNet/ModelNet40_normal_resampled/modelnet40_normal_resampled'

train_dataset = ModelNetDataLoader(root=DATA_PATH, args=args, split='train')
test_dataset = ModelNetDataLoader(root=DATA_PATH, args=args, split='test')

trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True,
                                              num_workers=args.num_workers)
testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False,
                                             num_workers=args.num_workers)

log_string('The number of training data is: %d' % len(train_dataset))
log_string('The number of test data is: %d' % len(test_dataset))

seed = 3
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

'''MODEL LOADING'''
import torch.backends.cudnn as cudnn

# device = 'cuda'


num_class = 40
classifier = PointConvDensityClsSsg(num_class).cuda()
# if device == 'cuda':
#     classifier = torch.nn.DataParallel(classifier)
#     cudnn.benchmark = True

if args.pretrain is not None:
    #     print('Use pretrain model...')
    log_string('Use pretrain model')
    checkpoint = torch.load(args.pretrain)
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
else:
    print('No existing model, starting training from scratch...')
    start_epoch = 0

if args.optimizer == 'SGD':
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
elif args.optimizer == 'Adam':
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
global_epoch = 0
global_step = 0
best_tst_accuracy = 0.0
best_instance_acc = 0.0
best_class_acc = 0.0
blue = lambda x: '\033[94m' + x + '\033[0m'

'''TRANING'''
log_string('Start training...')
train_loss, train_acc, train_class_acc, val_loss, val_acc, val_class_acc, lr_history = [], [], [], [], [], [], []
for epoch in range(start_epoch, args.epoch):
    #     print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
    log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
    #     mean_correct = []
    total_loss = 0
    train_pred = []
    train_true = []

    scheduler.step()
    lr = scheduler.get_last_lr()[0]
    # print learning rate
    print("Epoch:", epoch + 1, "LR:", lr)
    lr_history.append(lr)
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        points, target = data
        points = points.data.numpy()
        jittered_data = random_scale_point_cloud(points[:, :, 0:3], scale_low=2.0 / 3, scale_high=3 / 2.0)
        jittered_data = shift_point_cloud(jittered_data, shift_range=0.2)
        points[:, :, 0:3] = jittered_data
        points = random_point_dropout_v2(points)
        shuffle_points(points)
        points = torch.Tensor(points)
        #         target = target[:, 0]

        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()

        classifier = classifier.train()
        pred = classifier(points)

        #         pred = classifier(points[:, :3, :], points[:, 3:, :])
        loss = F.nll_loss(pred, target.long())
        pred_choice = pred.data.max(1)[1]

        train_true.append(target.cpu().numpy())
        train_pred.append(pred_choice.detach().cpu().numpy())

        #         correct = pred_choice.eq(target.long().data).cpu().sum()
        #         mean_correct.append(correct.item() / float(points.size()[0]))
        loss.backward()
        optimizer.step()
        global_step += 1

        total_loss += loss.item()

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)

    train_avg_loss = total_loss / len(trainDataLoader)
    log_string('Train Instance Loss: %f' % train_avg_loss)
    #     train_instance_acc = np.mean(mean_correct)
    train_instance_acc = accuracy_score(train_true, train_pred)
    #     log_string('Train Instance Accuracy: %f' % train_instance_acc)
    train_class_accuracy = balanced_accuracy_score(train_true, train_pred)
    #     log_string('Train Class Accuracy: %f' % train_class_accuracy)
    log_string('Train Instance Accuracy: %f, Class Accuracy: %f' % (train_instance_acc, train_class_accuracy))

    train_loss.append(train_avg_loss)
    #     print(f"Train Instance Loss: {train_loss[0]}")
    train_acc.append(train_instance_acc)
    #     print(f"train_acc:{train_acc}")
    train_class_acc.append(train_class_accuracy)

    #     train_acc = np.mean(mean_correct)
    #     print('Train Accuracy: %f' % train_acc)
    #     logger.info('Train Accuracy: %f' % train_acc)

    val_avg_loss, instance_acc, class_acc = test(classifier, testDataLoader)

    if (instance_acc >= best_instance_acc):
        best_instance_acc = instance_acc

    if (class_acc >= best_class_acc):
        best_class_acc = class_acc
    log_string('Test Instance Loss: %f' % val_avg_loss)
    log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    log_string('Best Instance Accuracy: %f, Best Class Accuracy: %f' % (best_instance_acc, best_class_acc))

    val_loss.append(val_avg_loss)
    val_acc.append(instance_acc)
    val_class_acc.append(class_acc)

    global_epoch += 1
# print('Best Accuracy: %f'%best_tst_accuracy)

log_string('End of training...')

# 400 epochs
all_preds = []
all_labels = []
with torch.no_grad():
    #     val_avg_loss, instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=num_class)
    for j, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred = classifier(points)
        # softmax计算后的值选出最大的一个
        pred_choice = pred.data.max(1)[1]

        all_preds += list(pred_choice.cpu().numpy())
        all_labels += list(target.cpu().numpy())



target_names = list(train_dataset.classes.keys())
print(classification_report(all_labels, all_preds, target_names=target_names))