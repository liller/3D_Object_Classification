
import sys
import torch.backends.cudnn as cudnn
import argparse
import logging
from PointTransformer.segmentation.models.PointTransformer import *
from PointTransformer.segmentation.provider import *
from PointTransformer.data_util.ShapeNetDataloader import *
from PointTransformer.utils import *


# 传入命令行参数
sys.argv = ['--model', '--batchsize', '32','--epoch', '200', '--gpu', '3']
args = parse_args()
torch.cuda.set_device(int(args.gpu))

torch.cuda.set_device(int(args.gpu))




classifier = PointTransformerSeg(args).cuda()
criterion = torch.nn.CrossEntropyLoss()


start_epoch = 0
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

LEARNING_RATE_CLIP = 1e-5
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = 20

best_acc = 0
global_epoch = 0
best_class_avg_iou = 0
best_inctance_avg_iou = 0

train_loss, train_acc, val_loss, val_acc, val_iou, val_class_iou, lr_history = [], [], [], [], [], [], []

for epoch in range(start_epoch, args.epoch):
    mean_correct = []
    total_loss = 0

    log_string('\nEpoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
    '''Adjust learning rate and BN momentum'''
    lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
    #     lr = scheduler.get_last_lr()[0]
    # print learning rate
    lr_history.append(lr)

    log_string('Learning rate:%f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    print('BN momentum updated to: %f' % momentum)
    classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
    classifier = classifier.train()

    '''learning one epoch'''
    for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
        optimizer.zero_grad()

        points = points.data.numpy()
        points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
        points = torch.Tensor(points)
        points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
        #         points = points.transpose(2, 1)

        seg_pred = classifier(
            torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]

        #         #to_categorical(label, num_classes)将大类作为额外特征在upsampling的过程中concat到所有点上
        #         seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
        #         #seg_pred被reshape成了维度为[-1, num_part]的张量，这样每行就对应一个点的预测结果，列数num_part对应可能的类别数目50。
        #         seg_pred = seg_pred.contiguous().view(-1, num_part)
        #         target = target.view(-1, 1)[:, 0]
        #         pred_choice = seg_pred.data.max(1)[1]

        correct = pred_choice.eq(target.data).cpu().sum()
        mean_correct.append(correct.item() / (args.batchsize * args.num_point))
        #         loss = criterion(seg_pred, target, trans_feat)
        loss = criterion(seg_pred, target)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_avg_loss = total_loss / len(trainDataLoader)
    log_string('Train Instance Loss: %f' % train_avg_loss)
    train_instance_acc = np.mean(mean_correct)
    train_loss.append(train_avg_loss)
    #     print(f"Train Instance Loss: {train_loss[0]}")
    train_acc.append(train_instance_acc)
    log_string('Train accuracy is: %.5f' % train_instance_acc)

    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_loss_test = 0
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        # seg_classes.keys()16类  记录各个类别的IoU（Intersection over Union）和从分割标签到类别的映射关系。
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader),
                                                      smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            #             points = points.transpose(2, 1)
            seg_pred = classifier(
                torch.cat([points, to_categorical(label, num_category).repeat(1, points.shape[1], 1)], -1))
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
            #             seg_pred, trans_feat = classifier(points, to_categorical(label, num_classes))
            #             cur_pred_val = seg_pred.cpu().data.numpy()
            #             cur_pred_val_logits = cur_pred_val
            #             cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target1 = target.view(-1, 1)[:, 0]
            #             loss = criterion(seg_pred, target1, trans_feat)
            loss = criterion(seg_pred, target1)
            total_loss_test += loss.item()

            target = target.cpu().data.numpy()

            # 我们可以得到模型对当前batch中每个样本每个像素的预测类别
            for i in range(cur_batch_size):
                # 该样本的真实类别
                cat = seg_label_to_cat[target[i, 0]]
                # 当前预测结果（cur_pred_val_logits）中取出第i个样本的logits，原始值
                logits = cur_pred_val_logits[i, :, :]
                # 计算了模型对第i个样本每个像素的预测类别
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            # 计算预测结果与实际目标匹配的像素数量，然后累计到总的正确数量中
            correct = np.sum(cur_pred_val == target)
            # 预测正确的像素数量
            total_correct += correct
            # 当前批次中的像素总数累加
            total_seen += (cur_batch_size * NUM_POINT)

            # 算每个类别（50）的预测准确度
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

            # 表示预测和目标之间的交集与并集的比值。
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :]
                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                for l in seg_classes[cat]:
                    if (np.sum(segl == l) == 0) and (
                            np.sum(segp == l) == 0):  # part is not present, no prediction as well
                        part_ious[l - seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                            np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))

        val_avg_loss = total_loss_test / len(testDataLoader)

        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])

        mean_shape_ious = np.mean(list(shape_ious.values()))
        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_accuracy'] = np.mean(
            np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        for cat in sorted(shape_ious.keys()):
            log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        val_loss.append(val_avg_loss)
        val_acc.append(test_metrics['accuracy'])
        val_iou.append(test_metrics['inctance_avg_iou'])
        #         print(f"val_acc{val_acc}")
        val_class_iou.append(test_metrics['class_avg_iou'])

    log_string('\nTrain Instance Loss: %f' % val_avg_loss)
    log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
        epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
    #     if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
    #         logger.info('Save model...')
    #         savepath = str(checkpoints_dir) + '/best_model.pth'
    #         log_string('Saving at %s' % savepath)
    #         state = {
    #             'epoch': epoch,
    #             'train_acc': train_instance_acc,
    #             'test_acc': test_metrics['accuracy'],
    #             'class_avg_iou': test_metrics['class_avg_iou'],
    #             'inctance_avg_iou': test_metrics['inctance_avg_iou'],
    #             'model_state_dict': classifier.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #         }
    #         torch.save(state, savepath)
    #         log_string('Saving model....')

    if test_metrics['accuracy'] > best_acc:
        best_acc = test_metrics['accuracy']
    if test_metrics['class_avg_iou'] > best_class_avg_iou:
        best_class_avg_iou = test_metrics['class_avg_iou']
    if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
        best_inctance_avg_iou = test_metrics['inctance_avg_iou']
    log_string('Best accuracy is: %.5f' % best_acc)
    log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
    log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
    global_epoch += 1

