import numpy as np
from tqdm import tqdm
import datetime
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
from PointMLP.segmentation.provider import *


class PointMLP(nn.Module):
    def __init__(self, num_classes=50,points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2],
                 gmp_dim=64,cls_dim=64, **kwargs):
        super(PointMLP, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_classes
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        en_dims = [last_channel]
        ### Building Encoder #####
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)
            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                             res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel
            en_dims.append(last_channel)


        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) ==len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], groups=groups, res_expansion=res_expansion,
                                           bias=bias, activation=activation)
            )

        self.act = get_activation(activation)

        # class label mapping
        self.cls_map = nn.Sequential(
            ConvBNReLU1D(16, cls_dim, bias=bias, activation=activation),
            ConvBNReLU1D(cls_dim, cls_dim, bias=bias, activation=activation)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(ConvBNReLU1D(en_dim, gmp_dim, bias=bias, activation=activation))
        self.gmp_map_end = ConvBNReLU1D(gmp_dim*len(en_dims), gmp_dim, bias=bias, activation=activation)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):
        xyz = x.permute(0, 2, 1)
#         x = torch.cat([x,norm_plt],dim=1)
        x = self.embedding(x)  # B,D,N

        xyz_list = [xyz]  # [B, N, 3]
        x_list = [x]  # [B, D, N]

        # here is the encoder
        for i in range(self.stages):
            # Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # [b,g,3]  [b,g,k,d]
            x = self.pre_blocks_list[i](x)  # [b,d,g]
            x = self.pos_blocks_list[i](x)  # [b,d,g]
            xyz_list.append(xyz)
            x_list.append(x)

        # here is the decoder
        xyz_list.reverse()
        x_list.reverse()
        x = x_list[0]
        for i in range(len(self.decode_list)):
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)

        # here is the global context
        gmp_list = []
        for i in range(len(x_list)):
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # [b, gmp_dim, 1]

        #here is the cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x


def pointMLP(num_classes=50, **kwargs) -> PointMLP:
    return PointMLP(num_classes=num_classes, points=2048, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="anchor",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4],
                 de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                 gmp_dim=64,cls_dim=64, **kwargs)


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
        #         points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
        #                                           Variable(norm_plt.float())
        points, label, target, norm_plt = points.float().cuda(), label.long().squeeze(1).cuda(), target.long().cuda(), \
                                          norm_plt.float().cuda()
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        #         points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
        #                                           target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        #         points, label, target, norm_plt = points.to(device), label.squeeze(1).to(device), target.to(device), norm_plt.to(device)

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
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]  # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

        # Note: We do not need to calculate per_class iou during training

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


def test_epoch(test_loader, model, epoch, num_part, num_classes):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = {}
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader),
                                                            smoothing=0.9):
        batch_size, num_point, _ = points.size()
        #         points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
        #                                           Variable(norm_plt.float())
        points, label, target, norm_plt = points.float().cuda(), label.long().squeeze(1).cuda(), target.long().cuda(), \
                                          norm_plt.float().cuda()
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        #         points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
        #                                           target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        #         points, label, target, norm_plt = points.to(device), label.squeeze(1).to(device), target.to(device), norm_plt.to(device)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(16):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[
                cat_idx]  # avg class iou across all samples

    test_avg_loss = test_loss * 1.0 / count

    metrics['accuracy'] = np.mean(accuracy)
    #     test_metrics = metrics['accuracy']

    metrics['shape_avg_iou'] = shape_ious * 1.0 / count
    #     total_per_cat_iou = metrics['shape_avg_iou']

    outstr = 'Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    printf(outstr)

    return test_avg_loss, metrics, final_total_per_cat_iou


