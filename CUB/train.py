"""
Train InceptionV3 Network using the CUB-200-2011 dataset
"""
import pdb
import os
import sys
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np
from analysis import Logger, AverageMeter, accuracy, binary_accuracy

from CUB import probe, tti, gen_cub_synthetic, hyperopt
from CUB.dataset import load_data, find_class_imbalance
from CUB.config import BASE_DIR, N_CLASSES, N_ATTRIBUTES, UPWEIGHT_RATIO, MIN_LR, LR_DECAY_SIZE
from CUB.models import ModelXtoCY, ModelXtoChat_ChatToY, ModelXtoY, ModelXtoC, ModelOracleCtoY, ModelXtoCtoY


def run_epoch_simple(model, optimizer, loader, loss_meter, acc_meter, criterion, args, is_training):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """
    if is_training:
        model.train()
    else:
        model.eval()
    for _, data in enumerate(loader):
        inputs, labels = data
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]
            inputs = torch.stack(inputs).t().float()
        inputs = torch.flatten(inputs, start_dim=1).float()
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var
        
        outputs = model(inputs_var)
        loss = criterion(outputs, labels_var)
        acc = accuracy(outputs, labels, topk=(1,))
        loss_meter.update(loss.item(), inputs.size(0))
        acc_meter.update(acc[0], inputs.size(0))

        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters
    return loss_meter, acc_meter

def run_epoch(model, optimizer, loader, loss_meter, acc_meter, criterion, attr_criterion, args, is_training):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """
    if is_training:
        model.train()
    else:
        model.eval()

    for _, data in enumerate(loader):
        if attr_criterion is None:
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            inputs, labels, attr_labels = data
            if args.n_attributes > 1:
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()#.float() #N x 312
            else:
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        if is_training and args.use_aux:
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck: #loss main is for the main task label (always the first output)
                loss_main = 1.0 * criterion(outputs[0], labels_var) + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * (1.0 * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]) \
                                                            + 0.4 * attr_criterion[i](aux_outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i])))
        else: #testing or no aux logits
            outputs = model(inputs_var)
            losses = []
            out_start = 0
            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1
            if attr_criterion is not None and args.attr_loss_weight > 0: #X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    losses.append(args.attr_loss_weight * attr_criterion[i](outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor), attr_labels_var[:, i]))

        if args.bottleneck: #attribute accuracy
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            acc = accuracy(outputs[0], labels, topk=(1,)) #only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        if attr_criterion is not None:
            if args.bottleneck:
                total_loss = sum(losses)/ args.n_attributes
            else: #cotraining, loss by class prediction and loss by attribute prediction have the same weight
                total_loss = losses[0] + sum(losses[1:])
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: #finetune
            total_loss = sum(losses)
        loss_meter.update(total_loss.item(), inputs.size(0))
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    return loss_meter, acc_meter

# 主训练函数
# model 已构造好，args 是 Namespace
def train(model, args):
    # === 准备阶段 ===

    # imbalance: list[float]
    imbalance = None

    if args.use_attr and not args.no_img and args.weighted_loss:
        # 训练数据路径
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')

        if args.weighted_loss == 'multiple':
            # list[float]，len=112，每个 attr 一个独立 ratio
            imbalance = find_class_imbalance(train_data_path, True)
        else:
            # "single" / 其他非空字符串
            # list[float]，len=112，但 112 个值都相同
            imbalance = find_class_imbalance(train_data_path, False)

    if os.path.exists(args.log_dir): # job restarted by cluster
        # 如果存在旧的日志目录
        for f in os.listdir(args.log_dir):
            # 将里面的每个文件都删除
            os.remove(os.path.join(args.log_dir, f))
    else:
        # 递归新建日志目录
        os.makedirs(args.log_dir)

    # Logger 对象，内部 open(log.txt, 'w') 覆盖式写
    logger = Logger(os.path.join(args.log_dir, 'log.txt'))

    logger.write('\n' + "[args]=========" + '\n')
    logger.write(str(args) + '\n')
    logger.write('\n' + "[imbalance]=========" + '\n')
    logger.write(str(imbalance) + '\n')

    # 强制让操作系统将 buffer 写到 磁盘，防止 crash 丢 log
    logger.flush()

    # 绑定 model 到 GPU
    model = model.cuda()

    # 主任务 y 的损失（对 logits <-> class_label 算 softmax-CE）
    criterion = torch.nn.CrossEntropyLoss()

    # 用 attr 且 用图，准备 attr 损失列表
    if args.use_attr and not args.no_img:
        # list[Loss]，长度 = n_attributes
        attr_criterion = [] #separate criterion (loss function) for each attribute
        
        if args.weighted_loss:
            # 加权 BCE 分支
            assert(imbalance is not None)
            for ratio in imbalance:
                # 正样本权重 = ratio
                attr_criterion.append(
                    torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
                )
        else:
            # 不加权分支
            for i in range(args.n_attributes):  # 112
                # 注意这里是 CE 不是 BCE，原作者设计
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    # run_epoch_simple 不读 attr_criterion 这个变量
    else:
        attr_criterion = None

    if args.optimizer == 'Adam':
        # Adam 优化器对象，内部维护（β1, β2, ε, m, v）状态
        # Adam 不接 momentum 参数（用内部 β1/β2 替代）
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )
    else:
        # SGD
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            momentum=0.9, 
            weight_decay=args.weight_decay
        )

    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, threshold=0.00001, min_lr=0.00001, eps=1e-08)
    
    # scheduler 对象，内部维护 _step_count（每次 .step() 自增）
    # 当 _step_count 是 step_size 的倍数时，lr *= gamma
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, 
        step_size=args.scheduler_step, 
        gamma=0.1
    )

    # MIN_LR = 1e-4，LR_DECAY_SIZE = 0.1，都在 CUB/config.py
    # LR_DECAY_SIZE 即 gamma
    # 数学含义：lr 衰减到 MIN_LR 需要 log_{gamma}(MIN_LR / args.lr) 这么多次 乘 gamma 
    # stop_epoch：到了这个 epoch 就不再调用 scheduler.step()
    stop_epoch = int( math.log(MIN_LR / args.lr) / math.log(LR_DECAY_SIZE) ) * args.scheduler_step

    print("\n[Stop epoch]=========")
    print("Stop epoch: ", stop_epoch)
    print("\n")

    # 训练数据集 路径
    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    # 验证数据集 路径
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    logger.write('\n' + "[train data path]=========" + '\n')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: #retraining
        train_loader = load_data(
            [train_data_path, val_data_path], 
            args.use_attr, 
            args.no_img, 
            args.batch_size, 
            args.uncertain_labels, 
            image_dir=args.image_dir, 
            n_class_attr=args.n_class_attr, 
            resampling=args.resampling
        )
        val_loader = None
    else:
        train_loader = load_data([train_data_path], args.use_attr, args.no_img, args.batch_size, args.uncertain_labels, image_dir=args.image_dir, \
                                 n_class_attr=args.n_class_attr, resampling=args.resampling)
        val_loader = load_data([val_data_path], args.use_attr, args.no_img, args.batch_size, image_dir=args.image_dir, n_class_attr=args.n_class_attr)

    best_val_epoch = -1
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in range(0, args.epochs):
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()
        if args.no_img:
            train_loss_meter, train_acc_meter = run_epoch_simple(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, args, is_training=True)
        else:
            train_loss_meter, train_acc_meter = run_epoch(model, optimizer, train_loader, train_loss_meter, train_acc_meter, criterion, attr_criterion, args, is_training=True)
 
        if not args.ckpt: # evaluate on val set
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, args, is_training=False)
                else:
                    val_loss_meter, val_acc_meter = run_epoch(model, optimizer, val_loader, val_loss_meter, val_acc_meter, criterion, attr_criterion, args, is_training=False)

        else: #retraining
            val_loss_meter = train_loss_meter
            val_acc_meter = train_acc_meter

        if best_val_acc < val_acc_meter.avg:
            best_val_epoch = epoch
            best_val_acc = val_acc_meter.avg
            logger.write('\n' + "[New model best model]=========" + '\n')
            logger.write('New model best model at epoch %d\n' % epoch)
            torch.save(model, os.path.join(args.log_dir, 'best_model_%d.pth' % args.seed))
            #if best_val_acc >= 100: #in the case of retraining, stop when the model reaches 100% accuracy on both train + val sets
            #    break

        train_loss_avg = train_loss_meter.avg
        val_loss_avg = val_loss_meter.avg
        
        logger.write('\n' + "[train info]=========" + '\n')
        logger.write('Epoch [%d]:\tTrain loss: %.4f\tTrain accuracy: %.4f\t'
                'Val loss: %.4f\tVal acc: %.4f\t'
                'Best val epoch: %d\n'
                % (epoch, train_loss_avg, train_acc_meter.avg, val_loss_avg, val_acc_meter.avg, best_val_epoch)) 
        logger.flush()
        
        if epoch <= stop_epoch:
            # scheduler.step(epoch) #scheduler step to update lr at the end of epoch     
            scheduler.step()
        #inspect lr
        if epoch % 10 == 0:
            print("\n[Current lr]=========")
            # print('Current lr:', scheduler.get_lr())
            print('Current lr:', scheduler.get_last_lr())
            print("\n")

        # if epoch % args.save_step == 0:
        #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        # 两种早停
        if epoch >= 100 and val_acc_meter.avg < 3:
            # 训了 100 epoch acc 仍然低的离谱
            print("Early stopping because of low accuracy")
            break
        if epoch - best_val_epoch >= 100:
            # 距离上次刷新 best 已经过了大于 100 个 epoch
            print("Early stopping because acc hasn't improved for a long time")
            break

def train_X_to_C(args):
    model = ModelXtoC(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                      n_attributes=args.n_attributes, expand_dim=args.expand_dim, three_class=args.three_class)
    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                            num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_Chat_to_y_and_test_on_Chat(args):
    model = ModelXtoChat_ChatToY(n_class_attr=args.n_class_attr, n_attributes=args.n_attributes,
                                 num_classes=N_CLASSES, expand_dim=args.expand_dim)
    train(model, args)

def train_X_to_C_to_y(args):
    model = ModelXtoCtoY(n_class_attr=args.n_class_attr, pretrained=args.pretrained, freeze=args.freeze,
                         num_classes=N_CLASSES, use_aux=args.use_aux, n_attributes=args.n_attributes,
                         expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
    train(model, args)

def train_X_to_y(args):
    model = ModelXtoY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux)
    train(model, args)

def train_X_to_Cy(args):
    model = ModelXtoCY(pretrained=args.pretrained, freeze=args.freeze, num_classes=N_CLASSES, use_aux=args.use_aux,
                       n_attributes=args.n_attributes, three_class=args.three_class, connect_CY=args.connect_CY)
    train(model, args)

def train_probe(args):
    probe.run(args)

def test_time_intervention(args):
    tti.run(args)

def robustness(args):
    gen_cub_synthetic.run(args)

def hyperparameter_optimization(args):
    hyperopt.run(args)

# ======================================

def parse_arguments(experiment):
    # 创建一个 argparse 解析器对象
    # Get argparse configs from user
    parser = argparse.ArgumentParser(description='CUB Training')

    # 声明 positional argument
    # positional argument：意味着 argparse 会从 sys.argv 的第一个非可选位置（也就是 sys.argv[1]）读取
    parser.add_argument(
        'dataset', 
        type=str, 
        help='Name of the dataset.'
    )

    # 声明 positional argument
    # 必须在 choices 的列表里
    parser.add_argument(
        'exp', 
        type=str, 
        choices=[
            'Concept_XtoC', 
            'Independent_CtoY', 
            'Sequential_CtoY', 
            'Standard', 
            'Multitask', 
            'Joint', 
            'Probe', 
            'TTI', 
            'Robustness', 
            'HyperparameterSearch'
        ], 
        help='Name of experiment to run.'
    )

    # 声明 optional argument
    parser.add_argument(
        '--seed', 
        required=True, 
        type=int, 
        help='Numpy and torch seed.'
    )

    if experiment == 'Probe':
        # 把 args 包成 1-tuple (args,) 返回
        return (probe.parse_arguments(parser),)

    elif experiment == 'TTI':
        return (tti.parse_arguments(parser),)

    elif experiment == 'Robustness':
        return (gen_cub_synthetic.parse_arguments(parser),)

    elif experiment == 'HyperparameterSearch':
        return (hyperopt.parse_arguments(parser),)

    else:
        # 通用训练分支

        # 指定保存 checkpoint / log 的目录
        parser.add_argument(
            '-log_dir', 
            default=None, 
            help='where the trained model is saved'
        )

        # batch size
        parser.add_argument(
            '-batch_size', 
            '-b', 
            type=int, 
            help='mini-batch size'
        )

        # 训练总 epoch 数
        parser.add_argument(
            '-epochs', 
            '-e', 
            type=int, 
            help='epochs for training process'
        )

        # 每隔多少 epoch 保存一次 checkpoint
        parser.add_argument(
            '-save_step', 
            default=1000, 
            type=int, 
            help='number of epochs to save model'
        )

        # 优化器学习率
        parser.add_argument(
            '-lr', 
            type=float, 
            help="learning rate"
        )

        # 优化器 L2 正则系数
        parser.add_argument(
            '-weight_decay', 
            type=float, 
            default=5e-5, 
            help='weight decay for optimizer'
        )

        # 是否加载 ImageNet 预训练的 Inception V3 权重
        # 默认使用预训练权重
        parser.add_argument(
            '-pretrained', 
            '-p', 
            action='store_true', 
            help='whether to load pretrained model & just fine-tune'
        )

        # 是否冻结 Inception 底层卷积（只 finetune 最后的 fc 层）
        # paper 不冻结，全网络训？
        parser.add_argument(
            '-freeze', 
            action='store_true', 
            help='whether to freeze the bottom part of inception network'
        )

        # 是否使用 Inception V3 的辅助分类头（auxiliary classifier，从中间层引出的小 head）
        # paper 训练时开了，loss = main_loss + 0.4 * aux_loss （template_model.py 的标准做法）
        # 能稍微提升收敛速度，对 deep network 中间层提供额外梯度信号
        parser.add_argument(
            '-use_aux', 
            action='store_true', 
            help='whether to use aux logits'
        )

        # 是否使用 attribute 标签（C）作 supervision
        # 所有 CBM 系列（Concept_XtoC / Independent / Sequential / Joint / Multitask）都需要
        # Standard baseline 不传（因为 Standard 只用 y label，不用 C label）
        parser.add_argument(
            '-use_attr', 
            action='store_true', 
            help='whether to use attributes (FOR COTRAINING ARCHITECTURE ONLY)'
        )

        # attribute 预测 loss 的权重 λ
        # Joint 模型用：total_loss = L_y + λ * L_c
        # paper Fig 2 做了 λ ∈ {0.001, 0.01, 0.1, 1} 的 sweep，对比 acc vs Interpretability 的 trade-off
        # Concept_XtoC 模式下因为只有 attribute loss （没 L_y），λ 无影响（也常传 1.0）
        parser.add_argument(
            '-attr_loss_weight', 
            default=1.0, 
            type=float, 
            help='weight for loss by predicting attributes'
        )

        # 是否完全不用图片，只用 attribute 预测类别
        # 给 Independent_CtoY / Sequential_CtoY 用
        # 这两种模式的 C -> Y 部分只是个 MLP ，不用 Inception
        # 传了之后 dataloader 不加载图片（直接走 attribute_label）
        # Concept_XtoC / Joint / Standard 必须用图片，不能传 -no_img
        parser.add_argument(
            '-no_img', 
            action='store_true', 
            help='if included, only use attributes (and not raw imgs) for class prediction'
        )

        # 模型是否走 “ 先 X -> C 再 C -> Y ” 的 bottleneck 结构
        # Concept_XtoC（只训 X -> C 那段）需要 -bottleneck，这时模型只有 attribute head。没有 class head
        # Independent_CtoY / Sequential_CtoY 也用 -bottleneck 
        # Joint / Standard / Multitask 不传，因为它们有 class head 直接输出 y
        parser.add_argument(
            '-bottleneck', 
            help='whether to predict attributes before class labels', 
            action='store_true'
        )

        # 是否对 attribute loss 用 class imbalance weight
        # 可选值：'' / 'single' / 'multiple'
        # '' = 不加权（default）
        # 'single' = 用单一 imbalance ratio 给所有 attribute
        # 'multiple' = 每个 attribute 一个 ratio
        parser.add_argument(
            '-weighted_loss', 
            default='', # note: may need to reduce lr，否则梯度尺度变大导致不稳
            help='Whether to use weighted loss for single attribute or multiple ones'
        )

        # 是否用 CUB 的 attribute_certainty 信息做 soft label（normalized 到 [0, 1]）
        # CUB 每个 attribute 标注有个 1-4 的 certainty 等级：
        # 1=not visible (鸟身体被遮挡看不到那个部位), 2=guessing, 3=probably, 4=definitely
        # 传了的话，attribute label 不再是 0 / 1，而是按 certainty 加权的 0 / 0.25 / 0.5 / 0.75 / 1 这种连续值
        # paper 默认不用
        parser.add_argument(
            '-uncertain_labels', 
            action='store_true', 
            help='whether to use (normalized) attribute certainties as labels'
        )

        # 用多少个 attribute 作 bottleneck
        parser.add_argument(
            '-n_attributes', 
            type=int, 
            default=N_ATTRIBUTES, 
            help='whether to apply bottlenecks to only a few attributes'
        )

        # 在 C -> Y MLP 中间加一个隐藏层的维度
        # 0 表示不加隐藏层（直接 linear，n_attr -> 200）
        # 非 0 则插入一个 n_attr -> expand_dim -> 200 的 2 层 MLP（带 ReLU）
        # paper 默认 0
        parser.add_argument(
            '-expand_dim', 
            type=int, 
            default=0, 
            help='dimension of hidden layer (if we want to increase model capacity) - for bottleneck only'
        )

        # attribute 预测是 binary（2 类：0 / 1）还是 ternary（3 类：0 / 1 / not visible）
        # 传 2：把 not visible 合并到 0（标准做法）
        # 传 3：把 not visible 单独算一类（更细的建模，但训练样本里 not visible 比例小，效果不一定好）
        # paper 默认 2
        parser.add_argument(
            '-n_class_attr', 
            type=int, 
            default=2, 
            help='whether attr prediction is a binary or triary classification'
        )

        # 训练数据（pkl 文件）所在目录
        # 里面要有 train.pkl / val.pkl / test.pkl 三个文件
        parser.add_argument(
            '-data_dir', 
            default='official_datasets', 
            help='directory to the training data'
        )

        # 图片文件夹名（dataset.py 用它来重写 pkl 里的图片路径）
        # normal 训练用 default 'images'（CUB_200_2011/images/ 那个文件夹）
        # Robustness 实验时换成 'AdversarialData/CUB_fixed/train' 走对抗数据
        # 具体替换逻辑在 dataset.py
        parser.add_argument(
            '-image_dir', 
            default='images', 
            help='test image folder to run inference on'
        )

        # 是否在 dataloader 里用 ImbalanceDatasetSampler 做样本重采样
        # （这个 flag 没啥用）
        parser.add_argument(
            '-resampling', 
            help='Whether to use resampling', 
            action='store_true'
        )

        # 是否走 Joint（X -> A -> Y 完整 end-to-end）训练
        # Joint 实验需要传 '-end2end'，模型会用 End2EndModel(Inception, MLP) 包装
        # 注意：Joint 训练命令跟 Multitask 几乎一样，只多 '-end2end' 这一个 flag
        parser.add_argument(
            '-end2end', 
            action='store_true', 
            help='Whether to train X -> A -> Y end to end. Train cmd is the same as cotraining + this arg'
        )
        
        # 优化器类型，可选 SGD / RMSProp / Adam
        parser.add_argument(
            '-optimizer', 
            default='SGD', 
            help='Type of optimizer to use, options incl SGD, RMSProp, Adam'
        )

        # 是否走 retraining 模式（在 train + val 合并集上重训，不留 val 评估）
        # 空字符串：普通 train（保留 val 作 early stopping / model selection）
        # 非空：retraining
        parser.add_argument(
            '-ckpt', 
            default='', 
            help='For retraining on both train + val set'
        )

        # 每多少个 epoch 把 lr 乘 0.1
        # paper 用 1000 或 20
        parser.add_argument(
            '-scheduler_step', 
            type=int, 
            default=1000, 
            help='Number of steps before decaying current learning rate by half'
        )

        # 是否对 total_loss 做归一化
        # 传了之后：total_loss = total_loss / ( 1 + args.attr_loss_weight * args.n_attributes )
        # 相当于让总 loss 跟单一 cross-entropy 在同一量级，方便跨 λ 比较
        # paper Joint 实验都用
        parser.add_argument(
            '-normalize_loss', 
            action='store_true', 
            help='Whether to normalize loss by taking attr_loss_weight into account'
        )

        # 在 C -> Y 之前是否对 ĉ logits 加 ReLU 激活
        # 即，给 Joint / Bottleneck 模型用，X -> C 输出是 raw logit，喂给 C -> Y MLP 之前做不做激活
        # paper 不用
        parser.add_argument(
            '-use_relu', 
            action='store_true', 
            help='Whether to include relu activation before using attributes to predict Y. '
                 'For end2end & bottleneck model'
        )
        
        # 在 C -> Y 之前是否对 ĉ logits 加 Sigmoid 激活
        # paper Joint 实验有“with sigmoid” 和 “without sigmoid”两个变体（Table 1 / Fig 2）
        # 用 sigmoid 后 ĉ 是 [0, 1] 的概率值，更接近“概念是否存在”的语义
        # 不用 sigmoid 则 ĉ 是 raw logit，更接近“概念的连续置信度”
        # 两者各有利弊，paper 比较过
        parser.add_argument(
            '-use_sigmoid', 
            action='store_true', 
            help='Whether to include sigmoid activation before using attributes to predict Y. '
                 'For end2end & bottleneck model'
        )

        # Multitask 模式下是否把 C 也接入 Y 的预测路径
        # 不传：纯 Multitask，X -> backbone -> (C_head, Y_head)，两个 head 完全并行，Y 不依赖 C
        # 传了：X -> backbone -> (C_head, Y_head)，且 Y_head 同时接 backbone feature 和 C_head 输出
        # 传了之后，Multitask 在结构上更接近 Joint 的混合体
        # paper 默认不传
        parser.add_argument(
            '-connect_CY', 
            action='store_true', 
            help='Whether to use concepts as auxiliary features (in multitasking) to predict Y'
        )
        
        # 把所有传入的 flag 解析成一个 Namespace 对象 args
        args = parser.parse_args()
        
        # 动态加一个属性，布尔值
        # 决定 attribute 预测头是 binary 还是 ternary 分类
        # 这是 argparse 之后的“后处理”
        args.three_class = (args.n_class_attr == 3)
        
        # 返回一个 1-tuple 包裹 args Namespace
        # tuple 形式跟其他 4 个特殊分支保持一致接口
        return (args,)
