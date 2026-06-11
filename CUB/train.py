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

# =====================

# import pickle

# =====================

def run_epoch_simple(
    model,        # C -> Y 的 MLP（ModelOracleCtoY），输入概念向量，输出 200 类 logits
    optimizer, 
    loader,       # no_img=True 的 loader，每个 batch 给 (概念，类别)，没有图像
    loss_meter,   # AverageMeter，按样本数加权累积本 epoch 平均 loss
    acc_meter,    # AverageMeter，累积平均 top-1 acc
    criterion,    # nn.CrossEntropyLoss，类别分类 loss
    args, 
    is_training   # True 走训练（反向传播 + step），False 走纯验证
):
    """
    A -> Y: Predicting class labels using only attributes with MLP
    """

    # 切 训练/验证 模式
    if is_training:
        model.train()
    else:
        model.eval()

    # batch 循环
    for _, data in enumerate(loader):
        # 只解包两项：概念 + 类别（对比 run_epoch 解包三项；这里没图，概念本身就是 inputs）
        inputs, labels = data

        # ===== inputs 整形：两种来源形状统一成 [B, D] =====
        if isinstance(inputs, list):
            #inputs = [i.long() for i in inputs]

            # 触发：n_class_attr=2 (默认)
            # __getitem__ 返回 112 长的 list （0/1 概念）
            # DataLoader 默认 collate 把“list 每一位”当独立字段：
            # inputs = [
            #     tensor([0,1,...,0]),   # 第 0 个概念在 B 样本上的值，[B]
            #     tensor([1,0,...,1]),   # 第 1 个概念
            #     ...                    # 共 112 项
            # ]
            # 
            # stack → [112, B]
            # .t() → [B, 112]
            inputs = torch.stack(inputs).t().float()

        # 这条对上面的 if 结果再次 flatten 无影响
        # 
        # 这里针对 n_class_attr=3 的情况：__getitem__ 返回 [112, 3] one-hot
        # -> collate 成 [B, 112, 3] 张量 ( 非 list，跳过上面 if ) -> flatten(start_dim=1) -> [B, 336]
        inputs = torch.flatten(inputs, start_dim=1).float()

        # 搬到 GPU
        inputs_var = torch.autograd.Variable(inputs).cuda()
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels).cuda()
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        # ===== 前向 + loss + acc =====

        # MLP 直接返回单个 [B, 200] logits (对比 run_epoch 返回 list，要 outputs[0] 取类别)
        outputs = model(inputs_var)

        # outputs 经 softmax-CE，返回标量
        loss = criterion(outputs, labels_var)

        # 返回 [top1_acc]，acc[0] 是百分数张量
        acc = accuracy(outputs, labels, topk=(1,))

        # loss 转 float
        loss_meter.update(loss.item(), inputs.size(0))
        # acc 直接塞 tensor（都能跑，只是不统一）
        acc_meter.update(acc[0], inputs.size(0))

        # 训练时反向传播，验证时只统计
        if is_training:
            optimizer.zero_grad() #zero the parameter gradients
            loss.backward()
            optimizer.step() #optimizer step to update parameters

    return loss_meter, acc_meter

def run_epoch(
    model, 
    optimizer, 
    loader, 
    # AverageMeter，累积本 epoch 的平均 loss
    loss_meter, 
    # AverageMeter，累积本 epoch 的平均 acc
    acc_meter, 
    # 类别分类 loss，通常是 nn.CrossEntropyLoss
    criterion, 
    # 概念 attr loss 的 list，长度 = n_attributes；纯 finetune 时为 None
    attr_criterion, 
    args, 
    # True 走训练（反向传播 + step），False 走纯验证
    is_training
):
    """
    For the rest of the networks (X -> A, cotraining, simple finetune)
    """

    # ====================================
    # 一个 epoch 的 训练 / 验证 循环，CBM 三种模式合一：
    # ① X -> A 独立训练（bottleneck=True，模型只输出 112 个属性 logits）
    # ② Cotrainning（X -> A, Y）（bottleneck=False，模型输出 属性 + class）
    # ③ Simple finetune（X -> Y）（attr_criterion=None，没有属性监督）
    # ====================================

    # 切换 训练 / 验证 模式
    if is_training:
        model.train()
    else:
        model.eval()

    # batch 循环
    for _, data in enumerate(loader):
        # ===== 数据解包 + attr_labels 整形 =====
        if attr_criterion is None:
            # 纯 finetune 路径
            # DataLoader 只返回 (图像，类别)
            inputs, labels = data
            attr_labels, attr_labels_var = None, None
        else:
            # 带属性监督 路径
            # DataLoader 返回 (图像，类别，属性)
            inputs, labels, attr_labels = data

            if args.n_attributes > 1:
                # 例：CUB 上 n_attribute = 112
                # 
                # DataLoader 把每个属性当作一个独立字段收集
                # 所以 attr_labels 是一个长度 112 的 list
                # list 中的每项是 shape=[B] 的 tensor ：
                # 
                # attr_labels = [
                #     tensor([1, 0, 1, ..., 0]),   # 第 0 个属性在 B 个样本上的 0/1
                #     tensor([0, 1, 0, ..., 1]),   # 第 1 个属性
                #     ...
                #     tensor([1, 1, 0, ..., 0]),   # 第 111 个属性
                # ]
                # 
                # .long() 统一转 long 类型
                # stack 成一个 2 维 tensor：[112, B]
                # 转置为 [B, 112]
                attr_labels = [i.long() for i in attr_labels]
                attr_labels = torch.stack(attr_labels).t()
            else:
                # 单属性场景（极少见，主要是测试用）
                if isinstance(attr_labels, list):
                    attr_labels = attr_labels[0]
                attr_labels = attr_labels.unsqueeze(1)

            # 转 float —— BCEWithLogitsLoss 要求 target 是 float（label 一开始是 long）
            # Variable 是老 API，新 PyTorch 直接用 tensor 即可，行为完全等价
            attr_labels_var = torch.autograd.Variable(attr_labels).float()
            attr_labels_var = attr_labels_var.cuda() if torch.cuda.is_available() else attr_labels_var

        # 输入图像 / 类别标签 搬上 GPU
        inputs_var = torch.autograd.Variable(inputs)
        inputs_var = inputs_var.cuda() if torch.cuda.is_available() else inputs_var
        labels_var = torch.autograd.Variable(labels)
        labels_var = labels_var.cuda() if torch.cuda.is_available() else labels_var

        # ========== 前向传播 + loss 列表 ==========

        if is_training and args.use_aux:
            # ---------- 训练 + 用 aux 头：Inception-v3 返回 (主头, aux 头) ----------

            # 非 bottleneck (cotraining 或 finetune)：
            # [class_logits,  attr1_logit,  attr2_logit,  ...,  attr112_logit]
            #     ↑ shape=[B,200]   ↑ shape=[B,1]
            # 
            # bottleneck (X -> A 独立训练)：
            # [attr1_logit,  attr2_logit,  ...,  attr112_logit]
            #
            # 用 out_start 这个"指针"标记 "属性 logit 从第几位开始"：
            # 非 bottleneck → out_start=1（要跳过占第 0 位的类别 logit）
            #    bottleneck → out_start=0（没有类别 logit，从头就是属性）
            # 
            # 这样 attr loss 那段循环 outputs[i+out_start] 就能两种模式共用
            outputs, aux_outputs = model(inputs_var)
            losses = []
            out_start = 0

            if not args.bottleneck: # loss main is for the main task label (always the first output)
                # 类别 loss：Inception-v3 标准复合形式
                # 主头权重 1.0，aux 头权重 0.4
                # 
                # aux 头物理意义：插在 backbone 中间层的小分类器，给深层梯度多一条短路
                # 训练时一并优化，测试时丢掉 —— 起正则化 + 稳定深层训练的作用
                loss_main =   1.0 * criterion(outputs[0],     labels_var) \
                            + 0.4 * criterion(aux_outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1

            if attr_criterion is not None and args.attr_loss_weight > 0: # X -> A, cotraining, end2end
                # 属性 loss：每个属性独立算一个 BCE，外面乘 attr_loss_weight 缩放
                # 
                # 关键操作分解（以第 i 个属性为例）：
                # outputs[i+out_start]                shape [B, 1]，第 i 个属性的 logit
                # .squeeze()                          shape [B]，去掉最后那个 1
                # .type(torch.cuda.FloatTensor)       确保是 float，BCE 要 float
                # attr_criterion[i](pred, target)     这一个属性的 BCEWithLogitsLoss
                # attr_labels_var[:, i]               shape [B]，第 i 个属性的 ground truth
                # 
                # 同样是 1.0 * main + 0.4 * aux 的 Inception 复合形式
                for i in range(len(attr_criterion)):
                    i_outputs_pred     =     outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor)
                    i_aux_outputs_pred = aux_outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor)

                    i_target = attr_labels_var[:, i]

                    losses.append(
                        args.attr_loss_weight * (   1.0 * attr_criterion[i](i_outputs_pred,     i_target) \
                                                  + 0.4 * attr_criterion[i](i_aux_outputs_pred, i_target)   )
                    )
        else: # testing or no aux logits
            # ---------- 验证 / 不用 aux：单输出路径 ----------
            # 
            # 走到这里有两种触发：
            # (a) is_training=False（验证时永远不用 aux）
            # (b) is_training=True 但 args.use_aux=False（用户主动关 aux）
            # 
            # 模型只返回 outputs 一个 list（结构同上面 outputs），loss 公式去掉 aux 那一项
            outputs = model(inputs_var)
            losses = []
            out_start = 0

            if not args.bottleneck:
                loss_main = criterion(outputs[0], labels_var)
                losses.append(loss_main)
                out_start = 1

            if attr_criterion is not None and args.attr_loss_weight > 0: # X -> A, cotraining, end2end
                for i in range(len(attr_criterion)):
                    i_outputs_pred = outputs[i+out_start].squeeze().type(torch.cuda.FloatTensor)

                    i_target = attr_labels_var[:, i]

                    losses.append(
                        args.attr_loss_weight * attr_criterion[i](i_outputs_pred, i_target)
                    )

        # ========== 准确率统计（两种口径） ==========

        # 是否独立训 概念瓶颈
        # X -> A
        if args.bottleneck: # attribute accuracy
            # bottleneck 模式（X→A 独立）：没有类别预测可比，只能算"属性预测对了几个"
            # 
            # outputs 是 [attr1_logit, attr2_logit, ..., attr112_logit]，每个 shape=[B,1]
            # 
            # torch.cat(dim=1)         沿属性维拼成单张大矩阵：shape [B, 112]
            # nn.Sigmoid()             每个位置压到 (0,1)，解释为 P(该属性=1)
            # binary_accuracy(p, gt)   阈值 0.5 二值化后和 attr_labels 比，返回平均正确率
            # 
            # 例：B=4，n_attr=112，sigmoid 后阈值化得到 [4,112] 的 0/1，
            # 和 ground truth [4,112] 逐位置比，4*112=448 个位置算正确率
            sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
            acc = binary_accuracy(sigmoid_outputs, attr_labels)
            acc_meter.update(acc.data.cpu().numpy(), inputs.size(0))
        else:
            # 非 bottleneck 模式（cotraining 或 finetune）：直接算类别 top-1
            # 
            # outputs[0] 是 class_logits，shape [B, 200]（CUB 200 类）
            # accuracy(..., topk=(1,)) 返回 [top1_acc]，取第 0 项
            # 
            # 注意：cotraining 模式下 attribute 预测的准不准这里不统计，只关心类别
            acc = accuracy(outputs[0], labels, topk=(1,)) # only care about class prediction accuracy
            acc_meter.update(acc[0], inputs.size(0))

        # ========== 总 loss 合成（3 个分支） ==========

        # 是否需要 概念损失
        if attr_criterion is not None:
            if args.bottleneck:
                # X→A 模式：losses 里全是 112 个属性 loss，没有类别 loss
                # 
                # 除以 n_attributes 相当于 求平均 而不是 求和
                # 这样属性数变化时（比如换数据集，n_attr 从 112 变 28）
                # 总 loss 量级不会跟着跳变，超参（lr / weight_decay）不用重新调
                total_loss = sum(losses) / args.n_attributes
            else: # cotraining, loss by class prediction and loss by attribute prediction have the same weight
                # Cotraining 模式：losses[0] 是类别 loss，losses[1:] 是 112 个属性 loss
                # 
                # 不平均，直接相加：
                # 类别 loss 量级 ~ O(1)（CrossEntropy 在 200 类上初始 ~log(200)≈5.3）
                # 属性 loss 已经被 attr_loss_weight 外乘缩放过了
                # 
                # 类别和属性这里 同权 进入 total —— 加权关系全靠 attr_loss_weight 控制
                total_loss = losses[0] + sum(losses[1:])

                # 可选的整体归一化：防止 "加更多属性 → 总 loss 无脑变大 → lr 要重调"
                # 
                # 例：attr_loss_weight=0.4，n_attributes=112
                # 分母 = 1 + 0.4 * 112 = 45.8
                # 把 total 拉回 ~O(1) 量级
                # 
                # 分母的形式 = (类别 loss 系数 1) + (单个 attr loss 系数 λ) * (属性数)
                # 物理意义：把总 loss 除以"理论上各 loss 项系数之和"做权重归一化
                if args.normalize_loss:
                    total_loss = total_loss / (1 + args.attr_loss_weight * args.n_attributes)
        else: # finetune
            # Finetune 模式：只有一个类别 loss，sum 单项等于自身
            total_loss = sum(losses)

        # 累积本 batch 的 loss 到 meter（按样本数加权求平均）
        loss_meter.update(total_loss.item(), inputs.size(0))

        # 训练时走"清空旧梯度 → 反传新梯度 → 更新参数"三步固定套路
        # 验证时跳过，只统计 loss / acc
        if is_training:
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

    return loss_meter, acc_meter

# 主训练函数
# model 已构造好，args 是 Namespace
def train(model, args):
    # === 准备阶段 ===

    imbalance = None

    # ===================================================

    # imbalance 在整个 CBM 训练链路里的位置
    #
    # train() 准备阶段：
    #   imbalance = [10.78, 15.77, ..., 3.93]
    #       │
    #       │ 灌进每个 attr 的 BCEWithLogitsLoss(weight=...)
    #       ▼
    #   attr_criterion = [BCE(weight=10.78), BCE(weight=15.77), ..., BCE(weight=3.93)]
    #       │
    #       │ 传给 run_epoch
    #       ▼
    #
    # run_epoch 每个 batch 内部：
    #   logits, attr_logits = model(x)
    #
    #   losses_list = []
    #   for i in range(112):
    #       l_i = attr_criterion[i](attr_logits[i], attr_labels[:, i])
    #       # l_i 已经被自己的 ratio 放大过
    #       losses_list.append(l_i)
    #
    #   total_loss = main_ce_loss + args.attr_loss_weight * sum(losses_list)
    #   # args.attr_loss_weight 就是 paper 的 λ
    #   # 在 ratio 缩放之上再加一层全局缩放
    #   # Joint 取 0.001/0.01；Standard 取 0；Concept_XtoC 走 bottleneck 跳过 main_ce_loss
    #
    #   total_loss.backward()
    #   # 稀有 attr 的梯度被放大 ratio 倍
    #   # 反传到 Inception backbone 和该 attr 的 head
    #   # 模型在鉴别该稀有 attr 上学得更狠

    # ===================================================

    # 哪些训练任务用 imbalance、哪些不用

    # Step 1 Concept_XtoC (Inception 训 X→C)：
    #   args.no_img=False & args.use_attr=True
    #   → 走 run_epoch，attr_criterion 在 BCE 里读 weight
    #   → 用 imbalance

    # Step 2 Independent_CtoY (MLP 训 C→Y)：
    #   args.no_img=True
    #   → 走 run_epoch_simple，根本不读 attr_criterion
    #   → 主任务是 class CE，attr 不参与 loss
    #   → 不用 imbalance

    # 三档 weighted_loss 对比：
    #
    # 'multiple' (paper 默认)：
    #   imbalance = 112 个独立 ratio
    #   每个 attr 按自己稀有度独立缩放
    #
    # 'single' / 其他非空：
    #   imbalance = [global_ratio] * 112，112 个相同值
    #   global_ratio 是把 4796 * 112 = 537152 个 (sample, attr) 当一个大池子算的全局负正比
    #   所有 attr 加同一权重，相当于把 attr_loss_weight 整体放大固定倍数
    #
    # ''（空字符串）：
    #   imbalance = None
    #   attr_criterion 用 BCE() 不带 weight，完全不做 balance

    # ===================================================

    if args.use_attr and not args.no_img and args.weighted_loss:
        # 用 attr 标签 + 用图 + 用加权损失
        # Step 1 Concept_XtoC 走这个分支
        # Step 2 Independent_CtoY ( no_img=True ) 不走这个分支

        # 训练数据路径
        train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')

        # === debug 读取 pkl 文件的内容 ===start

        # print("=== debug ===")
        # # 二进制读
        # with open('CUB_processed/class_attr_data_10/train.pkl', 'rb') as f:
        #     data = pickle.load(f)

        # # 看顶层类型
        # print(type(data))
        # # 例：<class 'list'>

        # # 看顶层大小（list / dict 才有 len）
        # print(len(data))
        # # 例：4796

        # # 看一条样本的类型 + 字段
        # print(type(data[0]))
        # # 例：<class 'dict'>

        # print(data[0].keys())
        # # 例：dict_keys(['id', 'img_path', 'class_label', 'attribute_label', 'attribute_certainty'])

        # # 逐字段看，长字段只看头几个值
        # for k, v in data[0].items():
        #     if hasattr(v, '__len__') and not isinstance(v, str):
        #         print(f"{k:25} → {type(v).__name__} len={len(v)}, 头几个: {list(v)[:5]}")
        #     else:
        #         print(f"{k:25} → {v}")
        # # 例：
        # # id                        → 5
        # # img_path                  → CUB_200_2011/images/001.Black_footed_Albatross/.../xxx.jpg
        # # class_label               → 0
        # # attribute_label           → list len=112, 头几个: [0, 1, 0, 0, 1]
        # # attribute_certainty       → list len=312, 头几个: [3, 4, 3, 3, 4]
        # print("=============")

        # === debug 读取 pkl 文件的内容 ===end

        if args.weighted_loss == 'multiple':
            # list[float]，len=112
            # 每个 attr 单独算自己的 ratio
            imbalance = find_class_imbalance(
                pkl_file=train_data_path, 
                multiple_attr=True
            )
        else:
            # 'single' 或 其他非空字符串
            # list[float]，len=112
            # 所有 attr 共享一个全局 ratio
            imbalance = find_class_imbalance(
                pkl_file=train_data_path, 
                multiple_attr=False
            )

    # 日志目录
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

    if args.use_attr and not args.no_img:
        # 用 attr 且 用图，则 准备 attr 损失列表

        # list[Loss]，长度 = n_attributes
        attr_criterion = [] # separate criterion (loss function) for each attribute

        if args.weighted_loss:
            # ratio 加权 BCE

            assert(imbalance is not None)

            # 循环结束后
            # attr_criterion 是 112 个 loss 对象的列表
            # 每个 loss 绑定自己的 ratio：
            # 例：
            # attr_criterion[0]   的 weight = tensor([10.78])
            # attr_criterion[1]   的 weight = tensor([15.77])
            # ...
            # attr_criterion[N]   的 weight = tensor([ 3.93])
            # 
            # 有个细节要注意：用的是 weight= 而不是 pos_weight=
            # 这两个参数在 BCEWithLogitsLoss 里语义不一样：
            # 
            # pos_weight=ratio       ：只把正样本那项 BCE 乘 ratio，不动负样本
            #                          即教科书的"上采样稀有正类"做法
            # 
            # weight=tensor([ratio]) ：被 broadcast 到 batch 维全是 ratio
            #                          正样本、负样本一起乘 ratio
            # 
            # CBM 源码走的是后者
            # 整个 attr 的 loss 被等倍放大，不是只放大正类
            # 源码注释里写的"正样本权重 = ratio"其实是描述偏差
            # （也可能作者本意想用 pos_weight 但敲错了）
            # 实际效果近似"提升该 attr 在多任务 loss 总和里的总权重"
            # 做研究复现/对比时要意识到这个区别
            for ratio in imbalance:
                # 正样本权重 = ratio
                attr_criterion.append(
                    torch.nn.BCEWithLogitsLoss(weight=torch.FloatTensor([ratio]).cuda())
                )
        else:
            # 不加权
            for i in range(args.n_attributes):  # 112
                # 注意这里是 CE 不是 BCE，原作者设计
                attr_criterion.append(torch.nn.CrossEntropyLoss())
    # run_epoch_simple 不需要 attr_criterion 这个变量
    else:
        attr_criterion = None

    # === 选择优化器 ===

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

    # === 配置 scheduler ===

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

    # === 拿到 训练集 和 验证集 ===

    # 训练数据集 路径
    train_data_path = os.path.join(BASE_DIR, args.data_dir, 'train.pkl')
    # 验证数据集 路径
    val_data_path = train_data_path.replace('train.pkl', 'val.pkl')

    logger.write('\n' + "[train data path]=========" + '\n')
    logger.write('train data path: %s\n' % train_data_path)

    if args.ckpt: # retraining
        # 传了 ckpt，train 和 val 数据集 合并训练
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
        # 不传 ckpt，train 和 val 数据集 分开
        train_loader = load_data(
            [train_data_path], 
            args.use_attr, 
            args.no_img, 
            args.batch_size, 
            args.uncertain_labels, 
            image_dir=args.image_dir, 
            n_class_attr=args.n_class_attr, 
            resampling=args.resampling
        )
        val_loader = load_data(
            [val_data_path], 
            args.use_attr, 
            args.no_img, 
            args.batch_size, 
            image_dir=args.image_dir, 
            n_class_attr=args.n_class_attr
        )

    # 后面没用到这个变量
    # best_val_loss = float('inf')

    # 在 验证集 上最好的 epoch
    best_val_epoch = -1
    # 最好的 epoch 对应的 acc
    best_val_acc = 0

    # epoch 循环
    for epoch in range(0, args.epochs):
        # 训练 loss 和 acc 统计工具
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        if args.no_img:
            # 训 C → Y
            train_loss_meter, train_acc_meter = run_epoch_simple(
                model, 
                optimizer, 
                train_loader, 
                train_loss_meter, 
                train_acc_meter, 
                criterion, 
                args, 
                is_training=True
            )
        else:
            # 训 X → C
            train_loss_meter, train_acc_meter = run_epoch(
                model=model, 
                optimizer=optimizer, 
                loader=train_loader, 
                loss_meter=train_loss_meter, 
                acc_meter=train_acc_meter, 
                # 分类损失
                criterion=criterion, 
                # 概念损失
                attr_criterion=attr_criterion, 
                args=args, 
                is_training=True
            )

        if not args.ckpt: # evaluate on val set
            # 验证 loss 和 acc 统计工具
            val_loss_meter = AverageMeter()
            val_acc_meter = AverageMeter()
        
            with torch.no_grad():
                if args.no_img:
                    val_loss_meter, val_acc_meter = run_epoch_simple(
                        model, 
                        optimizer, 
                        val_loader, 
                        val_loss_meter, 
                        val_acc_meter, 
                        criterion, 
                        args, 
                        is_training=False
                    )
                else:
                    val_loss_meter, val_acc_meter = run_epoch(
                        model=model, 
                        optimizer=optimizer, 
                        loader=val_loader, 
                        loss_meter=val_loss_meter, 
                        acc_meter=val_acc_meter, 
                        criterion=criterion, 
                        attr_criterion=attr_criterion, 
                        args=args, 
                        is_training=False
                    )

        else: # retraining
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
        # inspect lr
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

# ======================================

def train_X_to_C(args):
    model = ModelXtoC(
        pretrained=args.pretrained, 
        freeze=args.freeze, 
        num_classes=N_CLASSES, 
        use_aux=args.use_aux, 
        n_attributes=args.n_attributes, 
        expand_dim=args.expand_dim, 
        three_class=args.three_class
    )

    train(model, args)

def train_oracle_C_to_y_and_test_on_Chat(args):
    model = ModelOracleCtoY(
        n_class_attr=args.n_class_attr, 
        n_attributes=args.n_attributes, 
        num_classes=N_CLASSES, 
        expand_dim=args.expand_dim
    )

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
