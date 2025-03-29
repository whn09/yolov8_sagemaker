import os
import sys
import argparse
from pathlib import Path
from ultralytics import YOLO


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 基本训练参数
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov8n.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--time', type=float, default=None, help='maximum training time in hours')
    parser.add_argument('--patience', type=int, default=100, help='early stopping patience (epochs without improvement)')
    parser.add_argument('--batch', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu or mps')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    
    # 保存设置
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save', action='store_true', default=True, help='save train checkpoints')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    
    # 模型设置
    parser.add_argument('--pretrained', type=str, default=True, help='use pretrained model')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], default='auto', help='optimizer')
    parser.add_argument('--freeze', nargs='+', type=int, default=None, help='Freeze layers')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    
    # 数据处理
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image cache mode (ram, disk, or False)')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--classes', nargs='+', type=int, default=None, help='filter by class: --classes 0, or --classes 0 2 3')
    
    # 训练行为
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--amp', action='store_true', default=True, help='Automatic Mixed Precision training')
    parser.add_argument('--fraction', type=float, default=1.0, help='dataset fraction to use for training')
    parser.add_argument('--profile', action='store_true', help='profile ONNX and TensorRT speeds')
    parser.add_argument('--val', action='store_true', default=True, help='validate during training')
    parser.add_argument('--deterministic', action='store_true', default=True, help='deterministic training')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%')
    parser.add_argument('--plots', action='store_true', default=False, help='save plots during training')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    
    # 超参数
    parser.add_argument('--lr0', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01, help='final learning rate (fraction of lr0)')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD momentum/Adam beta1')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='optimizer weight decay')
    parser.add_argument('--warmup-epochs', type=float, default=3.0, help='warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8, help='warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1, help='warmup bias lr')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--close-mosaic', type=int, default=10, help='disable mosaic augmentation for final epochs')
    
    # 损失权重
    parser.add_argument('--box', type=float, default=7.5, help='box loss gain')
    parser.add_argument('--cls', type=float, default=0.5, help='cls loss gain')
    parser.add_argument('--dfl', type=float, default=1.5, help='dfl loss gain')
    parser.add_argument('--pose', type=float, default=12.0, help='pose loss gain')
    parser.add_argument('--kobj', type=float, default=1.0, help='keypoint objectness loss gain')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--nbs', type=int, default=64, help='nominal batch size')
    
    # 分割参数
    parser.add_argument('--overlap-mask', action='store_true', default=True, help='overlap masks')
    parser.add_argument('--mask-ratio', type=int, default=4, help='mask downsample ratio')
    
    # 数据增强参数
    parser.add_argument('--hsv-h', type=float, default=0.015, help='image HSV-Hue augmentation (fraction)')
    parser.add_argument('--hsv-s', type=float, default=0.7, help='image HSV-Saturation augmentation (fraction)')
    parser.add_argument('--hsv-v', type=float, default=0.4, help='image HSV-Value augmentation (fraction)')
    parser.add_argument('--degrees', type=float, default=0.0, help='image rotation (+/- deg)')
    parser.add_argument('--translate', type=float, default=0.1, help='image translation (+/- fraction)')
    parser.add_argument('--scale', type=float, default=0.5, help='image scale (+/- gain)')
    parser.add_argument('--shear', type=float, default=0.0, help='image shear (+/- deg)')
    parser.add_argument('--perspective', type=float, default=0.0, help='image perspective (+/- fraction)')
    parser.add_argument('--flipud', type=float, default=0.0, help='image flip up-down (probability)')
    parser.add_argument('--fliplr', type=float, default=0.5, help='image flip left-right (probability)')
    parser.add_argument('--mosaic', type=float, default=1.0, help='image mosaic (probability)')
    parser.add_argument('--mixup', type=float, default=0.0, help='image mixup (probability)')
    parser.add_argument('--copy-paste', type=float, default=0.0, help='segment copy-paste (probability)')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    # Load a model
    model = YOLO(opt.weights)  # load a pretrained model (recommended for training)

    # Use the model
    # model.train(data=opt.data, epochs=opt.epochs, batch=opt.batch, imgsz=opt.imgsz, project=opt.project, name=opt.name, save_period=opt.save_period)  # train the model
    model.train(
        data=opt.data,                      # 数据集配置文件路径
        epochs=opt.epochs,                  # 训练轮数
        time=opt.time,                      # 最大训练时间（小时）
        patience=opt.patience,              # 提前停止的耐心值
        batch=opt.batch,                    # 批次大小
        imgsz=opt.imgsz,                    # 图像尺寸
        device=opt.device,                  # 训练设备
        workers=opt.workers,                # 数据加载线程数
        
        # 保存设置
        project=opt.project,                # 保存结果的项目目录
        name=opt.name,                      # 实验名称
        exist_ok=opt.exist_ok,              # 是否覆盖已有实验目录
        save=opt.save,                      # 是否保存模型
        save_period=opt.save_period,        # 每x个epoch保存一次
        
        # 模型设置
        pretrained=opt.pretrained,          # 使用预训练权重
        optimizer=opt.optimizer,            # 优化器选择
        freeze=opt.freeze,                  # 冻结层
        dropout=opt.dropout,                # 使用dropout正则化
        
        # 数据处理
        rect=opt.rect,                      # 矩形训练
        cache=opt.cache,                    # 图像缓存模式
        single_cls=opt.single_cls,          # 是否作为单类别数据集训练
        classes=opt.classes,                # 过滤的类别
        
        # 训练行为
        resume=opt.resume,                  # 恢复中断的训练
        amp=opt.amp,                        # 自动混合精度
        fraction=opt.fraction,              # 数据集使用比例
        profile=opt.profile,                # 是否进行性能分析
        val=opt.val,                        # 是否进行验证
        deterministic=opt.deterministic,    # 确定性训练
        multi_scale=opt.multi_scale,        # 多尺度训练
        plots=opt.plots,                    # 是否保存绘图
        seed=opt.seed,                      # 随机种子
        
        # 超参数
        lr0=opt.lr0,                        # 初始学习率
        lrf=opt.lrf,                        # 最终学习率比例
        momentum=opt.momentum,              # SGD动量
        weight_decay=opt.weight_decay,      # 优化器权重衰减
        warmup_epochs=opt.warmup_epochs,    # 预热轮数
        warmup_momentum=opt.warmup_momentum, # 预热阶段动量
        warmup_bias_lr=opt.warmup_bias_lr,  # 预热阶段偏置学习率
        cos_lr=opt.cos_lr,                  # 余弦学习率调度
        close_mosaic=opt.close_mosaic,      # 最后几轮关闭mosaic增强
        
        # 损失权重
        box=opt.box,                        # 框损失增益
        cls=opt.cls,                        # 分类损失增益
        dfl=opt.dfl,                        # DFL损失增益
        pose=opt.pose,                      # 姿态损失增益
        kobj=opt.kobj,                      # 关键点目标损失增益
        label_smoothing=opt.label_smoothing, # 标签平滑系数
        nbs=opt.nbs,                        # 标称批量大小
        
        # 分割参数
        overlap_mask=opt.overlap_mask,      # 是否允许掩码重叠
        mask_ratio=opt.mask_ratio,          # 掩码下采样比例
        
        # 数据增强参数
        hsv_h=opt.hsv_h,                    # 色调增强
        hsv_s=opt.hsv_s,                    # 饱和度增强
        hsv_v=opt.hsv_v,                    # 亮度增强
        degrees=opt.degrees,                # 图像旋转
        translate=opt.translate,            # 图像平移
        scale=opt.scale,                    # 图像缩放
        shear=opt.shear,                    # 图像剪切
        perspective=opt.perspective,        # 图像透视
        flipud=opt.flipud,                  # 上下翻转图像概率
        fliplr=opt.fliplr,                  # 左右翻转图像概率
        mosaic=opt.mosaic,                  # 马赛克增强概率
        mixup=opt.mixup,                    # mixup增强概率
        copy_paste=opt.copy_paste,          # copy-paste增强概率
    )


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)