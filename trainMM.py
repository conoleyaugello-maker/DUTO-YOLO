
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('')
    # model = YOLOMM('')
    model.train(data="",
                epochs=300,batch=16,close_mosaic=20,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                # modality='rb', # 模态消融参数 非必要不得开启（慎重做模态消融实验）
                # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭
                device='0',
                # cache='cache', #内存超过64G 的可以考虑使用此方法加速训练，硬盘空间足够大的也可以考虑 设置为 disk 获取加速，都不满足的就关闭
                exist_ok=False,
                # afss_enabled=True, #启动afss训练采样机制
                project='ResTest',name='neck-ctrans')
