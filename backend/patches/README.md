# 第三方补丁

在 `third_party/` 下克隆对应仓库后执行 `git apply`：

- openlenda.patch（https://github.com/turingmotors/openlenda）：tools/demo.py 的 torch.load(weights_only=False) 与无界面 waitKey。
- CLRerNet.patch（https://github.com/hirotomusiker/CLRerNet）：libs/models/layers/nms/src 里 boxes.type() -> scalar_type()，适配新版 PyTorch。

site-packages 的两处修改（不在仓库内）：
- mmdet/__init__.py：mmcv_maximum_version 放宽到 2.3.0。
- mmengine/optim/optimizer/builder.py：Adafactor 注册加 force=True。
