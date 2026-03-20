passable_area_system- 
	data-
		labels- 
		raw- 
		masks-
		predictions-
	deep_learning- 
		model- 
			unet.py
			unet.pth 
		datasets.py 
		infer.py 
		train.py 
	evaulation- 
		metrics.py 
	traditional- 
		...... 
	utils- 
		........ 
	main.py





| 版本     | 模型   | 输入尺寸    | 数据增强   | 损失函数                       | 优化器           | 学习率调度             | 批量大小 | Epoch | 训练指标                                 |
| ------ | ---- | ------- | ------ | -------------------------- | ------------- | ----------------- | ---- | ----- | ------------------------------------ |
| **v1** | UNet | 256×256 | 无      | BCE                        | Adam(lr=1e-3) | 无                 | 4    | 40    | Loss≈30, Dice≈15                     |
| **v2** | UNet | 256×256 | 无      | BCE                        | Adam(lr=1e-3) | 无                 | 4    | 40    | Loss≈15.0687                         |
| **v3** | UNet | 256×256 | 无      | BCE                        | Adam(lr=1e-3) | 无                 | 4    | 40    | Loss≈15.0687, Dice≈15                |
| **v4** | UNet | 256×256 | 随机水平翻转 | BCE(pos_weight=3.0) + Dice | Adam(lr=5e-4) | ReduceLROnPlateau | 4    | 40    | Loss=1.4263, BCE=0.8026, Dice=0.6238 |
| **v5** | UNet | 256×256 | 随机水平翻转 | BCE(pos_weight=3.0) + Dice | Adam(lr=5e-4) | ReduceLROnPlateau | 4    | 40    | Loss=1.4199, BCE=0.7972, Dice=0.6227 |


本项目实现了一个面向室内扫地机器人的可通行区域识别系统。版本几不知道

系统采用多维度融合策略，包括：

1. 语义维度（Semantic）
   基于UNet模型对室内场景进行语义分割，提取地面区域。

2. 几何维度（Geometric）
   利用Canny边缘检测提取潜在障碍物结构信息。

3. 融合决策（Fusion）
   通过加权融合策略整合语义信息与几何信息，
   生成最终可通行区域。

此外，系统引入后处理优化（阈值调整与形态学操作），
增强区域连通性与鲁棒性。

实验结果表明：
- Floor IoU: 0.8497
- Passable IoU: 0.8473

在保证精度的同时，提升了系统对复杂场景的适应能力。