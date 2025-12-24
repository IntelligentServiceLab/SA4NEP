# SA4NEP：融合事件序列语义与属性关联语义的业务流程下一事件预测

本仓库是论文 **《融合事件序列语义与属性关联语义的业务流程下一事件预测》**

## 项目简介

预测性业务流程监控（PBPM）旨在利用历史事件日志预测正在进行的流程实例的未来行为。现有的方法往往难以兼顾**事件序列的上下文语义**与**事件属性间隐含的依赖关系**。

**SA4NEP** 提出了一种双分支深度学习架构来解决这一问题：

1.  **事件序列语义分支**：设计叙事模板将事件日志转换为“语义故事”，并利用大语言模型（BERT）提取长程上下文语义特征。
2.  **属性关联语义分支**：通过随机嵌入（Random Embedding）对多维属性进行编码，并利用**残差卷积神经网络（Res-CNN）**捕捉属性间隐含的交互与互斥关系。
3.  **语义融合模块**：在统一空间内深度融合两类特征，通过线性分类头完成下一事件预测。

## 环境依赖

本项目基于 Python 和 PyTorch 开发。请确保安装以下依赖库：

```bash
pip install torch pandas numpy scikit-learn imbalanced-learn transformers jinja2 tqdm
```

**目录结构说明**

`├── data/                   # 存放原始事件日志 CSV 文件 (例如: helpdesk.csv)`
`├── utility/`
`│   ├── Bert-medium/        # 预训练 BERT 模型文件 (需包含 config.json, pytorch_model.bin 等)`
`│   └── log_config.py       # [关键] 配置文件，定义叙事模板和属性映射`
`├── models/                 # 训练好的模型将保存于此`
`├── pro_data/               # 预处理后的数据 (代码自动生成)`
`├── log_history/            # 序列化后的日志和字典 (代码自动生成)`
`└── SA4BPM.py               # 主程序 (包含数据处理、训练和评估)`

代码默认加载本地的 BERT 模型。请下载 `bert-medium` (或其他 BERT 变体) 并放置在 `utility/Bert-medium/` 目录下，或者修改代码中的加载路径

BERT下载链接：https://huggingface.co/prajjwal1/bert-medium