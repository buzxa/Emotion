import numpy as np

def display_cm(cm, labels, hide_zeros=False, display_metrics=False):
    num_classes = len(labels)

    # 检查混淆矩阵的维度是否与标签数量一致
    if cm.shape != (num_classes, num_classes):
        raise ValueError(f"混淆矩阵应为 {num_classes}x{num_classes}，但实际形状是 {cm.shape}")

    # 计算指标
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    F1 = np.zeros(num_classes)

    for i in range(num_classes):
        # 计算 Recall（行指标）
        row_sum = cm.sum(axis=1)[i]
        recall[i] = cm[i, i] / row_sum if row_sum != 0 else 0

        # 计算 Precision（列指标）
        col_sum = cm.sum(axis=0)[i]
        precision[i] = cm[i, i] / col_sum if col_sum != 0 else 0

        # 计算 F1
        if precision[i] + recall[i] != 0:
            F1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    # 计算总指标（加权平均）
    total_samples = cm.sum()
    total_precision = (precision * cm.sum(axis=1)).sum() / total_samples
    total_recall = (recall * cm.sum(axis=1)).sum() / total_samples
    total_F1 = (F1 * cm.sum(axis=1)).sum() / total_samples

    # 格式化输出
    columnwidth = max([len(x) for x in labels] + [5])  # 列宽
    empty_cell = " " * columnwidth

    # 打印混淆矩阵头部
    print("    Pred", end=' ')
    for label in labels:
        print(f"%{columnwidth}s" % label, end=' ')
    print(f"%{columnwidth}s" % 'Total')
    print("    True")

    # 打印每一行
    for i, label1 in enumerate(labels):
        print(f"%{columnwidth}s" % label1, end=' ')
        for j in range(num_classes):
            cell = f"%{columnwidth}d" % cm[i, j]
            if hide_zeros and cm[i, j] == 0:
                cell = empty_cell
            print(cell, end=' ')
        print(f"%{columnwidth}d" % sum(cm[i, :]))

    # 打印指标（可选）
    if display_metrics:
        print("\nMetrics:")
        print(f"{'Precision':>{columnwidth}}", end=' ')
        for p in precision:
            print(f"%{columnwidth}.2f" % p, end=' ')
        print(f"%{columnwidth}.2f" % total_precision)

        print(f"{'Recall':>{columnwidth}}", end=' ')
        for r in recall:
            print(f"%{columnwidth}.2f" % r, end=' ')
        print(f"%{columnwidth}.2f" % total_recall)

        print(f"{'F1':>{columnwidth}}", end=' ')
        for f in F1:
            print(f"%{columnwidth}.2f" % f, end=' ')
        print(f"%{columnwidth}.2f" % total_F1)


