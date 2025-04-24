# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import pandas as pd                                         # 读取文件、处理数组
import csv
import numpy as np
import time
from Segment_ import Seg                                    # 数据预处理模块，包含分词等功能
import gensim                                               # 从gensim中导入Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec, KeyedVectors
from sklearn import svm                                     # sklearn导入支持向量机
from sklearn.model_selection import train_test_split        # 导入数据集划分工具
from sklearn.metrics import confusion_matrix                # 导入评价指标：混淆矩阵和f1值
from classification_utilities import display_cm             # 给混淆矩阵加表头
import joblib                                               # 储存或调用模型
import multiprocessing                                      # 多进程模块
import PySimpleGUI as sg                                    # gui工具包
import codecs
import warnings                                             # 忽略告警
import joblib
from joblib import parallel_backend
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def read_data(dataset_name):
    wds = Seg()
    target_path = f'./data/{dataset_name}.seg.txt'
    target = codecs.open(target_path, 'w', encoding='utf8')

    input_path = os.path.join('./data', f'{dataset_name}.txt')
    with open(input_path, encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 分割标签和评论文本
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue  # 无效行

            label, text = parts[0], parts[1]

            # 仅处理合法的分类标签
            try:
                if int(label) not in {0, 1, 2}:
                    continue
            except ValueError:
                continue  # 标签非整数

            # 仅对评论文本分词
            seg_list = wds.cut(text, cut_all=False)
            line_seg = ' '.join(seg_list)

            if len(line_seg) >= 25:  # 根据需求调整长度过滤
                target.write(f"{label} {line_seg}\n")  # 写入格式：标签 分词后文本

    target.close()


def getWordVecs(words, model):
    vecs = []
    for word in words:
        if word in model:  # 检查词是否存在
            vecs.append(model[word])  # 直接访问 model[word]
    return vecs


def buildVecs(data, model):
    fileVecs = []
    label = []

    for line in data:
        parts = line.strip().split()
        if len(parts) < 2:  # 至少包含标签和一个词
            continue

        label_part = parts[0]
        wordList = parts[1:]  # 实际分词内容

        try:
            label_val = int(label_part)
            if label_val not in {0, 1, 2}:
                continue
        except ValueError:
            continue

        # 转换为词向量
        vecs = getWordVecs(wordList, model)
        if len(vecs) == 0:
            continue

        # 取平均向量
        vecsArray = np.mean(vecs, axis=0)
        fileVecs.append(vecsArray)
        label.append(label_val)

    return fileVecs, label

def get_data_wordvec(dataset_name):
    inp = f'./data/{dataset_name}.seg.txt'
    f = codecs.open(inp, mode='r', encoding='utf-8')
    line = f.readlines()

    data = []
    for i in line:
        data.append(i)
    f.close()
    return data


def word2vec_(dataset_name):
    class CorpusGenerator:  # 自定义迭代器排除标签
        def __init__(self, filename):
            self.filename = filename

        def __iter__(self):
            with codecs.open(self.filename, 'r', 'utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:  # 确保有文本内容
                        yield parts[1:]  # 跳过标签，只返回分词后的文本

    input_path = f'./data/{dataset_name}.seg.txt'
    output_path = f'./data/{dataset_name}.seg.text.vector'

    # 使用自定义迭代器训练模型
    sentences = CorpusGenerator(input_path)
    model_ = Word2Vec(
        sentences,
        vector_size=300,  # 词向量维度
        window=8,  # 上下文窗口
        min_count=1,  # 过滤低频词
        sg=1,  # 使用Skip-Gram
        hs=1,  # 层次Softmax加速训练
        workers=multiprocessing.cpu_count()
    )
    model_.wv.save_word2vec_format(output_path, binary=False)

    # 构建特征向量
    data = get_data_wordvec(dataset_name)
    Input22, label = buildVecs(data, model_)

    # 保存时排除索引列
    df_x = pd.DataFrame(Input22)
    df_y = pd.DataFrame(label, columns=['label'])
    output_csv = f'./data/{dataset_name}_word2vec.csv'
    data = pd.concat([df_y, df_x], axis=1)
    data.to_csv(output_csv, index=False)

# 均衡
def classification1_(dataset_name):
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    # 读取标签和特征
    y = df['label']
    x = df.drop('label', axis=1)

    # 标签检查
    unique_labels = np.unique(y)
    print("实际存在的标签:", unique_labels)
    print("标签分布:\n", y.value_counts())

    # 定义类别标签,从0开始对应
    labels = ["好评", "中评", "差评"]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

    # 训练SVM
    print('训练支持向量机...')
    clf = svm.SVC(
        C=100,
        kernel='rbf',
        class_weight='balanced',
        probability=True
    )
    clf.fit(X_train, y_train)
    joblib.dump(clf, "./model/model_{dataset_name}.m")

    # 评估
    print('混淆矩阵')
    y_pred = clf.predict(X_test)
    cv_conf = confusion_matrix(y_test, y_pred)
    display_cm(cv_conf, labels, display_metrics=True)

    print('准确率: %.2f' % clf.score(x, y))


# 速度优先
def classification2_(dataset_name):
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    y = df['label']
    x = df.drop('label', axis=1)

    labels = ["表达开心", "表达伤心", "表达恶心", "表达生气", "表达害怕", "表达惊喜"]

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=42)
    print('支持向量机....')
    # 使用线性核SVM用于加速训练并提升高维稀疏数据表现
    from sklearn.svm import LinearSVC
    clf = LinearSVC(
        C=10,
        class_weight='balanced',
        max_iter=10000)

    # 加速训练
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    clf.fit(X_train, y_train)
    joblib.dump(clf, "./model/model_{dataset_name}.m")

    # 评估
    y_pred = clf.predict(X_test)
    print('混淆矩阵')
    cv_conf = confusion_matrix(y_test, y_pred)
    display_cm(cv_conf, labels, display_metrics=True)
    print(f"准确率: {clf.score(X_test, y_test):.2f}")

# 质量优先
def classification3_(dataset_name):
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    y = df['label'].astype(int)
    x = df.drop('label', axis=1)

    labels = ["好评", "中评", "差评"]

    # 标准化特征
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.7, random_state=42)

    # 检查模型是否存在
    model_path = f"./model/best_model_{dataset_name}.m"
    if os.path.exists(model_path):
        # 直接加载已有模型
        print("检测到已保存的最佳模型，直接加载...")
        best_clf = joblib.load(model_path)
    else:
        # 定义参数网格
        param_grid = [
            {
                'kernel': ['rbf'],
                'C': [10, 100, 300],
                'gamma': [0.1, 0.01, 'scale', 'auto'],
                'class_weight': ['balanced']
            },
            {
                'kernel': ['poly'],
                'C': [10, 100, 300],
                'gamma': [0.1, 0.01, 'scale'],
                'degree': [2, 3],
                'coef0': [0, 1.0],
                'class_weight': ['balanced']
            }
        ]

        # 执行网格搜索（仅在第一次运行或模型未保存时）
        from sklearn.model_selection import GridSearchCV
        grid_search = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,  # 使用所有CPU核心
            verbose=2
        )

        print("开始网格搜索...")
        grid_search.fit(X_train, y_train)

        # 保存最佳模型
        best_clf = grid_search.best_estimator_
        joblib.dump(best_clf, model_path)
        print(f"模型已保存至 {model_path}")

    # 评估模型（无论新旧）
    y_pred = best_clf.predict(X_test)

    # 混淆矩阵
    print('\n混淆矩阵:')
    cv_conf = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    print("混淆矩阵维度:", cv_conf.shape)
    # print(cv_conf)
    display_cm(cv_conf, labels, display_metrics=True)
    print(f"测试集准确率: {best_clf.score(X_test, y_test):.2f}")


def predict_(a, dataset_name):
    # 加载词向量模型
    inp = f"./data/{dataset_name}.seg.text.vector"
    model = KeyedVectors.load_word2vec_format(inp, binary=False)

    # 分词
    wds = Seg()
    seg_list = wds.cut(a, cut_all=False)
    print(f"分词结果: {seg_list}")  # 调试输出
    line_seg = seg_list  # 直接使用列表，避免重复 split/join

    # 获取词向量
    vecs = getWordVecs(line_seg, model)
    if len(vecs) == 0:
        return "无法识别（无有效词向量）"

    # 计算平均向量
    vecsArray = np.mean(vecs, axis=0).reshape(1, 300)

    # 加载分类模型（动态路径）
    model_path = f"./model/best_model_{dataset_name}.m"
    print(f"Loading model from: {model_path}")  # 调试输出路径
    try:
        clf = joblib.load(model_path)
    except FileNotFoundError:
        return f"模型文件 {model_path} 未找到！"

    # 预测
    kk = clf.predict(vecsArray)
    if kk[0] == 0:
        return "表达好评"
    elif kk[0] == 1:
        return "表达中评"
    elif kk[0] == 2:
        return "表达差评"
    else:
        return "无效预测结果"

def read_table_data(filename):
    with open(filename, "r", encoding='gbk') as infile:
        reader = csv.reader(infile)
        data = list(reader)  # read everything else into a list of rows
    return data

def make_window(theme):
    sg.theme(theme)
    # 菜单栏
    menu_def = [['Help', ['About...', ['你好']]], ]
    # 主界面之一：文本识别界面
    News_detection = [
        [sg.Menu(menu_def, tearoff=True)],
        [sg.Text('')],
        [sg.Multiline(s=(60, 20), key='_INPUT_news_', expand_x=True)],
        [sg.Text('')],
        [sg.Text('', s=(12)), sg.Text('识别结果：', font=("Helvetica", 15)),
         sg.Text('     ', key='_OUTPUT_news_', font=("Helvetica", 15))],
        [sg.Text('')],
        [sg.Text('', s=(12)), sg.Button('识别', font=("Helvetica", 15)), sg.Text('', s=(10)),
         sg.Button('清空', font=("Helvetica", 15)),
         sg.Text('', s=(4))],
        [sg.Text('')],
        [sg.Sizegrip()]
    ]
    # 主界面之二：文本识别内容的管理，可以查看自己识别的内容

    News_management = [
        [sg.Table(values=read_table_data('table_data.csv')[1:][:], headings=['文本内容', '识别时间', '识别结果'], max_col_width=30,
                  auto_size_columns=True,
                  display_row_numbers=False,
                  justification='center',
                  num_rows=20,
                  key='-TABLE_de-',
                  selected_row_colors='red on yellow',
                  enable_events=True,
                  expand_x=True,
                  expand_y=True,
                  vertical_scroll_only=False,
                  enable_click_events=True,  # Comment out to not enable header and other clicks
                  )
         ],

        [sg.Button('删除选中的结果', font=("Helvetica", 15)), sg.Button('查看识别结果', font=("Helvetica", 15))],
        [sg.Sizegrip()]

    ]
    empty = []

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [sg.Text('情感识别系统', size=(50, 1), justification='center', font=("Helvetica", 16),
                       relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True, expand_x=True)]]
    layout += [[sg.TabGroup([[
        sg.Tab(' 文 本 识 别 ', News_detection),
        sg.Tab('                                                     ', empty),
        sg.Tab(' 结 果 管 理  ', News_management,element_justification="right",)]], expand_x=True, expand_y=True,font=("Helvetica", 16)),

    ]]
    # layout[-1].append(sg.Sizegrip())
    window = sg.Window('情感识别系统', layout,
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window.set_min_size(window.size)
    return window

def main_WINDOW(dataset_name):

    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)

        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break

        elif event == '识别':
            kk = predict_(values['_INPUT_news_'], dataset_name)
            time2 = time.strftime('%Y-%m-%d %H:%M:%S')
            newuser = [values['_INPUT_news_'], time2, kk]
            with open('./data/table_data.csv', 'a', newline='') as studentDetailsCSV:
                writer = csv.writer(studentDetailsCSV, dialect='excel')
                writer.writerow(newuser)
            window['_OUTPUT_news_'].update(kk)
            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

        elif event == '清空':
            window['_OUTPUT_news_'].update(' ')
            window['_INPUT_news_'].update('')
        elif event == '查看识别结果':
            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

        elif event == '删除选中的结果':
            data = pd.read_csv('./data/table_data.csv', encoding='gbk')
            data.drop(data.index[int(values['-TABLE_de-'][0])], inplace=True)
            # 如果想要保存新的csv文件，则为
            data.to_csv("./data/table_data.csv", index=None, encoding="gbk")

            window["-TABLE_de-"].update(values=read_table_data('./data/table_data.csv')[1:][:])

    window.close()
    exit(0)

if __name__ == '__main__':
    # 从终端输入数据集名称（不带扩展名）
    dataset_name = input("请输入数据集名称（不带扩展名）：").strip()

    # read_data(dataset_name)
    # word2vec_(dataset_name)
    # classification3_(dataset_name)
    sg.theme()
    main_WINDOW(dataset_name)