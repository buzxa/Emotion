# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import threading
from queue import Queue
import sys
sys.path.append('D:\Mat\github\Emotion\JD\comment4.py')     # 添加comment4.py所在目录
# sys.path.append('D:\Mat\github\Emotion\LSTM')
sys.path.append(os.path.join(os.path.dirname(__file__), "..", 'LSTM'))
from LSTM.main import predict_lstm
from JD.comment4 import get_jd_comments

import os
import pandas as pd                                         # 读取文件、处理数组
import csv
import numpy as np
import time
from Segment_ import Seg                                    # 数据预处理模块，包含分词等功能
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
from joblib import parallel_backend

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
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
    # 判断模型类型，选择正确的访问方式
    if hasattr(model, 'wv'):  # 如果是 Word2Vec 对象
        word_vectors = model.wv
    else:  # 如果是 KeyedVectors 对象
        word_vectors = model
    for word in words:
        if word in word_vectors:
            vecs.append(word_vectors[word])
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
        window=10,  # 上下文窗口
        min_count=2,  # 过滤低频词
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
def classification_balance(dataset_name):
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    y = df['label'].values
    X = df.drop('label', axis=1).values
    labels = ["好评", "中评", "差评"]

    # 模型路径
    model_path = f"./model/model_balance_{dataset_name}.m"

    # 先分割数据
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    if os.path.exists(model_path):
        print("\n加载已保存的平衡模型和预处理组件...\n")
        components = joblib.load(model_path)
        clf = components['model']
        scaler = components['scaler']

        # 处理测试集
        X_test_processed = scaler.transform(X_raw_test)
    else:
        print('\n训练平衡模型...')
        # 标准化处理
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_raw_train)
        X_test_processed = scaler.transform(X_raw_test)

        # 训练模型
        clf = svm.SVC(
            C=100,
            kernel='rbf',
            class_weight='balanced',
            probability=True
        )
        clf.fit(X_train_processed, y_train)

        # 保存组件
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': clf,
            'scaler': scaler
        }, model_path)

    # 评估
    y_pred = clf.predict(X_test_processed)
    print('\n混淆矩阵_balance')
    cv_conf = confusion_matrix(y_test, y_pred)
    display_cm(cv_conf, labels, display_metrics=True)
    print(f"测试集准确率: {clf.score(X_test_processed, y_test):.2f}")

# 速度优先
def classification_speed(dataset_name):
    # 数据加载
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    # 准备数据
    y = df['label'].values
    X = df.drop('label', axis=1)
    labels = ["好评", "中评", "差评"]

    # 模型路径
    model_path = f"./model/model_speed_{dataset_name}.m"

    # 始终先进行数据分割（确保测试集一致性）
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.3, random_state=42)

    if os.path.exists(model_path):
        print("加载已保存的速度模型和预处理组件...")
        components = joblib.load(model_path)
        clf = components['model']
        scaler = components['scaler']
        pca = components['pca']

        # 使用保存的预处理处理测试集
        X_test_scaled = scaler.transform(X_raw_test)
        X_test_pca = pca.transform(X_test_scaled)
    else:
        print('训练速度模型...')
        # 训练集预处理流程
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_raw_train)

        pca = PCA(n_components=50)
        X_train_pca = pca.fit_transform(X_train_scaled)

        # 训练模型
        clf = LinearSVC(C=10, class_weight='balanced', max_iter=10000)
        clf.fit(X_train_pca, y_train)

        # 保存组件
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': clf,
            'scaler': scaler,
            'pca': pca
        }, model_path)

        # 处理测试集
        X_test_scaled = scaler.transform(X_raw_test)
        X_test_pca = pca.transform(X_test_scaled)

    # 评估模型
    y_pred = clf.predict(X_test_pca)
    print('\n混淆矩阵_speed')
    cv_conf = confusion_matrix(y_test, y_pred)
    display_cm(cv_conf, labels, display_metrics=True)
    print(f"测试集准确率: {clf.score(X_test_pca, y_test):.2f}")

# 质量优先
def classification_quality(dataset_name):
    input_csv = f'./data/{dataset_name}_word2vec.csv'
    df = pd.read_csv(input_csv)

    y = df['label'].astype(int).values
    X = df.drop('label', axis=1).values
    labels = ["好评", "中评", "差评"]

    model_path = f"./model/model_quality_{dataset_name}.m"
    X_raw_train, X_raw_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42)

    if os.path.exists(model_path):
        print("加载已保存的质量模型和预处理组件...")
        components = joblib.load(model_path)
        clf = components['model']
        scaler = components['scaler']
        X_test_processed = scaler.transform(X_raw_test)
    else:
        print('训练质量模型...')
        # 标准化处理
        scaler = StandardScaler()
        X_train_processed = scaler.fit_transform(X_raw_train)
        X_test_processed = scaler.transform(X_raw_test)

        # 参数网格和网格搜索
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

        grid_search = GridSearchCV(
            estimator=svm.SVC(probability=True),  # 概率支持
            param_grid=param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        print("开始网格搜索...")
        grid_search.fit(X_train_processed, y_train)

        # 保存组件
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': grid_search.best_estimator_,
            'scaler': scaler
        }, model_path)

    # 评估
    clf = joblib.load(model_path)['model']
    y_pred = clf.predict(X_test_processed)
    print('\n混淆矩阵_quality:')
    cv_conf = confusion_matrix(y_test, y_pred)
    display_cm(cv_conf, labels, display_metrics=True)
    print(f"测试集准确率: {clf.score(X_test_processed, y_test):.2f}")


def predict_svm(a, dataset_name, model_type):
    """
    参数说明：
    a: 待预测文本 (字符串)
    dataset_name: 数据集名称 (字符串)
    model_type: 模型类型，可选 'balance'/'speed'/'quality' (默认balance)
    返回：预测结果字符串（包含置信度）或错误信息
    """
    try:
        # 输入验证
        if not isinstance(a, str) or len(a.strip()) == 0:
            return "错误：输入文本无效"

        # 加载词向量模型
        word2vec_path = f"./data/{dataset_name}.seg.text.vector"
        if not os.path.exists(word2vec_path):
            return f"错误：词向量文件 {word2vec_path} 不存在"

        try:
            model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
        except Exception as e:
            return f"词向量加载失败：{str(e)}"

        # 文本预处理
        wds = Seg()
        try:
            seg_list = wds.cut(a.strip(), cut_all=False)
            if len(seg_list) == 0:
                return "无法识别：分词结果为空"
            # 终端观察输出结果
            print(f"SVM分词:{seg_list}")
        except Exception as e:
            return f"分词失败：{str(e)}"

        # 生成词向量
        try:
            vecs = getWordVecs(seg_list, model)
            if len(vecs) == 0:
                return "无法识别：无有效词向量"
            raw_vec = np.mean(vecs, axis=0).reshape(1, -1)
        except Exception as e:
            return f"词向量生成失败：{str(e)}"

        # 加载分类模型及预处理
        model_components = {}
        expected_dim = 300  # 默认维度
        try:
            model_paths = {
                'balance': f"./model/model_balance_{dataset_name}.m",
                'speed': f"./model/model_speed_{dataset_name}.m",
                'quality': f"./model/model_quality_{dataset_name}.m"
            }

            if model_type not in model_paths:
                return f"错误：无效模型类型 '{model_type}'"

            model_path = model_paths[model_type]
            if not os.path.exists(model_path):
                return f"模型文件 {model_path} 不存在"

            components = joblib.load(model_path)

            # 处理不同模型类型
            if model_type == 'speed':
                required_keys = ['model', 'scaler', 'pca']
                for key in required_keys:
                    if key not in components:
                        return f"损坏的模型文件：缺少 {key} 组件"
                model_components.update(components)
                expected_dim = 50
                scaled = components['scaler'].transform(raw_vec)
                processed_vec = components['pca'].transform(scaled)

            elif model_type == 'balance':
                if 'scaler' not in components:
                    return "损坏的模型文件：缺少scaler组件"
                model_components['model'] = components['model']
                model_components['scaler'] = components['scaler']
                processed_vec = components['scaler'].transform(raw_vec)

            elif model_type == 'quality':
                if 'scaler' not in components:
                    return "损坏的模型文件：缺少scaler组件"
                model_components['model'] = components['model']
                model_components['scaler'] = components['scaler']
                processed_vec = components['scaler'].transform(raw_vec)

        except Exception as e:
            return f"模型加载失败：{str(e)}"

        # 特征维度验证
        try:
            if processed_vec.shape[1] != expected_dim:
                return (
                    f"维度不匹配 (当前：{processed_vec.shape[1]}，"
                    f"需要：{expected_dim})\n"
                    f"可能原因：模型训练参数与当前不兼容"
                )
        except IndexError:
            return "特征矩阵形状异常"

        # 执行预测及置信度计算
        try:
            clf = model_components['model']
            pred = clf.predict(processed_vec)

            # 计算置信度
            if hasattr(clf, 'predict_proba'):
                # 概率模型（balance/quality）
                proba = clf.predict_proba(processed_vec)
                confidence = np.max(proba[0])
            else:
                # 非概率模型（speed）
                decision = clf.decision_function(processed_vec)
                # 应用softmax转换
                exp_decision = np.exp(decision - np.max(decision))
                softmax_proba = exp_decision / exp_decision.sum()
                confidence = np.max(softmax_proba)

            label_map = {
                0: "好评",
                1: "中评",
                2: "差评"
            }

            if pred[0] not in label_map:
                return f"未知预测结果：{pred[0]}"

            return (
                f"[{model_type}]预测结果：{label_map[pred[0]]}\n"
                f"置信度：{confidence:.2%}"
            )

        except Exception as e:
            return f"预测执行失败：{str(e)}"

    except Exception as e:
        return f"系统错误：{str(e)}"

# 爬虫线程类
class CrawlerThread(threading.Thread):
    def __init__(self, queue, progress_queue, params):
        super().__init__()
        self.queue = queue  # 结果队列
        self.progress_queue = progress_queue  # 进度队列
        self.params = params

    def run(self):
        try:
            # 定义进度回调函数
            def update_progress(percent):
                self.progress_queue.put(("PROGRESS", percent))

            type_dict = {"全部评价": 0, "追评": 1, "好评": 2, "中评": 3, "差评": 4}
            sort_dict = {"默认排序": 0, "时间排序": 1}

            # 执行爬虫并传入回调
            save_path = get_jd_comments(
                product_id=self.params["product_id"],
                pages=self.params["pages"],
                comment_type=type_dict[self.params["comment_type"]],
                sort_type=sort_dict[self.params["sort_type"]],
                progress_callback=update_progress  # 注入回调
            )

            self.queue.put(("SUCCESS", save_path))
        except Exception as e:
            self.queue.put(("ERROR", str(e)))


def read_table_data(file_path):
    try:
        with open(file_path, 'r', newline='', encoding='gbk') as file:
            reader = csv.reader(file)
            data = [row for row in reader if row]  # 过滤空行

            # 填充缺失列（兼容旧数据）
            for row in data:
                while len(row) < 5:
                    row.append('')
            return data
    except FileNotFoundError:
        return []

def make_window(theme):
    sg.theme(theme)
    # 菜单栏
    menu_def = [['Help', ['About...', ['你好']]], ]

    # 数据获取页面
    Date_collection = [
        [sg.Text("商品ID：    "), sg.Input(key="-PRODUCT_ID-", size=(15, 5))],
        [sg.Text("爬取页数："), sg.Input(key="-PAGES-", size=(15, 5))],
        [sg.Text("评价类型："), sg.Combo(["全部评价", "追评", "好评", "中评", "差评"],
                                        key="-COMMENT_TYPE-",
                                        default_value="全部评价",
                                        size=(15, 5))],
        [sg.Text("排序方式："), sg.Combo(["默认排序", "时间排序"],
                                        key="-SORT_TYPE-",
                                        default_value="默认排序",
                                        size=(15, 5))],
        [sg.Button("开始运行", key="-START_CRAWL-"),
         sg.ProgressBar(100, orientation='h', size=(20, 20), key="-PROGRESS-")],
        [sg.Text("保存路径："), sg.Text("", key="-SAVE_PATH-", size=(50, 5))],
        [sg.StatusBar("就绪", key="-STATUS-")]
    ]

    # 文本识别界面
    # 文本识别界面 News_detection 修改后的布局
    News_detection = [
        [sg.Menu(menu_def, tearoff=True)],
        [sg.Text('')],
        [sg.Text("选择模型："), sg.Combo(["SVM-质量优先", "SVM-速度优先", "SVM-均衡", "LSTM"],
                                        default_value="SVM-质量优先", key="-MODEL_TYPE-", size=(15, 1))],
        [sg.Multiline(s=(60, 20), key='_INPUT_news_', expand_x=True)],
        [sg.Text('')],
        [sg.Text('', s=(12)),
         sg.Text('识别结果：', font=("Helvetica", 15)),
         sg.Text('     ', key='_OUTPUT_RESULT_', font=("Helvetica", 15)),  # 新增结果显示
         sg.Text('置信度：', font=("Helvetica", 15)),
         sg.Text('     ', key='_OUTPUT_CONFIDENCE_', font=("Helvetica", 15))],  # 新增置信度显示
        [sg.Text('')],
        [sg.Text('', s=(12)), sg.Button('识别', font=("Helvetica", 15)), sg.Text('', s=(10)),
         sg.Button('清空', font=("Helvetica", 15)),
         sg.Text('', s=(4))],
        [sg.Text('')],
        [sg.Sizegrip()]
    ]

    # 文本识别内容的管理，可以查看自己识别的内容
    News_management = [
        [sg.Table(
            values=read_table_data('table_data.csv')[1:][:],
            headings=['文本内容', '识别模型', '识别结果', '置信度', '识别时间'],
            max_col_width=100,
            auto_size_columns=False,
            col_widths=[60, 15, 10, 10, 10],  # 调整列宽
            display_row_numbers=False,
            justification='left',
            num_rows=20,
            key='-TABLE_de-',
            selected_row_colors='red on yellow',
            enable_events=True,
            expand_x=True,
            expand_y=True,
            vertical_scroll_only=False,
            enable_click_events=True
        )],
        [sg.Button('删除选中的结果', font=("Helvetica", 15)), sg.Button('刷新', font=("Helvetica", 15))],
        [sg.Sizegrip()]
    ]

    layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
              [sg.Text('情感识别系统', size=(50, 1), justification='center', font=("Helvetica", 16),
                       relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True, expand_x=True)]]
    layout += [[sg.TabGroup([[
        sg.Tab(' 数 据 采 集 ', Date_collection),
        sg.Tab(' 文 本 识 别 ', News_detection),
        # sg.Tab('                                                     ', empty),
        sg.Tab(' 结 果 管 理  ', News_management, element_justification="right",)]], expand_x=True, expand_y=True, font=("Helvetica", 16)),

    ]]
    # layout[-1].append(sg.Sizegrip())
    window = sg.Window('情感识别系统', layout, size=(1200, 800),
                       right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0, 0),
                       use_custom_titlebar=True, finalize=True, keep_on_top=True)
    window.set_min_size(window.size)
    return window

def main_WINDOW(dataset_name):
    window = make_window(sg.theme())

    crawl_queue = Queue()  # 结果队列
    progress_queue = Queue()  # 进度队列
    crawl_thread = None

    while True:
        event, values = window.read(timeout=100)

        # 处理进度更新
        if not progress_queue.empty():
            msg_type, value = progress_queue.get()
            if msg_type == "PROGRESS":
                window["-PROGRESS-"].update(value)
                window["-STATUS-"].update(f"爬取中... {value}%")

        # 处理线程消息
        if not crawl_queue.empty():
            status, msg = crawl_queue.get()
            if status == "SUCCESS":
                window["-STATUS-"].update("爬取完成")
                window["-SAVE_PATH-"].update(msg)
                window["-PROGRESS-"].update(100)
                sg.popup(f"评论已保存到：\n{msg}")
            else:
                window["-STATUS-"].update("爬取失败")
                sg.popup_error(f"错误：{msg}")

        # 启动爬虫线程
        if event == "-START_CRAWL-":
            params = {
                "product_id": values["-PRODUCT_ID-"],
                "pages": int(values["-PAGES-"]),
                "comment_type": values["-COMMENT_TYPE-"],
                "sort_type": values["-SORT_TYPE-"]
            }

            # 参数验证
            if not all(params.values()):
                sg.popup_error("请填写所有参数！")
                continue

            # 初始化进度条
            window["-STATUS-"].update("爬取进行中...")
            window["-PROGRESS-"].update(0)

            # 启动线程（传入两个队列）
            crawl_thread = CrawlerThread(crawl_queue, progress_queue, params)
            crawl_thread.start()

        if event in (None, 'Exit'):
            print("[LOG] Clicked Exit!")
            break


        elif event == '识别':
            # 映射模型类型
            model_mapping = {
                "SVM-质量优先": "quality",
                "SVM-速度优先": "speed",
                "SVM-均衡": "balance",
                "LSTM": "lstm"
            }
            model_type = model_mapping.get(values["-MODEL_TYPE-"], "balance")

            input_text = values['_INPUT_news_']

            try:
                if model_type == "lstm":
                    from LSTM.main import predict_lstm

                    # 验证必要文件存在
                    lstm_base = os.path.join(os.path.dirname(__file__), "..", "LSTM")
                    required_files = [
                        os.path.join(lstm_base, "data", "embedding_Tencent.npz"),
                        os.path.join(lstm_base, "saved_dict", "lstm.ckpt"),
                        os.path.join(lstm_base, "data", "vocab.pkl")
                    ]
                    missing_files = [f for f in required_files if not os.path.exists(f)]
                    if missing_files:
                        sg.popup_error("LSTM模型文件缺失：\n" + "\n".join(missing_files))
                        continue
                    # 调用LSTM预测函数
                    label, confidence = predict_lstm(input_text)
                    prediction = f"{label}"
                    confidence = f"{confidence:.2%}"
                else:
                    # 原有SVM预测逻辑
                    raw_result = predict_svm(input_text, dataset_name, model_type)
                    result_lines = raw_result.split('\n')
                    prediction = result_lines[0].split('预测结果：')[-1].strip() if '预测结果：' in result_lines[
                        0] else raw_result
                    confidence = result_lines[1].split('置信度：')[-1].strip() if len(result_lines) > 1 else "N/A"
                # 更新UI显示
                window['_OUTPUT_RESULT_'].update(prediction)
                window['_OUTPUT_CONFIDENCE_'].update(confidence)

                # 保存结果到CSV
                time_str = time.strftime('%Y-%m-%d %H:%M:%S')
                new_record = [
                    values['_INPUT_news_'],
                    "\t" + prediction,
                    "\t" + model_type,
                    "\t" + confidence,
                    time_str
                ]
                # 确保文件存在并写入列头
                csv_path = '../JD/txtDate/table_data.csv'
                file_exists = os.path.exists(csv_path)

                with open(csv_path, 'a', newline='', encoding='gbk') as f:
                    writer = csv.writer(f)
                    # 如果文件不存在，写入表头
                    if not file_exists:
                        writer.writerow(['文本内容', '识别模型', '识别结果', '置信度', '识别时间'])
                    # 使用csv自带的分隔符处理
                    writer.writerow(new_record)
                # 刷新表格
                window["-TABLE_de-"].update(values=read_table_data(csv_path)[1:])

            except Exception as e:
                sg.popup_error(f"预测失败：{str(e)}")
                window['_OUTPUT_RESULT_'].update("预测失败")
                window['_OUTPUT_CONFIDENCE_'].update("N/A")

        elif event == '清空':
            window['_OUTPUT_RESULT_'].update('')
            window['_OUTPUT_CONFIDENCE_'].update('')
            window['_INPUT_news_'].update('')


        elif event == '删除选中的结果':
            csv_path = '../JD/txtDate/table_data.csv'
            try:
                # 使用更安全的csv读取方式
                with open(csv_path, 'r', encoding='gbk') as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                if values['-TABLE_de-'] and len(rows) > 1:  # 确保有数据
                    selected_row = values['-TABLE_de-'][0]
                    # 检查选中行号有效性
                    if 0 <= selected_row < len(rows) - 1:  # 减1因为包含表头
                        del rows[selected_row + 1]  # 加1因为rows[0]是表头
                        # 使用csv重新写入
                        with open(csv_path, 'w', newline='', encoding='gbk') as f:
                            writer = csv.writer(f)
                            writer.writerows(rows)
                        window["-TABLE_de-"].update(values=rows[1:])
                    else:
                        sg.popup_error("无效的行选择")

            except Exception as e:

                sg.popup_error(f"删除失败：{str(e)}")

        elif event == '刷新':
            window["-TABLE_de-"].update(values=read_table_data('../JD/txtDate/table_data.csv')[1:][:])

    window.close()
    exit(0)

if __name__ == '__main__':
    # 选择数据集
    # dataset_name = input("请输入数据集名称（不带扩展名）：").strip()
    dataset_name = "dateJD"

    # 特征化原始数据
    # read_data(dataset_name)
    # word2vec_(dataset_name)

    # 选择模型
    # classification_balance(dataset_name)
    # classification_speed(dataset_name)
    # classification_quality(dataset_name)

    # 测试模型预测
    # text_samples = [
    #     "这次购物体验真的超乎预期！商品质量非常好，细节做工精致，完全符合描述，甚至比想象中还要满意。",
    #     "不知道为什么，有时候切换到照相功能的时候就卡住，我是国补的时候买的，感觉还没我的13好用，不敢肯定是不是全新机了",
    #     "短时间内价格持续变动，不保值，建议大家换平台购买吧"
    # ]
    # for text in text_samples:
    #     print("平衡模型:", predict_svm(text, dataset_name, "balance"))
    #     print("速度模型:", predict_svm(text, dataset_name, "speed"))
    #     print("质量模型:", predict_svm(text, dataset_name, "quality"))
    #     print("-" * 50)

    # 可视化系统
    sg.theme()
    main_WINDOW(dataset_name)

