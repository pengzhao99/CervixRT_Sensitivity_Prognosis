"""
特征筛选、建模与评估程序 - 独立可配置的机器学习流水线

功能：
    - 支持多个特征文件的合并（取共同患者）
    - 支持指定标签文件
    - 数据划分为训练集和测试集（7:3）
    - 在训练集上进行特征筛选（单变量检验 → FDR校正 → 相关性分析 → LASSO）
    - 基于训练集使用五折交叉验证确定最佳超参数
    - 使用最优参数和筛选特征在整个训练集上重新训练最终模型
    - 在测试集上进行最终评估
    - 输出LASSO系数路径图和CV误差图、相关性热图、特征重要性图、
      AUC曲线图、校准曲线图、决策曲线图
    - 保存所有训练模型到 models/ 子目录，支持在新数据集上预测

使用方式：
    # 训练模式（默认）
    python model_pipeline.py

    # 预测模式 - 使用单个模型预测
    python model_pipeline.py predict --model <模型pkl路径> --data <新数据csv路径>

    # 预测模式 - 批量预测（目录下所有模型）
    python model_pipeline.py predict --models-dir <models目录> --data <新数据csv路径>

    # 预测模式 - 带真实标签评估
    python model_pipeline.py predict --models-dir <models目录> --data <新数据csv路径> --label <标签xlsx>
"""

import os
import logging
import warnings
import pickle
import json
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    brier_score_loss
)
from scipy import stats
from statsmodels.stats.multitest import multipletests
import xgboost as xgb

from plot_utils import (
    plot_lasso_figures,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_roc_curves,
    plot_calibration_curve,
    plot_decision_curve,
    plot_rfe_cv_curve,
)

# ============================================================================
# 参数设置区域 - 根据需要修改以下参数
# ============================================================================

PARAMS = {
    # ---- 输入数据 ----
    'feature_files': [
        r'D:\projects\CervixRT_Sensitivity_Prognosis\results\extracted_features\ct_features.csv',
    ],

    # 标签文件
    'label_file': r'E:\data\CR-PR\CT\sensitivity_label.xlsx',

    # 标签编码
    'label_encoding': {
        'CR': 0,
        'PR': 1,
    },

    # ---- 数据划分 ----
    'test_ratio': 0.20,            # 测试集比例（8:2划分）
    'random_state': 42,            # 随机种子
    # 数据划分文件（JSON格式）：若存在则从中读取训练集/测试集ID划分，若为None则重新划分数据集
    'data_split_file': r"D:\projects\CervixRT_Sensitivity_Prognosis\results\data_split.json",
    # 类别不平衡处理：'balanced' 自动根据类别比例调整权重，None 不调整
    'class_weight': 'balanced',

    # ---- 特征筛选参数 ----
    'feature_selection': {
        # 手动指定的分类变量列表（在列表中的特征使用卡方检验，不在列表中的使用Mann-Whitney U检验）
        # 设为 None 或空列表 [] 则全部特征使用 Mann-Whitney U 检验
        'categorical_features': ['Pathology', 'HPV', 'Targeted_therapy', 'Chemotherapy',
                                 'Immunotherapy', 'BinaryOfSyn', 'NACT', 'ACT', 'MT'],

        # ICC筛选特征文件路径（仅保留ICC满足要求的特征，None则跳过ICC筛选）
        'icc_features_file': None,

        # 排除包含以下关键词的特征（在筛选前先移除，设为空列表[]则不排除任何特征）
        # 例如: ['wavelet', 'log-sigma'] 表示排除所有wavelet和LoG特征
        'exclude_features': ['wavelet', 'log-sigma', 'MT'],
        'p_threshold': 0.05,       # Mann-Whitney U 检验 p值阈值
        'use_fdr_correction': True,  # 是否使用FDR校正
        'fdr_alpha': 0.05,           # FDR校正显著性水平
        'corr_threshold': 0.85,    # 相关系数阈值（|r| > 此值的特征剔除）
        # 相关性筛选中特征保留优先级（从高到低）
        # 匹配规则：按顺序匹配，越靠前优先级越高
        'feature_priority': [
            'CT_GTV_original_shape_VoxelVolume',          # GTV体积
            'CT_Ring_original_shape_VoxelVolume',         # Ring体积
            'GTV',
            'Ring',
            'log-sigma',           # LoG特征
            'wavelet',             # Wavelet特征
        ],
        # 不需要标准化的连续变量列表
        'no_scale_features': [],

        'use_lasso': True,         # 是否使用LASSO
        'lasso_cv_folds': 5,       # LASSO 交叉验证折数
        'lasso_alpha': None,       # LASSO alpha（None=自动选择）

        # 递归特征消除 (RFE) 配置
        'use_rfe': False,           # 是否使用递归特征消除
        'rfe_estimator': 'logistic_regression',  # RFE 使用的模型: 'logistic_regression' 或 'random_forest'
        'rfe_cv_folds': 5,         # RFE 交叉验证折数
        'rfe_n_features': None,    # 指定最终特征数（None=自动选择CV AUC最高的）
        'rfe_step': 1,             # 每次迭代移除的特征数
    },

    # ---- 模型配置 ----
    'models': {
        'Logistic Regression': {
            'enabled': True,
            'param_grid': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2'],
                'solver': ['liblinear'],
                'max_iter': [2000],
            }
        },
        'SVM': {
            'enabled': True,
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1],
                'kernel': ['rbf', 'linear'],
                'probability': [True],
            }
        },
        'Random Forest': {
            'enabled': True,
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [2, 3],
                'min_samples_split': [2, 3],
                'min_samples_leaf': [2, 3],
            }
        },
        'XGBoost': {
            'enabled': True,
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [2, 3],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            }
        },
        'TabPFN': {
            'enabled': True,
            'param_grid': None,
        },
    },

    # 模型训练通用设置
    'cv_folds': 5,                     # 交叉验证折数
    'scoring': 'roc_auc',              # 评估指标
    'n_jobs': -1,                      # 并行任务数

    # ---- 输出配置 ----
    'output_dir': r'D:\projects\CervixRT_Sensitivity_Prognosis\results\model_radiomics',
}

# ============================================================================
# 以下为程序逻辑，一般不需要修改
# ============================================================================

logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



def load_and_merge_features(params):
    """加载并合并多个特征文件"""
    feature_files = params['feature_files']
    label_file = params.get('label_file')
    label_encoding = params['label_encoding']

    if not feature_files:
        raise ValueError("必须指定至少一个特征文件！")

    dfs = []
    for fpath in feature_files:
        if not os.path.exists(fpath):
            logger.warning(f"特征文件不存在，跳过: {fpath}")
            continue
        df = pd.read_csv(fpath, converters={'ID': str})
        logger.info(f"读取特征文件: {fpath} ({df.shape[0]} 患者, {df.shape[1]} 列)")
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("没有有效的特征文件可加载！")

    if len(dfs) == 1:
        merged = dfs[0]
    else:
        merged = dfs[0]
        for i, df in enumerate(dfs[1:], start=2):
            meta_cols = ['ID', 'Label', 'Label_Encoded']
            feature_cols = [c for c in df.columns if c not in meta_cols]
            merge_df = df[['ID'] + feature_cols]
            merged = pd.merge(merged, merge_df, on='ID', how='inner')
            logger.info(f"合并第 {i} 个文件后: {merged.shape[0]} 共同患者, {merged.shape[1]} 列")

    # 确保有标签信息
    if 'Label_Encoded' not in merged.columns or merged['Label_Encoded'].isna().any():
        if label_file and os.path.exists(label_file):
            logger.info(f"从标签文件加载标签: {label_file}")
            labels_df = pd.read_excel(label_file, converters={'ID': str})
            labels_df['Label_Encoded'] = labels_df['Label'].map(label_encoding)
            merged = merged.drop(columns=['Label', 'Label_Encoded'], errors='ignore')
            merged = pd.merge(merged, labels_df[['ID', 'Label', 'Label_Encoded']], on='ID', how='inner')
        else:
            raise ValueError("特征文件缺少标签信息且未指定标签文件！")

    merged = merged.dropna(subset=['Label_Encoded'])
    merged['Label_Encoded'] = merged['Label_Encoded'].astype(int)

    logger.info(f"最终数据: {merged.shape[0]} 患者, 标签分布: CR={sum(merged['Label_Encoded']==0)}, PR={sum(merged['Label_Encoded']==1)}")
    return merged


def split_data(df, params):
    """
    将数据划分为训练集和测试集
    - 若 data_split_file 不为 None 且文件存在，则从 JSON 文件中读取已有的数据划分
    - 否则重新划分数据集，并将训练集和测试集 ID 保存为 JSON 文件到 output_dir
    """
    test_ratio = params['test_ratio']
    random_state = params['random_state']
    output_dir = params['output_dir']
    data_split_file = params.get('data_split_file', None)

    # 若指定了划分文件且存在，则从中读取
    if data_split_file is not None and os.path.exists(data_split_file):
        logger.info(f"从文件加载数据划分: {data_split_file}")
        with open(data_split_file, 'r', encoding='utf-8') as f:
            split_data_dict = json.load(f)

        train_ids = split_data_dict['train_ids']
        test_ids = split_data_dict['test_ids']

        train_df = df[df['ID'].isin(train_ids)].copy()
        test_df = df[df['ID'].isin(test_ids)].copy()

        # 检查是否有未匹配的ID
        loaded_ids = set(train_ids) | set(test_ids)
        all_ids = set(df['ID'].values)
        missing_ids = all_ids - loaded_ids
        if missing_ids:
            logger.warning(f"以下 {len(missing_ids)} 个患者ID未包含在划分文件中，已被忽略: {missing_ids}")

        logger.info(f"从划分文件加载完成:")
        logger.info(f"  训练集: {len(train_df)} 样本 (CR={sum(train_df['Label_Encoded']==0)}, PR={sum(train_df['Label_Encoded']==1)})")
        logger.info(f"  测试集: {len(test_df)} 样本 (CR={sum(test_df['Label_Encoded']==0)}, PR={sum(test_df['Label_Encoded']==1)})")
        return train_df, test_df

    # 重新划分数据集
    y = df['Label_Encoded'].values

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_ratio,
        random_state=random_state,
        stratify=y
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # 将训练集和测试集ID保存为JSON文件
    os.makedirs(output_dir, exist_ok=True)
    split_json_path = os.path.join(output_dir, 'data_split.json')
    split_dict = {
        'train_ids': train_df['ID'].tolist(),
        'test_ids': test_df['ID'].tolist(),
        'test_ratio': test_ratio,
        'random_state': random_state,
    }
    with open(split_json_path, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"数据划分已保存: {split_json_path}")

    logger.info(f"数据划分完成（{1-test_ratio:.0%}:{test_ratio:.0%}）:")
    logger.info(f"  训练集: {len(train_df)} 样本 (CR={sum(train_df['Label_Encoded']==0)}, PR={sum(train_df['Label_Encoded']==1)})")
    logger.info(f"  测试集: {len(test_df)} 样本 (CR={sum(test_df['Label_Encoded']==0)}, PR={sum(test_df['Label_Encoded']==1)})")

    return train_df, test_df


def get_feature_priority(feature_name, priority_list):
    """根据优先级列表返回特征的优先级分数（越小优先级越高）"""
    feature_lower = feature_name.lower()
    for idx, pattern in enumerate(priority_list):
        if pattern.lower() == feature_lower:
            return idx
        elif pattern.lower() in feature_lower:
            return idx
    return len(priority_list)


def classify_feature_types(feature_cols, categorical_list):
    """
    将特征列表分为分类变量和连续变量

    参数:
        feature_cols: 所有特征列名列表
        categorical_list: 手动指定的分类变量列表（None或[]则无分类变量）

    返回:
        tuple: (categorical_features, continuous_features)
    """
    if not categorical_list:
        return [], list(feature_cols)
    cat_set = set(categorical_list)
    cat_features = [f for f in feature_cols if f in cat_set]
    cont_features = [f for f in feature_cols if f not in cat_set]
    return cat_features, cont_features


def classify_categorical_detail(df, cat_features):
    """
    将分类变量细分为二分类和多分类

    返回:
        tuple: (binary_cat_features, multi_cat_features)
    """
    binary_cat = []
    multi_cat = []
    for col in cat_features:
        n_unique = df[col].dropna().nunique()
        if n_unique <= 2:
            binary_cat.append(col)
        else:
            multi_cat.append(col)
    return binary_cat, multi_cat


def build_mixed_feature_matrix(df, selected_features, cat_features, scaler=None, fit=False,
                               no_scale_features=None, multi_cat_categories=None):
    """
    构建混合类型特征矩阵:
      - 连续变量 → StandardScaler 标准化（no_scale_features 中的特征跳过标准化）
      - 二分类变量 → 保持原值（0/1编码）
      - 多分类变量 → one-hot编码（drop_first=True，类别与训练集保持一致）

    参数:
        df: 数据 DataFrame
        selected_features: 选中的特征列表
        cat_features: 分类变量列表（在selected_features中的分类变量）
        scaler: StandardScaler 对象（fit=True时可为None，fit=False时必须传入）
        fit: True=拟合scaler，False=仅transform
        no_scale_features: 不需要标准化的特征列表（保持原始值）
        multi_cat_categories: 多分类变量的类别字典 {col: sorted_categories_list}
                              fit=True时自动捕获并返回，fit=False时传入以保证one-hot编码一致

    返回:
        tuple: (X_matrix, expanded_feature_names, scaler, multi_cat_cols, multi_cat_categories)
    """
    binary_cat, multi_cat = classify_categorical_detail(df, cat_features)
    cont_features = [f for f in selected_features if f not in cat_features]
    # 仅保留在 selected_features 中的分类变量
    binary_cat = [f for f in binary_cat if f in selected_features]
    multi_cat = [f for f in multi_cat if f in selected_features]

    # 将连续变量分为需要标准化和不需要标准化的两组
    no_scale_set = set(no_scale_features) if no_scale_features else set()
    cont_scale = [f for f in cont_features if f not in no_scale_set]
    cont_no_scale = [f for f in cont_features if f in no_scale_set]

    parts = []
    expanded_names = []

    # 1a. 连续变量（需要标准化）：StandardScaler
    if cont_scale:
        X_cont = df[cont_scale].apply(pd.to_numeric, errors='coerce').values
        if fit:
            scaler = StandardScaler()
            X_cont_scaled = scaler.fit_transform(X_cont)
        else:
            X_cont_scaled = scaler.transform(X_cont)
        parts.append(X_cont_scaled)
        expanded_names.extend(cont_scale)

    # 1b. 连续变量（不需要标准化）：保持原始值
    if cont_no_scale:
        X_no_scale = df[cont_no_scale].apply(pd.to_numeric, errors='coerce').values
        parts.append(X_no_scale)
        expanded_names.extend(cont_no_scale)

    # 2. 二分类变量：保持原值
    if binary_cat:
        X_bin = df[binary_cat].apply(pd.to_numeric, errors='coerce').values
        parts.append(X_bin)
        expanded_names.extend(binary_cat)

    # 3. 多分类变量：one-hot编码（保证与训练集类别一致）
    multi_cat_cols = []
    if multi_cat_categories is None:
        multi_cat_categories = {}

    if multi_cat:
        for col in multi_cat:
            if fit:
                # 训练模式：捕获排序后的唯一类别值，确保可复现
                categories = sorted(df[col].dropna().unique().tolist())
                multi_cat_categories[col] = categories
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            else:
                # 预测模式：使用训练时保存的类别列表，保证 one-hot 列完全一致
                if col in multi_cat_categories:
                    categories = multi_cat_categories[col]
                    # 用 pd.Categorical 强制使用训练集的类别（含训练集中有但新数据没有的类别）
                    cat_series = pd.Series(
                        pd.Categorical(df[col].values, categories=categories),
                        index=df.index
                    )
                    # 检查新数据中是否出现训练集未见过的类别
                    new_vals = set(df[col].dropna().unique()) - set(categories)
                    if new_vals:
                        logger.warning(f"  多分类变量 '{col}' 存在训练集未见过的类别: {new_vals}，这些样本对应行将为全0")
                    dummies = pd.get_dummies(cat_series, prefix=col, drop_first=True)
                    # 确保列与训练集完全一致（补齐新数据缺失的虚拟列）
                    expected_cols = [f"{col}_{c}" for c in categories[1:]]  # drop_first 跳过第一个
                    for ec in expected_cols:
                        if ec not in dummies.columns:
                            dummies[ec] = 0
                    dummies = dummies[expected_cols]  # 保证列顺序一致
                else:
                    # 向后兼容：旧模型 pkl 中没有 multi_cat_categories，退化为原始逻辑
                    logger.warning(f"  多分类变量 '{col}' 缺少训练类别信息，使用当前数据编码（可能与训练集不一致）")
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            parts.append(dummies.values.astype(float))
            expanded_names.extend(dummies.columns.tolist())
            multi_cat_cols.append(col)

    if not parts:
        return np.array([]), [], scaler, multi_cat_cols, multi_cat_categories

    X_matrix = np.hstack(parts)
    return X_matrix, expanded_names, scaler, multi_cat_cols, multi_cat_categories


def recursive_feature_elimination(train_df, selected_features, y, params):
    """
    递归特征消除（RFE）+ 交叉验证，选择最优特征子集

    流程：
    1. 在全部训练数据上运行一次完整RFE(n_features_to_select=1)，得到唯一的特征淘汰排序
    2. 基于该排序，对每个特征数N，取top-N特征，用CV评估AUC
    3. 选中特征、排序文件、AUC曲线三者完全一致

    参数:
        train_df: 训练集 DataFrame
        selected_features: 当前已选中的特征列表（LASSO或相关性筛选后）
        y: 标签数组
        params: 全局参数字典

    返回:
        tuple: (rfe_selected_features, rfe_report_dict)
    """
    fs_params = params['feature_selection']
    rfe_estimator_name = fs_params.get('rfe_estimator', 'logistic_regression')
    rfe_cv_folds = fs_params.get('rfe_cv_folds', 5)
    rfe_n_features = fs_params.get('rfe_n_features', None)
    rfe_step = fs_params.get('rfe_step', 1)
    random_state = params['random_state']
    class_weight = params.get('class_weight')

    logger.info(f"\n--- 第4阶段: 递归特征消除 (RFE) ---")
    logger.info(f"  RFE 模型: {rfe_estimator_name}")
    logger.info(f"  交叉验证折数: {rfe_cv_folds}")
    logger.info(f"  输入特征数: {len(selected_features)}")

    if len(selected_features) <= 1:
        logger.warning("  输入特征数 <= 1，跳过RFE")
        return selected_features, {}

    # 准备数据
    X_rfe = train_df[selected_features].apply(pd.to_numeric, errors='coerce').values
    scaler_rfe = StandardScaler()
    X_rfe_scaled = scaler_rfe.fit_transform(X_rfe)

    # 初始化RFE基模型
    if rfe_estimator_name == 'logistic_regression':
        estimator = LogisticRegression(
            random_state=random_state,
            max_iter=2000,
            class_weight=class_weight,
            solver='liblinear',
            penalty='l2',
            C=1.0
        )
    elif rfe_estimator_name == 'random_forest':
        estimator = RandomForestClassifier(
            random_state=random_state,
            n_estimators=100,
            class_weight=class_weight,
            n_jobs=-1
        )
    else:
        raise ValueError(f"不支持的RFE模型: {rfe_estimator_name}，请选择 'logistic_regression' 或 'random_forest'")

    n_features_total = len(selected_features)

    # ===== 步骤1: 在全部训练数据上运行完整RFE，获取唯一的特征淘汰排序 =====
    logger.info(f"  在全部训练数据上运行完整RFE，确定特征淘汰顺序...")
    full_rfe = RFE(
        estimator=estimator,
        n_features_to_select=1,
        step=rfe_step
    )
    full_rfe.fit(X_rfe_scaled, y)
    full_ranking = full_rfe.ranking_  # rank=1最重要，rank=N最先被淘汰

    # 按ranking排序得到特征重要性顺序（rank小 = 重要）
    feature_order = np.argsort(full_ranking)  # 索引按ranking升序

    logger.info(f"  特征淘汰排序（从最重要到最先淘汰）:")
    for idx in feature_order:
        logger.info(f"    rank={full_ranking[idx]}: {selected_features[idx]}")

    # ===== 步骤2: 基于固定排序，对每个N用CV评估AUC =====
    cv = StratifiedKFold(n_splits=rfe_cv_folds, shuffle=True, random_state=random_state)
    min_features = 1
    max_features = n_features_total
    feature_range = list(range(min_features, max_features + 1))

    cv_auc_means = []
    cv_auc_stds = []
    cv_auc_all = []

    logger.info(f"  评估特征数范围: {min_features} ~ {max_features}")

    for n_feat in feature_range:
        # 取ranking <= n_feat的特征索引（即top n_feat个最重要的特征）
        feat_indices = [i for i in range(n_features_total) if full_ranking[i] <= n_feat]
        X_subset = X_rfe_scaled[:, feat_indices]

        # 交叉验证计算AUC
        fold_aucs = []
        for train_idx, val_idx in cv.split(X_subset, y):
            X_train_fold = X_subset[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X_subset[val_idx]
            y_val_fold = y[val_idx]

            if rfe_estimator_name == 'logistic_regression':
                clf = LogisticRegression(
                    random_state=random_state,
                    max_iter=2000,
                    class_weight=class_weight,
                    solver='liblinear',
                    penalty='l2',
                    C=1.0
                )
            else:
                clf = RandomForestClassifier(
                    random_state=random_state,
                    n_estimators=100,
                    class_weight=class_weight,
                    n_jobs=-1
                )

            clf.fit(X_train_fold, y_train_fold)
            y_val_proba = clf.predict_proba(X_val_fold)[:, 1]

            try:
                auc = roc_auc_score(y_val_fold, y_val_proba)
            except ValueError:
                auc = 0.5
            fold_aucs.append(auc)

        cv_auc_means.append(np.mean(fold_aucs))
        cv_auc_stds.append(np.std(fold_aucs))
        cv_auc_all.append(fold_aucs)

    cv_auc_means = np.array(cv_auc_means)
    cv_auc_stds = np.array(cv_auc_stds)

    # ===== 步骤3: 确定最优特征数，选中特征 =====
    if rfe_n_features is not None and 1 <= rfe_n_features <= n_features_total:
        optimal_n_features = rfe_n_features
        logger.info(f"  使用指定的特征数: {optimal_n_features}")
    else:
        best_idx = np.argmax(cv_auc_means)
        optimal_n_features = feature_range[best_idx]
        logger.info(f"  自动选择最优特征数: {optimal_n_features} (CV AUC = {cv_auc_means[best_idx]:.4f} +/- {cv_auc_stds[best_idx]:.4f})")

    # 从完整排序中取 top optimal_n_features（ranking <= optimal_n_features）
    rfe_selected_features = [
        selected_features[i] for i in range(n_features_total)
        if full_ranking[i] <= optimal_n_features
    ]

    logger.info(f"  RFE 最终选中特征: {len(rfe_selected_features)} 个")
    for i, f in enumerate(rfe_selected_features, 1):
        logger.info(f"    {i}. {f}")

    rfe_report = {
        'rfe_estimator': rfe_estimator_name,
        'rfe_cv_folds': rfe_cv_folds,
        'feature_range': feature_range,
        'cv_auc_means': cv_auc_means,
        'cv_auc_stds': cv_auc_stds,
        'cv_auc_all': cv_auc_all,
        'optimal_n_features': optimal_n_features,
        'rfe_selected_features': rfe_selected_features,
        'rfe_full_ranking': full_ranking,
        'input_features': selected_features,
    }

    return rfe_selected_features, rfe_report


def feature_selection(train_df, params):
    """
    在训练集上进行特征筛选（单变量 → FDR → 相关性 → LASSO）

    返回:
        tuple: (selected_features, selection_report_dict)
    """
    fs_params = params['feature_selection']
    meta_cols = ['ID', 'Label', 'Label_Encoded']
    feature_cols = [c for c in train_df.columns if c not in meta_cols]

    # 排除指定类型的特征
    exclude_patterns = fs_params.get('exclude_features', [])
    if exclude_patterns:
        before_count = len(feature_cols)
        feature_cols = [
            c for c in feature_cols
            if not any(pat.lower() in c.lower() for pat in exclude_patterns)
        ]
        excluded_count = before_count - len(feature_cols)
        logger.info(f"排除特征模式 {exclude_patterns}: 移除 {excluded_count} 个，剩余 {len(feature_cols)} 个")

    logger.info("=" * 60)
    logger.info(f"开始特征筛选（初始特征数: {len(feature_cols)}）")
    logger.info("=" * 60)

    y = train_df['Label_Encoded'].values
    report = {}

    # ---- ICC特征筛选 ----
    icc_features_file = fs_params.get('icc_features_file', None)
    if icc_features_file is not None:
        if os.path.exists(icc_features_file):
            icc_df = pd.read_csv(icc_features_file)
            icc_feature_list = set(icc_df['Feature'].astype(str).tolist())
            before_count = len(feature_cols)
            feature_cols = [c for c in feature_cols if c in icc_feature_list]
            removed_count = before_count - len(feature_cols)
            logger.info(f"ICC特征筛选: 仅保留ICC满足要求的特征，移除 {removed_count} 个，剩余 {len(feature_cols)} 个")
            report['icc_filter_applied'] = True
            report['icc_features_file'] = icc_features_file
            report['icc_removed_count'] = removed_count
        else:
            logger.warning(f"ICC特征文件不存在，跳过ICC筛选: {icc_features_file}")
            report['icc_filter_applied'] = False
    else:
        logger.info("ICC特征文件路径为None，跳过ICC筛选")
        report['icc_filter_applied'] = False

    # ---- 第0阶段：分类特征类型 ----
    cat_features_list = fs_params.get('categorical_features', None)
    cat_features, cont_features = classify_feature_types(feature_cols, cat_features_list)
    if cat_features:
        logger.info(f"分类变量 ({len(cat_features)}): {cat_features}")
    logger.info(f"连续变量 ({len(cont_features)}): {cont_features}")
    report['cat_features'] = cat_features
    report['cont_features'] = cont_features

    # ---- 第1阶段：单变量检验 ----
    p_values = {}
    univariate_stats = []  # 保存每个特征的统计量

    # 分类变量: 卡方检验
    if cat_features:
        logger.info(f"\n--- 第1a阶段: 卡方检验（分类变量）---")
        for col in cat_features:
            try:
                contingency = pd.crosstab(train_df[col], train_df['Label_Encoded'])
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                stat, p, dof, expected = stats.chi2_contingency(contingency)
                p_values[col] = p
                n = contingency.sum().sum()
                min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                cramers_v = np.sqrt(stat / (n * min_dim)) if min_dim > 0 else np.nan
                univariate_stats.append({
                    'Feature': col, 'Test': 'Chi-square', 'Stat': stat,
                    'P_value': p, 'Effect_Size': cramers_v,
                    'Significant': p < fs_params['p_threshold'],
                })
            except Exception:
                continue
        cat_selected = [f for f in cat_features if f in p_values and p_values[f] < fs_params['p_threshold']]
        logger.info(f"  卡方检验显著特征: {len(cat_selected)} / {len(cat_features)}")

    # 连续变量: Mann-Whitney U 检验
    logger.info(f"\n--- 第1b阶段: Mann-Whitney U 单变量检验（连续变量）---")
    for col in cont_features:
        vals = pd.to_numeric(train_df[col], errors='coerce').values
        group_0 = vals[y == 0]
        group_1 = vals[y == 1]
        group_0 = group_0[~np.isnan(group_0)]
        group_1 = group_1[~np.isnan(group_1)]

        if len(group_0) < 3 or len(group_1) < 3:
            continue
        try:
            _, p = stats.mannwhitneyu(group_0, group_1, alternative='two-sided')
            p_values[col] = p

            cr_median = np.median(group_0)
            cr_q1 = np.percentile(group_0, 25)
            cr_q3 = np.percentile(group_0, 75)
            pr_median = np.median(group_1)
            pr_q1 = np.percentile(group_1, 25)
            pr_q3 = np.percentile(group_1, 75)

            univariate_stats.append({
                'Feature': col, 'Test': 'Mann-Whitney U', 'Stat': _,
                'CR_Median': cr_median, 'CR_Q1': cr_q1, 'CR_Q3': cr_q3,
                'CR_IQR': f'{cr_median:.4f} ({cr_q1:.4f}-{cr_q3:.4f})',
                'PR_Median': pr_median, 'PR_Q1': pr_q1, 'PR_Q3': pr_q3,
                'PR_IQR': f'{pr_median:.4f} ({pr_q1:.4f}-{pr_q3:.4f})',
                'P_value': p,
                'Significant': p < fs_params['p_threshold'],
            })
        except Exception:
            continue

    selected_univariate = [f for f, p in p_values.items() if p < fs_params['p_threshold']]
    logger.info(f"  Mann-Whitney U 显著特征: {len([f for f in cont_features if f in p_values and p_values[f] < fs_params['p_threshold']])} / {len(cont_features)}")
    logger.info(f"  合计显著特征: {len(selected_univariate)} / {len(p_values)} 有效检验")
    report['univariate_count'] = len(selected_univariate)
    report['univariate_features'] = selected_univariate.copy()
    report['univariate_stats'] = univariate_stats

    # ---- 第1.5阶段：FDR校正 ----
    if fs_params.get('use_fdr_correction', False) and selected_univariate:
        logger.info(f"\n--- 第1.5阶段: FDR 校正 (alpha={fs_params['fdr_alpha']}) ---")
        # 对所有有效p值进行FDR校正
        all_features_tested = list(p_values.keys())
        all_p_vals = [p_values[f] for f in all_features_tested]

        reject, pvals_corrected, _, _ = multipletests(
            all_p_vals, alpha=fs_params['fdr_alpha'], method='fdr_bh'
        )

        selected_fdr = [f for f, r in zip(all_features_tested, reject) if r]
        logger.info(f"  FDR校正后显著特征: {len(selected_fdr)} / {len(all_features_tested)}")

        # 更新p_values为校正后的值（用于后续优先级判断）
        p_values_corrected = dict(zip(all_features_tested, pvals_corrected))

        # 将FDR校正后的p值添加到统计表中
        fdr_map = dict(zip(all_features_tested, pvals_corrected))
        fdr_reject_map = dict(zip(all_features_tested, reject))
        for row in univariate_stats:
            feat = row['Feature']
            row['FDR_P_value'] = fdr_map.get(feat, np.nan)
            row['FDR_Significant'] = bool(fdr_reject_map.get(feat, False))

        if selected_fdr:
            selected = selected_fdr
            # 使用校正后的p值
            p_values = p_values_corrected
        else:
            logger.warning("  FDR校正后无显著特征，回退使用未校正结果")
            selected = selected_univariate
        report['fdr_count'] = len(selected_fdr)
        report['fdr_features'] = selected_fdr.copy()
    else:
        selected = selected_univariate
        report['fdr_count'] = None

    if not selected:
        logger.warning("单变量/FDR筛选后无剩余特征！尝试放宽阈值至0.1")
        selected = [f for f, p in p_values.items() if p < 0.1]
        if not selected:
            logger.error("放宽阈值后仍无特征，返回空列表")
            return [], report

    # ---- 第2阶段：相关性分析（基于优先级）----
    logger.info(f"\n--- 第2阶段: 相关性分析 (阈值: {fs_params['corr_threshold']}) ---")
    priority_list = fs_params.get('feature_priority', [])

    X_sel = train_df[selected].apply(pd.to_numeric, errors='coerce')
    corr_matrix = X_sel.corr(method='pearson').abs()

    # 按优先级对特征排序（优先级高的排前面）
    feature_priorities = {f: get_feature_priority(f, priority_list) for f in selected}
    sorted_features = sorted(selected, key=lambda f: (feature_priorities[f], p_values.get(f, 1.0)))

    to_remove = set()
    corr_removal_report = {}  # {保留的特征: [被移除的特征列表]}

    for i, feat_i in enumerate(sorted_features):
        if feat_i in to_remove:
            continue
        removed_by_this = []
        for j in range(i + 1, len(sorted_features)):
            feat_j = sorted_features[j]
            if feat_j in to_remove:
                continue
            if feat_i in corr_matrix.columns and feat_j in corr_matrix.columns:
                corr_val = corr_matrix.loc[feat_i, feat_j]
                if corr_val > fs_params['corr_threshold']:
                    to_remove.add(feat_j)
                    removed_by_this.append(feat_j)
        if removed_by_this:
            corr_removal_report[feat_i] = removed_by_this

    selected = [f for f in sorted_features if f not in to_remove]
    logger.info(f"  移除高相关特征: {len(to_remove)} 个，剩余: {len(selected)} 个")
    report['corr_removed_count'] = len(to_remove)
    report['corr_removal_detail'] = corr_removal_report
    report['after_corr_features'] = selected.copy()

    if not selected:
        logger.error("相关性筛选后无剩余特征！")
        return [], report

    # ---- 第3阶段：LASSO ----
    lasso_cv_model = None
    if fs_params['use_lasso'] and len(selected) > 1:
        logger.info(f"\n--- 第3阶段: LASSO 特征选择（混合编码）---")
        y_lasso = y

        # 使用混合编码构建特征矩阵
        X_lasso, lasso_expanded_names, scaler_lasso, lasso_multi_cat, _ = build_mixed_feature_matrix(
            train_df, selected, cat_features, scaler=None, fit=True,
            no_scale_features=fs_params.get('no_scale_features')
        )

        lasso_cv = LassoCV(
            cv=fs_params['lasso_cv_folds'],
            random_state=params['random_state'],
            n_jobs=-1,
            max_iter=10000,
            n_alphas=100,
        )
        lasso_cv.fit(X_lasso, y_lasso)
        lasso_cv_model = lasso_cv

        best_alpha = lasso_cv.alpha_
        coefficients = lasso_cv.coef_
        logger.info(f"  LASSO 最优 alpha: {best_alpha:.6f}")

        # 将展开后的特征系数映射回原始特征名
        lasso_coefs = {}
        for i, c in enumerate(coefficients):
            if abs(c) > 1e-6 and i < len(lasso_expanded_names):
                lasso_coefs[lasso_expanded_names[i]] = c

        # 从展开名映射回原始特征名（多分类变量取最大绝对系数的one-hot列）
        original_coef = {}
        for feat_name, coef_val in lasso_coefs.items():
            # 检查是否是 one-hot 展开列（格式为 col_value）
            matched_multi = None
            for mc in lasso_multi_cat:
                if feat_name.startswith(mc + '_'):
                    matched_multi = mc
                    break
            if matched_multi:
                if matched_multi not in original_coef or abs(coef_val) > abs(original_coef[matched_multi]):
                    original_coef[matched_multi] = coef_val
            else:
                original_coef[feat_name] = coef_val

        lasso_selected = list(original_coef.keys())
        logger.info(f"  LASSO 选中特征: {len(lasso_selected)} / {len(selected)}")

        if lasso_selected:
            selected = lasso_selected
            report['lasso_coefs'] = original_coef
        else:
            logger.warning("  LASSO 未选出任何特征，保留相关性筛选后的全部特征")
            report['lasso_coefs'] = {}

        report['lasso_model'] = lasso_cv_model
        report['lasso_feature_names'] = lasso_expanded_names
        report['lasso_X_scaled'] = X_lasso
        report['lasso_y'] = y_lasso

    # ---- 第4阶段：递归特征消除 (RFE) ----
    if fs_params.get('use_rfe', False) and len(selected) > 1:
        rfe_selected, rfe_report = recursive_feature_elimination(
            train_df, selected, y, params
        )
        if rfe_selected:
            selected = rfe_selected
            report['rfe_report'] = rfe_report
        else:
            logger.warning("  RFE 未选出任何特征，保留上一阶段的全部特征")
            report['rfe_report'] = {}

    logger.info(f"\n最终选中特征数: {len(selected)}")
    for i, f in enumerate(selected, 1):
        p_val = p_values.get(f, None)
        if p_val is not None:
            logger.info(f"  {i}. {f} (p={p_val:.6f})")
        else:
            logger.info(f"  {i}. {f}")

    report['final_features'] = selected
    return selected, report


def initialize_model(model_name, random_state=42, class_weight=None, scale_pos_weight=None):
    """初始化模型（含class_weight处理类别不平衡）"""
    if model_name == 'Logistic Regression':
        return LogisticRegression(random_state=random_state, max_iter=2000, class_weight=class_weight)
    elif model_name == 'SVM':
        return SVC(random_state=random_state, probability=True, class_weight=class_weight)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(random_state=random_state, class_weight=class_weight)
    elif model_name == 'XGBoost':
        kwargs = {'random_state': random_state, 'eval_metric': 'logloss'}
        if scale_pos_weight is not None:
            kwargs['scale_pos_weight'] = scale_pos_weight
        return xgb.XGBClassifier(**kwargs)
    elif model_name == 'TabPFN':
        return None
    else:
        raise ValueError(f"未知模型: {model_name}")


def train_tabpfn(X_train, y_train, X_test, y_test, params):
    """使用TabPFN模型进行训练和预测"""
    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        logger.warning("TabPFN 未安装，跳过。请运行: pip install tabpfn")
        return None

    logger.info("  训练 TabPFN 模型（无需超参数调优）...")

    n_features = X_train.shape[1]
    if n_features > 100:
        logger.warning(f"  TabPFN 特征数限制为100，当前 {n_features} 个，将截断")
        X_train = X_train[:, :100]
        X_test = X_test[:, :100]

    if X_train.shape[0] > 1000:
        logger.warning(f"  TabPFN 训练样本限制为1000，当前 {X_train.shape[0]} 个，将随机采样")
        rng = np.random.RandomState(42)
        idx = rng.choice(X_train.shape[0], 1000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    try:
        model = TabPFNClassifier(device='auto', n_estimators=8)
        model.fit(X_train, y_train)

        # 训练集预测
        y_train_pred = model.predict(X_train)
        y_train_pred_proba = model.predict_proba(X_train)[:, 1]

        # 测试集预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        results = {
            'model_name': 'TabPFN',
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'train_auc': roc_auc_score(y_train, y_train_pred_proba),
            'train_brier_score': brier_score_loss(y_train, y_train_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_train_pred_proba': y_train_pred_proba,
            'y_train': y_train,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'best_params': 'N/A (TabPFN无超参数)',
        }

        logger.info(f"  TabPFN 训练集AUC: {results['train_auc']:.4f}")
        logger.info(f"  TabPFN 测试集AUC: {results['auc']:.4f}")

        return results
    except Exception as e:
        logger.error(f"  TabPFN 训练失败: {e}")
        return None


def train_and_evaluate(model_name, model_config, X_train, y_train, X_test, y_test, params):
    """
    训练单个模型：
    - 基于训练集使用5折交叉验证确定最佳超参数
    - 使用最优参数在整个训练集上重新训练最终模型
    - 在测试集上评估
    """
    if model_name == 'TabPFN':
        return train_tabpfn(X_train, y_train, X_test, y_test, params)

    logger.info(f"  训练模型: {model_name}")

    # 处理class_weight
    class_weight = params.get('class_weight')
    # XGBoost 使用 scale_pos_weight 代替 class_weight
    scale_pos_weight = None
    if class_weight == 'balanced' and model_name == 'XGBoost':
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        cw_for_model = None
    else:
        cw_for_model = class_weight

    model = initialize_model(model_name, params['random_state'],
                             class_weight=cw_for_model, scale_pos_weight=scale_pos_weight)
    param_grid = model_config.get('param_grid')

    cv = StratifiedKFold(n_splits=params['cv_folds'], shuffle=True, random_state=params['random_state'])

    if param_grid:
        # 基于训练集使用5折交叉验证确定最佳超参数
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=params['scoring'],
            n_jobs=params['n_jobs'],
            verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        logger.info(f"  最佳参数: {best_params}")
        logger.info(f"  交叉验证 {params['scoring']}: {cv_score:.4f}")

        # 使用最优参数在整个训练集上重新训练
        final_model = initialize_model(model_name, params['random_state'],
                                       class_weight=cw_for_model, scale_pos_weight=scale_pos_weight)
        final_model.set_params(**best_params)
        final_model.fit(X_train, y_train)
    else:
        # 无参数网格，直接训练
        final_model = model
        final_model.fit(X_train, y_train)
        best_params = {}
        cv_score = np.mean(
            [roc_auc_score(y_train[test_idx],
                          final_model.predict_proba(X_train[test_idx])[:, 1])
             for train_idx, test_idx in cv.split(X_train, y_train)]
        ) if hasattr(final_model, 'predict_proba') else 0.0

    # 训练集预测
    y_train_pred = final_model.predict(X_train)
    y_train_pred_proba = final_model.predict_proba(X_train)[:, 1]

    # 测试集预测
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    results = {
        'model_name': model_name,
        'model': final_model,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_pred_proba),
        'train_brier_score': brier_score_loss(y_train, y_train_pred_proba),
        'cv_auc': cv_score,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_train_pred_proba': y_train_pred_proba,
        'y_train': y_train,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'best_params': best_params,
    }

    logger.info(f"  训练集 AUC: {results['train_auc']:.4f}")
    logger.info(f"  测试集 Accuracy: {results['accuracy']:.4f}")
    logger.info(f"  测试集 AUC: {results['auc']:.4f}")
    logger.info(f"  测试集 Brier Score: {results['brier_score']:.4f}")

    return results


def save_patient_predictions(all_results, train_df, test_df, selected_features, cat_features, scaler, expanded_names, output_dir, no_scale_features=None, multi_cat_categories=None):
    """
    保存每个患者的预测概率结果（训练集和测试集）
    使用 build_mixed_feature_matrix 构建混合编码特征矩阵
    """
    # 训练集: 用已 fit 的 scaler 进行 transform
    X_train_matrix, _, _, _, _ = build_mixed_feature_matrix(
        train_df, selected_features, cat_features, scaler=scaler, fit=False,
        no_scale_features=no_scale_features, multi_cat_categories=multi_cat_categories
    )
    # 测试集
    X_test_matrix, _, _, _, _ = build_mixed_feature_matrix(
        test_df, selected_features, cat_features, scaler=scaler, fit=False,
        no_scale_features=no_scale_features, multi_cat_categories=multi_cat_categories
    )

    for model_name, result in all_results.items():
        model = result['model']

        # 训练集预测
        train_pred_proba = model.predict_proba(X_train_matrix)[:, 1]
        train_pred = model.predict(X_train_matrix)

        train_pred_df = pd.DataFrame({
            'ID': train_df['ID'].values,
            'Label': train_df['Label'].values,
            'Label_Encoded': train_df['Label_Encoded'].values,
            'Predicted_Probability': train_pred_proba,
            'Predicted_Label': train_pred,
            'Set': 'Train',
        })

        # 测试集预测
        test_pred_proba = model.predict_proba(X_test_matrix)[:, 1]
        test_pred = model.predict(X_test_matrix)

        test_pred_df = pd.DataFrame({
            'ID': test_df['ID'].values,
            'Label': test_df['Label'].values,
            'Label_Encoded': test_df['Label_Encoded'].values,
            'Predicted_Probability': test_pred_proba,
            'Predicted_Label': test_pred,
            'Set': 'Test',
        })

        pred_df = pd.concat([train_pred_df, test_pred_df], ignore_index=True)
        safe_name = model_name.replace(' ', '_')
        pred_path = os.path.join(output_dir, f'predictions_{safe_name}.csv')
        pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')

    logger.info(f"  患者预测概率已保存")


def save_results(all_results, selected_features, selection_report, train_df, test_df, scaler, expanded_names, params, multi_cat_categories=None):
    """保存所有结果"""
    output_dir = params['output_dir']
    cat_features = selection_report.get('cat_features', [])
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存模型比较表
    comparison_data = []
    for model_name, result in all_results.items():
        comparison_data.append({
            'Model': model_name,
            'CV_AUC': result.get('cv_auc', result.get('train_auc', 'N/A')),
            'Train_AUC': result['train_auc'],
            'Train_Brier': result['train_brier_score'],
            'Test_Accuracy': result['accuracy'],
            'Test_Precision': result['precision'],
            'Test_Recall': result['recall'],
            'Test_F1': result['f1'],
            'Test_AUC': result['auc'],
            'Test_Brier': result['brier_score'],
            'Best_Params': str(result['best_params']),
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Test_AUC', ascending=False)

    comparison_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    logger.info(f"模型比较结果已保存: {comparison_path}")

    # 打印结果表
    logger.info("\n" + "=" * 80)
    logger.info("模型比较结果（按测试集AUC排序）")
    logger.info("=" * 80)
    print(comparison_df.to_string(index=False))

    # 2. 保存所有训练模型（含预处理信息，便于后续在新数据集上预测）
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    for m_name, m_result in all_results.items():
        safe_m_name = m_name.replace(' ', '_')
        model_save_path = os.path.join(models_dir, f'{safe_m_name}_model.pkl')
        with open(model_save_path, 'wb') as f:
            pickle.dump({
                'model': m_result['model'],
                'model_name': m_name,
                'selected_features': selected_features,
                'cat_features': cat_features,
                'expanded_names': expanded_names,
                'scaler': scaler,
                'best_params': m_result['best_params'],
                'train_auc': m_result['train_auc'],
                'test_auc': m_result['auc'],
                'no_scale_features': params.get('feature_selection', {}).get('no_scale_features'),
                'multi_cat_categories': multi_cat_categories,
            }, f)
        logger.info(f"模型已保存: {model_save_path}")

    # 同时保存一份最佳模型的快捷引用（兼容旧逻辑）
    best_model_name = comparison_df.iloc[0]['Model']
    best_model = all_results[best_model_name]['model']
    model_path = os.path.join(output_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'auc': all_results[best_model_name]['auc'],
            'selected_features': selected_features,
            'best_params': all_results[best_model_name]['best_params'],
        }, f)
    logger.info(f"最佳模型快捷引用已保存: {model_path}")

    # 3. 保存选中的特征列表
    features_path = os.path.join(output_dir, 'selected_features.txt')
    with open(features_path, 'w', encoding='utf-8') as f:
        for feat in selected_features:
            f.write(feat + '\n')

    # 4. 生成评估报告（含相关性筛选详情）
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("模型评估报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"数据划分: 训练集 {1-params['test_ratio']:.0%} / 测试集 {params['test_ratio']:.0%}\n")
        f.write(f"特征筛选后特征数: {len(selected_features)}\n")
        f.write(f"随机种子: {params['random_state']}\n")
        f.write(f"交叉验证: {params['cv_folds']} 折\n\n")

        # 特征筛选报告
        f.write("-" * 80 + "\n")
        f.write("特征筛选过程:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  单变量检验后: {selection_report.get('univariate_count', 'N/A')} 个特征\n")
        if selection_report.get('fdr_count') is not None:
            f.write(f"  FDR校正后: {selection_report['fdr_count']} 个特征\n")
        f.write(f"  相关性筛选移除: {selection_report.get('corr_removed_count', 0)} 个特征\n")
        f.write(f"  相关性筛选后: {len(selection_report.get('after_corr_features', []))} 个特征\n")
        f.write(f"  LASSO筛选后（最终）: {len(selected_features)} 个特征\n\n")

        # 相关性筛选详情
        corr_detail = selection_report.get('corr_removal_detail', {})
        if corr_detail:
            f.write("-" * 80 + "\n")
            f.write("相关性特征筛选详情（保留特征 -> 移除的高相关特征）:\n")
            f.write("-" * 80 + "\n")
            for retained, removed_list in corr_detail.items():
                removed_str = ', '.join(removed_list)
                f.write(f"  {retained} -> {removed_str}\n")
            f.write("\n")

        # 模型性能
        f.write("-" * 80 + "\n")
        f.write("模型性能比较:\n")
        f.write("-" * 80 + "\n")
        f.write(comparison_df.to_string(index=False))
        f.write("\n\n")

        f.write(f"最佳模型: {best_model_name}\n")
        f.write(f"测试集AUC: {all_results[best_model_name]['auc']:.4f}\n")
        f.write(f"测试集Brier Score: {all_results[best_model_name]['brier_score']:.4f}\n\n")

        f.write("混淆矩阵（测试集）:\n")
        f.write(str(all_results[best_model_name]['confusion_matrix']))
        f.write("\n\n")

        f.write("分类报告（测试集）:\n")
        f.write(classification_report(
            all_results[best_model_name]['y_test'],
            all_results[best_model_name]['y_pred'],
            target_names=['CR', 'PR']
        ))
        f.write("\n\n")

        f.write("选中的特征:\n")
        for i, feat in enumerate(selected_features, 1):
            f.write(f"  {i}. {feat}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write(f"报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    logger.info(f"评估报告已保存: {report_path}")

    # 5. 保存患者预测概率
    save_patient_predictions(all_results, train_df, test_df, selected_features, cat_features, scaler, expanded_names, output_dir,
                             no_scale_features=params.get('feature_selection', {}).get('no_scale_features'),
                             multi_cat_categories=multi_cat_categories)

    return comparison_df


def run_pipeline(params):
    """执行完整的建模流水线"""
    output_dir = params['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # ===== 步骤1: 加载数据 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 1/6: 加载和合并特征数据")
    logger.info("=" * 80)
    df = load_and_merge_features(params)

    # ===== 步骤2: 数据划分 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 2/6: 数据划分（训练集 / 测试集）")
    logger.info("=" * 80)
    train_df, test_df = split_data(df, params)

    # 保存划分信息（CSV格式，兼容旧逻辑）
    split_info = pd.DataFrame({
        'ID': pd.concat([train_df['ID'], test_df['ID']]),
        'Split': ['train'] * len(train_df) + ['test'] * len(test_df),
    })
    split_info.to_csv(os.path.join(output_dir, 'data_split.csv'), index=False, encoding='utf-8-sig')

    # ===== 步骤3: 特征筛选（仅在训练集上）=====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 3/6: 特征筛选（仅在训练集上进行）")
    logger.info("=" * 80)
    selected_features, selection_report = feature_selection(train_df, params)

    if not selected_features:
        logger.error("特征筛选未选出任何特征，程序终止！")
        return

    # 保存单变量特征筛选统计量到CSV
    univariate_stats = selection_report.get('univariate_stats', [])
    if univariate_stats:
        stats_df = pd.DataFrame(univariate_stats)
        stats_df = stats_df.sort_values('P_value')
        stats_path = os.path.join(output_dir, 'univariate_test_statistics.csv')
        try:
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
            logger.info(f"  单变量检验统计量已保存: {stats_path}")
        except PermissionError:
            alt_path = os.path.join(output_dir, 'univariate_test_statistics_new.csv')
            stats_df.to_csv(alt_path, index=False, encoding='utf-8-sig')
            logger.warning(f"  原文件被占用，已保存到: {alt_path}")

    # 绘制LASSO图
    lasso_model = selection_report.get('lasso_model')
    lasso_feature_names = selection_report.get('lasso_feature_names', [])
    lasso_X_scaled = selection_report.get('lasso_X_scaled')
    lasso_y = selection_report.get('lasso_y')
    if lasso_model is not None:
        plot_lasso_figures(lasso_model, lasso_feature_names, output_dir,
                          X_scaled=lasso_X_scaled, y=lasso_y)

    # 绘制RFE交叉验证AUC曲线
    rfe_report = selection_report.get('rfe_report', {})
    if rfe_report:
        plot_rfe_cv_curve(rfe_report, output_dir)

        # 保存RFE特征排序文件（按淘汰顺序）
        rfe_full_ranking = rfe_report.get('rfe_full_ranking')
        input_features = rfe_report.get('input_features')
        cv_auc_means = rfe_report.get('cv_auc_means')
        feature_range = rfe_report.get('feature_range')
        optimal_n = rfe_report.get('optimal_n_features')

        if rfe_full_ranking is not None and input_features is not None:
            n_total = len(input_features)

            # full_ranking: rank=1最重要（最后被淘汰），rank=N最不重要（最先被淘汰）
            # 选择逻辑: ranking <= N 的特征会被选中
            # 所以 Min_N_Features_Included = rank（当特征数 >= rank 时，该特征被包含）
            rows = []
            for i, (feat, rank) in enumerate(zip(input_features, rfe_full_ranking)):
                min_n_included = int(rank)  # 曲线上 x >= rank 时该特征参与建模
                # 对应AUC曲线上的值
                if cv_auc_means is not None and feature_range is not None:
                    auc_idx = min_n_included - feature_range[0]
                    cv_auc_at_n = float(cv_auc_means[auc_idx]) if 0 <= auc_idx < len(cv_auc_means) else None
                else:
                    cv_auc_at_n = None
                rows.append({
                    'Feature': feat,
                    'RFE_Rank': int(rank),  # rank=1最重要，rank=N最先被淘汰
                    'Min_N_Features_Included': min_n_included,  # 曲线上x>=此值时该特征在模型中
                    'CV_AUC_At_N': cv_auc_at_n,
                    'Selected': 'Yes' if rank <= optimal_n else 'No',
                })

            ranking_df = pd.DataFrame(rows)
            # 按重要性排序：rank小的最重要（最后被淘汰）
            ranking_df = ranking_df.sort_values('RFE_Rank', ascending=True).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            ranking_df.index.name = 'Importance_Rank'

            ranking_path = os.path.join(output_dir, 'rfe_feature_ranking.csv')
            ranking_df.to_csv(ranking_path, encoding='utf-8-sig')
            logger.info(f"  RFE特征排序已保存: {ranking_path}")
            logger.info(f"  说明: RFE_Rank=1 表示最重要（最后被淘汰）")
            logger.info(f"  说明: 选择N个特征时，RFE_Rank<=N 的特征被选中")
            logger.info(f"  说明: 最优特征数={optimal_n}，即Selected=Yes的特征")

    # 绘制相关性热图
    plot_correlation_heatmap(train_df, selected_features, output_dir)

    # ===== 步骤4: 准备训练数据 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 4/6: 数据预处理（连续变量标准化 + 分类变量one-hot编码）")
    logger.info("=" * 80)

    cat_features_all = selection_report.get('cat_features', [])
    y_train = train_df['Label_Encoded'].values
    y_test = test_df['Label_Encoded'].values

    no_scale_features = params.get('feature_selection', {}).get('no_scale_features')
    # 训练集: fit + transform（同时捕获多分类变量的类别信息）
    X_train, expanded_names, scaler, multi_cat_cols, multi_cat_categories = build_mixed_feature_matrix(
        train_df, selected_features, cat_features_all, scaler=None, fit=True,
        no_scale_features=no_scale_features
    )
    # 测试集: transform（使用训练集的类别信息保证 one-hot 编码一致）
    X_test, _, _, _, _ = build_mixed_feature_matrix(
        test_df, selected_features, cat_features_all, scaler=scaler, fit=False,
        no_scale_features=no_scale_features, multi_cat_categories=multi_cat_categories
    )

    no_scale_set_final = set(no_scale_features) if no_scale_features else set()
    cont_features_final = [f for f in selected_features if f not in cat_features_all and f not in no_scale_set_final]
    cont_no_scale_final = [f for f in selected_features if f not in cat_features_all and f in no_scale_set_final]
    binary_cat_final = [f for f in selected_features
                        if f in cat_features_all and f not in multi_cat_cols]
    logger.info(f"连续变量（标准化）: {cont_features_final}")
    if cont_no_scale_final:
        logger.info(f"连续变量（不标准化，保持原始值）: {cont_no_scale_final}")
    logger.info(f"二分类变量（保持原值）: {binary_cat_final}")
    logger.info(f"多分类变量（one-hot编码）: {multi_cat_cols}")
    logger.info(f"展开后特征数: {len(expanded_names)}")
    logger.info(f"特征矩阵: X_train={X_train.shape}, X_test={X_test.shape}")

    # 保存scaler和特征名映射
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'expanded_names': expanded_names,
                     'selected_features': selected_features, 'cat_features': cat_features_all,
                     'no_scale_features': no_scale_features,
                     'multi_cat_categories': multi_cat_categories}, f)

    # ===== 步骤5: 训练和评估所有模型 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 5/6: 模型训练与评估（5折交叉验证调参 + 全训练集重训练）")
    logger.info("=" * 80)

    all_results = {}
    for model_name, model_config in params['models'].items():
        if not model_config.get('enabled', True):
            logger.info(f"\n跳过模型: {model_name}（未启用）")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"训练模型: {model_name}")
        logger.info(f"{'='*60}")

        result = train_and_evaluate(
            model_name, model_config,
            X_train, y_train, X_test, y_test,
            params
        )

        if result is not None:
            all_results[model_name] = result

    if not all_results:
        logger.error("没有模型训练成功！")
        return

    # 在训练集上预测，用于可视化和报告
    for model_name, result in all_results.items():
        model = result['model']
        train_pred_proba = model.predict_proba(X_train)[:, 1]
        train_pred = model.predict(X_train)
        result['y_train'] = y_train
        result['y_train_pred'] = train_pred
        result['y_train_pred_proba'] = train_pred_proba
        result['train_auc'] = roc_auc_score(y_train, train_pred_proba)
        result['train_brier_score'] = brier_score_loss(y_train, train_pred_proba)

    # ===== 逻辑回归系数解读 =====
    lr_result = all_results.get('Logistic Regression')
    if lr_result is not None and hasattr(lr_result['model'], 'coef_'):
        logger.info("\n" + "-" * 60)
        logger.info("逻辑回归特征系数解读")
        logger.info("-" * 60)
        lr_model = lr_result['model']
        coef_std = lr_model.coef_[0]

        # 反推原始尺度系数: 仅标准化后的连续变量需要反推，no_scale特征保持原系数
        coef_raw = coef_std.copy()
        n_cont_scaled = len(cont_features_final)
        if n_cont_scaled > 0 and scaler is not None:
            coef_raw[:n_cont_scaled] = coef_std[:n_cont_scaled] / scaler.scale_

        or_values = np.exp(coef_raw)
        coef_df = pd.DataFrame({
            'Feature': expanded_names,
            'Coef_Raw': coef_raw,
            'OR': or_values,
            'Coef_Std': coef_std,
        }).sort_values('Coef_Raw', ascending=False)

        for _, row in coef_df.iterrows():
            direction = '↑ PR风险' if row['Coef_Raw'] > 0 else '↓ PR风险'
            logger.info(f"  {row['Feature']:30s}  coef = {row['Coef_Raw']:+.4f}  "
                         f"OR = {row['OR']:.4f}  {direction}")

        coef_df.to_csv(os.path.join(output_dir, 'lr_coefficients.csv'),
                       index=False, encoding='utf-8-sig')
        logger.info("逻辑回归系数已保存至 lr_coefficients.csv")

    # ===== 步骤6: 保存结果和可视化 =====
    logger.info("\n" + "=" * 80)
    logger.info("步骤 6/6: 保存结果和生成可视化")
    logger.info("=" * 80)

    comparison_df = save_results(all_results, selected_features, selection_report, train_df, test_df, scaler, expanded_names, params,
                                  multi_cat_categories=multi_cat_categories)

    # 生成可视化图表
    plot_feature_importance(all_results, selected_features, train_df, output_dir,
                           expanded_names=expanded_names)
    plot_roc_curves(all_results, output_dir)

    # 最佳模型的校准曲线和决策曲线
    best_model_name = comparison_df.iloc[0]['Model']
    best_result = all_results[best_model_name]
    plot_calibration_curve(best_result, output_dir)
    plot_decision_curve(best_result, output_dir)

    # 最终汇总
    best_auc = comparison_df.iloc[0]['Test_AUC']
    logger.info("\n" + "=" * 80)
    logger.info("流水线执行完成！")
    logger.info(f"最佳模型: {best_model_name} (测试集AUC: {best_auc:.4f})")
    logger.info(f"结果目录: {output_dir}")
    logger.info("=" * 80)


def _evaluate_predictions(y_true, y_pred, y_pred_proba, model_name='Model', n_samples=None):
    """
    计算与训练流水线一致的评估指标，返回字典
    """
    metrics = {
        'model_name': model_name,
        'n_samples': n_samples if n_samples else len(y_true),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics['auc'] = None
    return metrics


def _generate_evaluation_report(all_eval_results, output_dir):
    """
    生成评估报告文本文件（含所有模型的指标和混淆矩阵）
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("新数据集模型评估报告\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"评估模型数: {len(all_eval_results)}\n\n")

        # 汇总表
        summary_rows = []
        for ev in all_eval_results:
            summary_rows.append({
                'Model': ev['model_name'],
                'N_Samples': ev['n_samples'],
                'Accuracy': ev['accuracy'],
                'Precision': ev['precision'],
                'Recall': ev['recall'],
                'F1': ev['f1'],
                'AUC': ev['auc'] if ev['auc'] is not None else 'N/A',
                'Brier': ev['brier_score'],
            })
        summary_df = pd.DataFrame(summary_rows)
        if 'AUC' in summary_df.columns:
            summary_df = summary_df.sort_values('AUC', ascending=False, na_position='last')

        f.write("-" * 80 + "\n")
        f.write("模型性能比较（按AUC排序）:\n")
        f.write("-" * 80 + "\n")
        f.write(summary_df.to_string(index=False))
        f.write("\n\n")

        # 每个模型的详细结果
        for ev in all_eval_results:
            f.write("=" * 80 + "\n")
            f.write(f"模型: {ev['model_name']}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"  样本数:    {ev['n_samples']}\n")
            f.write(f"  Accuracy:  {ev['accuracy']:.4f}\n")
            f.write(f"  Precision: {ev['precision']:.4f}\n")
            f.write(f"  Recall:    {ev['recall']:.4f}\n")
            f.write(f"  F1-Score:  {ev['f1']:.4f}\n")
            f.write(f"  AUC:       {ev['auc']:.4f}\n" if ev['auc'] is not None else "  AUC:       N/A（仅含单一类别）\n")
            f.write(f"  Brier:     {ev['brier_score']:.4f}\n\n")

            # 混淆矩阵
            cm = ev['confusion_matrix']
            f.write("  混淆矩阵:\n")
            f.write(f"                    Predicted CR    Predicted PR\n")
            f.write(f"    Actual CR       {cm[0][0]:>10}    {cm[0][1]:>10}\n")
            f.write(f"    Actual PR       {cm[1][0]:>10}    {cm[1][1]:>10}\n\n")

            # 分类报告
            y_true = ev['y_true']
            y_pred = ev['y_pred']
            f.write("  分类报告:\n")
            f.write(classification_report(y_true, y_pred, target_names=['CR', 'PR']))
            f.write("\n")

        f.write("\n" + "=" * 80 + "\n")

    logger.info(f"评估报告已保存: {report_path}")
    return report_path


def _plot_prediction_roc(all_eval_results, output_dir):
    """绘制所有模型的ROC曲线（新数据集）"""
    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, ev in enumerate(all_eval_results):
        y_true = ev['y_true']
        y_proba = ev['y_pred_proba']
        if len(set(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_val = roc_auc_score(y_true, y_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.5,
                label=f'{ev["model_name"]} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('1 - Specificity (False Positive Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('ROC Curves - New Dataset')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'roc_curves_new.png'))
    fig.savefig(os.path.join(plots_dir, 'roc_curves_new.pdf'))
    plt.close(fig)
    logger.info(f"  ROC曲线图已保存")


def _plot_prediction_calibration(all_eval_results, output_dir):
    """绘制所有模型的校准曲线（新数据集）"""
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, ev in enumerate(all_eval_results):
        y_true = ev['y_true']
        y_proba = ev['y_pred_proba']
        if len(set(y_true)) < 2:
            continue
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )
        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                color=colors[i % len(colors)], linewidth=1.5, markersize=5,
                label=ev['model_name'])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves - New Dataset')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'calibration_curves_new.png'))
    fig.savefig(os.path.join(plots_dir, 'calibration_curves_new.pdf'))
    plt.close(fig)
    logger.info(f"  校准曲线图已保存")


def _plot_prediction_dca(all_eval_results, output_dir):
    """绘制所有模型的决策曲线（新数据集）"""
    import matplotlib.pyplot as plt

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    fig, ax = plt.subplots(figsize=(5.5, 5))
    thresholds = np.linspace(0.01, 0.99, 200)

    # 使用第一个有标签的结果计算 prevalence（假设所有模型评估同一数据集）
    prevalence = None
    for ev in all_eval_results:
        if len(set(ev['y_true'])) >= 2:
            prevalence = np.mean(ev['y_true'])
            break
    if prevalence is None:
        plt.close(fig)
        return

    n = len(all_eval_results[0]['y_true'])

    # Treat All 线
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    treat_all_positive = nb_all > 0
    if treat_all_positive.any():
        ax.plot(thresholds[treat_all_positive], nb_all[treat_all_positive],
                color='#4575b4', linewidth=1.2, linestyle='--', label='Treat All', zorder=2)

    # Treat None 基线
    ax.axhline(y=0, color='#333333', linewidth=0.9, linestyle='-', label='Treat None', zorder=1)

    # 每个模型的决策曲线
    y_max_global = 0
    for i, ev in enumerate(all_eval_results):
        y_true = ev['y_true']
        y_proba = ev['y_pred_proba']
        if len(set(y_true)) < 2:
            continue

        net_benefits = []
        for thresh in thresholds:
            y_pred_t = (y_proba >= thresh).astype(int)
            tp = np.sum((y_pred_t == 1) & (y_true == 1))
            fp = np.sum((y_pred_t == 1) & (y_true == 0))
            nb = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefits.append(nb)
        net_benefits = np.array(net_benefits)

        # 仅保留净收益 > 0 的范围
        positive_mask = net_benefits > 0
        if positive_mask.any():
            pos_idx = np.where(positive_mask)[0]
            start_idx = max(0, pos_idx[0] - 2)
            end_idx = min(len(thresholds) - 1, pos_idx[-1] + 2)
            plot_t = thresholds[start_idx:end_idx + 1]
            plot_nb = net_benefits[start_idx:end_idx + 1]
            y_max_global = max(y_max_global, np.max(plot_nb))
            ax.plot(plot_t, plot_nb, color=colors[i % len(colors)], linewidth=1.8,
                    label=ev['model_name'], zorder=3)

    y_max = max(y_max_global, np.max(nb_all[treat_all_positive]) if treat_all_positive.any() else 0.05)
    ax.set_ylim(-0.005, y_max * 1.15 + 0.005)
    ax.set_xlim(0.01, 0.99)
    ax.set_xlabel('Threshold Probability', fontsize=10)
    ax.set_ylabel('Net Benefit', fontsize=10)
    ax.set_title('Decision Curve Analysis — New Dataset', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, edgecolor='black',
              fancybox=False, framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'decision_curves_new.png'), bbox_inches='tight')
    fig.savefig(os.path.join(plots_dir, 'decision_curves_new.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  决策曲线图已保存")


def predict_new_data(model_pkl_path, new_data_path, output_path=None, label_file=None, label_encoding=None):
    """
    使用已保存的模型对新数据集进行预测

    参数:
        model_pkl_path: 已保存的模型 pkl 文件路径（models/ 目录下的任一模型）
        new_data_path:  新数据集 CSV 文件路径（需包含与训练时相同的特征列）
        output_path:    预测结果保存路径（CSV），为 None 时自动保存在模型同目录
        label_file:     可选，新数据集的真实标签文件（xlsx），用于评估模型性能
        label_encoding: 可选，标签编码字典，如 {'CR': 0, 'PR': 1}

    返回:
        dict: {
            'result_df': 预测结果 DataFrame,
            'evaluation': 评估指标字典（有真实标签时）或 None,
        }
    """
    # 1. 加载已保存的模型
    logger.info(f"加载模型: {model_pkl_path}")
    with open(model_pkl_path, 'rb') as f:
        model_info = pickle.load(f)

    model = model_info['model']
    model_name = model_info.get('model_name', 'Unknown')
    selected_features = model_info['selected_features']
    cat_features = model_info.get('cat_features', [])
    expanded_names = model_info['expanded_names']
    scaler = model_info['scaler']
    no_scale_features = model_info.get('no_scale_features')
    multi_cat_categories = model_info.get('multi_cat_categories')

    logger.info(f"模型名称: {model_name}")
    logger.info(f"训练时使用的特征数: {len(selected_features)}")
    logger.info(f"展开后特征数: {len(expanded_names)}")

    # 2. 加载新数据
    logger.info(f"加载新数据: {new_data_path}")
    new_df = pd.read_csv(new_data_path, converters={'ID': str})
    logger.info(f"新数据: {new_df.shape[0]} 样本, {new_df.shape[1]} 列")

    # 3. 检查特征是否齐全
    missing_features = [f for f in selected_features if f not in new_df.columns]
    if missing_features:
        logger.warning(f"新数据缺少以下 {len(missing_features)} 个特征列:")
        for mf in missing_features:
            logger.warning(f"  - {mf}")
        logger.error(f"特征不匹配，无法预测！请确保新数据包含模型所需的全部 {len(selected_features)} 个特征。")
        return None

    logger.info(f"特征检查通过: {len(selected_features)} 个特征可用")

    # 4. 使用保存的预处理流程构建特征矩阵
    X_new, new_expanded_names, _, _, _ = build_mixed_feature_matrix(
        new_df, selected_features, cat_features, scaler=scaler, fit=False,
        no_scale_features=no_scale_features, multi_cat_categories=multi_cat_categories
    )

    if len(new_expanded_names) != len(expanded_names):
        logger.error(
            f"展开后特征数不一致: 训练时 {len(expanded_names)}，新数据 {len(new_expanded_names)}\n"
            f"可能原因: 多分类变量的类别分布不同"
        )
        return None

    logger.info(f"特征矩阵构建完成: {X_new.shape}")

    # 5. 执行预测
    logger.info(f"开始预测...")
    y_pred_proba = model.predict_proba(X_new)[:, 1]
    y_pred = model.predict(X_new)

    # 6. 构建结果表
    result_df = pd.DataFrame({
        'ID': new_df['ID'].values,
        'Predicted_Label': y_pred,
        'Predicted_Probability': y_pred_proba,
    })

    if 'Label' in new_df.columns:
        result_df.insert(1, 'True_Label', new_df['Label'].values)
    elif 'Label_Encoded' in new_df.columns:
        result_df.insert(1, 'True_Label_Encoded', new_df['Label_Encoded'].values)

    # 7. 若提供了真实标签文件，加载并评估
    evaluation = None
    if label_file and os.path.exists(label_file):
        logger.info(f"加载真实标签: {label_file}")
        labels_df = pd.read_excel(label_file, converters={'ID': str})
        if label_encoding is None:
            label_encoding = {'CR': 0, 'PR': 1}
        labels_df['Label_Encoded'] = labels_df['Label'].map(label_encoding)
        result_df = pd.merge(result_df, labels_df[['ID', 'Label', 'Label_Encoded']],
                             on='ID', how='left')

        eval_mask = result_df['Label_Encoded'].notna()
        if eval_mask.any():
            y_true = result_df.loc[eval_mask, 'Label_Encoded'].astype(int).values
            y_p = result_df.loc[eval_mask, 'Predicted_Label'].values
            y_pp = result_df.loc[eval_mask, 'Predicted_Probability'].values

            evaluation = _evaluate_predictions(y_true, y_p, y_pp, model_name=model_name)
            evaluation['y_true'] = y_true
            evaluation['y_pred'] = y_p
            evaluation['y_pred_proba'] = y_pp

            cm = evaluation['confusion_matrix']
            logger.info("=" * 60)
            logger.info(f"{model_name} - 新数据集评估结果（{eval_mask.sum()} 个有标签样本）")
            logger.info("=" * 60)
            logger.info(f"  Accuracy:  {evaluation['accuracy']:.4f}")
            logger.info(f"  Precision: {evaluation['precision']:.4f}")
            logger.info(f"  Recall:    {evaluation['recall']:.4f}")
            logger.info(f"  F1-Score:  {evaluation['f1']:.4f}")
            logger.info(f"  AUC:       {evaluation['auc']:.4f}" if evaluation['auc'] is not None else "  AUC:       N/A")
            logger.info(f"  Brier:     {evaluation['brier_score']:.4f}")
            logger.info(f"  混淆矩阵:")
            logger.info(f"                 Predicted CR    Predicted PR")
            logger.info(f"    Actual CR    {cm[0][0]:>10}    {cm[0][1]:>10}")
            logger.info(f"    Actual PR    {cm[1][0]:>10}    {cm[1][1]:>10}")
            logger.info("=" * 60)

    # 8. 保存预测结果
    if output_path is None:
        model_dir = os.path.dirname(model_pkl_path)
        safe_name = model_name.replace(' ', '_')
        output_path = os.path.join(model_dir, f'predictions_new_{safe_name}.csv')

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"预测结果已保存: {output_path}")
    logger.info(f"预测分布: CR={sum(y_pred == 0)}, PR={sum(y_pred == 1)}")

    return {'result_df': result_df, 'evaluation': evaluation}


def predict_all_models(models_dir, new_data_path, output_dir=None, label_file=None, label_encoding=None):
    """
    使用 models/ 目录下的所有模型对新数据集进行预测，并生成：
      - 评估报告（含混淆矩阵）
      - ROC 曲线图
      - 校准曲线图
      - 决策曲线图
      - 汇总 CSV

    参数:
        models_dir:     模型目录路径（训练时生成的 models/ 子目录）
        new_data_path:  新数据集 CSV 文件路径
        output_dir:     结果保存目录（None 则在 models_dir 同级创建 predictions_new/）
        label_file:     可选，真实标签文件
        label_encoding: 可选，标签编码字典

    返回:
        pd.DataFrame: 所有模型的预测结果汇总表
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(models_dir), 'predictions_new')
    os.makedirs(output_dir, exist_ok=True)

    # 查找所有模型 pkl 文件
    pkl_files = sorted([f for f in os.listdir(models_dir) if f.endswith('_model.pkl')])
    if not pkl_files:
        logger.error(f"在 {models_dir} 中未找到任何模型文件！")
        return None

    logger.info(f"找到 {len(pkl_files)} 个模型: {pkl_files}")

    all_predictions = []     # result_df 列表
    all_eval_results = []    # evaluation 字典列表（有真实标签时）
    summary_rows = []

    for pkl_file in pkl_files:
        pkl_path = os.path.join(models_dir, pkl_file)
        m_name = pkl_file.replace('_model.pkl', '').replace('_', ' ')
        logger.info(f"\n{'=' * 60}")
        logger.info(f"预测模型: {m_name} ({pkl_file})")
        logger.info(f"{'=' * 60}")

        out_path = os.path.join(output_dir, f'predictions_{pkl_file.replace("_model.pkl", ".csv")}')
        ret = predict_new_data(pkl_path, new_data_path, output_path=out_path,
                               label_file=label_file, label_encoding=label_encoding)

        if ret is None:
            continue

        result_df = ret['result_df']
        evaluation = ret['evaluation']
        result_df['Model'] = m_name
        all_predictions.append(result_df)

        # 汇总行
        row = {
            'Model': m_name,
            'N_Samples': len(result_df),
            'Predicted_CR': int((result_df['Predicted_Label'] == 0).sum()),
            'Predicted_PR': int((result_df['Predicted_Label'] == 1).sum()),
            'Mean_Probability': result_df['Predicted_Probability'].mean(),
        }
        if evaluation is not None:
            row['Accuracy'] = evaluation['accuracy']
            row['Precision'] = evaluation['precision']
            row['Recall'] = evaluation['recall']
            row['F1'] = evaluation['f1']
            row['AUC'] = evaluation['auc']
            row['Brier'] = evaluation['brier_score']
            all_eval_results.append(evaluation)

        summary_rows.append(row)

    if not all_predictions:
        logger.error("所有模型预测均失败！")
        return None

    # ===== 保存汇总表 =====
    summary_df = pd.DataFrame(summary_rows)
    if 'AUC' in summary_df.columns:
        summary_df = summary_df.sort_values('AUC', ascending=False, na_position='last')
    summary_path = os.path.join(output_dir, 'model_predictions_summary.csv')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n预测汇总已保存: {summary_path}")
    print(summary_df.to_string(index=False))

    # 合并所有预测结果
    combined = pd.concat(all_predictions, ignore_index=True)
    combined_path = os.path.join(output_dir, 'all_model_predictions.csv')
    combined.to_csv(combined_path, index=False, encoding='utf-8-sig')
    logger.info(f"合并预测结果已保存: {combined_path}")

    # ===== 评估报告 + 可视化（需有真实标签） =====
    if all_eval_results:
        logger.info("\n" + "=" * 80)
        logger.info("生成评估报告和可视化图表")
        logger.info("=" * 80)

        # 1. 评估报告（含混淆矩阵）
        _generate_evaluation_report(all_eval_results, output_dir)

        # 2. ROC 曲线
        _plot_prediction_roc(all_eval_results, output_dir)

        # 3. 校准曲线
        _plot_prediction_calibration(all_eval_results, output_dir)

        # 4. 决策曲线
        _plot_prediction_dca(all_eval_results, output_dir)

        logger.info("\n评估报告和可视化图表生成完成！")
    else:
        logger.info("\n无真实标签，跳过评估报告和可视化生成。")

    return combined


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='特征筛选、建模与评估程序')
    parser.add_argument('mode', nargs='?', default='train',
                        choices=['train', 'predict'],
                        help='运行模式: train（训练）或 predict（预测）')
    parser.add_argument('--model', type=str, default=None,
                        help='预测模式: 单个模型 pkl 文件路径')
    parser.add_argument('--models-dir', type=str,
                        default=r'D:\projects\CervixRT_Sensitivity_Prognosis\results\model_radiomics\models',
                        help='预测模式: 模型目录路径（批量预测所有模型）')
    parser.add_argument('--data', type=str,
                        default=r'D:\projects\CervixRT_Sensitivity_Prognosis\results\extracted_features\ct2_features.csv',
                        help='预测模式: 新数据集 CSV 文件路径')
    parser.add_argument('--output', type=str,
                        default=r'D:\projects\CervixRT_Sensitivity_Prognosis\results\model_radiomics\predictions_nii2',
                        help='预测模式: 预测结果保存路径')
    parser.add_argument('--label', type=str,
                        default=r'E:\data\CR-PR\CT\sensitivity_label2.xlsx',
                        help='预测模式: 真实标签文件（可选，用于评估）')
    args = parser.parse_args()

    if args.mode == 'train':
        # ===== 训练模式 =====
        logger.info("=" * 80)
        logger.info("特征筛选、建模与评估程序")
        logger.info("=" * 80)
        logger.info(f"特征文件: {PARAMS['feature_files']}")
        logger.info(f"数据划分: 训练 80% / 测试 20%")
        logger.info(f"启用的模型: {[k for k, v in PARAMS['models'].items() if v.get('enabled', True)]}")
        logger.info(f"输出目录: {PARAMS['output_dir']}")
        logger.info("=" * 80)

        run_pipeline(PARAMS)

    elif args.mode == 'predict':
        # ===== 预测模式 =====
        logger.info("=" * 80)
        logger.info("模型预测模式")
        logger.info("=" * 80)

        label_encoding = PARAMS.get('label_encoding', {'CR': 0, 'PR': 1})

        if args.model:
            # 单模型预测
            if not args.data:
                parser.error("预测模式必须通过 --data 指定新数据集路径")
            predict_new_data(
                model_pkl_path=args.model,
                new_data_path=args.data,
                output_path=args.output,
                label_file=args.label,
                label_encoding=label_encoding,
            )
        elif args.models_dir:
            # 批量预测（目录下所有模型）
            if not args.data:
                parser.error("预测模式必须通过 --data 指定新数据集路径")
            predict_all_models(
                models_dir=args.models_dir,
                new_data_path=args.data,
                output_dir=args.output,
                label_file=args.label,
                label_encoding=label_encoding,
            )
        else:
            # 默认使用 PARAMS 中的 output_dir/models 目录
            default_models_dir = os.path.join(PARAMS['output_dir'], 'models')
            if os.path.isdir(default_models_dir):
                if not args.data:
                    parser.error("预测模式必须通过 --data 指定新数据集路径")
                predict_all_models(
                    models_dir=default_models_dir,
                    new_data_path=args.data,
                    output_dir=args.output,
                    label_file=args.label,
                    label_encoding=label_encoding,
                )
            else:
                parser.error(
                    "预测模式需指定模型: --model <单个pkl路径> 或 --models-dir <模型目录>\n"
                    f"默认模型目录不存在: {default_models_dir}"
                )

