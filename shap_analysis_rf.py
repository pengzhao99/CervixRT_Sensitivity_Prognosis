"""
随机森林模型 SHAP 可解释性分析

功能：
    - 加载 model_pipeline.py 训练保存的随机森林模型（pkl文件）
    - 使用保存的预处理信息重建特征矩阵
    - 计算 SHAP 值
    - 绘制 SHAP 摘要图（summary plot）和特征重要性图（bar plot）
    - 同时输出训练集和测试集的 SHAP 分析结果

使用方式：
    # 默认分析 model_radiomics 目录下的随机森林模型
    python shap_analysis_rf.py

    # 指定模型目录
    python shap_analysis_rf.py --models-dir D:\\projects\\CervixRT_Sensitivity_Prognosis\\results\\model_clinic_ct

    # 指定输出目录
    python shap_analysis_rf.py --models-dir <模型目录> --output-dir <输出目录>
"""

import os
import sys
import pickle
import argparse
import logging
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入 model_pipeline 中的数据处理函数
from model_pipeline import (
    load_and_merge_features,
    split_data,
    build_mixed_feature_matrix,
    PARAMS,
)


def load_rf_model(model_pkl_path):
    """
    加载已保存的随机森林模型及其预处理信息

    返回:
        dict: 包含 model, selected_features, cat_features, expanded_names,
              scaler, no_scale_features, multi_cat_categories 等信息
    """
    logger.info(f"加载模型: {model_pkl_path}")
    with open(model_pkl_path, 'rb') as f:
        model_info = pickle.load(f)

    model_name = model_info.get('model_name', 'Unknown')
    logger.info(f"模型名称: {model_name}")
    logger.info(f"选中特征数: {len(model_info['selected_features'])}")
    logger.info(f"展开后特征数: {len(model_info['expanded_names'])}")

    return model_info


def prepare_data_for_shap(params, model_info):
    """
    使用与训练时完全一致的预处理流程，重建训练集和测试集的特征矩阵

    返回:
        tuple: (X_train, X_test, y_train, y_test, expanded_names, train_df, test_df)
    """
    # 1. 加载数据
    logger.info("加载和合并特征数据...")
    df = load_and_merge_features(params)

    # 2. 数据划分（使用与训练时相同的划分）
    logger.info("划分训练集/测试集...")
    train_df, test_df = split_data(df, params)

    # 3. 构建特征矩阵
    selected_features = model_info['selected_features']
    cat_features = model_info.get('cat_features', [])
    scaler = model_info['scaler']
    no_scale_features = model_info.get('no_scale_features')
    multi_cat_categories = model_info.get('multi_cat_categories')

    logger.info("构建训练集特征矩阵...")
    X_train, expanded_names, _, _, _ = build_mixed_feature_matrix(
        train_df, selected_features, cat_features,
        scaler=scaler, fit=False,
        no_scale_features=no_scale_features,
        multi_cat_categories=multi_cat_categories,
    )

    logger.info("构建测试集特征矩阵...")
    X_test, _, _, _, _ = build_mixed_feature_matrix(
        test_df, selected_features, cat_features,
        scaler=scaler, fit=False,
        no_scale_features=no_scale_features,
        multi_cat_categories=multi_cat_categories,
    )

    y_train = train_df['Label_Encoded'].values
    y_test = test_df['Label_Encoded'].values

    logger.info(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    logger.info(f"特征名数量: {len(expanded_names)}")

    return X_train, X_test, y_train, y_test, expanded_names, train_df, test_df


def compute_and_plot_shap(model, X, feature_names, output_dir, dataset_name="train", max_display=20):
    """
    计算 SHAP 值并绘制摘要图和重要性图（期刊发表级美化版）

    参数:
        model: 随机森林模型
        X: 特征矩阵 (numpy array)
        feature_names: 特征名列表
        output_dir: 输出目录
        dataset_name: 数据集名称（用于文件命名和标题）
        max_display: 最多显示的特征数
    """
    import shap
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    logger.info(f"\n{'='*60}")
    logger.info(f"SHAP 分析 - {dataset_name} 数据集")
    logger.info(f"{'='*60}")

    # 缩短特征名：去掉公共前缀 CT_ 和 _original_
    display_names = [name.replace('CT_', '').replace('_original_', '_') for name in feature_names]
    logger.info(f"特征名已缩短用于显示（原始名保留在CSV中）")

    # 创建 SHAP explainer
    logger.info("创建 TreeExplainer...")
    explainer = shap.TreeExplainer(model)

    # 计算 SHAP 值
    logger.info("计算 SHAP 值...")
    shap_values = explainer.shap_values(X)

    # 对于二分类问题，shap_values 可能是:
    #   - list: [class0_shap, class1_shap]（旧版 shap）
    #   - 3D array: (n_samples, n_features, n_classes)（新版 shap >= 0.45）
    #   - 2D array: (n_samples, n_features)（单输出模型）
    if isinstance(shap_values, list):
        # 取正类（PR, label=1）的 SHAP 值
        shap_vals = shap_values[1]
        logger.info(f"二分类模型（list格式），使用正类 (PR) SHAP 值")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 新版 shap 返回 (n_samples, n_features, n_classes)
        # 取正类（index=1）的 SHAP 值
        shap_vals = shap_values[:, :, 1]
        logger.info(f"二分类模型（3D数组格式 {shap_values.shape}），使用正类 (PR) SHAP 值")
    else:
        shap_vals = shap_values
        logger.info(f"SHAP 值直接使用，形状: {np.array(shap_vals).shape}")

    logger.info(f"SHAP 值形状: {shap_vals.shape}")

    # 保存 SHAP 值为 CSV
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_csv_path = os.path.join(output_dir, f'shap_values_{dataset_name}.csv')
    shap_df.to_csv(shap_csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"SHAP 值已保存: {shap_csv_path}")

    # 保存平均绝对 SHAP 值（特征重要性排序）
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap,
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)
    importance_df.index = importance_df.index + 1
    importance_df.index.name = 'Rank'

    importance_csv_path = os.path.join(output_dir, f'shap_feature_importance_{dataset_name}.csv')
    importance_df.to_csv(importance_csv_path, encoding='utf-8-sig')
    logger.info(f"SHAP 特征重要性已保存: {importance_csv_path}")
    logger.info(f"\nTop 10 重要特征 ({dataset_name}):")
    print(importance_df.head(10).to_string())

    # ===== 期刊级绘图全局设置 =====
    # 临时设置 matplotlib 全局参数（期刊风格）
    rc_params = {
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    original_rc = {k: mpl.rcParams.get(k) for k in rc_params}
    mpl.rcParams.update(rc_params)

    # 统一图表高度
    n_features_show = min(max_display, len(feature_names))
    fig_height = max(6, n_features_show * 0.38)

    # ===== 绘制 SHAP 摘要图 (Summary Plot / Beeswarm) =====
    logger.info("绘制 SHAP 摘要图...")
    fig_summary = plt.figure(figsize=(10, fig_height))

    shap.summary_plot(
        shap_vals, X,
        feature_names=display_names,
        max_display=max_display,
        show=False,
    )
    ax = plt.gca()
    ax.set_title(f'SHAP Summary Plot — {dataset_name.capitalize()} Set',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=12)

    # 美化 colorbar
    cb_axes = [c for c in fig_summary.axes if c != ax]
    for cax in cb_axes:
        cax.set_ylabel('Feature Value', fontsize=11, rotation=270, labelpad=12)
        cax.tick_params(labelsize=9)

    plt.tight_layout()
    summary_png = os.path.join(output_dir, f'shap_summary_{dataset_name}.png')
    summary_pdf = os.path.join(output_dir, f'shap_summary_{dataset_name}.pdf')
    plt.savefig(summary_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(summary_pdf, bbox_inches='tight', facecolor='white')
    plt.close('all')
    logger.info(f"  摘要图已保存: {summary_png}")

    # ===== 绘制 SHAP 特征重要性条形图 (Bar Plot) =====
    logger.info("绘制 SHAP 特征重要性图...")
    fig_bar = plt.figure(figsize=(10, fig_height))

    shap.summary_plot(
        shap_vals, X,
        feature_names=display_names,
        plot_type='bar',
        max_display=max_display,
        show=False,
        color='#C0392B',
    )
    ax_bar = plt.gca()
    ax_bar.set_title(f'SHAP Feature Importance — {dataset_name.capitalize()} Set',
                     fontsize=14, fontweight='bold', pad=12)
    ax_bar.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_linewidth(0.8)
    ax_bar.spines['bottom'].set_linewidth(0.8)

    plt.tight_layout()
    bar_png = os.path.join(output_dir, f'shap_importance_bar_{dataset_name}.png')
    bar_pdf = os.path.join(output_dir, f'shap_importance_bar_{dataset_name}.pdf')
    plt.savefig(bar_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(bar_pdf, bbox_inches='tight', facecolor='white')
    plt.close('all')
    logger.info(f"  重要性条形图已保存: {bar_png}")

    # 恢复 matplotlib 全局参数
    mpl.rcParams.update(original_rc)

    return shap_vals, importance_df


def run_shap_analysis(models_dir=None, output_dir=None, params=None):
    """
    执行完整的 SHAP 分析流程

    参数:
        models_dir: 模型所在目录（包含 Random_Forest_model.pkl）
        output_dir: SHAP 分析结果输出目录
        params: 流水线参数字典（None 则使用 model_pipeline.PARAMS）
    """
    if params is None:
        params = PARAMS.copy()

    # 确定模型路径
    if models_dir is None:
        models_dir = os.path.join(params['output_dir'], 'models')

    rf_model_path = os.path.join(models_dir, 'Random_Forest_model.pkl')
    if not os.path.exists(rf_model_path):
        logger.error(f"随机森林模型文件不存在: {rf_model_path}")
        logger.error("请确认模型目录正确，或先运行 model_pipeline.py 训练模型。")
        sys.exit(1)

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(models_dir), 'shap_analysis')
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 80)
    logger.info("随机森林模型 SHAP 可解释性分析")
    logger.info("=" * 80)
    logger.info(f"模型路径: {rf_model_path}")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)

    # 1. 加载模型
    model_info = load_rf_model(rf_model_path)
    rf_model = model_info['model']

    # 2. 准备数据
    X_train, X_test, y_train, y_test, expanded_names, train_df, test_df = \
        prepare_data_for_shap(params, model_info)

    # 3. 训练集 SHAP 分析
    compute_and_plot_shap(
        rf_model, X_train, expanded_names,
        output_dir, dataset_name="train", max_display=20,
    )

    # 4. 测试集 SHAP 分析
    compute_and_plot_shap(
        rf_model, X_test, expanded_names,
        output_dir, dataset_name="test", max_display=20,
    )

    # 最终汇总
    logger.info("\n" + "=" * 80)
    logger.info("SHAP 分析完成！")
    logger.info(f"结果目录: {output_dir}")
    logger.info("生成文件:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            logger.info(f"  {fname} ({size_kb:.1f} KB)")
    logger.info("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='随机森林模型 SHAP 可解释性分析')
    parser.add_argument('--models-dir', type=str, default=r"D:\projects\CervixRT_Sensitivity_Prognosis\results\model_radiomics\models",
                        help='模型目录路径（包含 Random_Forest_model.pkl），'
                             '默认为 model_pipeline.PARAMS 中 output_dir/models')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='SHAP 分析结果输出目录，默认为模型目录同级的 shap_analysis/')
    args = parser.parse_args()

    run_shap_analysis(
        models_dir=args.models_dir,
        output_dir=args.output_dir,
    )
