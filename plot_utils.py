"""
可视化工具模块 - 模型流水线绑定的绘图函数

包含：
    - LASSO系数路径图和CV误差图
    - 特征相关性热图
    - 特征重要性图（含方向）
    - ROC曲线图（训练集+测试集）
    - 校准曲线图
    - 决策曲线图
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.linear_model import lasso_path

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 绘图全局设置 - 符合期刊论文要求
# ============================================================================
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.linewidth': 1.0,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})


def plot_lasso_figures(lasso_cv_model, feature_names, output_dir, X_scaled=None, y=None):
    """
    绘制LASSO的两个经典图（符合期刊论文要求）：
    1. 特征系数随正则化参数的变化曲线
    2. 模型误差随正则化参数的变化曲线（带误差棒）

    参数:
        lasso_cv_model: 训练好的LassoCV模型
        feature_names: 特征名列表
        output_dir: 输出目录
        X_scaled: 标准化后的训练数据（用于计算系数路径）
        y: 标签数组
    """
    if lasso_cv_model is None:
        return

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    alphas = lasso_cv_model.alphas_

    # --- 图1: 特征系数路径图 ---
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 使用lasso_path计算系数路径
    if X_scaled is not None and y is not None:
        _, coef_path_computed, _ = lasso_path(X_scaled, y, alphas=alphas)
        # coef_path_computed shape: (n_features, n_alphas)
        coef_for_plot = coef_path_computed
    else:
        # fallback: 无法绘制系数路径图
        logger.warning("  未提供训练数据，无法绘制LASSO系数路径图")
        plt.close(fig)
        # 仍然绘制CV误差图
        fig, ax = plt.subplots(figsize=(6, 4.5))
        mse_path = lasso_cv_model.mse_path_
        mse_mean = np.mean(mse_path, axis=1)
        mse_std = np.std(mse_path, axis=1)
        ax.errorbar(np.log10(alphas), mse_mean, yerr=mse_std,
                    fmt='o-', markersize=3, linewidth=1.0, capsize=2, capthick=0.8,
                    color='#2c7bb6', ecolor='#abd9e9')
        ax.axvline(np.log10(lasso_cv_model.alpha_), color='k', linestyle='--', linewidth=1.0,
                   label=f'Optimal $\\lambda$ = {lasso_cv_model.alpha_:.4f}')
        ax.set_xlabel('log$_{10}$($\\lambda$)')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('LASSO Cross-Validation Error')
        ax.legend(loc='best', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, 'lasso_cv_error.png'))
        fig.savefig(os.path.join(plots_dir, 'lasso_cv_error.pdf'))
        plt.close(fig)
        return

    for i in range(coef_for_plot.shape[0]):
        ax.plot(np.log10(alphas), coef_for_plot[i, :], linewidth=0.8)

    ax.axvline(np.log10(lasso_cv_model.alpha_), color='k', linestyle='--', linewidth=1.0,
               label=f'Optimal $\\lambda$ = {lasso_cv_model.alpha_:.4f}')

    # 1SE规则（基于CV误差）
    mse_path = lasso_cv_model.mse_path_
    mse_mean = np.mean(mse_path, axis=1)
    mse_std = np.std(mse_path, axis=1)
    min_mse_idx = np.argmin(mse_mean)
    one_se_threshold = mse_mean[min_mse_idx] + mse_std[min_mse_idx]
    candidates = np.where(mse_mean <= one_se_threshold)[0]
    one_se_alpha = None
    if len(candidates) > 0:
        one_se_idx = candidates[0]
        one_se_alpha = alphas[one_se_idx]
        ax.axvline(np.log10(one_se_alpha), color='gray', linestyle=':', linewidth=1.0,
                   label=f'1-SE $\\lambda$ = {one_se_alpha:.4f}')

    ax.set_xlabel('log$_{10}$($\\lambda$)')
    ax.set_ylabel('Coefficients')
    ax.set_title('LASSO Coefficient Path')
    ax.legend(loc='best', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # 顶部标注非零系数数量
    ax2 = ax.twiny()
    n_nonzero = [np.sum(np.abs(coef_for_plot[:, j]) > 1e-6) for j in range(len(alphas))]

    tick_positions = np.linspace(0, len(alphas) - 1, min(8, len(alphas)), dtype=int)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(np.log10(alphas[tick_positions]))
    ax2.set_xticklabels([str(n_nonzero[i]) for i in tick_positions])
    ax2.set_xlabel('Number of non-zero coefficients')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'lasso_coefficient_path.png'))
    fig.savefig(os.path.join(plots_dir, 'lasso_coefficient_path.pdf'))
    plt.close(fig)
    logger.info(f"  LASSO系数路径图已保存")

    # --- 图2: CV误差随alpha变化（带误差棒）---
    fig, ax = plt.subplots(figsize=(6, 4.5))

    mse_path = lasso_cv_model.mse_path_  # (n_alphas, n_folds)
    mse_mean = np.mean(mse_path, axis=1)
    mse_std = np.std(mse_path, axis=1)

    ax.errorbar(np.log10(alphas), mse_mean, yerr=mse_std,
                fmt='o-', markersize=3, linewidth=1.0, capsize=2, capthick=0.8,
                color='#2c7bb6', ecolor='#abd9e9')
    ax.axvline(np.log10(lasso_cv_model.alpha_), color='k', linestyle='--', linewidth=1.0,
               label=f'Optimal $\\lambda$ = {lasso_cv_model.alpha_:.4f}')

    # 1SE规则
    min_mse_idx = np.argmin(mse_mean)
    one_se_threshold = mse_mean[min_mse_idx] + mse_std[min_mse_idx]
    candidates = np.where(mse_mean <= one_se_threshold)[0]
    if len(candidates) > 0:
        one_se_idx = candidates[0]
        ax.axvline(np.log10(alphas[one_se_idx]), color='gray', linestyle=':', linewidth=1.0,
                   label=f'1-SE $\\lambda$ = {alphas[one_se_idx]:.4f}')

    ax.set_xlabel('log$_{10}$($\\lambda$)')
    ax.set_ylabel('Mean Squared Error')
    ax.set_title('LASSO Cross-Validation Error')
    ax.legend(loc='best', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # 顶部标注非零系数数量
    if X_scaled is not None and y is not None:
        _, coef_path_for_top, _ = lasso_path(X_scaled, y, alphas=alphas)
        ax2 = ax.twiny()
        n_nonzero = [np.sum(np.abs(coef_path_for_top[:, j]) > 1e-6) for j in range(len(alphas))]
        tick_positions = np.linspace(0, len(alphas) - 1, min(8, len(alphas)), dtype=int)
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(np.log10(alphas[tick_positions]))
        ax2.set_xticklabels([str(n_nonzero[i]) for i in tick_positions])
        ax2.set_xlabel('Number of non-zero coefficients')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'lasso_cv_error.png'))
    fig.savefig(os.path.join(plots_dir, 'lasso_cv_error.pdf'))
    plt.close(fig)
    logger.info(f"  LASSO CV误差图已保存")


def plot_correlation_heatmap(train_df, selected_features, output_dir):
    """绘制选中特征的相关性热图"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    X_sel = train_df[selected_features].apply(pd.to_numeric, errors='coerce')
    corr_matrix = X_sel.corr(method='pearson')

    short_names = []
    for f in selected_features:
        name = f
        if len(name) > 30:
            name = name[:27] + '...'
        short_names.append(name)

    n_features = len(selected_features)
    fig_size = max(6, n_features * 0.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=n_features <= 15,
                fmt='.2f' if n_features <= 15 else '',
                cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5,
                xticklabels=short_names, yticklabels=short_names,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)

    ax.set_title('Correlation Heatmap of Selected Features')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    fig.savefig(os.path.join(plots_dir, 'feature_correlation_heatmap.png'))
    fig.savefig(os.path.join(plots_dir, 'feature_correlation_heatmap.pdf'))
    plt.close(fig)
    logger.info(f"  特征相关性热图已保存")


def _compute_feature_directions(expanded_names, selected_features, train_df, y):
    """
    为展开后的每个特征计算相关性方向（+1/-1）。
    - 展开后的 one-hot 列（如 'Stage_II'）回溯到原始列（'Stage'）计算相关性
    - 若展开列名在 train_df 中不存在，则使用原始列

    参数:
        expanded_names: 展开后的特征名列表（与模型特征一一对应）
        selected_features: 原始特征名列表
        train_df: 训练集 DataFrame
        y: 标签数组

    返回:
        list[float]: 每个展开特征的方向（+1 或 -1）
    """
    directions = []
    original_set = set(selected_features)

    for feat in expanded_names:
        # 若特征名直接存在于 train_df，直接使用
        col_to_use = feat if feat in train_df.columns else None

        if col_to_use is None:
            # 可能是 one-hot 展开列（格式: 'OriginalCol_Value'），尝试回溯原始列
            for orig in selected_features:
                if feat.startswith(orig + '_'):
                    col_to_use = orig
                    break

        if col_to_use is None:
            # 找不到对应原始列，默认方向为 +1
            directions.append(1)
            continue

        vals = pd.to_numeric(train_df[col_to_use], errors='coerce').values
        valid = ~np.isnan(vals)
        if valid.sum() > 2:
            corr, _ = stats.pointbiserialr(y[valid], vals[valid])
            directions.append(np.sign(corr) if corr != 0 else 1)
        else:
            directions.append(1)

    return directions


def plot_feature_importance(all_results, selected_features, train_df, output_dir,
                           expanded_names=None):
    """
    绘制特征重要性图（体现正相关还是负相关方向）

    参数:
        all_results: 所有模型结果字典
        selected_features: 原始特征名列表（用于方向计算）
        train_df: 训练集 DataFrame
        output_dir: 输出目录
        expanded_names: 展开后的特征名列表（与模型系数一一对应）。
                        若为 None 则使用 selected_features（无多分类变量时）
    """
    # expanded_names 与模型系数/重要性数组一一对应；若未传入则默认为 selected_features
    if expanded_names is None:
        expanded_names = list(selected_features)

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 优先使用Logistic Regression的系数（有方向性）
    if 'Logistic Regression' in all_results:
        model = all_results['Logistic Regression']['model']
        coefs = model.coef_[0]
        importance = coefs
        title = 'Feature Importance (Logistic Regression Coefficients)'
    elif 'XGBoost' in all_results:
        model = all_results['XGBoost']['model']
        importance = model.feature_importances_
        y = train_df['Label_Encoded'].values
        directions = _compute_feature_directions(expanded_names, selected_features, train_df, y)
        importance = importance * np.array(directions)
        title = 'Feature Importance (XGBoost, direction by correlation)'
    elif 'Random Forest' in all_results:
        model = all_results['Random Forest']['model']
        importance = model.feature_importances_
        y = train_df['Label_Encoded'].values
        directions = _compute_feature_directions(expanded_names, selected_features, train_df, y)
        importance = importance * np.array(directions)
        title = 'Feature Importance (Random Forest, direction by correlation)'
    else:
        logger.warning("  无法绘制特征重要性图：没有可用的模型")
        return

    # 使用 expanded_names 作为标签（与 importance 数组长度一致）
    sorted_idx = np.argsort(np.abs(importance))
    sorted_features = [expanded_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]

    short_names = []
    for f in sorted_features:
        name = f
        if len(name) > 35:
            name = name[:32] + '...'
        short_names.append(name)

    n_features = len(sorted_features)
    fig_height = max(4, n_features * 0.35)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    colors = ['#d73027' if v > 0 else '#4575b4' for v in sorted_importance]
    ax.barh(range(n_features), sorted_importance, color=colors, edgecolor='none', height=0.7)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(short_names)
    ax.set_xlabel('Coefficient / Importance')
    ax.set_title(title)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3, linewidth=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d73027', label='Positive (associated with PR)'),
        Patch(facecolor='#4575b4', label='Negative (associated with CR)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, edgecolor='black')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    fig.savefig(os.path.join(plots_dir, 'feature_importance.pdf'))
    plt.close(fig)
    logger.info(f"  特征重要性图已保存")


def plot_roc_curves(all_results, output_dir):
    """绘制所有模型的AUC曲线（训练集和测试集各一张）"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    # --- 训练集ROC ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (model_name, result) in enumerate(all_results.items()):
        y_train = result['y_train']
        y_train_proba = result['y_train_pred_proba']
        fpr, tpr, _ = roc_curve(y_train, y_train_proba)
        auc_val = roc_auc_score(y_train, y_train_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.5,
                label=f'{model_name} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('1 - Specificity (False Positive Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('ROC Curves - Training Set')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'roc_curves_train.png'))
    fig.savefig(os.path.join(plots_dir, 'roc_curves_train.pdf'))
    plt.close(fig)

    # --- 测试集ROC ---
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (model_name, result) in enumerate(all_results.items()):
        y_test = result['y_test']
        y_proba = result['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_val = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], linewidth=1.5,
                label=f'{model_name} (AUC = {auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.7)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('1 - Specificity (False Positive Rate)')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('ROC Curves - Test Set')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'roc_curves_test.png'))
    fig.savefig(os.path.join(plots_dir, 'roc_curves_test.pdf'))
    plt.close(fig)
    logger.info(f"  ROC曲线图已保存（训练集+测试集）")


def plot_calibration_curve(result, output_dir):
    """绘制最佳模型的校准曲线（训练集和测试集）"""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    model_name = result['model_name']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (y_true, y_proba, set_name) in enumerate([
        (result['y_train'], result['y_train_pred_proba'], 'Training Set'),
        (result['y_test'], result['y_pred_proba'], 'Test Set'),
    ]):
        ax = axes[idx]
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=10, strategy='uniform'
        )

        ax.plot(mean_predicted_value, fraction_of_positives, 's-',
                color='#377eb8', linewidth=1.5, markersize=5, label=model_name)
        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Perfectly calibrated')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration Curve - {set_name}')
        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'calibration_curve_best_model.png'))
    fig.savefig(os.path.join(plots_dir, 'calibration_curve_best_model.pdf'))
    plt.close(fig)
    logger.info(f"  校准曲线图已保存（最佳模型: {model_name}）")


def plot_forest_plot(X_train, y_train, expanded_names, output_dir, max_iter=500):
    """
    绘制单因素+多因素联合森林图（期刊论文发表水平）

    设计：左侧文字表格（GridSpec列1）+ 右侧森林图（GridSpec列2），
    每个特征占一行，单因素（蓝）和多因素（红）在同一行内上下偏移。

    参数:
        X_train: 标准化后的训练集特征矩阵 (numpy array)
        y_train: 训练集标签 (numpy array)
        expanded_names: 展开后的特征名列表
        output_dir: 输出目录
        max_iter: 逻辑回归最大迭代次数
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels 未安装，无法绘制森林图。请运行: pip install statsmodels")
        return

    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    logger.info("  计算单因素和多因素Logistic回归...")

    # ===== 单因素分析 =====
    uni_records = []
    for i, feat_name in enumerate(expanded_names):
        try:
            X_i = sm.add_constant(X_train[:, i])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model_i = sm.Logit(y_train, X_i)
                res_i = model_i.fit(disp=0, maxiter=max_iter)
            or_val = np.exp(res_i.params[1])
            conf_arr = np.asarray(res_i.conf_int())
            ci = np.exp(conf_arr[1, :])
            p_val = res_i.pvalues[1]
            uni_records.append({
                'Feature': feat_name, 'OR': or_val,
                'CI_low': ci[0], 'CI_high': ci[1], 'P_value': p_val
            })
        except Exception as e:
            logger.debug(f"  单因素分析失败 [{feat_name}]: {e}")
            uni_records.append({
                'Feature': feat_name, 'OR': np.nan,
                'CI_low': np.nan, 'CI_high': np.nan, 'P_value': np.nan
            })
    uni_df = pd.DataFrame(uni_records)

    # ===== 多因素分析 =====
    multi_records = []
    try:
        X_multi = sm.add_constant(X_train)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model_multi = sm.Logit(y_train, X_multi)
            res_multi = model_multi.fit(disp=0, maxiter=max_iter, method='bfgs')
        multi_params = np.asarray(res_multi.params)
        multi_conf = np.asarray(res_multi.conf_int())
        multi_pvalues = np.asarray(res_multi.pvalues)
        for i, feat_name in enumerate(expanded_names):
            param_idx = i + 1
            or_val = np.exp(multi_params[param_idx])
            ci = np.exp(multi_conf[param_idx, :])
            p_val = multi_pvalues[param_idx]
            multi_records.append({
                'Feature': feat_name, 'OR': or_val,
                'CI_low': ci[0], 'CI_high': ci[1], 'P_value': p_val
            })
    except Exception as e:
        logger.warning(f"  多因素Logistic回归拟合失败: {e}，尝试使用Newton法...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res_multi = model_multi.fit(disp=0, maxiter=max_iter, method='newton')
            multi_params = np.asarray(res_multi.params)
            multi_conf = np.asarray(res_multi.conf_int())
            multi_pvalues = np.asarray(res_multi.pvalues)
            for i, feat_name in enumerate(expanded_names):
                param_idx = i + 1
                or_val = np.exp(multi_params[param_idx])
                ci = np.exp(multi_conf[param_idx, :])
                p_val = multi_pvalues[param_idx]
                multi_records.append({
                    'Feature': feat_name, 'OR': or_val,
                    'CI_low': ci[0], 'CI_high': ci[1], 'P_value': p_val
                })
        except Exception as e2:
            logger.warning(f"  多因素分析最终失败: {e2}")
            for feat_name in expanded_names:
                multi_records.append({
                    'Feature': feat_name, 'OR': np.nan,
                    'CI_low': np.nan, 'CI_high': np.nan, 'P_value': np.nan
                })
    multi_df = pd.DataFrame(multi_records)

    # ===== 排序：按多因素P值升序 =====
    sort_order = multi_df.sort_values('P_value', ascending=True, na_position='last').index.tolist()
    uni_df = uni_df.loc[sort_order].reset_index(drop=True)
    multi_df = multi_df.loc[sort_order].reset_index(drop=True)

    n = len(expanded_names)

    # ===== 格式化函数 =====
    def _fmt_or_ci(row):
        if pd.isna(row['OR']):
            return '\u2014'
        return f"{row['OR']:.2f} ({row['CI_low']:.2f}\u2013{row['CI_high']:.2f})"

    def _fmt_p(row):
        if pd.isna(row['P_value']):
            return '\u2014'
        if row['P_value'] < 0.001:
            return '<0.001'
        return f"{row['P_value']:.3f}"

    # ===== 使用 GridSpec 布局：左表格 + 右森林图 =====
    row_height = 0.50  # 每行高度（英寸）
    fig_h = max(4.0, n * row_height + 1.8)
    fig_w = 8.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3.5, 3], wspace=0.03)

    ax_table = fig.add_subplot(gs[0])
    ax_forest = fig.add_subplot(gs[1])

    # --- 左侧表格区 (纯文字, 无坐标轴) ---
    ax_table.set_xlim(0, 1)
    ax_table.set_ylim(-0.5, n + 0.1)
    ax_table.axis('off')

    # 列位置定义 (x in 0~1)
    col_var = 0.01       # Variable
    col_uni_or = 0.50    # Univariate OR(CI)
    col_uni_p = 0.82     # Univariate P
    col_multi_or = 0.50  # Multivariate OR(CI) (same col, lower line)
    col_multi_p = 0.82   # Multivariate P

    # 表头（距黑线距离缩短）
    hdr_y = n - 0.15
    ax_table.text(col_var, hdr_y, 'Variable', fontsize=9, fontweight='bold',
                  ha='left', va='bottom')
    ax_table.text(col_uni_or, hdr_y, 'OR (95% CI)', fontsize=8, fontweight='bold',
                  ha='center', va='bottom', color='#2166ac')
    ax_table.text(col_uni_p, hdr_y, 'P value', fontsize=8, fontweight='bold',
                  ha='center', va='bottom', color='#2166ac')
    # 每行数据
    for i in range(n):
        y_pos = n - 1 - i  # 从上到下
        row_u = uni_df.iloc[i]
        row_m = multi_df.iloc[i]

        # 特征名（截断过长名称）
        feat_name = row_u['Feature']
        if len(feat_name) > 25:
            feat_name = feat_name[:22] + '...'

        # 判断显著性用于加粗
        sig_u = not pd.isna(row_u['P_value']) and row_u['P_value'] < 0.05
        sig_m = not pd.isna(row_m['P_value']) and row_m['P_value'] < 0.05

        # 特征名
        ax_table.text(col_var, y_pos + 0.12, feat_name, fontsize=8, ha='left', va='center')

        # 单因素行 (上半)
        ax_table.text(col_uni_or, y_pos + 0.12, _fmt_or_ci(row_u), fontsize=7.5,
                      ha='center', va='center', color='#2166ac',
                      fontweight='bold' if sig_u else 'normal', family='monospace')
        ax_table.text(col_uni_p, y_pos + 0.12, _fmt_p(row_u), fontsize=7.5,
                      ha='center', va='center', color='#2166ac',
                      fontweight='bold' if sig_u else 'normal', family='monospace')

        # 多因素行 (下半)
        ax_table.text(col_multi_or, y_pos - 0.18, _fmt_or_ci(row_m), fontsize=7.5,
                      ha='center', va='center', color='#b2182b',
                      fontweight='bold' if sig_m else 'normal', family='monospace')
        ax_table.text(col_multi_p, y_pos - 0.18, _fmt_p(row_m), fontsize=7.5,
                      ha='center', va='center', color='#b2182b',
                      fontweight='bold' if sig_m else 'normal', family='monospace')

        # 行间浅灰分隔线
        if i < n - 1:
            ax_table.axhline(y=y_pos - 0.5, color='#e8e8e8', linewidth=0.4)

    # 表头下划线
    ax_table.axhline(y=n - 0.5, color='black', linewidth=0.8)

    # --- 右侧森林图区 ---
    # 计算OR值范围
    all_ci_vals = []
    for df_tmp in [uni_df, multi_df]:
        valid = df_tmp.dropna(subset=['OR', 'CI_low', 'CI_high'])
        if len(valid) > 0:
            vals = np.concatenate([valid['CI_low'].values, valid['CI_high'].values])
            all_ci_vals.extend(vals[vals > 0].tolist())
    if not all_ci_vals:
        plt.close(fig)
        return

    or_min_val, or_max_val = min(all_ci_vals), max(all_ci_vals)
    x_lo = max(0.01, or_min_val * 0.5)
    x_hi = or_max_val * 2.0

    ax_forest.set_xscale('log')
    ax_forest.set_xlim(x_lo, x_hi)
    ax_forest.set_ylim(-0.5, n + 0.1)
    ax_forest.set_yticks([])

    # 垂直参考线 OR=1
    ax_forest.axvline(x=1.0, color='#555555', linestyle='--', linewidth=0.9, zorder=1)

    # 绘制每个特征的OR和CI
    y_offset_uni = 0.13
    y_offset_multi = -0.13

    for i in range(n):
        y_base = n - 1 - i

        # 背景色 (交替)
        if i % 2 == 0:
            ax_forest.axhspan(y_base - 0.5, y_base + 0.5,
                              color='#f7f7f7', zorder=0)

        # 单因素
        row_u = uni_df.iloc[i]
        if not pd.isna(row_u['OR']):
            y_u = y_base + y_offset_uni
            sig_u = not pd.isna(row_u['P_value']) and row_u['P_value'] < 0.05
            # CI 线
            ax_forest.plot([row_u['CI_low'], row_u['CI_high']], [y_u, y_u],
                           color='#2166ac', linewidth=1.8 if sig_u else 1.0,
                           solid_capstyle='round', zorder=2)
            # OR 点
            ax_forest.scatter(row_u['OR'], y_u, color='#2166ac',
                              s=50 if sig_u else 25, zorder=4,
                              marker='s', edgecolors='white', linewidths=0.4)

        # 多因素
        row_m = multi_df.iloc[i]
        if not pd.isna(row_m['OR']):
            y_m = y_base + y_offset_multi
            sig_m = not pd.isna(row_m['P_value']) and row_m['P_value'] < 0.05
            # CI 线
            ax_forest.plot([row_m['CI_low'], row_m['CI_high']], [y_m, y_m],
                           color='#b2182b', linewidth=1.8 if sig_m else 1.0,
                           solid_capstyle='round', zorder=2)
            # OR 点
            ax_forest.scatter(row_m['OR'], y_m, color='#b2182b',
                              s=50 if sig_m else 25, zorder=4,
                              marker='D', edgecolors='white', linewidths=0.4)

    # x轴设置
    import matplotlib.ticker as mticker
    ax_forest.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_forest.xaxis.get_major_formatter().set_scientific(False)
    # 分开标注：Odds Ratio用黑色，Favors用浅色小字
    ax_forest.set_xlabel('Odds Ratio (log scale)', fontsize=9, labelpad=5)
    ax_forest.text(0.12, -0.07, '\u2190 Favors CR', transform=ax_forest.transAxes,
                   fontsize=7.5, ha='center', va='top', color='#999999', fontstyle='italic')
    ax_forest.text(0.88, -0.07, 'Favors PR \u2192', transform=ax_forest.transAxes,
                   fontsize=7.5, ha='center', va='top', color='#999999', fontstyle='italic')

    # 隐藏不需要的边框
    ax_forest.spines['top'].set_visible(False)
    ax_forest.spines['right'].set_visible(False)
    ax_forest.spines['left'].set_visible(False)
    ax_forest.tick_params(axis='y', left=False)
    ax_forest.tick_params(axis='x', labelsize=8.5)

    # 图例（仅区分单因素/多因素颜色，底端与左图黑线平齐）
    legend_elements = [
        Line2D([0], [0], marker='s', color='#2166ac', linewidth=1.5,
               markersize=6, label='Univariate'),
        Line2D([0], [0], marker='D', color='#b2182b', linewidth=1.5,
               markersize=6, label='Multivariate'),
    ]
    # 左图黑线在 y=n-0.5，对应右图 axes分数 = n/(n+0.6)
    legend_bottom_frac = n / (n + 0.6)
    ax_forest.legend(handles=legend_elements, loc='lower right', fontsize=8,
                     frameon=True, edgecolor='#cccccc', fancybox=False, ncol=1,
                     bbox_to_anchor=(0.98, legend_bottom_frac))

    fig.suptitle('Forest Plot: Univariate and Multivariate Logistic Regression',
                 fontsize=11.5, fontweight='bold', y=0.98)

    fig.subplots_adjust(bottom=0.10)
    fig.savefig(os.path.join(plots_dir, 'forest_plot.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(plots_dir, 'forest_plot.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("  森林图已保存（单因素+多因素联合分析）")

    # 保存统计结果到CSV
    report_df = pd.DataFrame({
        'Feature': uni_df['Feature'],
        'Univariate_OR': uni_df['OR'],
        'Univariate_CI_Lower': uni_df['CI_low'],
        'Univariate_CI_Upper': uni_df['CI_high'],
        'Univariate_P': uni_df['P_value'],
        'Multivariate_OR': multi_df['OR'],
        'Multivariate_CI_Lower': multi_df['CI_low'],
        'Multivariate_CI_Upper': multi_df['CI_high'],
        'Multivariate_P': multi_df['P_value'],
    })
    csv_path = os.path.join(output_dir, 'forest_plot_statistics.csv')
    report_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"  森林图统计数据已保存: {csv_path}")


def plot_nomogram(lr_model, X_train, y_train, expanded_names, output_dir,
                  scaler=None, selected_features=None, cat_features=None, train_df=None,
                  nomogram_config=None):
    """
    绘制逻辑回归列线图 (Nomogram) — 期刊论文发表水平

    标准列线图设计：
    - Points标尺 (0~100)
    - 每个特征标尺（原始尺度，对齐Points轴）
    - Total Points标尺
    - Predicted Probability标尺

    参数:
        lr_model: 训练好的 LogisticRegression 模型
        X_train: 训练集特征矩阵 (numpy array, 已标准化/编码)
        y_train: 训练集标签
        expanded_names: 展开后的特征名列表
        output_dir: 输出目录
        scaler: StandardScaler 对象（用于反标准化连续变量）
        selected_features: 原始特征名列表
        cat_features: 分类变量列表
        train_df: 原始训练集 DataFrame
        nomogram_config: 列线图显示配置字典，可选键：
            - 'display_order': 特征显示排序（特征名列表）
            - 'display_names': {特征名: 显示名} 映射
            - 'binary_labels': {特征名: {0: '标签', 1: '标签'}} 二分类自定义标签
            - 'continuous_ticks': {特征名: [刻度值列表]} 自定义连续变量刻度
            - 'tick_formatters': {特征名: {值: '显示文本'}} 用于连续变量的刻度值→文本映射
            - 'total_points_step': Total Points轴刻度步长（默认自动）
    """
    if lr_model is None or not hasattr(lr_model, 'coef_'):
        logger.warning("  未提供有效的逻辑回归模型，跳过列线图")
        return

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    cfg = nomogram_config or {}

    coefs = lr_model.coef_[0]
    intercept = lr_model.intercept_[0]
    n_features = len(expanded_names)
    cat_features = cat_features or []

    # ===== 特征类型分类 =====
    cont_features_list = [f for f in selected_features if f not in cat_features] if selected_features else []
    binary_cat_list = []
    multi_cat_list = []
    if selected_features and cat_features:
        for f in selected_features:
            if f in cat_features and train_df is not None and f in train_df.columns:
                n_unique = train_df[f].dropna().nunique()
                if n_unique <= 2:
                    binary_cat_list.append(f)
                else:
                    multi_cat_list.append(f)
            elif f in cat_features:
                binary_cat_list.append(f)

    n_cont = len(cont_features_list)
    n_binary = len(binary_cat_list)

    # ===== 多分类变量配置 =====
    multi_cat_config = cfg.get('multi_cat_config', {})
    # 构建 expanded_col -> group_name 映射
    col_to_multicat_group = {}
    for group_orig_name, group_cfg in multi_cat_config.items():
        for col_name in group_cfg.get('categories', {}):
            if col_name != 'reference':
                col_to_multicat_group[col_name] = group_orig_name

    # ===== 计算每个特征的标准化范围和原始范围 =====
    feat_info = []
    for i, feat in enumerate(expanded_names):
        col_vals = X_train[:, i]
        v_min, v_max = float(np.min(col_vals)), float(np.max(col_vals))

        if i < n_cont:
            if scaler is not None and i < len(scaler.scale_):
                orig_min = v_min * scaler.scale_[i] + scaler.mean_[i]
                orig_max = v_max * scaler.scale_[i] + scaler.mean_[i]
            else:
                orig_min, orig_max = v_min, v_max
            if train_df is not None and feat in train_df.columns:
                col_orig = pd.to_numeric(train_df[feat], errors='coerce').dropna()
                orig_min, orig_max = float(col_orig.min()), float(col_orig.max())
            feat_info.append({'orig_min': orig_min, 'orig_max': orig_max, 'is_binary': False,
                              'label': feat, 'v_min': v_min, 'v_max': v_max, 'idx': i})
        elif i < n_cont + n_binary:
            feat_info.append({'orig_min': 0, 'orig_max': 1, 'is_binary': True,
                              'label': feat, 'v_min': 0.0, 'v_max': 1.0, 'idx': i})
        else:
            feat_info.append({'orig_min': 0, 'orig_max': 1, 'is_binary': True,
                              'label': feat, 'v_min': 0.0, 'v_max': 1.0, 'idx': i})

    # ===== 构建显示项目列表 =====
    # 每个显示项目可以是：
    # - 单个特征 {'type': 'single', 'feat_idx': i}
    # - 多分类组 {'type': 'multicat', 'group_name': ..., 'col_indices': [...], 'categories': {...}}
    display_items = []
    grouped_indices = set()

    for i, fi in enumerate(feat_info):
        feat_name = fi['label']
        if feat_name in col_to_multicat_group:
            group_orig = col_to_multicat_group[feat_name]
            if group_orig not in [d.get('group_name') for d in display_items if d['type'] == 'multicat']:
                # 第一次遇到该组，创建组显示项
                group_cfg = multi_cat_config[group_orig]
                col_indices = []
                for j, fj in enumerate(feat_info):
                    if fj['label'] in group_cfg.get('categories', {}) and fj['label'] != 'reference':
                        col_indices.append(j)
                        grouped_indices.add(j)
                display_items.append({
                    'type': 'multicat',
                    'group_name': group_orig,
                    'col_indices': col_indices,
                    'categories': group_cfg.get('categories', {}),
                    'display_name': group_cfg.get('display_name', group_orig),
                })
            # 已处理，跳过
            grouped_indices.add(i)
        elif i not in grouped_indices:
            display_items.append({'type': 'single', 'feat_idx': i})

    # ===== 标准列线图数学 =====
    # 计算每个显示项的最大LP贡献
    item_max_contributions = []
    for item in display_items:
        if item['type'] == 'single':
            fi = feat_info[item['feat_idx']]
            idx = item['feat_idx']
            range_i = fi['v_max'] - fi['v_min']
            item_max_contributions.append(abs(coefs[idx]) * range_i)
        else:
            # 多分类变量：每个类别的LP贡献 = coef * 1，reference=0
            lp_vals = [0.0]  # reference category LP = 0
            for col_idx in item['col_indices']:
                lp_vals.append(coefs[col_idx] * 1.0)
            item_max_contributions.append(max(lp_vals) - min(lp_vals))

    max_single_contrib = max(item_max_contributions) if item_max_contributions else 1.0
    if max_single_contrib < 1e-10:
        max_single_contrib = 1.0

    POINTS_MAX = 100
    points_per_lp = POINTS_MAX / max_single_contrib

    # 每个显示项的点数范围和最小LP
    item_points_range = []
    item_min_lp = []
    for item in display_items:
        if item['type'] == 'single':
            fi = feat_info[item['feat_idx']]
            idx = item['feat_idx']
            lp_at_min = coefs[idx] * fi['v_min']
            lp_at_max = coefs[idx] * fi['v_max']
            min_lp = min(lp_at_min, lp_at_max)
            max_lp = max(lp_at_min, lp_at_max)
            item_min_lp.append(min_lp)
            item_points_range.append((max_lp - min_lp) * points_per_lp)
        else:
            lp_vals = [0.0]
            for col_idx in item['col_indices']:
                lp_vals.append(coefs[col_idx] * 1.0)
            min_lp = min(lp_vals)
            max_lp = max(lp_vals)
            item_min_lp.append(min_lp)
            item_points_range.append((max_lp - min_lp) * points_per_lp)

    total_points_max = sum(item_points_range)

    # ===== 确定显示顺序 =====
    display_order = cfg.get('display_order', None)
    if display_order:
        ordered_item_indices = []
        for name in display_order:
            for idx, item in enumerate(display_items):
                item_label = item.get('display_name') or item.get('group_name') or ''
                if item['type'] == 'single':
                    item_label = feat_info[item['feat_idx']]['label']
                if item_label == name or (item['type'] == 'multicat' and item['group_name'] == name):
                    ordered_item_indices.append(idx)
                    break
        # 未在display_order中的项追加到末尾
        for idx in range(len(display_items)):
            if idx not in ordered_item_indices:
                ordered_item_indices.append(idx)
    else:
        ordered_item_indices = list(range(len(display_items)))

    # ===== 绘制列线图 =====
    n_display = len(ordered_item_indices)
    n_rows = n_display + 3  # Points + features + Total Points + Risk
    row_h = 0.60
    fig_h = max(5.5, n_rows * row_h + 1.2)
    fig_w = 10.5

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h),
                             gridspec_kw={'hspace': 0.05})
    if n_rows == 1:
        axes = [axes]

    # --- 统一设置 ---
    for ax in axes:
        ax.set_xlim(0, POINTS_MAX)
        ax.set_ylim(-0.3, 0.3)
        ax.tick_params(axis='x', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', left=False, labelleft=False)
        for sp in ax.spines.values():
            sp.set_visible(False)

    # --- 第1行: Points 标尺 (0~100) ---
    ax_pts = axes[0]
    ax_pts.axhline(y=0, color='black', linewidth=1.4)
    tick_vals = list(range(0, POINTS_MAX + 1, 10))
    for v in tick_vals:
        ax_pts.plot([v, v], [0, -0.10], color='black', linewidth=1.0)
    for v in range(0, POINTS_MAX + 1, 5):
        if v % 10 != 0:
            ax_pts.plot([v, v], [0, -0.06], color='black', linewidth=0.6)
    for v in tick_vals:
        ax_pts.text(v, -0.14, str(v), fontsize=9.5, ha='center', va='top')
    ax_pts.set_ylabel('Points', fontsize=11, fontweight='bold', rotation=0,
                       ha='right', va='center', labelpad=55)
    ax_pts.set_title('Nomogram for Predicting PR Risk', fontsize=13, fontweight='bold', pad=12)

    # --- 每个特征标尺（按display_order显示）---
    display_names = cfg.get('display_names', {})
    binary_labels = cfg.get('binary_labels', {})
    continuous_ticks = cfg.get('continuous_ticks', {})
    tick_formatters = cfg.get('tick_formatters', {})

    for row_idx, item_idx in enumerate(ordered_item_indices):
        ax = axes[row_idx + 1]
        item = display_items[item_idx]
        max_pts_i = item_points_range[item_idx]
        color = 'black'

        # 绘制基线
        ax.plot([0, max_pts_i], [0, 0], color=color, linewidth=1.4)

        if item['type'] == 'multicat':
            # === 多分类变量：在一条轴上显示所有类别 ===
            label = item['display_name']
            ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=0,
                           ha='right', va='center', labelpad=55)

            categories = item['categories']
            min_lp_group = item_min_lp[item_idx]

            # Reference category: LP = 0
            ref_label = categories.get('reference', 'Ref')
            ref_pos = (0.0 - min_lp_group) * points_per_lp
            ax.plot([ref_pos, ref_pos], [0, -0.10], color=color, linewidth=1.0)
            ax.text(ref_pos, -0.14, ref_label, fontsize=9.5, ha='center', va='top', color=color)

            # Other categories
            for col_idx in item['col_indices']:
                col_name = feat_info[col_idx]['label']
                cat_label = categories.get(col_name, col_name)
                lp_val = coefs[col_idx] * 1.0
                pts_pos = (lp_val - min_lp_group) * points_per_lp
                if -1 <= pts_pos <= max_pts_i + 1:
                    ax.plot([pts_pos, pts_pos], [0, -0.10], color=color, linewidth=1.0)
                    ax.text(pts_pos, -0.14, cat_label, fontsize=9.5, ha='center', va='top', color=color)

        elif item['type'] == 'single':
            # === 单个特征 ===
            feat_idx = item['feat_idx']
            fi = feat_info[feat_idx]
            feat_name = fi['label']
            min_lp_feat = item_min_lp[item_idx]

            # 显示名（支持自定义）
            label = display_names.get(feat_name, feat_name)
            if len(label) > 28:
                label = label[:25] + '...'
            ax.set_ylabel(label, fontsize=11, fontweight='bold', rotation=0,
                           ha='right', va='center', labelpad=55)

            if fi['is_binary']:
                # 二分类变量
                if coefs[feat_idx] >= 0:
                    pos_0, pos_1 = 0, max_pts_i
                else:
                    pos_0, pos_1 = max_pts_i, 0

                ax.plot([pos_0, pos_0], [0, -0.10], color=color, linewidth=1.0)
                ax.plot([pos_1, pos_1], [0, -0.10], color=color, linewidth=1.0)

                # 自定义标签
                lbl_map = binary_labels.get(feat_name, {0: '0', 1: '1'})
                ax.text(pos_0, -0.14, str(lbl_map.get(0, '0')), fontsize=9.5,
                        ha='center', va='top', color=color)
                ax.text(pos_1, -0.14, str(lbl_map.get(1, '1')), fontsize=9.5,
                        ha='center', va='top', color=color)
            else:
                # 连续变量
                custom_ticks = continuous_ticks.get(feat_name, None)
                formatter = tick_formatters.get(feat_name, None)

                if custom_ticks is not None:
                    orig_ticks = np.array(custom_ticks, dtype=float)
                    if scaler is not None and feat_idx < n_cont and feat_idx < len(scaler.scale_):
                        std_ticks = (orig_ticks - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
                    else:
                        std_ticks = np.interp(orig_ticks,
                                              [fi['orig_min'], fi['orig_max']],
                                              [fi['v_min'], fi['v_max']])
                else:
                    val_range = fi['orig_max'] - fi['orig_min']
                    if val_range <= 5:
                        n_ticks = 5
                    elif val_range <= 20:
                        n_ticks = 6
                    else:
                        n_ticks = 7
                    orig_ticks = np.linspace(fi['orig_min'], fi['orig_max'], n_ticks)
                    std_ticks = np.linspace(fi['v_min'], fi['v_max'], n_ticks)

                # 绘制刻度
                for j, (orig_val, std_val) in enumerate(zip(orig_ticks, std_ticks)):
                    lp_val = coefs[feat_idx] * std_val
                    pts_pos = (lp_val - min_lp_feat) * points_per_lp
                    if -1 <= pts_pos <= max_pts_i + 1:
                        ax.plot([pts_pos, pts_pos], [0, -0.10],
                                color=color, linewidth=1.0)
                        if formatter and orig_val in formatter:
                            txt = formatter[orig_val]
                        elif formatter:
                            int_val = int(round(orig_val))
                            txt = formatter.get(int_val, f'{orig_val:.1f}')
                        else:
                            val_range = fi['orig_max'] - fi['orig_min']
                            if val_range > 100:
                                txt = f'{orig_val:.0f}'
                            else:
                                txt = f'{orig_val:.1f}'
                        ax.text(pts_pos, -0.14, txt, fontsize=9.5, ha='center', va='top',
                                color=color)

    # --- Total Points 标尺 ---
    ax_total = axes[n_display + 1]
    ax_total.plot([0, POINTS_MAX], [0, 0], color='black', linewidth=1.4)

    total_display_scale = POINTS_MAX / total_points_max if total_points_max > 0 else 1.0

    # Total Points 刻度：使用整十步长
    total_step = cfg.get('total_points_step', None)
    if total_step is None:
        raw_step = total_points_max / 10
        if raw_step <= 5:
            total_step = 5
        elif raw_step <= 10:
            total_step = 10
        elif raw_step <= 25:
            total_step = 20
        elif raw_step <= 50:
            total_step = 50
        else:
            total_step = 100

    total_tick_vals = np.arange(0, total_points_max + total_step, total_step)
    total_tick_vals = total_tick_vals[total_tick_vals <= total_points_max * 1.05]

    for tv in total_tick_vals:
        tp = tv * total_display_scale
        if 0 <= tp <= POINTS_MAX:
            ax_total.plot([tp, tp], [0, -0.10], color='black', linewidth=1.0)
            ax_total.text(tp, -0.14, f'{int(tv)}', fontsize=9.5, ha='center', va='top')

    ax_total.set_ylabel('Total Points', fontsize=11, fontweight='bold', rotation=0,
                         ha='right', va='center', labelpad=55)

    # --- Risk of PR 概率标尺 ---
    ax_risk = axes[n_display + 2]
    ax_risk.set_ylim(-0.55, 0.3)
    ax_risk.plot([0, POINTS_MAX], [0, 0], color='black', linewidth=1.4)

    # 计算base_lp：所有显示项的最小LP之和 + intercept
    base_lp = intercept + sum(item_min_lp)

    target_probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    risk_pos_map = {}  # 保存概率对应的位置用于风险分层
    for p in target_probs:
        lp_needed = np.log(p / (1 - p))
        total_points_val = (lp_needed - base_lp) * points_per_lp
        display_pos = total_points_val * total_display_scale
        risk_pos_map[p] = display_pos
        if 0 <= display_pos <= POINTS_MAX:
            ax_risk.plot([display_pos, display_pos], [0, -0.10],
                         color='black', linewidth=1.0)
            if p < 0.1:
                txt = f'{p:.2f}'
            elif p > 0.9:
                txt = f'{p:.2f}'
            else:
                txt = f'{p:.1f}'
            ax_risk.text(display_pos, -0.14, txt, fontsize=9.5, ha='center', va='top')

    ax_risk.set_ylabel('Risk of PR', fontsize=11, fontweight='bold', rotation=0,
                        ha='right', va='center', labelpad=55)

    # --- 风险分层标注 (Risk stratification) ---
    risk_threshold = cfg.get('risk_threshold', 0.5)
    # 计算阈值对应的显示位置
    lp_thresh = np.log(risk_threshold / (1 - risk_threshold))
    thresh_total_pts = (lp_thresh - base_lp) * points_per_lp
    thresh_display_pos = thresh_total_pts * total_display_scale
    thresh_display_pos = np.clip(thresh_display_pos, 0, POINTS_MAX)

    # 背景条的y范围（在刻度文字下方）
    bar_y_bottom = -0.50
    bar_y_top = -0.35
    from matplotlib.patches import FancyBboxPatch
    # Low risk 绿色条（左侧）
    if thresh_display_pos > 0:
        ax_risk.axhspan(bar_y_bottom, bar_y_top, xmin=0/POINTS_MAX,
                        xmax=thresh_display_pos/POINTS_MAX,
                        color='#c8e6c9', alpha=0.85, zorder=1)
        low_center = thresh_display_pos / 2
        ax_risk.text(low_center, (bar_y_bottom + bar_y_top) / 2, 'Low Risk',
                     fontsize=9, fontweight='bold', color='#2e7d32',
                     ha='center', va='center', zorder=2)
    # High risk 红色条（右侧）
    if thresh_display_pos < POINTS_MAX:
        ax_risk.axhspan(bar_y_bottom, bar_y_top, xmin=thresh_display_pos/POINTS_MAX,
                        xmax=1.0,
                        color='#ffcdd2', alpha=0.85, zorder=1)
        high_center = (thresh_display_pos + POINTS_MAX) / 2
        ax_risk.text(high_center, (bar_y_bottom + bar_y_top) / 2, 'High Risk',
                     fontsize=9, fontweight='bold', color='#c62828',
                     ha='center', va='center', zorder=2)
    # 阈值分界线
    ax_risk.plot([thresh_display_pos, thresh_display_pos], [bar_y_bottom, bar_y_top],
                 color='#424242', linewidth=1.2, zorder=3)

    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.04, hspace=0.05)
    fig.savefig(os.path.join(plots_dir, 'nomogram.png'), bbox_inches='tight', dpi=300)
    fig.savefig(os.path.join(plots_dir, 'nomogram.pdf'), bbox_inches='tight')
    plt.close(fig)
    logger.info("  列线图已保存")


def plot_decision_curve(result, output_dir):
    """
    绘制最佳模型的决策曲线（Decision Curve Analysis, DCA）
    仅展示净收益 >= 0 的区域，符合论文发表规范
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    model_name = result['model_name']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for idx, (y_true, y_proba, set_name) in enumerate([
        (result['y_train'], result['y_train_pred_proba'], 'Training Set'),
        (result['y_test'], result['y_pred_proba'], 'Test Set'),
    ]):
        ax = axes[idx]
        # 使用更细粒度的阈值，临床常用范围 0.01-0.99
        thresholds = np.linspace(0.01, 0.99, 200)
        net_benefits_model = []
        net_benefits_all = []

        n = len(y_true)
        prevalence = np.mean(y_true)

        for thresh in thresholds:
            y_pred_t = (y_proba >= thresh).astype(int)
            tp = np.sum((y_pred_t == 1) & (y_true == 1))
            fp = np.sum((y_pred_t == 1) & (y_true == 0))
            nb_model = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefits_model.append(nb_model)

            nb_all = prevalence - (1 - prevalence) * (thresh / (1 - thresh))
            net_benefits_all.append(nb_all)

        net_benefits_model = np.array(net_benefits_model)
        net_benefits_all = np.array(net_benefits_all)

        # ===== 核心优化：仅保留净收益 > 0 的阈值范围 =====
        # 找出模型净收益 > 0 的阈值
        positive_mask = net_benefits_model > 0
        if positive_mask.any():
            # 找到净收益 > 0 的连续范围
            positive_indices = np.where(positive_mask)[0]
            start_idx = positive_indices[0]
            end_idx = positive_indices[-1]
            # 向两端稍微扩展一点以展示下降趋势
            start_idx = max(0, start_idx - 2)
            end_idx = min(len(thresholds) - 1, end_idx + 2)

            plot_thresholds = thresholds[start_idx:end_idx + 1]
            plot_nb_model = net_benefits_model[start_idx:end_idx + 1]
            plot_nb_all = net_benefits_all[start_idx:end_idx + 1]
        else:
            # 回退：使用全部阈值但截断负值
            plot_thresholds = thresholds
            plot_nb_model = net_benefits_model
            plot_nb_all = net_benefits_all

        # 绘制模型曲线
        ax.plot(plot_thresholds, plot_nb_model, color='#d73027', linewidth=1.8,
                label=model_name, zorder=3)

        # Treat All 曲线（仅在净收益 > 0 的部分显示）
        treat_all_positive = plot_nb_all > 0
        if treat_all_positive.any():
            ax.plot(plot_thresholds[treat_all_positive], plot_nb_all[treat_all_positive],
                    color='#4575b4', linewidth=1.2, linestyle='--', label='Treat All', zorder=2)

        # Treat None 基线 (y=0)
        ax.axhline(y=0, color='#333333', linewidth=0.9, linestyle='-', label='Treat None', zorder=1)

        # ===== 设置坐标轴 =====
        ax.set_xlim(plot_thresholds[0], plot_thresholds[-1])
        # Y轴从0开始，上界为最大净收益的1.15倍
        y_max = max(np.max(plot_nb_model), np.max(plot_nb_all[treat_all_positive]) if treat_all_positive.any() else 0.05)
        ax.set_ylim(-0.005, y_max * 1.15 + 0.005)

        ax.set_xlabel('Threshold Probability', fontsize=10)
        ax.set_ylabel('Net Benefit', fontsize=10)
        ax.set_title(f'Decision Curve Analysis — {set_name}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, edgecolor='black',
                  fancybox=False, framealpha=0.95, fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)

        # 添加阈值刻度优化
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
        ax.tick_params(axis='both', which='major', labelsize=9)

        # 添加边框美化
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'decision_curve_best_model.png'),
                bbox_inches='tight')
    fig.savefig(os.path.join(plots_dir, 'decision_curve_best_model.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  决策曲线图已保存（最佳模型: {model_name}，仅展示净收益>0区域）")


def plot_rfe_cv_curve(rfe_report, output_dir):
    """
    绘制RFE交叉验证AUC随特征数量的变化曲线（带误差bar）

    参数:
        rfe_report: RFE报告字典，包含 feature_range, cv_auc_means, cv_auc_stds, optimal_n_features
        output_dir: 输出目录
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    feature_range = rfe_report['feature_range']
    cv_auc_means = rfe_report['cv_auc_means']
    cv_auc_stds = rfe_report['cv_auc_stds']
    optimal_n = rfe_report['optimal_n_features']
    estimator_name = rfe_report.get('rfe_estimator', 'Unknown')

    # 格式化模型名称用于标题
    estimator_display = {
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
    }.get(estimator_name, estimator_name)

    fig, ax = plt.subplots(figsize=(7, 5))

    # 绘制均值曲线
    ax.plot(feature_range, cv_auc_means, color='#2166ac', linewidth=1.5,
            marker='o', markersize=3, label='Mean CV AUC')

    # 绘制误差带
    ax.fill_between(
        feature_range,
        cv_auc_means - cv_auc_stds,
        cv_auc_means + cv_auc_stds,
        alpha=0.2, color='#2166ac', label=r'$\pm$ 1 SD'
    )

    # 标记最优特征数
    optimal_idx = feature_range.index(optimal_n)
    ax.axvline(x=optimal_n, color='#e41a1c', linestyle='--', linewidth=1.0,
               label=f'Optimal: {optimal_n} features')
    ax.plot(optimal_n, cv_auc_means[optimal_idx], 'r*', markersize=12, zorder=5)

    # 在最优点旁标注AUC值
    ax.annotate(
        f'AUC={cv_auc_means[optimal_idx]:.3f}',
        xy=(optimal_n, cv_auc_means[optimal_idx]),
        xytext=(optimal_n + max(1, len(feature_range) * 0.05),
                cv_auc_means[optimal_idx] + 0.01),
        fontsize=9, color='#e41a1c',
        arrowprops=dict(arrowstyle='->', color='#e41a1c', lw=0.8)
    )

    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Cross-Validation AUC')
    ax.set_title(f'RFE Feature Selection (Estimator: {estimator_display})')
    ax.legend(loc='lower right', frameon=True, edgecolor='black')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    # 设置x轴为整数
    if len(feature_range) <= 20:
        ax.set_xticks(feature_range)
    else:
        step = max(1, len(feature_range) // 15)
        ticks = list(range(feature_range[0], feature_range[-1] + 1, step))
        if optimal_n not in ticks:
            ticks.append(optimal_n)
            ticks.sort()
        ax.set_xticks(ticks)

    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'rfe_cv_auc_curve.png'))
    fig.savefig(os.path.join(plots_dir, 'rfe_cv_auc_curve.pdf'))
    plt.close(fig)
    logger.info(f"  RFE交叉验证AUC曲线已保存")
