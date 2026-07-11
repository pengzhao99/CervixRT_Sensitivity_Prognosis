"""
影像预处理模块 - 基于ROI的百分位数截断和Z-score标准化
支持CT和MRI影像预处理
"""

import numpy as np
import SimpleITK as sitk
from scipy import stats


def percentile_truncation(image_array, lower_percentile=0.5, upper_percentile=99.5):
    """
    基于百分位数的强度截断
    
    Args:
        image_array: 输入影像数组
        lower_percentile: 下限百分位数 (默认0.5%)
        upper_percentile: 上限百分位数 (默认99.5%)
    
    Returns:
        截断后的影像数组
    """
    lower_bound = np.percentile(image_array, lower_percentile)
    upper_bound = np.percentile(image_array, upper_percentile)
    
    # 截断极值
    truncated = np.clip(image_array, lower_bound, upper_bound)
    
    return truncated


def zscore_normalize(image_array, target_mean=0, target_std=1):
    """
    Z-score标准化
    
    Args:
        image_array: 输入影像数组
        target_mean: 目标均值 (默认0)
        target_std: 目标标准差 (默认1)
    
    Returns:
        标准化后的影像数组
    """
    mean = np.mean(image_array)
    std = np.std(image_array)
    
    if std == 0:
        # 避免除以零
        return np.zeros_like(image_array) + target_mean
    
    normalized = (image_array - mean) / std
    normalized = normalized * target_std + target_mean
    
    return normalized


def extract_roi_pixels(image_array, mask_array):
    """
    提取ROI区域的像素值
    
    Args:
        image_array: 影像数组
        mask_array: 掩码数组 (0表示背景，>0表示ROI)
    
    Returns:
        ROI区域的像素值数组
    """
    roi_pixels = image_array[mask_array > 0]
    return roi_pixels


def percentile_truncation_roi(image_array, mask_array, lower_percentile=0.5, upper_percentile=99.5):
    """
    基于ROI的百分位数截断
    只考虑ROI区域内的像素值来计算百分位数
    
    Args:
        image_array: 输入影像数组
        mask_array: ROI掩码数组
        lower_percentile: 下限百分位数
        upper_percentile: 上限百分位数
    
    Returns:
        截断后的影像数组
    """
    # 提取ROI像素
    roi_pixels = extract_roi_pixels(image_array, mask_array)
    
    if len(roi_pixels) == 0:
        # 如果ROI为空，返回原图
        return image_array
    
    # 基于ROI计算百分位数
    lower_bound = np.percentile(roi_pixels, lower_percentile)
    upper_bound = np.percentile(roi_pixels, upper_percentile)
    
    # 截断
    truncated = np.clip(image_array, lower_bound, upper_bound)
    
    return truncated


def zscore_normalize_roi(image_array, mask_array, target_mean=0, target_std=1):
    """
    基于ROI的Z-score标准化
    使用ROI区域的均值和标准差进行标准化
    
    Args:
        image_array: 输入影像数组
        mask_array: ROI掩码数组
        target_mean: 目标均值
        target_std: 目标标准差
    
    Returns:
        标准化后的影像数组
    """
    # 提取ROI像素
    roi_pixels = extract_roi_pixels(image_array, mask_array)
    
    if len(roi_pixels) == 0:
        # 如果ROI为空，返回原图
        return image_array
    
    # 基于ROI计算统计量
    roi_mean = np.mean(roi_pixels)
    roi_std = np.std(roi_pixels)
    
    if roi_std == 0:
        # 避免除以零，返回目标均值
        return np.full_like(image_array, target_mean)
    
    # 对整个图像进行标准化（基于ROI的统计量）
    normalized = (image_array - roi_mean) / roi_std
    normalized = normalized * target_std + target_mean
    
    return normalized


def preprocess_ct(ct_image, gtv_mask,
                  lower_bound=-150, upper_bound=250):
    """
    CT影像预处理流程 - 固定范围截断
    截断到 -150 到 250 HU，不进行 Z-score 标准化

    Args:
        ct_image: SimpleITK CT影像对象
        gtv_mask: SimpleITK GTV掩码对象（用于保持接口一致，实际不使用）
        lower_bound: CT HU 下限，默认 -150
        upper_bound: CT HU 上限，默认 250

    Returns:
        预处理后的SimpleITK影像对象
    """
    # 转换为numpy数组
    ct_array = sitk.GetArrayFromImage(ct_image)

    # 固定范围截断
    ct_clipped = np.clip(ct_array, lower_bound, upper_bound)

    # 转换回SimpleITK影像
    preprocessed_image = sitk.GetImageFromArray(ct_clipped)
    preprocessed_image.CopyInformation(ct_image)

    return preprocessed_image


def preprocess_mri(mri_image, roi_mask,
                   lower_percentile=1.0, upper_percentile=99.0,
                   target_mean=0, target_std=1):
    """
    MRI影像预处理流程 - 百分位数截断 + Z-score 标准化
    使用传入的ROI掩码（已合并的GTV+Ring区域）进行预处理

    Args:
        mri_image: SimpleITK MRI影像对象
        roi_mask: SimpleITK ROI掩码对象（已合并的所有ROI区域）
        lower_percentile: 下限百分位数，默认 1%
        upper_percentile: 上限百分位数，默认 99%
        target_mean: Z-score 目标均值，默认 0
        target_std: Z-score 目标标准差，默认 1

    Returns:
        预处理后的SimpleITK影像对象
    """
    # 转换为numpy数组
    mri_array = sitk.GetArrayFromImage(mri_image)
    roi_mask_array = sitk.GetArrayFromImage(roi_mask)

    # 确保掩码是二值的
    roi_mask_array = (roi_mask_array > 0).astype(np.uint8)

    # 步骤1: 基于ROI的百分位数截断（去除极值）
    mri_truncated = percentile_truncation_roi(
        mri_array, roi_mask_array,
        lower_percentile, upper_percentile
    )

    # 步骤2: 基于ROI进行 Z-score 标准化
    mri_normalized = zscore_normalize_roi(
        mri_truncated, roi_mask_array,
        target_mean, target_std
    )

    # 转换回SimpleITK影像
    preprocessed_image = sitk.GetImageFromArray(mri_normalized)
    preprocessed_image.CopyInformation(mri_image)

    return preprocessed_image


def preprocess_image(image, mask, modality='CT', config=None):
    """
    通用影像预处理接口

    Args:
        image: SimpleITK影像对象
        mask: SimpleITK掩码对象（已合并的ROI区域）
        modality: 模态类型 ('CT' 或 'MRI')
        config: 预处理配置字典，如果为None则使用默认配置

    Returns:
        预处理后的SimpleITK影像对象
    """
    if modality.upper() == 'CT':
        if config is None:
            config = {'lower_bound': -150, 'upper_bound': 250}
        return preprocess_ct(
            image, mask,
            config.get('lower_bound', -150),
            config.get('upper_bound', 250)
        )
    elif modality.upper() == 'MRI':
        if config is None:
            config = {
                'percentile_low': 1.0,
                'percentile_high': 99.0,
                'target_mean': 0,
                'target_std': 1
            }
        return preprocess_mri(
            image, mask,
            config.get('percentile_low', 1.0),
            config.get('percentile_high', 99.0),
            config.get('target_mean', 0),
            config.get('target_std', 1),
        )
    else:
        raise ValueError(f"不支持的模态类型: {modality}。请选择 'CT' 或 'MRI'")


def get_preprocessing_stats(image, mask):
    """
    获取影像预处理的统计信息
    
    Args:
        image: 原始影像数组
        mask: ROI掩码数组
    
    Returns:
        包含统计信息的字典
    """
    roi_pixels = extract_roi_pixels(image, mask)
    
    stats_dict = {
        'roi_voxels': len(roi_pixels),
        'original_min': np.min(image),
        'original_max': np.max(image),
        'original_mean': np.mean(image),
        'original_std': np.std(image),
        'roi_min': np.min(roi_pixels) if len(roi_pixels) > 0 else 0,
        'roi_max': np.max(roi_pixels) if len(roi_pixels) > 0 else 0,
        'roi_mean': np.mean(roi_pixels) if len(roi_pixels) > 0 else 0,
        'roi_std': np.std(roi_pixels) if len(roi_pixels) > 0 else 0,
        'percentile_0_5': np.percentile(roi_pixels, 0.5) if len(roi_pixels) > 0 else 0,
        'percentile_99_5': np.percentile(roi_pixels, 99.5) if len(roi_pixels) > 0 else 0
    }
    
    return stats_dict


if __name__ == '__main__':
    # 简单测试
    print("影像预处理模块加载成功")
    print("支持的功能:")
    print("  - percentile_truncation: 百分位数截断")
    print("  - percentile_truncation_roi: 基于ROI的百分位数截断")
    print("  - zscore_normalize: Z-score标准化")
    print("  - zscore_normalize_roi: 基于ROI的Z-score标准化")
    print("  - preprocess_ct: CT预处理")
    print("  - preprocess_mri: MRI预处理")
