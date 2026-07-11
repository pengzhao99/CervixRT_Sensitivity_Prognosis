"""
影像组学特征提取程序 - 独立可配置的特征提取工具

功能：
    - 支持指定图像文件和多个ROI文件
    - 支持CT、MRI（T1, T2, DWI, ADC）和Dose（剂量）不同的预处理方式
    - Dose模态不进行预处理，直接使用原始剂量图像提取特征
    - 预处理时使用所有ROI的合并区域计算百分位数截断和Z-score归一化
    - 分别提取每个ROI的影像组学特征
    - 支持原图、小波滤波和高斯拉普拉斯（LoG）滤波图像的特征提取
    - 输出宽格式特征CSV文件

使用方式：
    直接修改下方 PARAMS 字典中的参数，然后运行：
    python extract_features.py
"""

import os
import logging
import pandas as pd
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor

from image_preprocessing import preprocess_ct, preprocess_mri

# ============================================================================
# 参数设置区域 - 根据需要修改以下参数
# ============================================================================

PARAMS = {
    # ---- 数据路径 ----
    'data_dir': r'E:\data\CR-PR\CT\nii',        # 图像数据目录（每个患者一个子文件夹）

    # ---- 图像和ROI文件名 ----
    'image_file': 'ct.nii.gz',     # 图像文件名
    'roi_files': {                   # ROI文件名字典（键为ROI名称，值为文件名）
        'GTV': 'gtv.nii.gz',        # 分别提取每个ROI的特征
        'Ring': 'ctv1-gtv.nii.gz',   # 取消注释以提取Ring区域特征
    },

    # ---- 模态和预处理 ----
    # 模态类型: 'CT', 'MRI' 或 'Dose'
    # CT: 固定范围截断（不做Z-score）
    # MRI: 百分位数截断 + Z-score标准化（适用于T1, T2, DWI, ADC）
    # Dose: 不进行预处理，直接使用原始剂量图像
    'modality': 'CT',
    'modality_prefix': '',  # 特征名称前缀（例如 'CT_', 'MRI_T2_'），留空则自动使用modality名

    # CT 预处理参数
    'ct_preprocessing': {
        'lower_bound': -150,       # CT HU 下限
        'upper_bound': 250,        # CT HU 上限
    },

    # MRI 预处理参数（适用于T1, T2, DWI, ADC）
    'mri_preprocessing': {
        'percentile_low': 0.5,     # 下限百分位数
        'percentile_high': 99.5,   # 上限百分位数
        'target_mean': 0,          # Z-score 目标均值
        'target_std': 1,           # Z-score 目标标准差
    },

    # ---- 特征提取参数 ----
    # CT 和 MRI 使用不同的默认 radiomics 参数
    'radiomics_settings_ct': {
        'binWidth': 15,                            # CT灰度量化宽度（HU值范围大，建议10-25）
        'interpolator': 'sitkBSpline',             # 插值方式
        'resampledPixelSpacing': [1.0, 1.0, 1.0],  # 重采样体素大小 (mm)
        'normalize': False,                         # 已外部预处理，不使用内置归一化
    },
    'radiomics_settings_mri': {
        'binWidth': 0.2,                           # MRI Z-score后灰度量化宽度（值域小，建议0.1-0.5）
        'interpolator': 'sitkBSpline',             # 插值方式
        'resampledPixelSpacing': [1.0, 1.0, 1.0],  # 重采样体素大小 (mm)
        'normalize': False,                         # 已外部预处理，不使用内置归一化
    },
    'radiomics_settings_dose': {
        'binWidth': 0.5,                           # 剂量灰度量化宽度（Gy值范围较小）
        'interpolator': 'sitkBSpline',             # 插值方式
        'resampledPixelSpacing': [1.0, 1.0, 1.0],  # 重采样体素大小 (mm)
        'normalize': False,                         # 剂量图像不需要归一化
    },

    # 图像类型（滤波器）配置
    # Original: 原始图像
    # Wavelet: 小波滤波（会生成多个子带图像：HHH, HHL, HLH, HLL, LHH, LHL, LLH, LLL）
    # LoG: 高斯拉普拉斯滤波（需指定sigma值列表）
    'image_types': {
        'Original': {},
        # 'Wavelet': {},
        # 'LoG': {'sigma': [1.0, 2.0, 3.0, 4.0, 5.0]},  # sigma值列表(mm)
    },

    # 特征类别配置
    'feature_classes': {
        'firstorder': [],    # 空列表 = 提取该类全部特征
        'shape': [],
        'glcm': [],
        'glrlm': [],
        'glszm': [],
        'ngtdm': [],
    },

    # ---- 输出配置 ----
    'output_dir': r'D:\projects\CervixRT_Sensitivity_Prognosis\results\extracted_features',
    'output_filename': 'ct_features.csv',   # 输出文件名

    # ---- 测试配置 ----
    'max_patients': None,  # 最大处理患者数（设为None或0则处理全部）
}

# ============================================================================
# 以下为程序逻辑，一般不需要修改
# ============================================================================

logging.basicConfig(
    level='INFO',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_radiomics_settings(params):
    """
    根据模态类型获取对应的 radiomics 参数

    参数:
        params: 参数字典

    返回:
        radiomics 设置字典
    """
    modality = params['modality'].upper()
    if modality == 'CT':
        return params['radiomics_settings_ct'].copy()
    elif modality == 'DOSE':
        return params['radiomics_settings_dose'].copy()
    else:
        return params['radiomics_settings_mri'].copy()


def build_extractor(params):
    """
    根据参数构建pyradiomics特征提取器

    参数:
        params: 参数字典

    返回:
        RadiomicsFeatureExtractor 实例
    """
    settings = get_radiomics_settings(params)
    image_types = params['image_types']
    feature_classes = params['feature_classes']

    # 使用settings初始化提取器
    extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update(settings)

    # 显式禁用所有默认的图像类型，再逐个启用指定的类型
    extractor.disableAllImageTypes()
    for image_type, custom_args in image_types.items():
        extractor.enableImageTypeByName(image_type, customArgs=custom_args)

    # 显式禁用所有默认的特征类别，再逐个启用指定的类别
    extractor.disableAllFeatures()
    for feature_class in feature_classes:
        extractor.enableFeatureClassByName(feature_class)

    logger.info(f"特征提取器配置:")
    logger.info(f"  模态: {params['modality']}")
    logger.info(f"  binWidth: {settings['binWidth']}")
    logger.info(f"  重采样: {settings['resampledPixelSpacing']}")
    logger.info(f"  图像类型: {list(image_types.keys())}")
    logger.info(f"  特征类别: {list(feature_classes.keys())}")

    # 验证已启用的图像类型
    enabled_types = extractor.enabledImagetypes
    logger.info(f"  已启用的图像类型: {enabled_types}")

    return extractor


def load_all_roi_masks(patient_dir, params):
    """
    加载所有ROI掩码文件

    参数:
        patient_dir: 患者数据目录
        params: 参数字典

    返回:
        dict[str, sitk.Image]: ROI名称 -> SimpleITK掩码对象
    """
    masks = {}
    for roi_name, roi_filename in params['roi_files'].items():
        roi_path = os.path.join(patient_dir, roi_filename)
        if not os.path.exists(roi_path):
            continue
        try:
            masks[roi_name] = sitk.ReadImage(roi_path)
        except Exception as e:
            logger.warning(f"  读取ROI {roi_name} 失败: {e}")
    return masks


def compute_combined_mask(masks):
    """
    将所有ROI掩码合并为一个联合掩码（用于预处理时的百分位数截断和Z-score归一化）

    参数:
        masks: dict[str, sitk.Image] ROI名称->掩码

    返回:
        sitk.Image: 合并后的二值掩码
    """
    if not masks:
        return None

    mask_list = list(masks.values())
    # 以第一个掩码为基础
    combined_array = (sitk.GetArrayFromImage(mask_list[0]) > 0).astype(np.uint8)

    for m in mask_list[1:]:
        m_array = (sitk.GetArrayFromImage(m) > 0).astype(np.uint8)
        combined_array = ((combined_array + m_array) > 0).astype(np.uint8)

    combined_mask = sitk.GetImageFromArray(combined_array)
    combined_mask.CopyInformation(mask_list[0])
    return combined_mask


def preprocess_image(image, combined_mask, params):
    """
    根据模态类型进行影像预处理
    使用所有ROI的合并区域进行百分位数截断和Z-score归一化

    参数:
        image: SimpleITK 图像对象
        combined_mask: SimpleITK 合并ROI掩码对象（所有ROI的并集）
        params: 参数字典

    返回:
        预处理后的SimpleITK图像对象
    """
    modality = params['modality'].upper()

    if modality == 'DOSE':
        # 剂量图像不需要预处理，直接返回原始图像
        logger.info("  Dose模态: 跳过预处理，使用原始剂量图像")
        return image
    elif modality == 'CT':
        ct_config = params['ct_preprocessing']
        return preprocess_ct(
            image, combined_mask,
            lower_bound=ct_config['lower_bound'],
            upper_bound=ct_config['upper_bound']
        )
    else:
        # MRI (T1, T2, DWI, ADC) 统一预处理方式
        mri_config = params['mri_preprocessing']

        return preprocess_mri(
            image, combined_mask,
            lower_percentile=mri_config['percentile_low'],
            upper_percentile=mri_config['percentile_high'],
            target_mean=mri_config['target_mean'],
            target_std=mri_config['target_std'],
        )


def extract_single_patient(patient_id, patient_dir, extractor, params):
    """
    为单个患者提取所有ROI的特征
    预处理使用所有ROI的合并区域，特征分别从每个ROI中提取

    参数:
        patient_id: 患者ID
        patient_dir: 患者数据目录
        extractor: 特征提取器
        params: 参数字典

    返回:
        list[dict]: 特征记录列表（长格式），失败则返回None
    """
    image_path = os.path.join(patient_dir, params['image_file'])
    if not os.path.exists(image_path):
        logger.warning(f"  患者 {patient_id}: 图像文件不存在 {image_path}")
        return None

    # 读取图像
    try:
        image = sitk.ReadImage(image_path)
    except Exception as e:
        logger.error(f"  患者 {patient_id}: 读取图像失败 - {e}")
        return None

    # 加载所有ROI掩码
    masks = load_all_roi_masks(patient_dir, params)
    if not masks:
        logger.warning(f"  患者 {patient_id}: 没有可用的ROI掩码文件")
        return None

    # 计算合并ROI掩码（用于预处理）
    combined_mask = compute_combined_mask(masks)

    # 使用合并ROI区域进行预处理（百分位数截断 + Z-score归一化）
    try:
        processed_image = preprocess_image(image, combined_mask, params)
    except Exception as e:
        logger.error(f"  患者 {patient_id}: 预处理失败 - {e}")
        processed_image = image  # fallback: 使用原始图像

    # 分别提取每个ROI的特征
    results = []
    for roi_name, mask in masks.items():
        try:
            feature_result = extractor.execute(processed_image, mask, label=1)
        except Exception as e:
            logger.error(f"  患者 {patient_id}: 提取 {roi_name} 特征失败 - {e}")
            continue

        for key, value in feature_result.items():
            if key.startswith('diagnostics_'):
                continue
            results.append({
                'ID': patient_id,
                'ROI_Type': roi_name,
                'Feature_Name': key,
                'Feature_Value': float(value) if not isinstance(value, str) else np.nan,
            })

    return results if results else None


def run_extraction(params):
    """
    执行批量特征提取的主流程（只负责提取特征，不处理标签）

    参数:
        params: 参数字典

    返回:
        pd.DataFrame: 宽格式特征表
    """
    data_dir = params['data_dir']
    output_dir = params['output_dir']
    output_filename = params['output_filename']

    os.makedirs(output_dir, exist_ok=True)

    # 扫描数据目录
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return None

    patient_dirs = []
    for name in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, name)
        if os.path.isdir(full_path):
            patient_dirs.append((name, full_path))

    logger.info(f"数据目录中共有 {len(patient_dirs)} 个患者文件夹")

    # 限制处理患者数（用于测试）
    max_patients = params.get('max_patients', None)
    if max_patients and max_patients > 0:
        patient_dirs = patient_dirs[:max_patients]
        logger.info(f"测试模式: 仅处理前 {max_patients} 个患者")

    if not patient_dirs:
        logger.error("没有可处理的患者！")
        return None

    # 构建特征提取器
    extractor = build_extractor(params)

    # 批量提取
    all_records = []
    success_count = 0
    fail_count = 0

    for idx, (patient_id, patient_dir) in enumerate(patient_dirs):
        logger.info(f"[{idx+1}/{len(patient_dirs)}] 处理患者: {patient_id}")

        records = extract_single_patient(patient_id, patient_dir, extractor, params)
        if records:
            all_records.extend(records)
            success_count += 1
        else:
            fail_count += 1

    logger.info(f"特征提取完成: 成功 {success_count}, 失败 {fail_count}")

    if not all_records:
        logger.error("没有提取到任何特征！")
        return None

    # 转换为DataFrame（长格式）
    long_df = pd.DataFrame(all_records)

    # 确保ID列为字符串类型，保留前导零
    long_df['ID'] = long_df['ID'].astype(str)

    # 确定特征名称前缀
    prefix = params.get('modality_prefix', '')
    if not prefix:
        prefix = params['modality'].upper() + '_'

    # 构建列名: prefix + ROI_Type + Feature_Name
    long_df['Feature_Column'] = prefix + long_df['ROI_Type'] + '_' + long_df['Feature_Name']

    # 透视为宽格式
    wide_df = long_df.pivot_table(
        index='ID',
        columns='Feature_Column',
        values='Feature_Value',
        aggfunc='first'
    ).reset_index()

    # 清理：移除全NaN列和含NaN/Inf的列
    feature_cols = [c for c in wide_df.columns if c != 'ID']
    cols_to_drop = []
    for col in feature_cols:
        vals = wide_df[col]
        if vals.isna().all():
            cols_to_drop.append(col)
        elif vals.isna().any():
            cols_to_drop.append(col)
        elif np.isinf(vals.astype(float)).any():
            cols_to_drop.append(col)

    if cols_to_drop:
        logger.warning(f"移除 {len(cols_to_drop)} 个含NaN/Inf的特征列")
        wide_df = wide_df.drop(columns=cols_to_drop)

    # 重新排列列：ID, 特征...
    feature_cols = [c for c in wide_df.columns if c != 'ID']
    wide_df = wide_df[['ID'] + sorted(feature_cols)]

    logger.info(f"最终特征表: {len(wide_df)} 患者, {len(feature_cols)} 特征")

    # 保存结果
    output_path = os.path.join(output_dir, output_filename)
    wide_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"特征已保存到: {output_path}")

    # 同时保存长格式（便于调试）
    long_output_path = os.path.join(output_dir, 'features_long_format.csv')
    long_df.to_csv(long_output_path, index=False, encoding='utf-8-sig')

    return wide_df


if __name__ == '__main__':
    logger.info("=" * 80)
    logger.info("影像组学特征提取程序")
    logger.info("=" * 80)
    logger.info(f"模态: {PARAMS['modality']}")
    logger.info(f"数据目录: {PARAMS['data_dir']}")
    logger.info(f"图像文件: {PARAMS['image_file']}")
    logger.info(f"ROI文件: {PARAMS['roi_files']}")
    logger.info(f"图像类型(滤波器): {list(PARAMS['image_types'].keys())}")
    logger.info(f"Radiomics参数: {get_radiomics_settings(PARAMS)}")
    logger.info(f"输出目录: {PARAMS['output_dir']}")
    logger.info("=" * 80)

    result = run_extraction(PARAMS)

    if result is not None:
        logger.info("特征提取成功完成！")
    else:
        logger.error("特征提取失败！")
