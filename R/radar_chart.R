# ==========================================
# 三个模型性能雷达图 (fmsb::radarchart)
# 指标: ACC, AUC, F1, Specificity, Sensitivity
# 分类阈值: Youden 最优阈值
# ==========================================
library(fmsb)
suppressMessages(library(pROC))

# ---- 输出目录 ----
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\radar_chart.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 读取三个模型的预测结果 ----
clinic    <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic\predictions_Logistic_Regression.csv)")
radiomics <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_original\predictions_Random_Forest.csv)")
combined  <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic_ct\predictions_Logistic_Regression.csv)")

# ---- 计算五个指标（使用 Youden 最优阈值） ----
compute_metrics <- function(y_true, y_prob) {
  roc_obj <- suppressMessages(roc(y_true, y_prob, quiet = TRUE))
  auc_val <- as.numeric(auc(roc_obj))

  # Youden 最优阈值
  coords_all <- coords(roc_obj, "best", best.method = "youden",
                       ret = c("threshold"))
  thr <- as.numeric(coords_all[["threshold"]])[1]
  cat("  Youden threshold:", round(thr, 4), "\n")

  y_pred <- ifelse(y_prob >= thr, 1, 0)

  tp <- sum(y_pred == 1 & y_true == 1)
  tn <- sum(y_pred == 0 & y_true == 0)
  fp <- sum(y_pred == 1 & y_true == 0)
  fn <- sum(y_pred == 0 & y_true == 1)

  acc         <- (tp + tn) / (tp + tn + fp + fn)
  sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), 0)
  specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), 0)
  precision   <- ifelse((tp + fp) > 0, tp / (tp + fp), 0)
  f1          <- ifelse((precision + sensitivity) > 0,
                        2 * precision * sensitivity / (precision + sensitivity), 0)

  c(ACC = acc, AUC = auc_val, F1_score = f1,
    Specificity = specificity, Sensitivity = sensitivity)
}

# ---- 构建雷达图数据框（fmsb 格式） ----
# fmsb 要求: 第1行=max, 第2行=min, 后续行=数据
build_radar_df <- function(subset_name) {
  d1 <- clinic[clinic$Set == subset_name, ];    d1 <- d1[order(d1$ID), ]
  d2 <- radiomics[radiomics$Set == subset_name, ]; d2 <- d2[order(d2$ID), ]
  d3 <- combined[combined$Set == subset_name, ];   d3 <- d3[order(d3$ID), ]

  cat("  Clinical model:");          m1 <- compute_metrics(d1$Label_Encoded, d1$Predicted_Probability)
  cat("  Radiomics model:");          m2 <- compute_metrics(d2$Label_Encoded, d2$Predicted_Probability)
  cat("  Radiomics-clinical model:");  m3 <- compute_metrics(d3$Label_Encoded, d3$Predicted_Probability)

  # 数据行
  df <- as.data.frame(rbind(m1, m2, m3))
  rownames(df) <- c("Clinical model", "Radiomics model", "Radiomics-clinical model")
  # 插入 max/min 行（前2行）
  max_row <- rep(1.0, ncol(df))
  min_row <- rep(0.0, ncol(df))
  df <- rbind(max_row, min_row, df)
  rownames(df)[1:2] <- c("max", "min")
  df
}

# ---- 颜色配置（与其他图统一） ----
model_colors <- c("#4DAF4A", "#E41A1C", "#377EB8")  # 绿、红、蓝

# ---- 绘图函数 ----
plot_radar <- function(subset_name, filename) {
  df <- build_radar_df(subset_name)

  # ---- PNG ----
  png(paste0(plot_dir, "/", filename, ".png"), width = 1200, height = 1200, res = 200)

  # 1. 居中：左右边距相同
  par(mar = c(4, 2, 3, 2))  # bottom, left, top, right

  # 使用自带百分比标签 (axistype=1)
  radarchart(
    df,
    axistype    = 1,
    caxislabels = sprintf("%.1f", seq(0, 1, by = 0.2)),
    seg         = 5,
    pcol        = model_colors,
    plwd        = 2,
    plty        = 1,
    pfcol       = adjustcolor(model_colors, alpha.f = 0.1),
    cglty       = 2,
    cglwd       = 1,
    cglcol      = "gray60",
    axislabcol  = "black",
    title       = paste0("Model Performance - ", subset_name, " Set"),
    centerzero  = FALSE,
    calcex      = 1.0
  )


  # 2. legend 位于正下方，水平方向留更多边距
  legend(x     = "bottom",
         legend = rownames(df)[3:5],
         col    = model_colors,
         lwd    = 2,
         cex    = 0.9,
         bty    = "n",
         horiz  = FALSE,
         xpd    = TRUE,
         inset  = c(0, -0.05))
  dev.off()

  # ---- PDF ----
  pdf(paste0(plot_dir, "/", filename, ".pdf"), width = 6, height = 6)
  par(mar = c(4, 2, 3, 2))
  radarchart(
    df,
    axistype    = 1,
    caxislabels = sprintf("%.1f", seq(0, 1, by = 0.2)),
    seg         = 5,
    pcol        = model_colors,
    plwd        = 2,
    plty        = 1,
    pfcol       = adjustcolor(model_colors, alpha.f = 0.1),
    cglty       = 2,
    cglwd       = 1,
    cglcol      = "gray60",
    axislabcol  = "black",
    title       = paste0("Model Performance - ", subset_name, " Set"),
    centerzero  = FALSE,
    calcex      = 1.0
  )
  legend(x     = "bottom",
         legend = rownames(df)[3:5],
         col    = model_colors,
         lwd    = 2,
         cex    = 0.9,
         bty    = "n",
         horiz  = FALSE,
         xpd    = TRUE,
         inset  = c(0, -0.05))
  dev.off()

  # 打印指标数值
  cat("\n===", subset_name, "Set ===\n")
  print(round(df[3:5, ], 4))
}

# ---- 生成训练集和测试集图 ----
plot_radar("Train", "radar_train")
plot_radar("Test",  "radar_test")

cat("\n雷达图已保存至:", plot_dir, "\n")
