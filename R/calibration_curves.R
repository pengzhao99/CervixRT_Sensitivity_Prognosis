# ==========================================
# 三个模型校准曲线 (rms::val.prob + loess 平滑)
# 训练集 + 测试集各一张图
# ==========================================
library(rms)

# ---- 输出目录 ----
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\calibration_curves.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 读取三个模型的预测结果 ----
clinic    <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic\predictions_Logistic_Regression.csv)")
radiomics <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_original\predictions_Random_Forest.csv)")
combined  <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic_ct\predictions_Logistic_Regression.csv)")

# ---- 模型名称与颜色 ----
model_names  <- c("Clinical model", "Radiomics model", "Radiomics-clinical model")
model_colors <- c("#4DAF4A", "#E41A1C", "#377EB8")  # 绿、红、蓝

# ---- 单个模型校准曲线绘制函数 ----
plot_cal_curve <- function(data, label_col, prob_col, col) {
  p <- data[[prob_col]]
  y <- data[[label_col]]

  # 用 val.prob 评估校准统计量（pl=FALSE 不绘图）
  val.prob(p, y, pl = FALSE, logistic.cal = TRUE, smooth = FALSE,
           lim = c(0, 1))

  # 用 loess 拟合平滑校准曲线
  lo <- loess(y ~ p, span = 0.75)
  x_seq <- seq(0, 1, length.out = 200)
  y_pred <- predict(lo, newdata = data.frame(p = x_seq))
  # 限制在 [0,1] 范围
  y_pred <- pmin(pmax(y_pred, 0), 1)

  lines(x_seq, y_pred, col = col, lwd = 2.5)
}

# ---- 绘图主函数 ----
plot_calibration <- function(subset_name, title_text) {
  d1 <- clinic[clinic$Set == subset_name, ]
  d2 <- radiomics[radiomics$Set == subset_name, ]
  d3 <- combined[combined$Set == subset_name, ]

  # 初始化画布
  plot(NULL, xlim = c(0, 1), ylim = c(0, 1),
       xlab = "Predicted Probability",
       ylab = "Observed Proportion",
       main = title_text,
       cex.main = 1.3, cex.lab = 1.3, cex.axis = 1.1,
       asp = 1)

  # 完美校准参考线
  abline(0, 1, lty = 2, col = "gray40", lwd = 1.5)

  # 绘制三个模型的校准曲线
  plot_cal_curve(d1, "Label_Encoded", "Predicted_Probability", model_colors[1])
  plot_cal_curve(d2, "Label_Encoded", "Predicted_Probability", model_colors[2])
  plot_cal_curve(d3, "Label_Encoded", "Predicted_Probability", model_colors[3])

  # 图例
  legend("bottomright",
         legend = c("Ideal", model_names),
         col    = c("gray40", model_colors),
         lty    = c(2, 1, 1, 1),
         lwd    = c(1.5, 2.5, 2.5, 2.5),
         cex    = 1.0,
         bty    = "n")
}

# ---- 保存训练集 ----
png(file = file.path(plot_dir, "calibration_train.png"),
    width = 7, height = 7, units = "in", res = 300)
plot_calibration("Train", "Calibration Curves - Training Set")
dev.off()

pdf(file = file.path(plot_dir, "calibration_train.pdf"),
    width = 7, height = 7)
plot_calibration("Train", "Calibration Curves - Training Set")
dev.off()

# ---- 保存测试集 ----
png(file = file.path(plot_dir, "calibration_test.png"),
    width = 7, height = 7, units = "in", res = 300)
plot_calibration("Test", "Calibration Curves - Test Set")
dev.off()

pdf(file = file.path(plot_dir, "calibration_test.pdf"),
    width = 7, height = 7)
plot_calibration("Test", "Calibration Curves - Test Set")
dev.off()

cat("校准曲线已保存至:", plot_dir, "\n")
