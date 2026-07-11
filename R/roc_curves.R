# ==========================================
# 三个模型 ROC 曲线对比 (训练集 + 测试集)
# ==========================================
# 依赖包: pROC (如未安装: install.packages("pROC"))
library(pROC)

# ---- 输出目录 ----
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\roc_curves.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 读取三个模型的预测结果 ----
clinic    <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic\predictions_Logistic_Regression.csv)")
radiomics <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_original\predictions_Random_Forest.csv)")
combined  <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic_ct\predictions_Logistic_Regression.csv)")

# ---- 定义模型名称与颜色 ----
model_names  <- c("Clinical model", "Radiomics model", "Radiomics-clinical model")
model_colors <- c("#4DAF4A", "#E41A1C", "#377EB8")  # 绿、红、蓝

# ---- 绘图函数 ----
plot_roc <- function(subset_name, title_text) {
  # 按 Set 筛选数据
  d1 <- clinic[clinic$Set == subset_name, ]
  d2 <- radiomics[radiomics$Set == subset_name, ]
  d3 <- combined[combined$Set == subset_name, ]

  # 构建 ROC 对象
  roc1 <- roc(d1$Label_Encoded, d1$Predicted_Probability, quiet = TRUE)
  roc2 <- roc(d2$Label_Encoded, d2$Predicted_Probability, quiet = TRUE)
  roc3 <- roc(d3$Label_Encoded, d3$Predicted_Probability, quiet = TRUE)

  roc_list <- list(roc1, roc2, roc3)
  auc_vals <- sapply(roc_list, auc)
  ci_vals  <- sapply(roc_list, function(r) ci(r, conf.level = 0.95))

  # 开始绘图
  plot(roc1, col = model_colors[1], lwd = 2,
       main = title_text,
       xlab = "1 - Specificity", ylab = "Sensitivity",
       legacy.axes = TRUE, asp = 1,
       cex.main = 1.3, cex.lab = 1.3, cex.axis = 1.1)
  lines(roc2, col = model_colors[2], lwd = 2)
  lines(roc3, col = model_colors[3], lwd = 2)

  # 构建图例文字: Model Name (AUC = x.xx, 95% CI: x.xx-x.xx)
  legend_labels <- sapply(seq_along(model_names), function(i) {
    sprintf("%s (AUC = %.3f, 95%% CI: %.3f-%.3f)",
            model_names[i], auc_vals[i], ci_vals[1, i], ci_vals[3, i])
  })

  legend("bottomright",
         legend  = legend_labels,
         col     = model_colors,
         lwd     = 2,
         cex     = 1.0,
         bty     = "n",
         inset   = c(0.02, 0.02))
}

# ---- 保存为 PNG ----
png(file = file.path(plot_dir, "roc_train.png"),
    width = 7, height = 7, units = "in", res = 300)
plot_roc("Train", "ROC Curves - Training Set")
dev.off()

png(file = file.path(plot_dir, "roc_test.png"),
    width = 7, height = 7, units = "in", res = 300)
plot_roc("Test", "ROC Curves - Test Set")
dev.off()

# ---- 保存为 PDF ----
pdf(file = file.path(plot_dir, "roc_train.pdf"),
    width = 7, height = 7)
plot_roc("Train", "ROC Curves - Training Set")
dev.off()

pdf(file = file.path(plot_dir, "roc_test.pdf"),
    width = 7, height = 7)
plot_roc("Test", "ROC Curves - Test Set")
dev.off()

cat("ROC 曲线已保存至:", plot_dir, "\n")
