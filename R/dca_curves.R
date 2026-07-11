# ==========================================
# 决策曲线分析 (rmda::plot_decision_curve)
# 自动生成 Cost/Benefit Ratio 次级轴
# ==========================================
library(rmda)

# ---- 输出目录 ----
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\dca_curves.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 读取三个模型的预测结果 ----
clinic    <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic\predictions_Logistic_Regression.csv)")
radiomics <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_original\predictions_Random_Forest.csv)")
combined  <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\model_clinic_ct\predictions_Logistic_Regression.csv)")

# ---- 阈值范围 ----
thresholds <- seq(0.01, 0.99, by = 0.01)

# ---- 从预测概率手动计算 Net Benefit ----
compute_nb <- function(p, y, thresholds) {
  N   <- length(y)
  prev <- mean(y == 1)
  res  <- data.frame(thresholds = thresholds, NB = NA_real_, sNB = NA_real_)
  for (i in seq_along(thresholds)) {
    pos <- p >= thresholds[i]
    tp  <- sum(pos & y == 1)
    fp  <- sum(pos & y == 0)
    nb  <- tp / N - fp / N * (thresholds[i] / (1 - thresholds[i]))
    res$NB[i]  <- nb
    res$sNB[i] <- nb / prev
  }
  # Treat All
  nb_all  <- prev - (1 - prev) * (thresholds / (1 - thresholds))
  s_nb_all <- nb_all / prev
  # Treat None
  nb_none  <- rep(0, length(thresholds))
  s_nb_none <- rep(0, length(thresholds))
  # 占位 cost.benefit.ratio（rmda 内部重新计算显示值）
  cbr <- sapply(thresholds, function(pt) {
    r <- pt / (1 - pt)
    paste0(round(r * 100), ":100")
  })
  list(
    model_data = res,
    all_data   = data.frame(thresholds = thresholds, NB = nb_all,  sNB = s_nb_all),
    none_data  = data.frame(thresholds = thresholds, NB = nb_none, sNB = s_nb_none),
    cbr        = cbr
  )
}

# ---- 构建单个 decision_curve 对象 ----
build_dc <- function(nb_result, model_name) {
  md <- nb_result$model_data;  md$model <- model_name; md$cost.benefit.ratio <- nb_result$cbr
  ad <- nb_result$all_data;    ad$model <- "All";      ad$cost.benefit.ratio <- nb_result$cbr
  nd <- nb_result$none_data;   nd$model <- "None";     nd$cost.benefit.ratio <- nb_result$cbr
  dd <- rbind(md, ad, nd)
  dd$sNB <- pmax(pmin(dd$sNB, 1), 0)   # clamp to [0, 1]
  structure(list(derived.data = dd, confidence.intervals = NULL,
                 policy = "opt-in", call = NULL),
            class = "decision_curve")
}

# ---- 固定显示的 Cost/Benefit Ratio 标签 ----
cb_labels <- c("1:100", "1:4", "2:3", "3:2", "4:1", "100:1")

# ---- 绘图函数 ----
plot_dca_rmda <- function(subset_name, filename) {
  d1 <- clinic[clinic$Set == subset_name, ];    d1 <- d1[order(d1$ID), ]
  d2 <- radiomics[radiomics$Set == subset_name, ]; d2 <- d2[order(d2$ID), ]
  d3 <- combined[combined$Set == subset_name, ];   d3 <- d3[order(d3$ID), ]
  y  <- d1$Label_Encoded

  nb_clin <- compute_nb(d1$Predicted_Probability, y, thresholds)
  nb_rad  <- compute_nb(d2$Predicted_Probability, y, thresholds)
  nb_comb <- compute_nb(d3$Predicted_Probability, y, thresholds)

  dc_list <- list(
    build_dc(nb_clin, "m1"),
    build_dc(nb_rad,  "m2"),
    build_dc(nb_comb, "m3")
  )

  # ---- PNG ----
  png(paste0(plot_dir, "/", filename, ".png"), width = 1400, height = 1050, res = 200)
  plot_decision_curve(
    dc_list,
    curve.names       = c("Clinical model", "Radiomics model", "Radiomics-clinical model"),
    cost.benefit.axis = TRUE,
    cost.benefits     = cb_labels,
    standardize       = TRUE,
    col               = c("#4DAF4A", "#E41A1C", "#377EB8"),
    lwd               = 2,
    lty               = c(1, 1, 1),
    xlim              = c(0, 1),
    ylim              = c(0, 1),
    xlab              = "High Risk Threshold",
    ylab              = "Standardized Net Benefit",
    legend.position   = "topright"
  )
  dev.off()

  # ---- PDF ----
  pdf(paste0(plot_dir, "/", filename, ".pdf"), width = 7, height = 5.25)
  plot_decision_curve(
    dc_list,
    curve.names       = c("Clinical model", "Radiomics model", "Radiomics-clinical model"),
    cost.benefit.axis = TRUE,
    cost.benefits     = cb_labels,
    standardize       = TRUE,
    col               = c("#4DAF4A", "#E41A1C", "#377EB8"),
    lwd               = 2,
    lty               = c(1, 1, 1),
    xlim              = c(0, 1),
    ylim              = c(0, 1),
    xlab              = "High Risk Threshold",
    ylab              = "Standardized Net Benefit",
    legend.position   = "topright"
  )
  dev.off()
}

# ---- 生成训练集和测试集图 ----
plot_dca_rmda("Train", "dca_train")
plot_dca_rmda("Test",  "dca_test")

cat("决策曲线已保存至:", plot_dir, "\n")
