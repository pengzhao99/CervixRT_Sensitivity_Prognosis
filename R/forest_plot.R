# ==========================================
# 森林图 (forestploter)
# 上部: 单变量 Logistic 回归 (所有变量)
# 下部: 多变量 Logistic 回归 (仅显著变量)
# ==========================================
suppressMessages(library(forestploter))
suppressMessages(library(grid))

# ---- 输出目录 ----
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\forest_plot.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# ---- 读取数据 ----
df <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\extracted_features\clinic_and_radscore-rf.csv)")
# Rad.score 已缩小100倍，恢复原始尺度
df$Rad.score <- df$Rad.score * 100
train <- df[df$Set == "Train", ]
cat("Train 样本量:", nrow(train), "\n")

# ---- 预测变量 ----
predictors <- c("Age", "Pathology", "HPV", "Stage",
                "Chemotherapy", "BinaryOfSyn", "NumberOfBrachy",
                "NACT", "ACT", "Rad.score")

var_labels <- c(
  "Age"              = "Age",
  "Pathology"        = "Pathology type (CSCC vs. Non-CSCC)",
  "HPV"              = "HPV",
  "Stage"            = "FIGO Stage",
  "Chemotherapy"     = "Concurrent chemotherapy",
  "BinaryOfSyn"      = "Concurrent cycles (≥4 vs. <4)",
  "NumberOfBrachy"   = "Number of Brachytherapy",
  "NACT"             = "Neoadjuvant chemotherapy",
  "ACT"              = "Adjuvant chemotherapy",
  "Rad.score"        = "Radiomics score "
)

# ---- 单变量 Logistic 回归 ----
run_uni <- function(var, data) {
  fml <- as.formula(paste("Label ~", var))
  fit <- glm(fml, data = data, family = binomial)
  s   <- summary(fit)
  co  <- s$coefficients[2, ]
  list(
    Variable = var, Label = var_labels[var],
    OR = exp(co["Estimate"]),
    Lower = exp(co["Estimate"] - 1.96 * co["Std. Error"]),
    Upper = exp(co["Estimate"] + 1.96 * co["Std. Error"]),
    P = co["Pr(>|z|)"]
  )
}
uni_list <- lapply(predictors, run_uni, data = train)
uni <- do.call(rbind.data.frame, uni_list)

cat("\n=== 单变量分析 ===\n")
for (i in seq_len(nrow(uni))) {
  sig <- ifelse(uni$P[i] < 0.05, "*", "")
  cat(sprintf("%-25s OR=%.3f (%.3f\u2013%.3f) P=%.4f %s\n",
              uni$Label[i], uni$OR[i], uni$Lower[i], uni$Upper[i], uni$P[i], sig))
}

# ---- 筛选显著变量 ----
sig_idx <- which(uni$P < 0.05)
sig_vars <- uni$Variable[sig_idx]
cat("\n显著变量 (p<0.05):", paste(sig_vars, collapse = ", "), "\n")

# ---- 多变量 Logistic 回归 ----
multi_fml <- as.formula(paste("Label ~", paste(sig_vars, collapse = " + ")))
multi_fit <- glm(multi_fml, data = train, family = binomial)
multi_s   <- summary(multi_fit)

multi <- data.frame(
  Variable = rownames(multi_s$coefficients)[-1],
  OR       = exp(multi_s$coefficients[-1, "Estimate"]),
  Lower    = exp(multi_s$coefficients[-1, "Estimate"] - 1.96 * multi_s$coefficients[-1, "Std. Error"]),
  Upper    = exp(multi_s$coefficients[-1, "Estimate"] + 1.96 * multi_s$coefficients[-1, "Std. Error"]),
  P        = multi_s$coefficients[-1, "Pr(>|z|)"]
)
multi$Label <- var_labels[multi$Variable]

cat("\n=== 多变量分析 ===\n")
for (i in seq_len(nrow(multi))) {
  cat(sprintf("%-25s OR=%.3f (%.3f\u2013%.3f) P=%.4f\n",
              multi$Label[i], multi$OR[i], multi$Lower[i], multi$Upper[i], multi$P[i]))
}

# ==========================================
# 构建 forestploter 数据框
# 参考格式: 变量名 | OR(95%CI) | CI图 | P值
# ==========================================

# 格式化函数
fmt_or <- function(or, lo, hi) sprintf("%.3f (%.3f\u2013%.3f)", or, lo, hi)
fmt_p  <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

# CI 显示列宽度（空格数控制宽度）
ci_blank <- paste(rep(" ", 25), collapse = "")

# ---- 上部：单变量 ----
n_uni <- nrow(uni)

# 单变量分区标题
dt_uni_header <- data.frame(
  Subgroup      = "Univariate logistic regression",
  `OR (95% CI)` = "",
  ` `           = "",
  `P value`     = "",
  check.names   = FALSE,
  stringsAsFactors = FALSE
)

dt_uni <- data.frame(
  Subgroup      = uni$Label,
  `OR (95% CI)` = fmt_or(uni$OR, uni$Lower, uni$Upper),
  ` `           = rep(ci_blank, n_uni),
  `P value`     = fmt_p(uni$P),
  check.names   = FALSE,
  stringsAsFactors = FALSE
)

# ---- 分隔行 ----
dt_sep <- data.frame(
  Subgroup      = "",
  `OR (95% CI)` = "",
  ` `           = "",
  `P value`     = "",
  check.names   = FALSE,
  stringsAsFactors = FALSE
)

# ---- 多变量分区标题 ----
dt_multi_header <- data.frame(
  Subgroup      = "Multivariate logistic regression",
  `OR (95% CI)` = "",
  ` `           = "",
  `P value`     = "",
  check.names   = FALSE,
  stringsAsFactors = FALSE
)

# ---- 下部：多变量 ----
n_multi <- nrow(multi)
dt_multi <- data.frame(
  Subgroup      = multi$Label,
  `OR (95% CI)` = fmt_or(multi$OR, multi$Lower, multi$Upper),
  ` `           = rep(ci_blank, n_multi),
  `P value`     = fmt_p(multi$P),
  check.names   = FALSE,
  stringsAsFactors = FALSE
)

# 合并: Univariate标题 + 单变量 + 空行 + Multivariate标题 + 多变量
dt <- rbind(dt_uni_header, dt_uni, dt_sep, dt_multi_header, dt_multi)

# est/lower/upper（标题行和分隔行用 NA）
est_all   <- c(NA, uni$OR,    NA, NA, multi$OR)
lower_all <- c(NA, uni$Lower, NA, NA, multi$Lower)
upper_all <- c(NA, uni$Upper, NA, NA, multi$Upper)

# ---- 主题设置（参考图风格） ----
tm <- forest_theme(
  base_size      = 10,
  # 蓝色实线参考线 (OR=1)
  refline_gp     = gpar(col = "#2166AC", lty = 1, lwd = 1.5),
  # 红色方块点估计
  ci_pch         = 15,
  ci_col         = "#D62728",
  ci_lwd         = 1.8,
  # 置信区间横线
  ci_line_col    = "#333333",
  footnote_gp    = gpar(col = "gray40", fontface = "italic", fontsize = 9),
  xaxis_gp       = gpar(fontsize = 9)
)

# ---- 绘制森林图 ----
p <- forest(
  dt[, c("Subgroup", "OR (95% CI)", " ", "P value")],
  est        = est_all,
  lower      = lower_all,
  upper      = upper_all,
  ci_column  = 3,
  ref_line   = 1,
  x_trans    = "log",
  xlim       = c(0.05, 15),
  ticks_at   = c(0.1, 0.5, 1, 2, 5, 10),
  ticks_digits = 1L,
  xlab       = "Odds Ratio",
  theme      = tm
)

# ---- 保存 ----
png(file.path(plot_dir, "forest_plot.png"), width = 7, height = 5, units = "in", res = 300)
plot(p)
dev.off()

pdf(file.path(plot_dir, "forest_plot.pdf"), width = 7, height = 5)
plot(p)
dev.off()

cat("\n森林图已保存至:", plot_dir, "\n")
