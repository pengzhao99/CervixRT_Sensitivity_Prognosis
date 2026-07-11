# ==========================================
# 第一步：环境准备与数据读取
# ==========================================
# 如果未安装 rms 包，请先运行: install.packages("rms")
library(rms)

# 创建输出目录
plot_dir <- file.path(dirname(r"(D:\projects\CervixRT_Sensitivity_Prognosis\R\combined_model.R)"), "plot")
dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

# 读取数据 (假设文件名为 data.csv，请根据实际情况修改路径)
df <- read.csv(r"(D:\projects\CervixRT_Sensitivity_Prognosis\results\extracted_features\clinic_and_radscore-rf.csv)", header = TRUE, stringsAsFactors = FALSE)

# 查看前几行，确认列名是否正确
head(df)

# ==========================================
# 第二步：数据预处理 (关键步骤)
# ==========================================

# 1. 修正列名 (防止因为空格或特殊字符导致报错)
# 如果你的CSV里列名有空格，R会自动把 "Rad-scoreSet" 读成 "Rad.scoreSet"
names(df) <- make.names(names(df))

# 2. 划分数据集
# 根据你的描述，利用 Rad-scoreSet 这一列的值来区分 Train 和 Test
train_data <- subset(df, Set == "Train")
test_data  <- subset(df, Set != "Train") # 或者具体指定 "Test" / "Val"

cat("训练集样本量:", nrow(train_data), "\n")
cat("测试集样本量:", nrow(test_data), "\n")

# 3. 变量类型转换 (非常重要！)
# R 会把 0/1/2 当作数字，必须转为 Factor 才能作为分类变量处理
train_data$Label        <- as.factor(train_data$Label)       # 结局变量

# 1. Pathology: 0 -> Non-CSCC, 1 -> CSCC
train_data$Pathology.types <- factor(train_data$Pathology, 
                                      levels = c(0, 1), 
                                      labels = c("Non-CSCC", "CSCC"))

# 2. Chemotherapy: 0 -> No, 1 -> Yes
train_data$Concurrent.chemotherapy <- factor(train_data$Chemotherapy, 
                                              levels = c(0, 1), 
                                              labels = c("No", "Yes"))

# 3. FIGO stage: 1->I, 2->II, 3->III, 4->IV (注意：列名可能带点号)
train_data$Stage <- factor(train_data$Stage, 
                                levels = c(1, 2, 3, 4), 
                                labels = c("I", "II", "III", "IV"))

# 4. Rad-score 恢复原始尺度（CSV 中已缩小100倍）
train_data$Rad.score <- train_data$Rad.score * 100

# ---- 设置变量标签（诺莫图显示用）----
# 注意：必须对 data.frame 中的列直接设置 attr，不能用 label(df$col) <-
attr(train_data[['Rad.score']], 'label') <- 'Radiomics score'
attr(train_data[['Stage']], 'label') <- 'FIGO stage'
attr(train_data[['Pathology.types']], 'label') <- 'Pathology types'
attr(train_data[['Concurrent.chemotherapy']], 'label') <- 'Concurrent chemotherapy'

# ==========================================
# 第三步：构建模型与绘制诺莫图
# ==========================================

# 设置 rms 包的环境参数 (必须运行，用于后续画图)
ddist <- datadist(train_data)
options(datadist = "ddist")

# 拟合多因素 Logistic 回归模型
# x=TRUE, y=TRUE 是为了保存数据以便后续验证
fit <- lrm(Label ~ Rad.score + Stage + Pathology.types + Concurrent.chemotherapy,
           data = train_data, x = TRUE, y = TRUE)

# 打印模型摘要，检查 P 值
print(fit)

# 绘制诺莫图
nom <- nomogram(fit,
         fun = plogis,          # 将线性预测值转换为概率
         fun.at = seq(0.1, 0.9, by = 0.1),  # 概率刻度间隔0.1
         lp = FALSE,            # 不显示 Linear Predictor 轴
         funlabel = "Risk of PR", # 概率轴的标签
         maxscale = 100)        # 总分轴的最大值

# ---- 保存为 PNG (300 DPI, 期刊发表质量) ----
png(file = file.path(plot_dir, "nomogram_combined_model.png"),
    width = 10, height = 6, units = "in", res = 300)
# par(mar = c(1.5, 1.5, 1.5, 1.5))
plot(nom, cex.axis = 0.75)
dev.off()

# ---- 保存为 PDF (矢量图, 期刊发表质量) ----
pdf(file = file.path(plot_dir, "nomogram_combined_model.pdf"),
    width = 10, height = 6)
# par(mar = c(1.5, 1.5, 1.5, 1.5))
plot(nom, cex.axis = 0.75)
dev.off()

cat("诺莫图已保存至:", plot_dir, "\n")