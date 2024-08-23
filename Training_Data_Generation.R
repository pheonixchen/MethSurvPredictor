library(readr)

mathylation <- read.csv("pca_principal_components.csv")

data <- read_tsv("TCGA-LGG.survival.tsv", na = c("NaN", "null", ""))
data = data[data$OS == 1, ]

colnames(mathylation)[1] = 'sample'


data = merge(data, mathylation, by = 'sample')
data = data[, -c(1, 2, 3)]
data[is.na(data)] <- 0

write.csv(data, 'data.csv')
