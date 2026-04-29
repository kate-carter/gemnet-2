#Normal Distribution of correlated matrices
#written on 4/6/2026 by Kate Carter because her goddamn matrices are negative

#install necessary packages
install.packages("readxl") # reads xcel files
install.packages("lsa") # package for cosine similarity
install.packages("tidyverse") # data organization my precious
install.packages("ggplot2") # datavis
install.packages("vtable") #summary plots
install.packages('viridis') #colors
install.packages('matrixNormal') #matrix stats


#load em
library(lsa)
library(tidyverse)
library(readxl)
library(ggplot2)
library(lubridate)
library(vtable)
library(viridis)
library(RColorBrewer)
library(matrixNormal)

#load test data:
s <- read.csv("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/spearman/s_corr/banana_256_spearman.csv", header = FALSE)
noneg <- read.csv("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/old_pipeline/take3/w_corr/banana_w_corr.csv", header = FALSE)
c <- read.csv("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/cosine/c_corr/banana_256_cosine.csv", header = FALSE)
mg <- read.csv("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/mahalanobis/mg_corr/banana_256_mahalanobis.csv", header = FALSE)
me <- read.csv("/Users/Kate/Documents/Binghamton-SPiNS/word-embedding-project/gemnet-2/gemini-2/mahalanobis/me_corr/banana_256_mahalanobis.csv", header = FALSE)

#spearman test
s_values <- as.numeric(as.matrix(s))
total_n <- length(s_values)

p_neg <- sum(s_values < 0, na.rm = TRUE) / total_n
p_1 <- sum(s_values >= 0 & s_values < 0.1, na.rm = TRUE) / total_n
p_2 <- sum(s_values >= 0.1 & s_values < 0.2, na.rm = TRUE) / total_n
p_3 <- sum(s_values >= 0.2 & s_values < 0.3, na.rm = TRUE) / total_n
p_4 <- sum(s_values >= 0.3 & s_values < 0.4, na.rm = TRUE) / total_n
p_5 <- sum(s_values >= 0.4 & s_values < 0.5, na.rm = TRUE) / total_n
p_6 <- sum(s_values >= 0.5 & s_values < 0.6, na.rm = TRUE) / total_n
p_7 <- sum(s_values >= 0.6 & s_values < 0.7, na.rm = TRUE) / total_n
p_8 <- sum(s_values >= 0.7 & s_values < 0.8, na.rm = TRUE) / total_n
p_9 <- sum(s_values >= 0.8 & s_values < 0.9, na.rm = TRUE) / total_n
pn1 <- sum(s_values < 0 & s_values > -0.1, na.rm = TRUE) / total_n
pn2 <- sum(s_values <= -0.1 & s_values > -0.2, na.rm = TRUE) / total_n
pn3 <- sum(s_values <= -0.2 & s_values > -0.3, na.rm = TRUE) / total_n
pn4 <- sum(s_values <= -0.3 & s_values > -0.4, na.rm = TRUE) / total_n
pn5 <- sum(s_values <= -0.4 & s_values > -0.5, na.rm = TRUE) / total_n
pn6 <- sum(s_values <= -0.5 & s_values > -0.6, na.rm = TRUE) / total_n
pn7 <- sum(s_values <= -0.6 & s_values > -0.7, na.rm = TRUE) / total_n
pn8 <- sum(s_values <= -0.7 & s_values > -0.8, na.rm = TRUE) / total_n
pn9 <- sum(s_values <= -0.8 & s_values > -0.9, na.rm = TRUE) / total_n
pn10 <- sum(s_values <= -0.9 & s_values >= -1, na.rm = TRUE) / total_n

#create dataframe
category <- c("pn10", "pn9", "pn8", "pn7", "pn6", "pn5", "pn4", "pn3", "pn2", "pn1", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9")
s_dist <- c(pn10, pn9, pn8, pn7, pn6, pn5, pn4, pn3, pn2, pn1, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)
range <- seq(-0.9, 0.9, 0.1)
df <- data.frame(
  category = factor(category, levels = category),
  s_dist = as.numeric(s_dist),
  range = as.numeric(range),
  stringsAsFactors = FALSE
)

#debugging
print(df)
print(class(df))
print(class(s_dist))
print(class(category))

#vis
ggplot(df, aes(range, s_dist)) +
  geom_col(fill = 'skyblue')+
  geom_text(aes(label = round(s_dist, digits=3)), angle = 90, size = 4)+
  labs(title = "Spearman Correlation Matrix Distribution",
       x = 'value range (by 0.1)',
       y = 'decimal')

#noneg test (spearman)
n_values <- as.numeric(as.matrix(noneg))
total_n_n <- length(n_values)

p_neg <- sum(n_values < 0, na.rm = TRUE) / total_n
p_1 <- sum(n_values >= 0 & n_values < 0.1, na.rm = TRUE) / total_n_n
p_2 <- sum(n_values >= 0.1 & n_values < 0.2, na.rm = TRUE) / total_n_n
p_3 <- sum(n_values >= 0.2 & n_values < 0.3, na.rm = TRUE) / total_n_n
p_4 <- sum(n_values >= 0.3 & n_values < 0.4, na.rm = TRUE) / total_n_n
p_5 <- sum(n_values >= 0.4 & n_values < 0.5, na.rm = TRUE) / total_n_n
p_6 <- sum(n_values >= 0.5 & n_values < 0.6, na.rm = TRUE) / total_n_n
p_7 <- sum(n_values >= 0.6 & n_values < 0.7, na.rm = TRUE) / total_n_n
p_8 <- sum(n_values >= 0.7 & n_values < 0.8, na.rm = TRUE) / total_n_n
p_9 <- sum(n_values >= 0.8 & n_values < 0.9, na.rm = TRUE) / total_n_n
pn1 <- sum(n_values < 0 & n_values > -0.1, na.rm = TRUE) / total_n
pn2 <- sum(n_values <= -0.1 & n_values > -0.2, na.rm = TRUE) / total_n_n
pn3 <- sum(n_values <= -0.2 & n_values > -0.3, na.rm = TRUE) / total_n_n
pn4 <- sum(n_values <= -0.3 & n_values > -0.4, na.rm = TRUE) / total_n_n
pn5 <- sum(n_values <= -0.4 & n_values > -0.5, na.rm = TRUE) / total_n_n
pn6 <- sum(n_values <= -0.5 & n_values > -0.6, na.rm = TRUE) / total_n_n
pn7 <- sum(n_values <= -0.6 & n_values > -0.7, na.rm = TRUE) / total_n_n
pn8 <- sum(n_values <= -0.7 & n_values > -0.8, na.rm = TRUE) / total_n_n
pn9 <- sum(n_values <= -0.8 & n_values > -0.9, na.rm = TRUE) / total_n_n
pn10 <- sum(n_values <= -0.9 & n_values >= -1, na.rm = TRUE) / total_n_n

#create dataframe
category_n <- c("pn10", "pn9", "pn8", "pn7", "pn6", "pn5", "pn4", "pn3", "pn2", "pn1", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9")
n_dist <- c(pn10, pn9, pn8, pn7, pn6, pn5, pn4, pn3, pn2, pn1, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)
dfn <- data.frame(
  category_n = factor(category_n, levels = category_n),
  range = as.numeric(range),
  n_dist = as.numeric(n_dist),
  stringsAsFactors = FALSE
)

#vis
ggplot(dfn) +
  geom_col(aes(range, n_dist)) +
  geom_text(aes(range, n_dist, label = round(n_dist, digits=3)), angle = 90, size = 4)+
  labs(title = "Pearson Correlation Matrix Distribution",
       x = 'value range (by 0.1)',
       y = 'decimal')

#cosine
c_values <- as.numeric(as.matrix(c))
total_n <- length(c_values)

p_neg <- sum(c_values < 0, na.rm = TRUE) / total_n
p_1 <- sum(c_values >= 0 & c_values < 0.1, na.rm = TRUE) / total_n
p_2 <- sum(c_values >= 0.1 & c_values < 0.2, na.rm = TRUE) / total_n
p_3 <- sum(c_values >= 0.2 & c_values < 0.3, na.rm = TRUE) / total_n
p_4 <- sum(c_values >= 0.3 & c_values < 0.4, na.rm = TRUE) / total_n
p_5 <- sum(c_values >= 0.4 & c_values < 0.5, na.rm = TRUE) / total_n
p_6 <- sum(c_values >= 0.5 & c_values < 0.6, na.rm = TRUE) / total_n
p_7 <- sum(c_values >= 0.6 & c_values < 0.7, na.rm = TRUE) / total_n
p_8 <- sum(c_values >= 0.7 & c_values < 0.8, na.rm = TRUE) / total_n
p_9 <- sum(c_values >= 0.8 & c_values < 0.9, na.rm = TRUE) / total_n
pn1 <- sum(c_values < 0 & c_values > -0.1, na.rm = TRUE) / total_n
pn2 <- sum(c_values <= -0.1 & c_values > -0.2, na.rm = TRUE) / total_n
pn3 <- sum(c_values <= -0.2 & c_values > -0.3, na.rm = TRUE) / total_n
pn4 <- sum(c_values <= -0.3 & c_values > -0.4, na.rm = TRUE) / total_n
pn5 <- sum(c_values <= -0.4 & c_values > -0.5, na.rm = TRUE) / total_n
pn6 <- sum(c_values <= -0.5 & c_values > -0.6, na.rm = TRUE) / total_n
pn7 <- sum(c_values <= -0.6 & c_values > -0.7, na.rm = TRUE) / total_n
pn8 <- sum(c_values <= -0.7 & c_values > -0.8, na.rm = TRUE) / total_n
pn9 <- sum(c_values <= -0.8 & c_values > -0.9, na.rm = TRUE) / total_n
pn10 <- sum(c_values <= -0.9 & c_values >= -1, na.rm = TRUE) / total_n
#create dataframe
category <- c("pn10", "pn9", "pn8", "pn7", "pn6", "pn5", "pn4", "pn3", "pn2", "pn1", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9")
c_dist <- c(pn10, pn9, pn8, pn7, pn6, pn5, pn4, pn3, pn2, pn1, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)
df <- data.frame(
  category = factor(category, levels = category),
  c_dist = as.numeric(c_dist),
  range = as.numeric(range),
  stringsAsFactors = FALSE
)

#debugging
print(df)
print(class(df))
print(class(s_dist))
print(class(category))

#vis
ggplot(df, aes(range, c_dist)) +
  geom_col(fill = 'skyblue')+
  geom_text(aes(label = round(c_dist, digits=3)), angle = 90, size = 4)+
  labs(title = "Cosine Correlation Matrix Distribution",
       x = 'value range (by 0.1)',
       y = 'decimal')

#mahalanobis (euclidian)
me_values <- as.numeric(as.matrix(me))
total_n <- length(me_values)

#p_neg <- sum(me_values < 0, na.rm = TRUE) / total_n
p_1 <- sum(me_values >= 0 & me_values < 0.1, na.rm = TRUE) / total_n
p_2 <- sum(me_values >= 0.1 & me_values < 0.2, na.rm = TRUE) / total_n
p_3 <- sum(me_values >= 0.2 & me_values < 0.3, na.rm = TRUE) / total_n
p_4 <- sum(me_values >= 0.3 & me_values < 0.4, na.rm = TRUE) / total_n
p_5 <- sum(me_values >= 0.4 & me_values < 0.5, na.rm = TRUE) / total_n
p_6 <- sum(me_values >= 0.5 & me_values < 0.6, na.rm = TRUE) / total_n
p_7 <- sum(me_values >= 0.6 & me_values < 0.7, na.rm = TRUE) / total_n
p_8 <- sum(me_values >= 0.7 & me_values < 0.8, na.rm = TRUE) / total_n
p_9 <- sum(me_values >= 0.8 & me_values < 0.9, na.rm = TRUE) / total_n
pn1 <- sum(me_values < 0 & me_values > -0.1, na.rm = TRUE) / total_n
pn2 <- sum(me_values <= -0.1 & me_values > -0.2, na.rm = TRUE) / total_n
pn3 <- sum(me_values <= -0.2 & me_values > -0.3, na.rm = TRUE) / total_n
pn4 <- sum(me_values <= -0.3 & me_values > -0.4, na.rm = TRUE) / total_n
pn5 <- sum(me_values <= -0.4 & me_values > -0.5, na.rm = TRUE) / total_n
pn6 <- sum(me_values <= -0.5 & me_values > -0.6, na.rm = TRUE) / total_n
pn7 <- sum(me_values <= -0.6 & me_values > -0.7, na.rm = TRUE) / total_n
pn8 <- sum(me_values <= -0.7 & me_values > -0.8, na.rm = TRUE) / total_n
pn9 <- sum(me_values <= -0.8 & me_values > -0.9, na.rm = TRUE) / total_n
pn10 <- sum(me_values <= -0.9 & me_values >= -1, na.rm = TRUE) / total_n
#create data frame
category <- c("pn10", "pn9", "pn8", "pn7", "pn6", "pn5", "pn4", "pn3", "pn2", "pn1", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9")
me_dist <- c(pn10, pn9, pn8, pn7, pn6, pn5, pn4, pn3, pn2, pn1, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)
df <- data.frame(
  category = factor(category, levels = category),
  range = as.numeric(range),
  me_dist = as.numeric(me_dist),
  stringsAsFactors = FALSE
)

#debugging
print(df)
print(class(df))
print(class(mg_dist))
print(class(category))

#vis
ggplot(df, aes(range, me_dist)) +
  geom_col(fill = 'skyblue')+
  geom_text(aes(label = round(me_dist, digits=3)), angle = 90, size = 4)+
  labs(title = "Maha/Euclidian Correlation Matrix Distribution",
       x = 'value range (by 0.1)',
       y = 'decimal')


#mahalanobis (gaussian)
mg_values <- as.numeric(as.matrix(mg))
total_n <- length(mg_values)

#p_neg <- sum(mg_values < 0, na.rm = TRUE) / total_n
p_1 <- sum(mg_values >= 0 & mg_values < 0.1, na.rm = TRUE) / total_n
p_2 <- sum(mg_values >= 0.1 & mg_values < 0.2, na.rm = TRUE) / total_n
p_3 <- sum(mg_values >= 0.2 & mg_values < 0.3, na.rm = TRUE) / total_n
p_4 <- sum(mg_values >= 0.3 & mg_values < 0.4, na.rm = TRUE) / total_n
p_5 <- sum(mg_values >= 0.4 & mg_values < 0.5, na.rm = TRUE) / total_n
p_6 <- sum(mg_values >= 0.5 & mg_values < 0.6, na.rm = TRUE) / total_n
p_7 <- sum(mg_values >= 0.6 & mg_values < 0.7, na.rm = TRUE) / total_n
p_8 <- sum(mg_values >= 0.7 & mg_values < 0.8, na.rm = TRUE) / total_n
p_9 <- sum(mg_values >= 0.8 & mg_values < 0.9, na.rm = TRUE) / total_n
pn1 <- sum(mg_values <= 0 & mg_values > -0.1, na.rm = TRUE) / total_n
pn2 <- sum(mg_values <= -0.1 & mg_values > -0.2, na.rm = TRUE) / total_n
pn3 <- sum(mg_values <= -0.2 & mg_values > -0.3, na.rm = TRUE) / total_n
pn4 <- sum(mg_values <= -0.3 & mg_values > -0.4, na.rm = TRUE) / total_n
pn5 <- sum(mg_values <= -0.4 & mg_values > -0.5, na.rm = TRUE) / total_n
pn6 <- sum(mg_values <= -0.5 & mg_values > -0.6, na.rm = TRUE) / total_n
pn7 <- sum(mg_values <= -0.6 & mg_values > -0.7, na.rm = TRUE) / total_n
pn8 <- sum(mg_values <= -0.7 & mg_values > -0.8, na.rm = TRUE) / total_n
pn9 <- sum(mg_values <= -0.8 & mg_values > -0.9, na.rm = TRUE) / total_n
pn10 <- sum(mg_values <= -0.9 & mg_values >= -1, na.rm = TRUE) / total_n



#create dataframe
category <- c("pn10", "pn9", "pn8", "pn7", "pn6", "pn5", "pn4", "pn3", "pn2", "pn1", "p_1", "p_2", "p_3", "p_4", "p_5", "p_6", "p_7", "p_8", "p_9")
mg_dist <- c(pn10, pn9, pn8, pn7, pn6, pn5, pn4, pn3, pn2, pn1, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9)
df <- data.frame(
  category = factor(category, levels = category),
  mg_dist = as.numeric(mg_dist),
  range = as.numeric(range),
  stringsAsFactors = FALSE
)

#debugging
print(df)
print(class(df))
print(class(mg_dist))
print(class(category))

#vis
ggplot(df, aes(range, mg_dist)) +
  geom_col(fill = 'skyblue')+
  geom_text(aes(label = round(mg_dist, digits=3)), angle = 90, size = 4)+
  labs(title = "Maha/Gaussian Matrix Distribution",
       x = 'value range (by 0.1)',
       y = 'decimal')


# -------------------------------------------------------------------------
# Threshold-aligned view (matches auto_threshold.py behavior)
# - negatives are set to 0 first
# - threshold keeps values >= t
# - bins use 0.05 increments to match threshold sweep
# -------------------------------------------------------------------------
threshold_bins <- seq(0, 1, by = 0.05)
threshold_values <- seq(0.05, 0.90, by = 0.05)

threshold_profile <- function(mat, label = "matrix") {
  vals <- as.numeric(as.matrix(mat))
  vals_noneg <- ifelse(vals < 0, 0, vals)
  total_n <- length(vals_noneg)

  # 0.05 bins (distribution after negative clipping)
  bin_left <- threshold_bins[-length(threshold_bins)]
  bin_right <- threshold_bins[-1]
  bin_pct <- sapply(seq_along(bin_left), function(i) {
    l <- bin_left[i]
    r <- bin_right[i]
    if (i < length(bin_left)) {
      sum(vals_noneg >= l & vals_noneg < r, na.rm = TRUE) / total_n
    } else {
      sum(vals_noneg >= l & vals_noneg <= r, na.rm = TRUE) / total_n
    }
  })

  # Retention exactly as auto_threshold.py interprets it
  # matrix_thresholded = where(value < threshold, 0, value)
  retention_pct <- sapply(threshold_values, function(t) {
    sum(vals_noneg >= t, na.rm = TRUE) / total_n
  })

  list(
    bins = data.frame(
      label = label,
      left = bin_left,
      right = bin_right,
      pct = bin_pct
    ),
    retention = data.frame(
      label = label,
      threshold = threshold_values,
      retention_pct = retention_pct
    )
  )
}

# Build threshold-aligned summaries (Mahalanobis + Spearman + Cosine)
prof_me <- threshold_profile(me, "me (Euclidean kernel)")
prof_mg <- threshold_profile(mg, "mg (Gaussian kernel)")
prof_s <- threshold_profile(s, "spearman")
prof_c <- threshold_profile(c, "cosine")

df_bins <- bind_rows(prof_me$bins, prof_mg$bins)
df_ret <- bind_rows(
  prof_me$retention,
  prof_mg$retention,
  prof_s$retention,
  prof_c$retention
)

print(df_ret)

ggplot(df_bins, aes(x = left, y = pct, fill = label)) +
  geom_col(position = "dodge") +
  geom_text(
    aes(label = round(pct, 3)),
    position = position_dodge(width = 0.9),
    angle = 90,
    size = 3
  ) +
  labs(
    title = "Threshold-Aligned Distribution (0.05 bins, negatives clipped)",
    x = "value bin left edge",
    y = "decimal"
  )

ggplot(df_ret, aes(x = threshold, y = retention_pct, color = label)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  geom_text(aes(label = round(retention_pct, 3)), vjust = -0.7, size = 3) +
  labs(
    title = "Retention vs Threshold (matches auto_threshold.py)",
    x = "threshold (keep values >= threshold)",
    y = "retention decimal"
  )