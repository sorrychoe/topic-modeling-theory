# import librarys
library(topicsovertime) # for ToT
library(tidyverse) # for Data preprocessing(dplyr, tidyr), visualization(ggplot2) & use pipeline operator(magrittr)
library(tidytext) # for Text preprocessing
library(showtext) # for set the Korean font
library(readxl) # for load excel data

font_add_google(name = "Nanum Gothic", family = "nanumgothic") # korean font setup
showtext_auto()

# set the seed number
set.seed(42)

# data load
# The data is related to Handong University, 
# which was reported in major Korean daily newspapers from January 1995 to September 2024.
df <- read_xlsx("../data/NewsResult_19950101-20240930.xlsx")

# ToT uses the publication date itself as a continuous timestamp.
# Unlike DTM, there is no need to cut the data into time slices.
df$Date <- as.Date(df$일자, format = "%Y%m%d")
df$doc_id <- seq_len(nrow(df))

# data preprocessing
# split the comma-separated keywords into one word per row
tokens <- df |>
  select(doc_id, 키워드) |>
  separate_longer_delim(키워드, ",") |>
  rename(word = 키워드) |>
  filter(word != "")

# same vocabulary filter as the DTM notebook (min_cf = 5, rm_top = 10)
word_freq <- tokens |> count(word, sort = TRUE)
vocab <- word_freq |> filter(n >= 5) |> slice(-(1:10)) |> pull(word)

# make the document-term matrix
dtm <- tokens |>
  filter(word %in% vocab) |>
  count(doc_id, word) |>
  cast_sparse(doc_id, word, n)

dates <- df$Date[as.integer(rownames(dtm))]

cat("Num docs:", nrow(dtm), ", Num Vocabs:", ncol(dtm), ", Total Words:", sum(dtm), "\n")
cat("Removed top 10 words:", head(word_freq$word, 10), "\n")

# define & train the model
tot_model <- tot(dtm, dates,
                 k = 7,
                 n_iter = 300,
                 time_weight = 0.3)

tot_model

# words per Topics
terms(tot_model, 10)

# Tidy the result into a data.frame
# Extract the phi value (word probability distribution of each topic)
td_beta <- as_tibble(t(tot_model$phi), rownames = "term") |>
  pivot_longer(-term, names_to = "topic", values_to = "beta")

# visualization of topic's value
td_beta |>
  group_by(topic) |>
  slice_max(beta, n = 7) |>
  ungroup() |>
  ggplot(aes(x = beta, 
             y = reorder_within(term, beta, topic),
             fill = topic)) +
  geom_col(show.legend = F) +
  scale_y_reordered() +
  facet_wrap(~topic, scales = "free") +
  labs(x = expression("word probability distribution: "~phi), y = NULL,
       title = "word probability distribution per topic") +
  theme(plot.title = element_text(size = 15))

# Beta distribution over time per topic (psi)
# psi is estimated on the normalized time scale (0 to 1), so convert it back to dates.
date_range <- range(dates)
span <- as.numeric(diff(date_range))
psi <- tot_model$psi

topic_labels <- apply(terms(tot_model, 3), 2, paste, collapse = ", ")
topic_labels <- paste0(names(topic_labels), ": ", topic_labels)

topic_period <- tibble(
  topic = topic_labels,
  shape1 = psi[, 1],
  shape2 = psi[, 2],
  mean_date = date_range[1] + span * shape1 / (shape1 + shape2),
  sd_years = sqrt(shape1 * shape2 / ((shape1 + shape2)^2 * (shape1 + shape2 + 1))) * span / 365.25,
  share = tot_model$topic_size / sum(tot_model$topic_size)
) |>
  arrange(mean_date)

topic_period

# show the time distribution of each topic
date_grid <- seq(date_range[1], date_range[2], by = "month")

td_psi <- map_dfr(seq_len(tot_model$k), function(k) {
  tibble(topic = topic_labels[k],
         date = date_grid,
         density = dbeta(as.numeric(date_grid - date_range[1]) / span, psi[k, 1], psi[k, 2]))
})

td_psi |>
  ggplot(aes(x = date, y = density, fill = topic)) +
  geom_area(show.legend = F, alpha = 0.8) +
  facet_wrap(~topic, ncol = 2, scales = "free_y") +
  labs(x = NULL, y = expression("density: "~psi),
       title = "time distribution per topic") +
  theme(plot.title = element_text(size = 15))

# Changes in topic percentage over time: P(topic | time)
td_trend <- topic_time(tot_model, date_grid) |>
  as_tibble() |>
  setNames(topic_labels) |>
  mutate(date = date_grid) |>
  pivot_longer(-date, names_to = "topic", values_to = "prob")

td_trend |>
  ggplot(aes(x = date, y = prob, fill = topic)) +
  geom_area() +
  scale_y_continuous(labels = scales::percent) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(x = NULL, y = "percent(%)", fill = NULL,
       title = "Changes in topic percentage over time") +
  theme(plot.title = element_text(size = 15),
        axis.text.x = element_text(angle = 45, hjust = 1))

# split the data into train(90%) & test(10%)
test_idx <- sample(nrow(dtm), round(nrow(dtm) * 0.1))

train_model <- tot(dtm[-test_idx, ], dates[-test_idx],
                   k = 7,
                   n_iter = 300,
                   time_weight = 0.3)

# predict the date of unseen articles
pred_date <- predict(train_model, dtm[test_idx, ], type = "time")

pred_result <- tibble(actual = dates[test_idx], predicted = pred_date) |>
  mutate(error_years = abs(as.numeric(predicted - actual)) / 365.25)

cat("correlation:", round(cor(as.numeric(pred_result$actual), as.numeric(pred_result$predicted)), 3), "\n")
cat("median absolute error (years):", round(median(pred_result$error_years), 2), "\n")
cat("baseline, always predicting the median date (years):",
    round(median(abs(as.numeric(dates[test_idx] - median(dates[-test_idx]))) / 365.25), 2), "\n")

# show result as a plot
pred_result |>
  ggplot(aes(x = actual, y = predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, colour = "red", linetype = "dashed") +
  labs(x = "actual date", y = "predicted date",
       title = "Predicted vs. actual date of unseen articles") +
  theme(plot.title = element_text(size = 15))
