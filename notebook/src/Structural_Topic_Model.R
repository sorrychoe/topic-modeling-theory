library(stm)
library(tidyverse)
library(tidytext)
library(tm)
library(showtext)

font_add_google(name = "Nanum Gothic", family = "nanumgothic")
showtext_auto()

df <- read_xlsx("data/NewsResult_19950101-20240930.xlsx")

df$키워드 <- gsub(",", " ", df$키워드)

df$Date <- as.Date(df$일자, format = "%Y%m%d")

df$Press <- as.factor(df$언론사)

processed <- textProcessor(documents = df$키워드, metadata = df)

out <- prepDocuments(processed$documents, processed$vocab, processed$meta)

out$meta$Date <- df$Date
out$meta$Press <- df$Press

topicN <- seq(3, 10)
topicN_storage <- searchK(out$documents, out$vocab, K = topicN)
plot(topicN_storage)

stm_model <- stm(documents = out$documents, 
                 vocab = out$vocab, 
                 K = 7, 
                 prevalence = ~ Press + Date, 
                 data = out$meta, 
                 max.em.its = 100, 
                 init.type = "Spectral")

plot(stm_model, type = "summary")

labelTopics(stm_model)

td_beta <- stm_model %>% tidy(matrix = 'beta') 

td_beta %>% 
  group_by(topic) %>% 
  slice_max(beta, n = 7) %>% 
  ungroup() %>% 
  mutate(topic = str_c("topic ", topic)) %>% 
  ggplot(aes(x = beta, 
             y = reorder_within(term, beta, topic),
             fill = topic)) +
  geom_col(show.legend = F) +
  scale_y_reordered() +
  facet_wrap(~topic, scales = "free") +
  labs(x = expression("word probability distribution: "~beta), y = NULL,
       title = "word probability distribution per topic") +
  theme(plot.title = element_text(size = 15))
