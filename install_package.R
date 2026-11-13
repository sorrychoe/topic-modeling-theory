if (!require("tm")) {
    install.packages("tm")
}
if (!require("tidyverse")) {
    install.packages("tidyverse")
}
if (!require("tidytext")) {
    install.packages("tidytext")
}
if (!require("showtext")) {
    install.packages("showtext")
}
if (!require("stm")) {
    install.packages("stm")
}
if (!require("readxl")) {
    install.packages("readxl")
}
if (!require("topicovertime")) {
    # not on CRAN yet; install from GitHub
    if (!require("remotes")) install.packages("remotes")
    remotes::install_github("sorrychoe/topicovertime")
}
