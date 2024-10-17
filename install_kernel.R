if (!require("devtools")) {
    install.packages("devtools")
}
if (!require("IRkernel")) {
    devtools::install_github('IRkernel/IRkernel')
}
IRkernel::installspec()