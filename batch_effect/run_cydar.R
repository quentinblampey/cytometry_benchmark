library("cydar")

foldername = commandArgs(trailingOnly=TRUE)

all_poised.x <- list()
for (b in paste0("B", c(2, 4, 5, 6, 7, 8, 10))) {
    sample.names <- list.files(paste0(foldername, "/", b))
    x <- list()
    for (file in sample.names) {
        x[[file]] <- read.csv(paste0(foldername, "/", b, "/", file))[,-1]
    }
    all_poised.x[[b]] <- x
}

batch_poised.comp <- list(
    factor(c(1, 2)),
    factor(c(1, 2, 1, 2)),
    factor(c(1, 2)),
    factor(c(1, 2, 1, 2)),
    factor(c(1, 2, 1, 2)),
    factor(c(1, 2)),
    factor(c(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2))
)

corrected <- normalizeBatch(all_poised.x, batch_poised.comp, mode="range")

for (b in paste0("B", c(2, 4, 5, 6, 7, 8, 10))) {
    sample.names <- list.files(paste0(foldername, "/", b))
    for (file in sample.names) {
        write.csv(corrected[[b]][[file]], file=paste0(foldername, "_out", "/", b, "/", file))
    }
}