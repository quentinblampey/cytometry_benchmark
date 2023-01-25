source('CyTOF_LDAtrain.R')
source('CyTOF_LDApredict.R')

for (x in list("2", "4", "5", "6", "7", "8", "10")) {
    print(paste("Running on batch", x))

    path_train <- paste("../POISED/LDA/B", x, "_train", sep="")
    path_test <- paste("../POISED/LDA/B", x, "_test", sep="")
    path_pred <- paste("../predictions/lda_poised_B", x, ".csv", sep="")

    LDA.Model <- CyTOF_LDAtrain(TrainingSamplesExt = path_train, TrainingLabelsExt = '',mode = 'CSV',
                                RelevantMarkers =  c(2:40), LabelIndex = 41, Transformation = FALSE)

    df <- CyTOF_LDApredict(LDA.Model,TestingSamplesExt = path_test, mode = 'CSV', RejectionThreshold = 0)

    write.csv(df, file=path_pred, row.names = FALSE)
}