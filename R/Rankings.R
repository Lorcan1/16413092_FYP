###### R script for deriving various rankings for AIF360 pipelines ########

library(dplyr)

#### Functions ####

#reads in csvs and aggregates by dataset and combination
parseResultFiles <- function(nameTemplate, endTemplate="output.csv") {
  files <- list.files(path=getwd())
  files <- files[startsWith(files, nameTemplate) & endsWith(files, endTemplate)]
  
  df <- data.frame()
  for (name in files) {
    df <- rbind(df, read.csv(paste0(getwd(),"/", name)))
  }
  
  df$Time <- gsub('\\{', '', df$Time)
  df$Time <- gsub('\\}', '', df$Time)
  df$Time <- as.numeric(df$Time)
  
  df$Combo <- paste0(df$Pre, '+', df$In_p, '+', df$Post, '+', df$Classifier)
  df$Combo <- factor(df$Combo)
  df$Score <- NULL
  df$Rank <- NULL
  
  aggDF <- aggregate(df, by=list(df$Combo, df$Dataset), FUN = mean)
  s <- data.frame(do.call(rbind, strsplit(as.character(aggDF$Group.1), '\\+')))
  names(s) <- c(names(df)[1:4])
  
  aggDF$Pre <- s$Pre
  aggDF$In_p <- s$In_p
  aggDF$Post <- s$Post
  aggDF$Classifier <- s$Classifier
  aggDF$Dataset <- aggDF$Group.2
  aggDF$Group.2 <- NULL
  aggDF$Combo <- aggDF$Group.1
  aggDF$Group.1 <- NULL
  aggDF$Sens_Attr <- NULL
  aggDF$Valid <- NULL
  
  return(aggDF)
}

#derives an equally weighted rank for both fairness and performance measures
#derives ranks per dataset then averages them to produce an average rank per combo
#depends on: computePerfRank and computeFairRank
deriveEqualRank <- function(df) {
  perfDF <- data.frame()
  fairDF <- data.frame()
  
  for (dataset in levels(df$Dataset)) {
    perfDF <- rbind(perfDF, computePerfRank(df[df$Dataset == dataset, ]))
    fairDF <- rbind(fairDF, computeFairRank(df[df$Dataset == dataset, ]))
  }
  
  perfAgg <- aggregate(perfDF$PerfRank, by=list(perfDF$Combo), FUN=mean)
  fairAgg <- aggregate(fairDF$FairRank, by=list(fairDF$Combo), FUN=mean)
  
  names(perfAgg) <- c("Combo", "PerfRank")
  names(fairAgg) <- c("Combo", "FairRank")
  
  rankDF <- merge(perfAgg, fairAgg, by="Combo")
  rankDF$EqualWeightRank <- (rankDF$PerfRank + rankDF$FairRank) / 2 
  
  return(rankDF)
}

# derives an performance rank 
# assumes df contains results for just one dataset
# considers all four performance variables equal
computePerfRank <- function(df) {
  
  perfRank <- rank(1- df$Accuracy, ties.method= "min") + 
    rank(1- df$AUC, ties.method= "min") + 
    rank(1- df$Precision, ties.method= "min") + 
    rank(1 - df$Recall, ties.method= "min")
  
  perfRank <- perfRank / 4
  perfRank <- rank(perfRank, ties.method= "min")
  
  return (data.frame(Combo=df$Combo, PerfRank = perfRank))
}

# derives an performance rank 
# assumes df contains results for just one dataset
# considers all five fairness measures equal
computeFairRank <- function(df) {
  fairRank <- rank(df$Mean.Difference, ties.method= "min") + 
    rank(1- df$Disparate.Impact, ties.method= "min") + 
    rank(df$Theil.Index, ties.method= "min") + 
    rank(df$Average.Odds.Difference, ties.method= "min") + 
    rank(df$Equal.Opportunity.Difference, ties.method= "min") 
  
  fairRank <- fairRank / 5
  fairRank <- rank(fairRank, ties.method= "min")
  
  return (data.frame(Combo=df$Combo, FairRank = fairRank))
}

#### Run #####

setwd("/Users/scaton/Documents/Papers/FairMLComp/16413092_FYP/data/basic")

stratResults <- deriveEqualRank(parseResultFiles("SingleStratSamples"))
randomResults <- deriveEqualRank(parseResultFiles("SingleSamples"))

combined <- merge(stratResults, randomResults, by="Combo")
combined$Diff <- combined$EqualWeightRank.x - combined$EqualWeightRank.y
hist(combined$Diff, breaks=15)
summary(combined$Diff)






