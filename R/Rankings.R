###### R script for deriving various rankings for AIF360 pipelines ########

library(dplyr)

#### Functions ####

#reads in csvs and aggregates by dataset and combination
parseResultFiles <- function(nameTemplate, endTemplate="output.csv",pathname) {
  # files <- list.files(path=getwd())
  files <- list.files(path=pathname)
  files <- files[startsWith(files, nameTemplate) & endsWith(files, endTemplate)]
  
  df <- data.frame()
  for (name in files) {
    # df <- rbind(df, read.csv(paste0(getwd(),"/", name)))
    df <- rbind(df, read.csv(paste0(pathname,"/", name)))
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
  # fairRank <- rank(df$Mean.Difference, ties.method= "min") + 
  #   rank(1- df$Disparate.Impact, ties.method= "min") +
  #   rank(df$Theil.Index, ties.method= "min") + 
  #   rank(df$Average.Odds.Difference, ties.method= "min") + 
  #   rank(df$Equal.Opportunity.Difference, ties.method= "min") 
  fairRank <- rank(abs(df$Mean.Difference), ties.method= "min") + 
    rank(abs(1- df$Disparate.Impact), ties.method= "min") +
    rank(df$Theil.Index, ties.method= "min") + 
    rank(abs(df$Average.Odds.Difference), ties.method= "min") + 
    rank(abs(df$Equal.Opportunity.Difference), ties.method= "min") 
  
  fairRank <- fairRank / 5
  fairRank <- rank(fairRank, ties.method= "min")
  
  return (data.frame(Combo=df$Combo, FairRank = fairRank))
}

#### Run #####

# setwd("/Users/scaton/Documents/Papers/FairMLComp/16413092_FYP/data/basic")
tryCatch({
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}, error=function(cond){message(paste("cannot change working directory"))
})

## With validation
stratResults <- deriveEqualRank(parseResultFiles("LogDefaultStratValid",pathname='../data/basic/'))
randomResults <- deriveEqualRank(parseResultFiles("LogDefaultRadValid",pathname='../data/basic/'))

#Without validation
stratResults <- deriveEqualRank(parseResultFiles("SingleStratSamples",pathname='../data/basic/'))
randomResults <- deriveEqualRank(parseResultFiles("SingleSamples",pathname='../data/basic/'))


combined <- merge(stratResults, randomResults, by="Combo")
combined$Diff <- combined$EqualWeightRank.x - combined$EqualWeightRank.y
hist(combined$Diff, breaks=15)
summary(combined$Diff) 

library(ggplot2)

ggplot(combined, aes(x=Diff)) + geom_histogram()

# a basic paired Wilcoxon test for differences in ranks
wilcox.test(combined$EqualWeightRank.x, combined$EqualWeightRank.y, paired = TRUE) 
# p value 0.65, meaning that there's no systematic positive or negative change in ranks

# add some stats for comparing combination of approaches
combined$Combo <- as.character(combined$Combo)

combined$Pre = ""
combined$In = ""
combined$Post = ""
combined$Algo = ""

for (i in 1:nrow(combined)){
  approach <- strsplit(combined$Combo[i], split='\\+')
  combined$Pre[i] = approach[[1]][1]
  combined$In[i] = approach[[1]][2]
  combined$Post[i] = approach[[1]][3]
  combined$Algo[i] = approach[[1]][4]  
}

only_pre <- combined %>% filter(Pre != "-", In == "-", Post == '-')
only_in <- combined %>% filter(In != "-", Pre == "-", Post == '-')
only_post <- combined %>% filter(Post != "-", In == "-", Pre == '-')
pre_in <- combined %>% filter(Pre != "-", In != "-", Post == '-')
pre_post <- combined %>% filter(Pre != "-", In == "-", Post != '-')
in_post <- combined %>% filter(Pre == "-", In != "-", Post != '-')
pre_in_post <- combined %>% filter(Pre != "-", In != "-", Post != '-')
only_classification <- combined %>% filter(Pre == "-", In == "-", Post == '-')

only_pre$Scenario = 'only_pre'
only_in$Scenario = 'only_in'
only_post$Scenario = 'only_post'
pre_in$Scenario = 'pre_in'
pre_post$Scenario = 'pre_post'
in_post$Scenario = 'in_post'
pre_in_post$Scenario = 'pre_in_post'
only_classification$Scenario = 'only_classification'

combined_new <- rbind(only_pre, only_in, only_post, pre_in, pre_post, in_post, pre_in_post, only_classification)

ggplot(combined_new, aes(x=Scenario,y=EqualWeightRank.x, fill=Scenario)) + geom_boxplot()

