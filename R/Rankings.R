###### R script for deriving various rankings for AIF360 pipelines ########

library(dplyr)
library(splitstackshape)
library(ggplot2)
library(xtable)

#### Functions ####

#reads in csvs and aggregates by dataset and combination
parseResultFiles <- function(nameTemplate, endTemplate="output.csv",pathname) {
  # files <- list.files(path=getwd())
  files <- list.files(path=pathname)
  
  if (length(nameTemplate) == 1) {
    files <- files[startsWith(files, nameTemplate) & endsWith(files, endTemplate)]
  } else {
    fileset <- c()
    for (name in nameTemplate) {
      fileset <- rbind(fileset, files[startsWith(files, name) & endsWith(files, endTemplate)])
    }
    files <- fileset
  }

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
  
  write.csv(df, file=paste0(getwd(), "/fairCombos.csv"), row.names = F)
  
  t <- table(df$Combo, df$Dataset)
  write.table(t, file=paste0(getwd(), "/combos.csv"), sep = ",")
  
  return (aggregateResults(df))
}

getRuntime <- function() {
  runtime <-  read.csv(file=paste0(getwd(), "/fairCombos.csv"))
  return (sum(runtime$Time))
}

aggregateResults <- function(df) {
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

#draws a sample of n observations of each combination
sampleNObs <- function(df, n=5, cols) {
  
  strat <- stratified(df, cols, n)
  
  print(sum(strat$Time) / 3600)
  
  return(strat)
  
}

#derives an equally weighted rank for both fairness and performance measures
#derives ranks per dataset then averages them to produce an average rank per combo
#depends on: computePerfRank and computeFairRank
deriveEqualRank <- function(df) {
  perfDF <- data.frame()
  fairDF <- data.frame()
  runtimeDF <- data.frame()
  
  for (dataset in levels(df$Dataset)) {
    perfDF <- rbind(perfDF, computePerfRank(df[df$Dataset == dataset, ]))
    fairDF <- rbind(fairDF, computeFairRank(df[df$Dataset == dataset, ]))
    runtimeDF <- rbind(runtimeDF, computeRuntimeRank(df[df$Dataset == dataset, ]))
  }
  
  perfAgg <- aggregate(perfDF$PerfRank, by=list(perfDF$Combo), FUN=mean)
  fairAgg <- aggregate(fairDF$FairRank, by=list(fairDF$Combo), FUN=mean)
  fairAggIndiv <- aggregate(fairDF$IndivFairRank, by=list(fairDF$Combo), FUN=mean)
  fairAggGroup <- aggregate(fairDF$GroupFairRank, by=list(fairDF$Combo), FUN=mean)
  runtimeAgg <- aggregate(runtimeDF$RuntimeRank, by=list(runtimeDF$Combo), FUN=mean)
  
  names(perfAgg) <- c("Combo", "PerfRank")
  names(fairAgg) <- c("Combo", "FairRank")
  names(fairAggIndiv) <- c("Combo", "FairRankIndiv")
  names(fairAggGroup) <- c("Combo", "FairRankGroup")
  names(runtimeAgg) <- c("Combo", "RuntimeRank")
  
  rankDF <- merge(perfAgg, fairAgg, by="Combo")
  rankDF <- merge(rankDF, fairAggIndiv, by="Combo")
  rankDF <- merge(rankDF, fairAggGroup, by="Combo")
  rankDF <- merge(rankDF, runtimeAgg, by="Combo")
  rankDF$EqualWeightRank <- (rankDF$PerfRank + rankDF$FairRank + rankDF$RuntimeRank) / 3 
  rankDF$PerfFairEqualRank <- (rankDF$PerfRank + rankDF$FairRank) / 2
  
  return(rankDF)
}

# derives an performance rank 
# assumes df contains results for just one dataset
# considers all four performance variables equal
computePerfRank <- function(df) {
  
  perfRank <- rank(1- df$Accuracy, ties.method= "min") + 
    #rank(1- df$AUC, ties.method= "min") + 
    rank(1- df$Precision, ties.method= "min") + 
    rank(1 - df$Recall, ties.method= "min")
  
  perfRank <- perfRank / 3
  #perfRank <- rank(perfRank, ties.method= "min")
  
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
  
  fairIndivRank <- rank(df$Theil.Index, ties.method= "min")
  
  fairGroupRank <- rank(abs(df$Mean.Difference), ties.method= "min") + 
    rank(abs(1- df$Disparate.Impact), ties.method= "min") +
    rank(abs(df$Average.Odds.Difference), ties.method= "min") + 
    rank(abs(df$Equal.Opportunity.Difference), ties.method= "min") 
  
  fairRank <- fairRank / 5
  #fairRank <- rank(fairRank, ties.method= "min")
  
  fairGroupRank <- fairGroupRank / 4
  #fairGroupRank <- rank(fairGroupRank, ties.method= "min")
  
  return (data.frame(Combo=df$Combo, FairRank = fairRank, IndivFairRank = fairIndivRank, GroupFairRank = fairGroupRank))
}

#ranks on the basis of runtime
computeRuntimeRank <- function(df) {
  #remove combos that failed (they are hardcoded to have a runtime of 0)
  df$Time[df$Time == 0] <- NA
  
  runtimeRank <-  rank(df$Time, ties.method = "min")
  return (data.frame(Combo=df$Combo, RuntimeRank = runtimeRank))
}

#### Run #####

# setwd("/Users/scaton/Documents/Papers/FairMLComp/16413092_FYP/data/basic")
tryCatch({
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}, error=function(cond){message(paste("cannot change working directory"))
})

## With validation
stratResults <- deriveEqualRank(parseResultFiles("SamplingStratComp",pathname='../data/StratVsRand/'))
randomResults <- deriveEqualRank(parseResultFiles("SamplingRandComp",pathname='../data/StratVsRand/'))

#Without validation
stratResults <- deriveEqualRank(parseResultFiles("SingleStratSamples",pathname='../data/basic/'))
randomResults <- deriveEqualRank(parseResultFiles("SingleSamples",pathname='../data/basic/'))

#New results
run1 <- deriveEqualRank(parseResultFiles("Run2Strat",pathname='../../logs/', endTemplate=".csv"))
run2 <- deriveEqualRank(parseResultFiles("ErrorHandlingStrat",pathname='../../logs/', endTemplate=".csv"))
  
runCombined <- deriveEqualRank(parseResultFiles(nameTemplate = c("Run2Strat", "ErrorHandlingStrat"),pathname='../../logs/', endTemplate=".csv"))
  
  
combined <- merge(stratResults, randomResults, by="Combo")
combined$DiffEqual <- combined$EqualWeightRank.x - combined$EqualWeightRank.y
hist(combined$Diff, breaks=15)
summary(combined$Diff) 

#### random vs stratified ####

ggplot(combined, aes(x=Diff)) + geom_histogram()

# a basic paired Wilcoxon test for differences in ranks
wilcox.test(combined$EqualWeightRank.x, combined$EqualWeightRank.y, paired = TRUE) 
# p value 0.8906, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$FairRank.x, combined$FairRank.y, paired = TRUE) 
# p value 0.4019, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$FairRankIndiv.x, combined$FairRankIndiv.y, paired = TRUE) 
# p value 0.9874, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$FairRankGroup.x, combined$FairRankGroup.y, paired = TRUE) 
# p value 0.2645, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$RuntimeRank.x, combined$RuntimeRank.y, paired = TRUE) 
# p value 0.3461, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$PerfRank.x, combined$PerfRank.y, paired = TRUE) 
# p value 0.7529, meaning that there's no systematic positive or negative change in ranks
wilcox.test(combined$PerfFairEqualRank.x, combined$PerfFairEqualRank.y, paired = TRUE) 
# p value 0.7694, meaning that there's no systematic positive or negative change in ranks


# add some stats for comparing combination of approaches
combined$Combo <- as.character(combined$Combo)

#### sampled results ####

df <- read.csv(paste0(getwd(), "/fairCombos.csv"), stringsAsFactors = T)
df <- sampleNObs(df, 5, c("Combo", "Dataset", "Sens_Attr"))
combined <- deriveEqualRank(aggregateResults(df))
combined <- combined[order(combined$PerfFairEqualRank), ]

#### tidy up and rebuild ####

combined$Combo <- as.character(combined$Combo)
combined$Pre = ""
combined$In = ""
combined$Post = ""
combined$Classifier = ""

for (i in 1:nrow(combined)){
  approach <- strsplit(combined$Combo[i], split='\\+')
  combined$Pre[i] = approach[[1]][1]
  combined$In[i] = approach[[1]][2]
  combined$Post[i] = approach[[1]][3]
  combined$Classifier[i] = approach[[1]][4]  
}

combined$Combo <- NULL
combined$Classifier[combined$Classifier == "Logistic Regression"] <- "LR"
combined$Classifier[combined$Classifier == "Naive Bayes"] <- "NB"
combined$Classifier[combined$Classifier == "Random Forest"] <- "RF"

combined <- combined[, c(8, 9, 11, 10, 1:7)]
View(combined)

row.names(combined) <- NULL
print(xtable(combined), file=paste0(getwd(), "/results.tex"), compress=F)

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
combined_new$Scenario <- factor(combined_new$Scenario, 
                                levels=c('only_classification', 'only_pre', 'only_in', 'only_post', 
                                         'pre_in', 'pre_post', 'in_post', 'pre_in_post'),
                                labels=c('only_classification', 'only_pre', 'only_in', 'only_post', 
                                         'pre_in', 'pre_post', 'in_post', 'pre_in_post'),
                                ordered=T)

#### plots ####

ggplot(combined_new, aes(x=Scenario,y=EqualWeightRank, fill=Scenario)) + geom_boxplot() + ggtitle("Mean Weighted Rank: Perf + Fair + Runtime")
ggplot(combined_new, aes(x=Scenario,y=PerfFairEqualRank, fill=Scenario)) + geom_boxplot() + ggtitle("Mean Weighted Rank: Perf + Fair")
ggplot(combined_new, aes(x=Scenario,y=PerfRank, fill=Scenario)) + geom_boxplot() + ggtitle("Mean Rank: Perf")
ggplot(combined_new, aes(x=Scenario,y=FairRank, fill=Scenario)) + geom_boxplot() + ggtitle("Mean Rank: Fair")
ggplot(combined_new, aes(x=Scenario,y=RuntimeRank, fill=Scenario)) + geom_boxplot() + ggtitle("Mean Rank: Runtime")
